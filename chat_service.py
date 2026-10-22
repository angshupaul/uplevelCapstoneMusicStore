"""Chat service: runs one chat turn on the app graph, independent of any UI.

The Gradio app (app.py) calls this layer, so all graph logic can be tested
without a browser.

- Every browser session has its own thread_id (new_thread_id). It goes in
  the graph config, so the checkpointer keeps each session's history apart.
  One compiled graph (get_graph) is shared by all sessions.
- stream_turn decides between resuming a paused graph (Command(resume=...))
  and sending a new message, streams progress, and yields the reply as soon
  as it exists: before create_memory saves preferences.
- Only the final reply is ever returned: never system or tool messages,
  handoffs, the customer_id or memory.
"""

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, StateSnapshot

from graph import build_app
from verification import MAX_VERIFICATION_ATTEMPTS

logger = logging.getLogger(__name__)

Status = Literal["success", "waiting", "ended", "error", "ignored"]

ERROR_MESSAGE = "Sorry, something went wrong on our side. Please try again in a moment."

# Shown while the graph works; each one describes the step that comes next.
FIRST_STAGE = "Checking your identity…"
NEXT_STAGE = {
    "verify_info": "Looking that up…",
    "supervisor": "Saving your preferences…",
}


@dataclass(frozen=True)
class TurnResult:
    reply: str
    status: Status
    elapsed_seconds: float
    # Technical detail for the log; never shown to the customer.
    error: str | None = None


@dataclass(frozen=True)
class Progress:
    stage: str


@dataclass(frozen=True)
class Reply:
    text: str


@dataclass(frozen=True)
class Done:
    result: TurnResult


Event = Progress | Reply | Done


def new_thread_id() -> str:
    """A fresh conversation id for one browser session."""
    return str(uuid4())


@lru_cache(maxsize=1)
def get_graph() -> CompiledStateGraph:
    """The app graph, built once per process and shared by every session."""
    return build_app()


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def pending_question(snapshot: StateSnapshot) -> str | None:
    """The question the graph is waiting on, or None if it isn't waiting.

    Checks for an actual interrupt, not just snapshot.next: a turn that
    crashed midway can leave pending work with no question, and the next
    message must then start a new turn rather than be sent as a resume value.
    """
    if not snapshot.next or not snapshot.interrupts:
        return None
    value = snapshot.interrupts[0].value
    return value.get("question", "") if isinstance(value, dict) else str(value)


def final_reply(messages: list) -> str:
    """The latest assistant reply after the customer's latest message."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if isinstance(message, AIMessage) and not message.tool_calls and message.text.strip():
            return message.text.strip()
    return ""


def _is_locked_out(values: dict) -> bool:
    """Verification failed for good in this conversation."""
    return (
        values.get("customer_id") is None
        and values.get("verification_attempts", 0) >= MAX_VERIFICATION_ATTEMPTS
    )


def stream_turn(graph: CompiledStateGraph, thread_id: str, text: str) -> Iterator[Event]:
    """Run one customer message through the graph, yielding events.

    Yields Progress(stage) while working, Reply(text) as soon as the reply
    exists, and always ends with Done(TurnResult). Never raises.
    """
    start = time.perf_counter()
    text = (text or "").strip()
    if not text:
        yield Done(TurnResult(reply="", status="ignored", elapsed_seconds=0.0))
        return

    config = _config(thread_id)
    reply, reply_elapsed = "", None
    try:
        if pending_question(graph.get_state(config)) is not None:
            payload = Command(resume=text)
        else:
            payload = {"messages": [HumanMessage(text)]}

        yield Progress(FIRST_STAGE)
        for update in graph.stream(payload, config, stream_mode="updates"):
            for node, values in update.items():
                if node == "supervisor" and not reply:
                    reply = final_reply((values or {}).get("messages", []))
                    if reply:
                        reply_elapsed = time.perf_counter() - start
                        yield Reply(reply)
                verified = node != "verify_info" or (values or {}).get("customer_id") is not None
                if node in NEXT_STAGE and verified:
                    yield Progress(NEXT_STAGE[node])

        snapshot = graph.get_state(config)
        question = pending_question(snapshot)
        if question is not None:
            status, text_out = "waiting", question
        else:
            status = "ended" if _is_locked_out(snapshot.values) else "success"
            text_out = final_reply(snapshot.values.get("messages", [])) or reply

        if text_out != reply:
            reply_elapsed = time.perf_counter() - start
            yield Reply(text_out)
        elapsed = reply_elapsed if reply_elapsed is not None else time.perf_counter() - start
        yield Done(TurnResult(reply=text_out, status=status, elapsed_seconds=round(elapsed, 2)))

    except Exception as exc:  # the UI must never crash
        logger.exception("Chat turn failed for thread %s", thread_id)
        yield Done(TurnResult(
            reply=ERROR_MESSAGE,
            status="error",
            elapsed_seconds=round(time.perf_counter() - start, 2),
            error=f"{type(exc).__name__}: {exc}",
        ))


def forget_thread(graph: CompiledStateGraph, thread_id: str) -> None:
    """Delete a conversation's checkpoints (e.g. on New conversation).

    Removes the main graph's and any paused subgraph's checkpoints for that
    thread only; other sessions are untouched. Saved music preferences are
    per customer, in the store, and are kept. Never raises: a failure is
    logged and the conversation is simply left behind.
    """
    checkpointer = getattr(graph, "checkpointer", None)
    if not thread_id or checkpointer is None:
        return
    try:
        checkpointer.delete_thread(thread_id)
    except Exception:
        logger.warning("Could not delete conversation %s", thread_id, exc_info=True)


def run_turn(graph: CompiledStateGraph, thread_id: str, text: str) -> TurnResult:
    """Run one turn and return only the result (for tests and scripts)."""
    for event in stream_turn(graph, thread_id, text):
        if isinstance(event, Done):
            return event.result
    raise RuntimeError("stream_turn ended without a result")  # unreachable
