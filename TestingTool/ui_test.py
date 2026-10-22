"""Tests for the web chat: chat_service.py and the Gradio handlers in app.py.

Everything runs offline: the graph is the real app graph with the fakes from
app_test.py (keyword router, always-handoff supervisor model, recording fake
agents, regex verification extractor, fake preference extractor).

Run from the project root with either:
    python TestingTool/ui_test.py
    python -m pytest TestingTool/ui_test.py -v
"""

import sys
import uuid
from pathlib import Path

import gradio as gr
import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import START, StateGraph

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app as A  # noqa: E402
import chat_service as C  # noqa: E402
import memory  # noqa: E402
import supervisor as S  # noqa: E402
import verification as V  # noqa: E402
from app_test import (  # noqa: E402
    INVOICE_ANSWER,
    MUSIC_ANSWER,
    AgentState,
    HandoffModel,
    KeywordRouter,
    fake_agent,
)
from chat_service import Done, Progress, Reply, TurnResult  # noqa: E402
from fakes import FakeChat, FakeExtractor, FakePreferenceExtractor  # noqa: E402
from graph import build_app  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:create_react_agent has been moved")

QUESTION = FakeChat().invoke([]).content  # what the fake verification asks
VERIFIED_MUSIC = "I'm customer 14. What Queen songs do you have?"
PHONE = "my phone is +1 (780) 434-4554"


@pytest.fixture(autouse=True)
def fake_llms(monkeypatch):
    monkeypatch.setattr(V, "get_extractor", FakeExtractor)
    monkeypatch.setattr(V, "get_llm", lambda: FakeChat())
    monkeypatch.setattr(memory, "get_preference_extractor", lambda: FakePreferenceExtractor())


class Agents:
    """Fake specialists; set ``fail_next`` to make the next call raise."""

    def __init__(self):
        self.log = []
        self.fail_next = False

    def music(self):
        def respond(state):
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("LLM timed out")
            self.log.append((S.MUSIC_AGENT, state))
            return {"messages": [AIMessage(MUSIC_ANSWER, name=S.MUSIC_AGENT)]}

        builder = StateGraph(AgentState)
        builder.add_node("respond", respond)
        builder.add_edge(START, "respond")
        return builder.compile(name=S.MUSIC_AGENT)


@pytest.fixture
def agents():
    return Agents()


@pytest.fixture
def graph(agents):
    supervisor = S.build_supervisor(
        model=HandoffModel(),
        router=KeywordRouter(),
        music_agent=agents.music(),
        invoice_agent=fake_agent(S.INVOICE_AGENT, INVOICE_ANSWER, agents.log),
    )
    return build_app(supervisor=supervisor)


def thread() -> str:
    return f"t-{uuid.uuid4().hex}"


def snapshot(graph, thread_id: str):
    return graph.get_state({"configurable": {"thread_id": thread_id}})


# --- chat_service: one turn ----------------------------------------------------

def test_verified_request_returns_only_the_final_reply(graph):
    result = C.run_turn(graph, thread(), VERIFIED_MUSIC)

    assert result.status == "success"
    assert result.reply == MUSIC_ANSWER
    assert result.error is None
    assert result.elapsed_seconds >= 0


def test_reply_arrives_before_preferences_are_saved(graph):
    events = list(C.stream_turn(graph, thread(), VERIFIED_MUSIC))
    kinds = [type(e).__name__ if not isinstance(e, Progress) else e.stage for e in events]

    assert kinds.index("Reply") < kinds.index("Saving your preferences…")
    assert isinstance(events[-1], Done)


def test_interrupt_then_resume_with_command(graph):
    t = thread()
    waiting = C.run_turn(graph, t, "Show my invoices")
    assert (waiting.status, waiting.reply) == ("waiting", QUESTION)
    assert C.pending_question(snapshot(graph, t)) == QUESTION

    answered = C.run_turn(graph, t, PHONE)  # resumes the paused graph
    assert (answered.status, answered.reply) == ("success", INVOICE_ANSWER)
    assert C.pending_question(snapshot(graph, t)) is None


def test_three_failures_end_the_conversation_for_good(graph):
    t = thread()
    statuses = [C.run_turn(graph, t, text).status for text in ["hi", "99991", "99992", "99993"]]
    assert statuses == ["waiting", "waiting", "waiting", "ended"]

    later = C.run_turn(graph, t, "I'm customer 14")
    assert later.status == "ended"
    assert later.reply == V.VERIFICATION_FAILED_MESSAGE


@pytest.mark.parametrize("text", ["", "   ", None])
def test_blank_message_is_ignored(graph, text):
    t = thread()
    assert C.run_turn(graph, t, text).status == "ignored"
    assert snapshot(graph, t).values == {}  # the graph never ran


def test_error_is_friendly_and_the_next_turn_starts_fresh(graph, agents):
    t = thread()
    C.run_turn(graph, t, "I'm customer 14. Show my invoices")  # verified
    agents.fail_next = True

    failed = C.run_turn(graph, t, "What Queen songs do you have?")
    assert failed.status == "error"
    assert failed.reply == C.ERROR_MESSAGE
    assert "LLM timed out" in failed.error
    # The crash left pending work but no question: the next message must
    # start a new turn, not be sent as a resume value.
    state = snapshot(graph, t)
    assert state.next and not state.interrupts
    assert C.pending_question(state) is None

    retried = C.run_turn(graph, t, "What Queen songs do you have?")
    assert (retried.status, retried.reply) == ("success", MUSIC_ANSWER)


def test_error_in_the_graph_itself_never_raises():
    class Broken:
        def get_state(self, config):
            raise RuntimeError("checkpointer down")

    result = C.run_turn(Broken(), thread(), "hello")
    assert result.status == "error"
    assert "checkpointer down" in result.error


# --- chat_service: sessions ------------------------------------------------------

def test_sessions_are_isolated(graph, agents):
    s1, s2 = thread(), thread()
    C.run_turn(graph, s1, VERIFIED_MUSIC)  # verified, answered
    waiting = C.run_turn(graph, s2, "What Queen songs do you have?")  # unverified

    assert waiting.status == "waiting"  # s1's verification doesn't carry over
    assert snapshot(graph, s1).values["customer_id"] == 14
    assert snapshot(graph, s2).values.get("customer_id") is None
    s2_text = " ".join(m.text for m in snapshot(graph, s2).values["messages"])
    assert MUSIC_ANSWER not in s2_text and "customer 14" not in s2_text
    assert C.pending_question(snapshot(graph, s1)) is None  # s2's pause isn't s1's


def test_new_thread_ids_are_unique():
    ids = {C.new_thread_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(uuid.UUID(i) for i in ids)


def test_forget_thread_deletes_only_that_conversation(graph):
    gone, kept = thread(), thread()
    C.run_turn(graph, gone, "Show my invoices")
    C.run_turn(graph, kept, "Show my invoices")

    C.forget_thread(graph, gone)

    assert snapshot(graph, gone).values == {}
    assert C.pending_question(snapshot(graph, kept)) == QUESTION
    C.forget_thread(graph, "never-existed")  # harmless
    C.forget_thread(object(), "x")  # no checkpointer: harmless


# --- app.py handlers -------------------------------------------------------------

def test_add_user_message_echoes_at_once_and_locks_the_input():
    history, box, pending, send, reset, status = A.add_user_message("  Hello  ", [])

    assert history == [{"role": "user", "content": "Hello"}]
    assert (box.value, box.interactive) == ("", False)
    assert pending == "Hello"
    assert send.interactive is False and reset.interactive is False
    assert "status-processing" in status


def test_add_user_message_ignores_blank_input():
    history = [{"role": "user", "content": "earlier"}]
    out = A.add_user_message("   ", history)
    assert out[0] == history
    assert out[2] == ""
    assert all(x == gr.skip() for x in (out[1], out[3], out[4], out[5]))


def run_respond(graph, thread_id, text, history=()):
    step1 = A.add_user_message(text, list(history))
    outputs = list(A.respond(graph, thread_id, step1[2], step1[0]))
    return outputs, outputs[-1]


def test_respond_streams_progress_then_reply_then_final_status(graph):
    outputs, last = run_respond(graph, thread(), VERIFIED_MUSIC)

    statuses = [o[1] for o in outputs if isinstance(o[1], str)]
    assert any("status-processing" in s for s in statuses)
    assert last[0][-1] == {"role": "assistant", "content": MUSIC_ANSWER}
    assert "status-success" in last[1] and "Answered in" in last[1]
    box, send, reset = last[2:]
    assert (box.interactive, box.placeholder) == (True, A.MESSAGE_PLACEHOLDER)
    assert send.interactive and reset.interactive


def test_respond_shows_waiting_then_ended_states(graph):
    t, history = thread(), []
    _, last = run_respond(graph, t, "hi")
    assert "status-waiting" in last[1]
    assert last[2].placeholder == A.WAITING_PLACEHOLDER

    for text in ["99991", "99992", "99993"]:
        _, last = run_respond(graph, t, text, last[0])
    assert "status-ended" in last[1]
    box, send, reset = last[2:]
    assert box.interactive is False and box.placeholder == A.ENDED_PLACEHOLDER
    assert send.interactive is False
    assert reset.interactive is True  # the only way forward


def test_a_second_reply_replaces_the_first(monkeypatch):
    def two_replies(graph, thread_id, text):
        yield Reply("first draft")
        yield Reply("final answer")
        yield Done(TurnResult("final answer", "success", 1.0))

    monkeypatch.setattr(A, "stream_turn", two_replies)
    *_, last = A.respond(None, "t", "hi", [{"role": "user", "content": "hi"}])

    assert last[0] == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "final answer"}]


def test_error_without_a_reply_still_shows_the_friendly_message(monkeypatch):
    def failing(graph, thread_id, text):
        yield Progress("Checking your identity…")
        yield Done(TurnResult(C.ERROR_MESSAGE, "error", 0.1, error="RuntimeError: boom"))

    monkeypatch.setattr(A, "stream_turn", failing)
    *_, last = A.respond(None, "t", "hi", [])

    assert last[0] == [{"role": "assistant", "content": C.ERROR_MESSAGE}]
    assert "status-error" in last[1] and "boom" not in last[1]  # no technical detail shown


def test_reset_gives_a_new_thread_and_a_clean_page(graph):
    old = thread()
    for text in ["hi", "99991", "99992", "99993"]:
        C.run_turn(graph, old, text)  # locked out

    new_id, chat, box, send, pending, status = A.reset_conversation(graph, old)

    assert new_id != old and uuid.UUID(new_id)
    assert chat == [] and pending == ""
    assert (box.value, box.interactive, box.placeholder) == ("", True, A.MESSAGE_PLACEHOLDER)
    assert send.interactive is True
    assert "status-idle" in status
    assert snapshot(graph, old).values == {}  # old conversation deleted


# --- Status badges -----------------------------------------------------------------

@pytest.mark.parametrize(
    "result, css_class, text",
    [
        (TurnResult("x", "success", 3.24), "status-success", "Answered in 3.2 s"),
        (TurnResult("x", "waiting", 0.1), "status-waiting", "Waiting for your input"),
        (TurnResult("x", "ended", 0.1), "status-ended", "Conversation ended"),
        (TurnResult("x", "error", 0.1), "status-error", "Something went wrong"),
        (TurnResult("", "ignored", 0.0), "status-idle", "Ready."),
    ],
)
def test_final_status_badges(result, css_class, text):
    badge = A.final_status(result)
    assert css_class in badge and text in badge
    assert 'role="status"' in badge and 'aria-live="polite"' in badge


def test_badges_have_icons_not_just_colour():
    assert "status-spinner" in A.status_html("processing", "Working…")
    for state in ("success", "waiting", "error", "ended"):
        assert "status-icon" in A.status_html(state, "x")


def test_status_text_is_escaped():
    badge = A.status_html("error", "<img src=x onerror=alert(1)>")
    assert "<img" not in badge and "&lt;img" in badge


def test_css_covers_every_state_and_reduced_motion():
    for state in ("idle", "processing", "success", "waiting", "error", "ended"):
        assert f".status-{state}" in A.APP_CSS
    assert "prefers-reduced-motion" in A.APP_CSS


# --- Page --------------------------------------------------------------------------

def test_build_ui_does_not_build_the_graph():
    C.get_graph.cache_clear()
    demo = A.build_ui()
    assert C.get_graph.cache_info().currsize == 0
    assert set(demo.components) >= {"thread_id", "chatbot", "message", "send", "new_conversation", "status"}


def test_each_session_gets_its_own_thread_id():
    state = A.build_ui().components["thread_id"]
    assert callable(state.value)  # Gradio calls it per browser session
    assert len({state.value() for _ in range(5)}) == 5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
