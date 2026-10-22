"""Tests for the supervisor in supervisor.py.

Everything runs offline: the router and supervisor models are scripted fakes,
and the specialists are small fake agents that record when they run. One
integration test uses the real invoice agent (with a scripted model) to show
the verified customer_id reaches its tools through the supervisor.

The supervisor model is often scripted to misbehave (wrong agent, wrong order,
stopping early, delegating again) to prove routing is enforced in code, not
just requested in the prompt.

Run from the project root with either:
    python TestingTool/supervisor_test.py
    python -m pytest TestingTool/supervisor_test.py -v
"""

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import supervisor as S  # noqa: E402
from fakes import ScriptedChatModel, asks_for, call  # noqa: E402
from invoice_agent import build_invoice_agent  # noqa: E402
from verification import VerifiedCustomer, verified_message  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:create_react_agent has been moved")

MARK = 14
MUSIC_ANSWER = "Showing 10 of 45 Queen tracks."
INVOICE_ANSWER = "Your last invoice was 2025-05-11 for $13.86."

TO_MUSIC = S._handoff_tool(S.MUSIC_AGENT)
TO_INVOICE = S._handoff_tool(S.INVOICE_AGENT)


class FakeRouter:
    """Stands in for the structured-output router; records each call."""

    def __init__(self, needs_invoice: bool, needs_music: bool):
        self.plan = S.RoutePlan(needs_invoice=needs_invoice, needs_music=needs_music)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.plan


class AgentState(MessagesState):
    customer_id: int | None


def fake_agent(name: str, answer: str, log: list):
    """A named specialist that records the state it receives and answers."""

    def respond(state):
        log.append((name, state))
        return {"messages": [AIMessage(answer, name=name)]}

    builder = StateGraph(AgentState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    return builder.compile(name=name)


def run(question: str, router: FakeRouter, *replies: AIMessage, history=(), invoice_agent=None):
    """Run the supervisor for one request; returns (final state, agent log, model)."""
    log = []
    model = ScriptedChatModel(replies=list(replies))
    graph = S.build_supervisor(
        model=model,
        router=router,
        music_agent=fake_agent(S.MUSIC_AGENT, MUSIC_ANSWER, log),
        invoice_agent=invoice_agent or fake_agent(S.INVOICE_AGENT, INVOICE_ANSWER, log),
    )
    state = {
        "messages": [
            verified_message(VerifiedCustomer(MARK, "Mark")),
            *history,
            HumanMessage(question),
        ],
        "customer_id": MARK,
    }
    out = graph.invoke(state)
    assert not model.replies, "the supervisor stopped before using every scripted reply"
    return out, log, model


def agents_run(log: list) -> list[str]:
    return [name for name, _ in log]


def handoff(agent_tool: str, call_id: str) -> AIMessage:
    return asks_for(call(agent_tool, call_id=call_id))


# --- Prompts and plan ------------------------------------------------------

@pytest.mark.parametrize(
    "rule",
    [
        "Tool-only responses",
        "Exact number quoting",
        "Honest failures",
        "Scope boundaries",
        "No memory-based answers",
        "Truncation transparency",
    ],
)
def test_supervisor_prompt_has_every_grounding_rule(rule):
    assert rule in S.SUPERVISOR_PROMPT


def test_supervisor_prompt_states_the_routing_rules():
    prompt = " ".join(S.SUPERVISOR_PROMPT.split())
    assert f"Catalog or music questions go to {S.MUSIC_AGENT}" in prompt
    assert f"Invoice, billing or purchase questions go to {S.INVOICE_AGENT}" in prompt
    assert f"Mixed requests go to {S.INVOICE_AGENT} first and {S.MUSIC_AGENT} second" in prompt
    assert "Off-topic requests are answered directly without delegating" in prompt


def test_router_prompt_treats_taste_statements_as_music():
    prompt = " ".join(S.ROUTER_PROMPT.split())
    assert 'A customer stating their music taste counts as needs_music, e.g. "I love jazz"' in prompt


@pytest.mark.parametrize(
    "needs_invoice, needs_music, order",
    [
        (False, True, [S.MUSIC_AGENT]),
        (True, False, [S.INVOICE_AGENT]),
        (True, True, [S.INVOICE_AGENT, S.MUSIC_AGENT]),  # invoice always first
        (False, False, []),
    ],
)
def test_route_plan_order(needs_invoice, needs_music, order):
    plan = S.RoutePlan(needs_invoice=needs_invoice, needs_music=needs_music)
    assert S.route_plan(plan) == order


def test_supervisor_model_gets_only_the_two_handoff_tools():
    _, _, model = run("Any Queen?", FakeRouter(False, True), handoff(TO_MUSIC, "a"), AIMessage(MUSIC_ANSWER))
    assert sorted(t["function"]["name"] for t in model.bound_tools) == sorted([TO_MUSIC, TO_INVOICE])


# --- Music routing ---------------------------------------------------------

def test_music_request_goes_to_music_agent():
    out, log, _ = run(
        "What Queen songs do you have?",
        FakeRouter(needs_invoice=False, needs_music=True),
        handoff(TO_MUSIC, "a"),
        AIMessage(MUSIC_ANSWER),
    )

    assert agents_run(log) == [S.MUSIC_AGENT]
    assert out["messages"][-1].content == MUSIC_ANSWER


def test_music_request_cannot_be_sent_to_invoice_agent():
    # The model picks the wrong agent; the hook redirects to the music agent.
    out, log, _ = run(
        "What Queen songs do you have?",
        FakeRouter(needs_invoice=False, needs_music=True),
        handoff(TO_INVOICE, "a"),
        AIMessage(MUSIC_ANSWER),
    )

    assert agents_run(log) == [S.MUSIC_AGENT]


# --- Invoice routing -------------------------------------------------------

def test_invoice_request_goes_to_invoice_agent_with_verified_id():
    out, log, _ = run(
        "When was my last order?",
        FakeRouter(needs_invoice=True, needs_music=False),
        handoff(TO_INVOICE, "a"),
        AIMessage(INVOICE_ANSWER),
    )

    assert agents_run(log) == [S.INVOICE_AGENT]
    [(_, received)] = log
    assert received["customer_id"] == MARK
    assert out["messages"][-1].content == INVOICE_ANSWER


def test_invoice_request_reaches_real_invoice_tools_for_verified_customer():
    invoice_model = ScriptedChatModel(replies=[
        asks_for(call("get_customer_invoices", {"customer_id": "5"})),  # tries another customer
        AIMessage("You have 7 invoices."),
    ])
    out, _, _ = run(
        "How many invoices do I have?",
        FakeRouter(needs_invoice=True, needs_music=False),
        handoff(TO_INVOICE, "a"),
        AIMessage("You have 7 invoices."),
        invoice_agent=build_invoice_agent(invoice_model),
    )

    # The invoice agent's tool result is in its own run, not in the
    # supervisor's history (output_mode="last_message"), so check what its
    # model was sent.
    tool_result = next(
        m for m in invoice_model.calls[1]
        if isinstance(m, ToolMessage) and m.name == "get_customer_invoices"
    )
    assert json.loads(tool_result.content)["CustomerId"] == MARK
    assert out["messages"][-1].content == "You have 7 invoices."


# --- Mixed requests: invoice first, then music -------------------------------

def test_mixed_request_runs_invoice_then_music():
    out, log, model = run(
        "What did I buy last, and what other albums does that artist have?",
        FakeRouter(needs_invoice=True, needs_music=True),
        handoff(TO_INVOICE, "a"),
        handoff(TO_MUSIC, "b"),
        AIMessage(f"{INVOICE_ANSWER} {MUSIC_ANSWER}"),
    )

    assert agents_run(log) == [S.INVOICE_AGENT, S.MUSIC_AGENT]
    # The music agent sees the invoice agent's answer (e.g. which artist).
    _, music_state = log[1]
    assert any(m.content == INVOICE_ANSWER for m in music_state["messages"])
    # The final reply is written after both answers are in.
    assert {INVOICE_ANSWER, MUSIC_ANSWER} <= {m.content for m in model.calls[-1]}
    assert out["messages"][-1].content == f"{INVOICE_ANSWER} {MUSIC_ANSWER}"


def test_mixed_request_order_is_enforced_when_model_starts_with_music():
    _, log, _ = run(
        "Recommend jazz and show my invoices",
        FakeRouter(needs_invoice=True, needs_music=True),
        handoff(TO_MUSIC, "a"),  # wrong order: rewritten to invoice
        handoff(TO_MUSIC, "b"),
        AIMessage("Combined."),
    )

    assert agents_run(log) == [S.INVOICE_AGENT, S.MUSIC_AGENT]


def test_mixed_request_cannot_stop_after_one_agent():
    _, log, _ = run(
        "Recommend jazz and show my invoices",
        FakeRouter(needs_invoice=True, needs_music=True),
        handoff(TO_INVOICE, "a"),
        AIMessage("Here are your invoices."),  # tries to finish early: forced on to music
        AIMessage("Combined."),
    )

    assert agents_run(log) == [S.INVOICE_AGENT, S.MUSIC_AGENT]


def test_mixed_request_never_runs_an_agent_twice():
    out, log, _ = run(
        "Recommend jazz and show my invoices",
        FakeRouter(needs_invoice=True, needs_music=True),
        handoff(TO_INVOICE, "a"),
        handoff(TO_MUSIC, "b"),
        handoff(TO_INVOICE, "c"),  # tries to delegate again: removed
    )

    assert agents_run(log) == [S.INVOICE_AGENT, S.MUSIC_AGENT]
    # The fallback reply is the agents' own answers, invoice first.
    final = out["messages"][-1]
    assert not final.tool_calls
    assert final.content == f"{INVOICE_ANSWER}\n\n{MUSIC_ANSWER}"


# --- Off-topic: rejected without any agent ------------------------------------

@pytest.mark.parametrize("question", ["What's the weather in Paris?", "Write me a Python script", "hi"])
def test_off_topic_is_rejected_without_any_agent(question):
    router = FakeRouter(needs_invoice=False, needs_music=False)
    out, log, model = run(question, router)  # no supervisor replies scripted

    assert log == []  # neither agent ran
    assert model.calls == []  # the supervisor model was never called
    assert len(router.calls) == 1
    assert out["messages"][-1].content == S.OFF_TOPIC_MESSAGE
    assert out["route"] == []


# --- Follow-up requests ----------------------------------------------------

def test_earlier_request_does_not_count_as_done():
    # A previous turn already used the music agent; the new request must run it again.
    previous_turn = [
        HumanMessage("Any Queen?"),
        AIMessage("", tool_calls=[call(TO_MUSIC, call_id="old")]),
        ToolMessage("Successfully transferred", name=TO_MUSIC, tool_call_id="old"),
        AIMessage("Old answer.", name=S.MUSIC_AGENT),
    ]
    _, log, _ = run(
        "And some jazz?",
        FakeRouter(needs_invoice=False, needs_music=True),
        handoff(TO_MUSIC, "a"),
        AIMessage(MUSIC_ANSWER),
        history=previous_turn,
    )

    assert agents_run(log) == [S.MUSIC_AGENT]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
