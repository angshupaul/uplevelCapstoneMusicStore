"""Tests for the identity verification flow in verification.py.

The LLM is replaced with simple fakes, so these run offline and cost nothing.
The database is the real in-memory Chinook instance.

Run from the project root with either:
    python TestingTool/verification_test.py
    python -m pytest TestingTool/verification_test.py -v
"""

import json
import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Make the project root importable when this file is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402
from langgraph.types import Command  # noqa: E402

import verification as V  # noqa: E402
from tools import get_customer_invoices, get_invoice_line_items  # noqa: E402
from verification import UserInput  # noqa: E402
from fakes import FakeChat, FakeExtractor  # noqa: E402

# Customers 14 and 5 in Chinook.
MARK = {"id": 14, "email": "mphilips12@shaw.ca", "phone": "+1 (780) 434-4554"}
FRANTISEK = {"id": 5, "email": "frantisekw@jetbrains.com", "phone": "+420 2 4172 5555"}


@pytest.fixture
def fake_chat(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(V, "get_extractor", FakeExtractor)
    monkeypatch.setattr(V, "get_llm", lambda: chat)
    return chat


@pytest.fixture
def downstream():
    """States received by the "next" node, standing in for a downstream agent."""
    return []


@pytest.fixture
def graph(fake_chat, downstream):
    def next_agent(state):
        downstream.append(state)
        return {}

    builder = StateGraph(V.VerificationState)
    builder.add_node("next", next_agent)
    builder.add_edge("next", END)
    V.add_verification_flow(builder, next_node="next")
    return builder.compile(checkpointer=MemorySaver())


def config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def found(identifier: str, kind: str):
    return V.find_customer(UserInput(identifier=identifier, identifier_type=kind))


def assert_verified(out: dict, customer_id: int) -> None:
    """The run finished verified: ID in state and in exactly one SystemMessage."""
    assert "__interrupt__" not in out
    assert out["customer_id"] == customer_id
    system = [m for m in out["messages"] if isinstance(m, SystemMessage)]
    assert len(system) == 1
    assert f"CustomerId={customer_id}" in system[0].content


# --- Extraction guard ------------------------------------------------------

@pytest.mark.parametrize(
    "returned, message, expected",
    [
        (("5", "customer_id"), "my id is 5", "5"),
        (("5", "customer_id"), "my id is 15", None),  # not a whole-number match
        (("007", "customer_id"), "id 007", "7"),
        (("A@B.com", "email"), "mail a@b.com", "A@B.com"),
        (("x@y.com", "email"), "no email here", None),  # invented by the LLM
        (("+1 (780) 434-4554", "phone"), "call 1 780 434 4554", "+1 (780) 434-4554"),
        (("123", "phone"), "123", None),  # too short for a phone number
        (("abc", "customer_id"), "abc", None),
        ((None, None), "hello", None),
    ],
)
def test_extraction_guard(returned, message, expected):
    result = V._validated(UserInput(identifier=returned[0], identifier_type=returned[1]), message)
    assert result.identifier == expected


# --- Database lookup -------------------------------------------------------

def test_lookup_by_customer_id():
    assert found("14", "customer_id").customer_id == MARK["id"]


def test_lookup_by_email_ignores_case():
    assert found(MARK["email"].upper(), "email").customer_id == MARK["id"]


def test_lookup_by_email_with_non_ascii_letters():
    # SQLite's LOWER() only folds A-Z; the input is lowercased in Python.
    assert found("STANISŁAW.WÓJCIK@WP.PL", "email").customer_id == 49


@pytest.mark.parametrize("phone", ["+1 (780) 434-4554", "17804344554", "(780) 434-4554", "780.434.4554"])
def test_lookup_by_phone_in_any_format(phone):
    assert found(phone, "phone").customer_id == MARK["id"]


@pytest.mark.parametrize(
    "identifier, kind",
    [("999", "customer_id"), ("nobody@x.com", "email"), ("434-4554", "phone"),
     ("x' OR '1'='1", "email"), ("", "email")],
)
def test_lookup_not_found(identifier, kind):
    assert found(identifier, kind) is None


# --- Verification node -----------------------------------------------------

def test_node_sets_customer_id_and_system_message(fake_chat):
    result = V.verify_customer({"messages": [HumanMessage(f"I'm {MARK['email']}")]})
    assert result["customer_id"] == MARK["id"]
    [message] = result["messages"]
    assert isinstance(message, SystemMessage)
    assert "CustomerId=14" in message.content


def test_node_skips_when_already_verified(fake_chat):
    state = {"messages": [HumanMessage("I'm customer 3")], "customer_id": MARK["id"]}
    assert V.verify_customer(state) == {}


def test_opening_message_without_identifier_is_not_an_attempt(fake_chat):
    assert V.verify_customer({"messages": [HumanMessage("show my invoices")]}) == {}


def test_wrong_identifier_counts_as_attempt(fake_chat):
    result = V.verify_customer({"messages": [HumanMessage("my id is 999")]})
    assert result == {"verification_attempts": 1}


def test_reply_after_being_asked_counts_as_attempt(fake_chat):
    state = {"messages": [HumanMessage("hi"), AIMessage("Your ID?"), HumanMessage("no idea")]}
    assert V.verify_customer(state) == {"verification_attempts": 1}


# --- Routing ---------------------------------------------------------------

@pytest.mark.parametrize(
    "state, route",
    [
        ({"customer_id": 14, "verification_attempts": 3}, "next"),
        ({"verification_attempts": 0}, "ask_for_identifier"),
        ({}, "ask_for_identifier"),
        ({"customer_id": None, "verification_attempts": 2}, "ask_for_identifier"),
        ({"verification_attempts": 3}, "verification_failed"),
    ],
)
def test_route_after_verification(state, route):
    assert V.route_after_verification(state) == route


# --- Full graph: interrupt and resume ---------------------------------------

def test_identifier_in_first_message_needs_no_interrupt(graph):
    out = graph.invoke({"messages": [HumanMessage("customer id 14")]}, config("a"))
    assert "__interrupt__" not in out
    assert out["customer_id"] == MARK["id"]


def test_pause_then_resume_until_verified(graph, fake_chat):
    cfg = config("b")
    out = graph.invoke({"messages": [HumanMessage("show my invoices")]}, cfg)
    assert out["__interrupt__"][0].value["question"].startswith("Could you share")

    out = graph.invoke(Command(resume="nobody@nowhere.com"), cfg)
    assert "__interrupt__" in out
    assert out.get("customer_id") is None

    out = graph.invoke(Command(resume=f"my phone is {MARK['phone']}"), cfg)
    assert out["customer_id"] == MARK["id"]
    # One fallback LLM call per question: resuming never repeats it.
    assert len(fake_chat.calls) == 2


def test_gives_up_after_max_attempts(graph):
    cfg = config("c")
    out = graph.invoke({"messages": [HumanMessage("hi")]}, cfg)
    for reply in ["999", "no idea", "still no"]:
        assert "__interrupt__" in out
        out = graph.invoke(Command(resume=reply), cfg)

    assert "__interrupt__" not in out
    assert out.get("customer_id") is None
    assert out["verification_attempts"] == V.MAX_VERIFICATION_ATTEMPTS
    assert out["messages"][-1].content == V.VERIFICATION_FAILED_MESSAGE


# --- Requirements: end-to-end through the graph -----------------------------

# 1. A valid customer ID verifies and injects the ID into state.
@pytest.mark.parametrize("message", ["5", "my customer ID is 5", "Customer #5, hi!"])
def test_req1_valid_customer_id_verifies(graph, message):
    out = graph.invoke({"messages": [HumanMessage(message)]}, config("req1"))
    assert_verified(out, FRANTISEK["id"])


# 2. A valid email verifies regardless of case.
@pytest.mark.parametrize(
    "email",
    ["frantisekw@jetbrains.com", "FRANTISEKW@JETBRAINS.COM", "FrantisekW@JetBrains.com"],
)
def test_req2_valid_email_verifies_case_insensitively(graph, email):
    out = graph.invoke({"messages": [HumanMessage(f"my email is {email}")]}, config("req2"))
    assert_verified(out, FRANTISEK["id"])


# 3. A valid phone number verifies in any formatting, after normalization.
@pytest.mark.parametrize(
    "phone",
    ["+420 2 4172 5555", "+420241725555", "420-2-4172-5555", "+420.2.4172.5555", "(420) 2 4172 5555"],
)
def test_req3_valid_phone_verifies_in_any_format(graph, phone):
    out = graph.invoke({"messages": [HumanMessage(f"call me on {phone}")]}, config("req3"))
    assert_verified(out, FRANTISEK["id"])


# 4. An invalid ID pauses the graph and asks the user to try again.
def test_req4_invalid_id_interrupts_and_asks_again(graph, fake_chat, downstream):
    cfg = config("req4")
    out = graph.invoke({"messages": [HumanMessage("my customer ID is 99999")]}, cfg)

    assert "__interrupt__" in out
    assert out.get("customer_id") is None
    assert out["verification_attempts"] == 1
    assert downstream == []  # no agent runs for an unverified user

    # The question shown to the user is the fallback reply, built from the
    # fallback prompt that asks for any of the three identifiers.
    question = out["__interrupt__"][0].value["question"]
    assert question == out["messages"][-1].content
    fallback_prompt = fake_chat.calls[-1][-1]
    assert isinstance(fallback_prompt, SystemMessage)
    for option in ("Customer ID", "email", "phone number"):
        assert option in fallback_prompt.content

    # The user's next reply re-runs verification.
    out = graph.invoke(Command(resume="sorry, it's 5"), cfg)
    assert_verified(out, FRANTISEK["id"])


# 5. After verification, downstream agents receive customer_id in state.
def test_req5_downstream_agent_receives_customer_id(graph, downstream):
    cfg = config("req5")
    graph.invoke({"messages": [HumanMessage("what did I buy?")]}, cfg)
    assert downstream == []  # still waiting for an identifier

    graph.invoke(Command(resume=FRANTISEK["email"]), cfg)

    [state] = downstream
    assert state["customer_id"] == FRANTISEK["id"]
    assert any(
        isinstance(m, SystemMessage) and f"CustomerId={FRANTISEK['id']}" in m.content
        for m in state["messages"]
    )


# 6. The ID is injected via a SystemMessage, never in user-facing text.
def test_req6_customer_id_only_in_system_message(graph):
    cfg = config("req6")
    first = graph.invoke({"messages": [HumanMessage("hi")]}, cfg)
    out = graph.invoke(Command(resume=FRANTISEK["phone"]), cfg)
    assert_verified(out, FRANTISEK["id"])

    # The verification adds exactly one new message, and it is a SystemMessage.
    verified_turn = out["messages"][len(first["messages"]) + 1:]
    assert [type(m) for m in verified_turn] == [SystemMessage]

    # Nothing shown to the user (AI replies, interrupt questions) carries the ID.
    user_facing = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    user_facing.append(first["__interrupt__"][0].value["question"])
    for text in user_facing:
        assert "CustomerId" not in text
        assert not re.search(rf"\b{FRANTISEK['id']}\b", text)


# --- Tools only see the verified customer -----------------------------------

def test_customer_id_hidden_from_tool_schema():
    assert "customer_id" not in get_customer_invoices.tool_call_schema.model_json_schema()["properties"]
    assert "customer_id" not in get_invoice_line_items.tool_call_schema.model_json_schema()["properties"]


def test_tool_uses_state_customer_not_llm_argument():
    builder = StateGraph(V.VerificationState)
    builder.add_node("tools", ToolNode([get_customer_invoices, get_invoice_line_items]))
    builder.set_entry_point("tools")
    tools_graph = builder.compile()

    llm_call = AIMessage("", tool_calls=[
        {"name": "get_customer_invoices", "args": {"customer_id": "5"}, "id": "1"},
        {"name": "get_invoice_line_items", "args": {"invoice_id": "90"}, "id": "2"},
    ])
    out = tools_graph.invoke({"messages": [llm_call], "customer_id": MARK["id"]})
    invoices, line_items = (json.loads(m.content) for m in out["messages"][1:])

    assert invoices["CustomerId"] == MARK["id"]  # not customer 5
    assert "No invoice found" in line_items["message"]  # invoice 90 is customer 5's


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
