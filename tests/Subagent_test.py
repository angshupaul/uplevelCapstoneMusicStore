"""Live sub-agent evaluation: routing, real data and grounding.

Unlike the other test files, these tests call the real LLM (gpt-4o-mini via
config) against the real in-memory database, because fabrication can only be
judged on genuine model output. They are skipped when no API key is
configured, cost a few cents per run, and, like any LLM evaluation, can vary
between runs.

Each test runs the full application graph (build_app): the customer first
verifies as customer 14, then asks the question. Every tool result is
captured with a callback, so the reply can be checked against what the tools
actually returned:
- every price or amount in the reply must appear in a tool result,
- every album/track title in title position (bold, quoted or starting a list
  item) must appear in a tool result from this turn.

Each test prints its input, the response and PASSED/FAILED.

Run from the project root:
    python -m pytest tests/Subagent_test.py -v
    python -m pytest tests/Subagent_test.py -m live -v -s   # the live tests (real LLM)
"""

import json
import re
import uuid

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

import supervisor as S
from config import ConfigError, get_settings
from database import execute_query
from graph import build_app

try:
    get_settings()
    _NO_LLM = ""
except ConfigError as error:
    _NO_LLM = f"live LLM tests need an API key: {error}"

pytestmark = pytest.mark.filterwarnings("ignore:create_react_agent has been moved")

CUSTOMER_ID = 14

DB_TITLES = {
    row["Title"].lower()
    for table, column in (("Album", "Title"), ("Track", "Name"))
    for row in json.loads(execute_query(f"SELECT {column} AS Title FROM {table}"))["rows"]
}

# Words that mark a bold/quoted span as a label ("Total spent", "Invoice ID
# 362") rather than a title.
_LABEL_WORDS = re.compile(
    r"\b(invoices?|total|spent|amount|date|items?|tracks?|albums?|artists?|genres?|"
    r"price|purchases?|showing|customer|support|rep|page|id|duration|composer|"
    r"songs?|count|billing|order)\b",
    re.IGNORECASE,
)
_MONEY = re.compile(r"\$\s?(\d[\d,]*\.\d+)|\b(\d+\.\d{2})\b")
_BOLD_OR_QUOTED = re.compile(r"\*\*(.+?)\*\*|\"([^\"\n]+)\"|“([^”\n]+)”")
_LIST_HEAD = re.compile(r"^\s*(?:\d+\.|[-*•])\s+(.+)$", re.MULTILINE)
_HEAD_SPLIT = re.compile(r"\s+(?:-|–|—|by|from)\s+|\s*[(:]\s*")


class ToolRecorder(BaseCallbackHandler):
    """Records every specialist tool call (name, input, output) in a run."""

    def __init__(self):
        self.calls = []
        self._names = {}

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self._names[run_id] = ((serialized or {}).get("name") or kwargs.get("name"), input_str)

    def on_tool_end(self, output, *, run_id, **kwargs):
        name, input_str = self._names.pop(run_id, (None, None))
        if name and not name.startswith("transfer_"):
            content = getattr(output, "content", output)
            self.calls.append({"name": name, "input": input_str, "output": str(content)})

    def names(self) -> list[str]:
        return [c["name"] for c in self.calls]

    def results(self) -> list:
        parsed = []
        for c in self.calls:
            try:
                parsed.append(json.loads(c["output"]))
            except ValueError:
                parsed.append(c["output"])
        return parsed


def _walk(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from _walk(v)
    else:
        yield value


def grounding_problems(reply: str, recorder: ToolRecorder) -> list[str]:
    """Claims in the reply that no tool result from this turn supports."""
    values = [v for result in recorder.results() for v in _walk(result)]
    numbers = {f"{float(v):.2f}" for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)}
    strings = " \n ".join(str(v).lower() for v in values if isinstance(v, str))
    problems = []

    for match in _MONEY.finditer(reply):
        amount = (match.group(1) or match.group(2)).replace(",", "")
        if f"{float(amount):.2f}" not in numbers:
            problems.append(f"amount {match.group(0).strip()} is not in any tool result")

    spans = [next(g for g in m.groups() if g) for m in _BOLD_OR_QUOTED.finditer(reply)]
    spans += [_HEAD_SPLIT.split(m.group(1).replace("*", "").replace('"', ""))[0] for m in _LIST_HEAD.finditer(reply)]
    for span in spans:
        text = span.strip(" .,!?'\"*").lower()
        if not re.search(r"[a-z]", text) or _MONEY.search(text) or _LABEL_WORDS.search(text):
            continue
        if text in strings:
            continue
        if text in DB_TITLES:
            problems.append(f"'{span}' is a real title but no tool returned it this turn (memory-based)")
        elif len(text.split()) >= 2:
            problems.append(f"'{span}' looks like a title but is in no tool result (possibly fabricated)")
    return list(dict.fromkeys(problems))  # a bold list head matches twice


@pytest.fixture
def ask():
    """Verify as customer 14, then ask; returns a function giving the outcome."""
    if _NO_LLM:
        pytest.skip(_NO_LLM)
    app = build_app()
    config = {"configurable": {"thread_id": f"subagent-{uuid.uuid4().hex}"}}
    out = app.invoke({"messages": [HumanMessage(f"Hi, my customer ID is {CUSTOMER_ID}")]}, config)
    assert out.get("customer_id") == CUSTOMER_ID, "verification setup failed"

    def _ask(question: str) -> dict:
        recorder = ToolRecorder()
        out = app.invoke(
            {"messages": [HumanMessage(question)]},
            {**config, "callbacks": [recorder]},
        )
        return {
            "question": question,
            "reply": out["messages"][-1].content,
            "route": out["route"],
            "agents": S.agents_called(out["messages"]),
            "tools": recorder,
            "customer_id": out.get("customer_id"),
        }

    return _ask


def report(result: dict, checks: list[tuple[str, bool]]) -> None:
    """Print the input, the response and each check, then PASSED or FAILED."""
    print("\n" + "=" * 78)
    print(f"INPUT:    {result['question']}")
    print(f"ROUTE:    {result['route']}   AGENTS RUN: {result['agents']}")
    print(f"TOOLS:    {result['tools'].names()}")
    print("RESPONSE:")
    print("  " + result["reply"].replace("\n", "\n  "))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'X '}] {name}")
    print(f"RESULT:   {'PASSED' if not failed else 'FAILED'}")
    assert not failed, f"failed checks: {failed}"


def grounded_check(result: dict) -> tuple[str, bool]:
    problems = grounding_problems(result["reply"], result["tools"])
    for problem in problems:
        print(f"  grounding: {problem}")
    return ("no fabricated album names, prices or amounts", not problems)


# --- The grounding checker itself (offline) ------------------------------------

def _recorder_with(*outputs: dict) -> ToolRecorder:
    recorder = ToolRecorder()
    recorder.calls = [{"name": "tool", "input": "", "output": json.dumps(o)} for o in outputs]
    return recorder


AC_DC = {"rows": [
    {"AlbumTitle": "For Those About To Rock We Salute You", "TrackCount": 10},
    {"AlbumTitle": "Let There Be Rock", "TrackCount": 8},
]}


def test_checker_accepts_a_grounded_reply():
    reply = (
        "AC/DC has:\n"
        "1. **For Those About To Rock We Salute You** - 10 tracks\n"
        "2. **Let There Be Rock** - 8 tracks"
    )
    assert grounding_problems(reply, _recorder_with(AC_DC)) == []


@pytest.mark.parametrize(
    "reply, problem",
    [
        ("1. **Highway To Hell** - 10 tracks", "possibly fabricated"),  # not in the database at all
        ("1. **Back In Black** - 10 tracks", "possibly fabricated"),
        ('You might like "Big Ones" too.', "memory-based"),  # real Aerosmith album, not returned
        ("Your total was $42.17.", "amount $42.17"),  # amount no tool returned
    ],
)
def test_checker_flags_fabricated_titles_and_amounts(reply, problem):
    problems = grounding_problems(reply, _recorder_with(AC_DC))
    print(f"\nINPUT:    {reply}\nFLAGGED:  {problems}")
    assert any(problem in p for p in problems)
    print("RESULT:   PASSED")


# --- Live tests ------------------------------------------------------------

@pytest.mark.live
def test_rock_albums_route_to_music_agent_with_real_data(ask):
    result = ask("What rock albums do you have?")
    results = result["tools"].results()
    db_titles_in_tools = {
        v.lower() for r in results for v in _walk(r) if isinstance(v, str) and v.lower() in DB_TITLES
    }
    cited = [t for t in db_titles_in_tools if t in result["reply"].lower()]

    report(result, [
        ("routed to music_agent only", result["agents"] == [S.MUSIC_AGENT]),
        ("music tools were called", bool(result["tools"].names())),
        ("tools returned rows from the database", bool(db_titles_in_tools)),
        ("reply cites titles the tools returned", bool(cited)),
        grounded_check(result),
    ])


@pytest.mark.live
def test_invoices_route_to_invoice_agent_with_verified_id(ask):
    result = ask("Show me my invoices")
    invoice_results = [
        r for c, r in zip(result["tools"].calls, result["tools"].results())
        if c["name"] == "get_customer_invoices"
    ]
    totals = {f"{inv['Total']:.2f}" for r in invoice_results for inv in r.get("invoices", [])}

    report(result, [
        ("routed to invoice_agent only", result["agents"] == [S.INVOICE_AGENT]),
        ("get_customer_invoices was called", bool(invoice_results)),
        (f"tool used verified customer_id {CUSTOMER_ID}",
         bool(invoice_results) and all(r.get("CustomerId") == CUSTOMER_ID for r in invoice_results)),
        ("reply quotes real invoice totals", any(t in result["reply"] for t in totals)),
        grounded_check(result),
    ])


@pytest.mark.live
def test_weather_is_rejected_without_any_sub_agent(ask):
    result = ask("What's the weather?")

    report(result, [
        ("no route planned", result["route"] == []),
        ("no sub-agent ran", result["agents"] == []),
        ("no tool was called", result["tools"].names() == []),
        ("supervisor's fixed rejection reply", result["reply"] == S.OFF_TOPIC_MESSAGE),
    ])


@pytest.mark.live
def test_mixed_request_runs_invoice_then_music(ask):
    result = ask("Show my invoices and also what jazz albums you have")
    names = result["tools"].names()
    invoice_tools = {"get_customer_invoices", "get_customer_purchased_tracks",
                     "get_invoice_support_rep", "get_invoice_line_items"}
    first_music = next((i for i, n in enumerate(names) if n not in invoice_tools), None)
    last_invoice = max((i for i, n in enumerate(names) if n in invoice_tools), default=None)

    report(result, [
        ("routed to invoice_agent then music_agent",
         result["agents"] == [S.INVOICE_AGENT, S.MUSIC_AGENT]),
        ("invoice tools ran before music tools",
         first_music is not None and last_invoice is not None and last_invoice < first_music),
        grounded_check(result),
    ])


@pytest.mark.live
@pytest.mark.parametrize(
    "question",
    [
        "Which albums does AC/DC have, and how many tracks are on each?",
        "How much is the song Bohemian Rhapsody?",
        "How much have I spent in total, and what was my most recent invoice?",
    ],
)
def test_agents_never_fabricate_names_or_amounts(ask, question):
    result = ask(question)
    report(result, [
        ("tools were called", bool(result["tools"].names())),
        grounded_check(result),
    ])


_NOTHING_FOUND = re.compile(
    r"\b(no|not|couldn't|could not|don't|do not|unable|none|isn't|wasn't|doesn't)\b", re.IGNORECASE
)


@pytest.mark.live
@pytest.mark.parametrize(
    "question",
    [
        "Do you have any albums by Taylor Swift?",
        "What's on my invoice 99999?",
    ],
)
def test_empty_results_are_reported_honestly(ask, question):
    result = ask(question)
    results = result["tools"].results()
    empty = [
        r for r in results
        if isinstance(r, dict) and (
            "message" in r or r.get("rows") == [] or r.get("tracks") == []
        )
    ]

    report(result, [
        ("a tool returned an empty / not-found result", bool(empty)),
        ("reply says nothing was found", bool(_NOTHING_FOUND.search(result["reply"]))),
        grounded_check(result),
    ])
