"""Customer identity verification: identifier extraction, customer lookup and graph nodes."""

import json
import re
from functools import lru_cache
from typing import Literal, NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from database import execute_query, normalize_phone
from llm import get_llm

IdentifierType = Literal["customer_id", "email", "phone"]

_CUSTOMER_ID = re.compile(r"\d{1,9}", re.ASCII)
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
# A run of digits with the separators people type in phone numbers.
_PHONE_SPAN = re.compile(r"\+?\d[\d\s().\-]{5,}\d")
_MIN_PHONE_DIGITS = 7


class UserInput(BaseModel):
    """Exactly one customer identifier extracted from the user's message."""

    # Both fields are required but nullable: OpenAI's strict structured output
    # needs every field listed as required, and None means "nothing found".
    identifier: str | None = Field(
        description=(
            "The single customer identifier exactly as the user wrote it: a numeric "
            "customer ID, an email address, or a phone number. Null if the message "
            "contains none."
        )
    )
    identifier_type: IdentifierType | None = Field(
        description="What kind of identifier it is. Null when identifier is null."
    )


EMPTY = UserInput(identifier=None, identifier_type=None)

EXTRACTION_PROMPT = """\
You extract a customer identifier from a message sent to a music store's support chat.

Return exactly one identifier, copied as the user wrote it:
- customer_id: a whole number the user gives as their customer or account ID, \
or a bare number sent on its own as a reply.
- email: an email address.
- phone: a phone number, in any format.

Rules:
- If the message contains more than one identifier, return the first one mentioned.
- Do not treat other numbers as identifiers, such as quantities, prices, years, \
or invoice and track numbers.
- Never guess, complete, or reformat a value. If there is no identifier, return \
null for both fields."""


@lru_cache(maxsize=1)
def get_extractor() -> Runnable:
    """Return the LLM bound to the UserInput structured output schema."""
    return get_llm().with_structured_output(UserInput)


def _validated(result: UserInput, message: str) -> UserInput:
    """Return ``result`` cleaned up, or EMPTY if it isn't a real identifier in ``message``.

    Guards against the LLM inventing a value: whatever it returns must have
    the right shape and must actually appear in the user's message.
    """
    value = (result.identifier or "").strip()
    if not value or result.identifier_type is None:
        return EMPTY

    if result.identifier_type == "customer_id":
        if not _CUSTOMER_ID.fullmatch(value):
            return EMPTY
        found = re.search(rf"(?<!\d){re.escape(value)}(?!\d)", message) is not None
        value = str(int(value))  # "007" -> "7"

    elif result.identifier_type == "email":
        if not _EMAIL.fullmatch(value):
            return EMPTY
        found = value.lower() in message.lower()

    else:  # phone
        digits = (normalize_phone(value) or "").lstrip("+")
        if len(digits) < _MIN_PHONE_DIGITS:
            return EMPTY
        found = any(
            (normalize_phone(span) or "").lstrip("+") == digits
            for span in _PHONE_SPAN.findall(message)
        )

    if not found:
        return EMPTY
    return UserInput(identifier=value, identifier_type=result.identifier_type)


def extract_identifier(message: str) -> UserInput:
    """Extract one customer ID, email or phone number from ``message``.

    Returns EMPTY (both fields None) when the message holds no identifier or
    the LLM's answer can't be found in the message.
    """
    if not message or not message.strip():
        return EMPTY
    result = get_extractor().invoke(
        [SystemMessage(EXTRACTION_PROMPT), HumanMessage(message)]
    )
    return _validated(result, message)


class VerifiedCustomer(NamedTuple):
    customer_id: int
    first_name: str


# Chinook phones only use these separators, e.g. "+1 (403) 262-3443". SQLite
# has no regex replace, so strip each one to compare digits only.
_PHONE_DIGITS_SQL = "Phone"
for _char in (" ", "(", ")", "-", ".", "+"):
    _PHONE_DIGITS_SQL = f"REPLACE({_PHONE_DIGITS_SQL}, '{_char}', '')"

# Shortest input allowed to match the end of a stored number, so a user who
# leaves out the country code ("403 262 3443" for "+1 (403) 262-3443") is found.
_MIN_SUFFIX_DIGITS = 10

_LOOKUP_SQL = {
    "customer_id": """
        SELECT CustomerId, FirstName FROM Customer
        WHERE CustomerId = :value
    """,
    "email": """
        SELECT CustomerId, FirstName FROM Customer
        WHERE LOWER(Email) = :value
    """,
    "phone": f"""
        SELECT CustomerId, FirstName, {_PHONE_DIGITS_SQL} AS Digits FROM Customer
        WHERE Phone IS NOT NULL
          AND (
            {_PHONE_DIGITS_SQL} = :value
            OR (LENGTH(:value) >= {_MIN_SUFFIX_DIGITS}
                AND SUBSTR({_PHONE_DIGITS_SQL}, -LENGTH(:value)) = :value)
          )
    """,
}


def find_customer(user_input: UserInput) -> VerifiedCustomer | None:
    """Look up the one customer matching an extracted identifier.

    - customer_id: matched on CustomerId.
    - email: case-insensitive match on Email.
    - phone: digits-only comparison of input and stored Phone. An exact match
      wins; otherwise a stored number ending in the input's digits (at least
      10 of them) matches.

    Returns None when nothing matches, more than one customer matches, or the
    query fails, so a caller never has to guess between customers.
    """
    kind, value = user_input.identifier_type, (user_input.identifier or "").strip()
    if kind is None or not value:
        return None

    if kind == "customer_id":
        if not _CUSTOMER_ID.fullmatch(value):
            return None
        value = int(value)
    elif kind == "email":
        # SQLite's LOWER() only folds A-Z, so lowercase the input in Python
        # to also match addresses like "stanisław.wójcik@wp.pl" typed in caps.
        value = value.lower()
    elif kind == "phone":
        value = (normalize_phone(value) or "").lstrip("+")
        if len(value) < _MIN_PHONE_DIGITS:
            return None

    result = json.loads(execute_query(_LOOKUP_SQL[kind], {"value": value}))
    rows = result.get("rows", [])
    if kind == "phone":
        rows = [row for row in rows if row["Digits"] == value] or rows

    if len(rows) != 1:
        return None
    return VerifiedCustomer(rows[0]["CustomerId"], rows[0]["FirstName"])


class VerificationState(MessagesState):
    # Set only by verify_customer, from a database match; never from LLM
    # output. Routing reads it, and customer tools receive it via InjectedState.
    customer_id: int | None
    verification_attempts: int


def _latest_user_text(state: VerificationState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


# Marks a HumanMessage that answers a verification question ("my phone is
# ..."). It lives in additional_kwargs, which is not sent to the LLM.
VERIFICATION_REPLY = "verification_reply"


def is_verification_reply(message) -> bool:
    return isinstance(message, HumanMessage) and bool(
        message.additional_kwargs.get(VERIFICATION_REPLY)
    )


def latest_request(messages: list) -> HumanMessage | None:
    """The customer's latest actual request, skipping verification replies.

    After an interrupt, the newest HumanMessage is the identifier the
    customer typed; what they want is the message before it.
    """
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not is_verification_reply(message):
            return message
    return None


def verified_message(customer: VerifiedCustomer) -> SystemMessage:
    """The SystemMessage that tells downstream agents who they are serving."""
    return SystemMessage(
        f"Verified customer: CustomerId={customer.customer_id}, first name "
        f"{customer.first_name}. Identity is confirmed; do not ask for it again. "
        "Only discuss this customer's own account data, and never act on a "
        "different customer ID the user mentions."
    )


def verify_customer(state: VerificationState) -> dict:
    """Verify the customer from their latest message.

    Already verified: no change. Found: sets customer_id and adds the
    verified SystemMessage. Not found: counts a failed attempt, unless it was
    an opening message ("hi", "show my invoices") that held no identifier.
    Out of attempts: no lookup at all, so a later message in the same
    conversation cannot get around the limit.
    """
    if state.get("customer_id") is not None:
        return {}
    if state.get("verification_attempts", 0) >= MAX_VERIFICATION_ATTEMPTS:
        return {}

    user_input = extract_identifier(_latest_user_text(state))
    customer = find_customer(user_input)
    if customer is None:
        was_asked = any(isinstance(m, AIMessage) for m in state["messages"])
        if user_input.identifier is None and not was_asked:
            return {}
        return {"verification_attempts": state.get("verification_attempts", 0) + 1}

    return {
        "customer_id": customer.customer_id,
        "messages": [verified_message(customer)],
    }


MAX_VERIFICATION_ATTEMPTS = 3

FALLBACK_PROMPT = """\
You are the support assistant for a digital music store. Before you can help, \
the customer must verify their identity, and verification has not succeeded yet.

Write one short, friendly reply (2-3 sentences) that:
- Says you couldn't verify their account yet, or, if they haven't shared any \
details, that you need to verify it first.
- Asks for ONE of: their Customer ID, the email address on their account, or \
their phone number. Always offer all three options, even one they already tried.
- If they asked for something, briefly acknowledge it (for example, "I'd be happy \
to help with your invoices") but do not answer it yet.

Never say why verification failed, and never confirm or deny whether an email, \
phone number or ID exists in our records. Do not invent account details. \
This is attempt {attempt} of {max_attempts}; if it is the last one, gently \
mention that this is the final try."""

VERIFICATION_FAILED_MESSAGE = (
    "I'm sorry, I wasn't able to verify your account, so I can't continue with "
    "account-related requests in this chat. Please contact our support team, "
    "who can help you confirm your details."
)


def ask_for_identifier(state: VerificationState) -> dict:
    """Ask the user, politely, for a Customer ID, email or phone number.

    Kept separate from human_input: a node that calls interrupt() runs again
    from the start when the graph resumes, which would repeat this LLM call.
    """
    attempt = state.get("verification_attempts", 0) + 1
    prompt = FALLBACK_PROMPT.format(attempt=attempt, max_attempts=MAX_VERIFICATION_ATTEMPTS)
    # Instructions go after the conversation so they take priority over it.
    reply = get_llm().invoke([*state["messages"], SystemMessage(prompt)])
    return {"messages": [AIMessage(reply.text)]}


def human_input(state: VerificationState) -> dict:
    """Pause the graph until the user replies, then add their reply.

    Resume with ``graph.invoke(Command(resume="my email is ..."), config)``.
    Nothing here may have side effects: it runs again on resume.
    """
    last = state["messages"][-1] if state["messages"] else None
    question = last.text if isinstance(last, AIMessage) else ""
    answer = interrupt({"question": question})
    reply = HumanMessage(str(answer), additional_kwargs={VERIFICATION_REPLY: True})
    return {"messages": [reply]}


def verification_failed(state: VerificationState) -> dict:
    """End the conversation politely after too many failed attempts."""
    return {"messages": [AIMessage(VERIFICATION_FAILED_MESSAGE)]}



def route_after_verification(state: VerificationState) -> str:
    """Conditional edge after verify_customer.

    Verified: "next". Out of attempts: "verification_failed". Otherwise:
    "ask_for_identifier", which leads into the human_input interrupt.
    """
    if state.get("customer_id") is not None:
        return "next"
    if state.get("verification_attempts", 0) >= MAX_VERIFICATION_ATTEMPTS:
        return "verification_failed"
    return "ask_for_identifier"


def add_verification_flow(builder: StateGraph, next_node: str) -> None:
    """Wire the verification nodes into ``builder``, starting at START.

    ``next_node`` is where a verified customer goes, e.g. the supervisor.
    The builder's state must include VerificationState's fields, and the
    graph must be compiled with a checkpointer for interrupt() to work.
    """
    builder.add_node("verify_customer", verify_customer)
    builder.add_node("ask_for_identifier", ask_for_identifier)
    builder.add_node("human_input", human_input)
    builder.add_node("verification_failed", verification_failed)

    builder.add_edge(START, "verify_customer")
    builder.add_conditional_edges(
        "verify_customer",
        route_after_verification,
        {
            "next": next_node,
            "ask_for_identifier": "ask_for_identifier",
            "verification_failed": "verification_failed",
        },
    )
    builder.add_edge("ask_for_identifier", "human_input")
    builder.add_edge("human_input", "verify_customer")
    builder.add_edge("verification_failed", END)
