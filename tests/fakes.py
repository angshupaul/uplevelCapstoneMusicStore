"""Offline stand-ins for LLMs, shared by the agent tests."""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from memory import UserProfile
from verification import UserInput


class ScriptedChatModel(BaseChatModel):
    """Fake chat model: returns the given replies in order, records each call.

    A real BaseChatModel, so LangGraph utilities that call bind_tools()
    (create_react_agent, create_supervisor) accept it.
    """

    replies: list[AIMessage]
    calls: list[list] = Field(default_factory=list)
    bound_tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [convert_to_openai_tool(t) for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])


def call(name: str, args: dict | None = None, call_id: str = "1") -> dict:
    """A tool call as it appears in AIMessage.tool_calls."""
    return {"name": name, "args": args or {}, "id": call_id}


def asks_for(*tool_calls: dict) -> AIMessage:
    """An AIMessage that only makes tool calls."""
    return AIMessage("", tool_calls=list(tool_calls))


class FakeExtractor:
    """Stands in for the structured-output LLM: finds the first identifier."""

    def invoke(self, messages):
        text = messages[-1].content
        if m := re.search(r"[^@\s]+@[^@\s]+\.\w+", text):
            return UserInput(identifier=m.group(), identifier_type="email")
        if m := re.search(r"\+?\d[\d\s().\-]{5,}\d", text):
            return UserInput(identifier=m.group(), identifier_type="phone")
        if m := re.search(r"\d+", text):
            return UserInput(identifier=m.group(), identifier_type="customer_id")
        return UserInput(identifier=None, identifier_type=None)


class FakeChat:
    """Stands in for the chat LLM used by ask_for_identifier."""

    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage("Could you share your Customer ID, email or phone number?")


class FakePreferenceExtractor:
    """Stands in for the UserProfile structured-output LLM.

    Saves what follows "I love" in Customer lines of the transcript (a crude
    stand-in; real extraction quality is checked by the live tests). Returns
    a wrong customer_id on purpose: callers must ignore it. Set ``error`` to
    make every call raise.
    """

    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        transcript = messages[-1].content
        found = re.findall(r"^Customer: .*?\bI love ([\w/&' -]+)", transcript, re.MULTILINE)
        prefs = [p for item in found for p in re.split(r"\s+and\s+|,\s*", item.strip()) if p]
        return UserProfile(customer_id=999, music_preferences=prefs)
