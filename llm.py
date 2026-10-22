"""The configured chat model, shared by every agent."""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """Return the chat model from config (LLM_MODEL, LLM_TEMPERATURE, ...)."""
    settings = get_settings()
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.api_key,
        base_url=settings.api_base_url,
        temperature=settings.temperature,
    )
