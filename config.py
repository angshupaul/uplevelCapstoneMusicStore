"""Environment-driven application configuration, loaded via python-dotenv.

Variables (read from the process environment, or from ``.env`` in the project root):

    LLM_API_KEY       required  (falls back to OPENAI_API_KEY)
    LLM_MODEL         required  (falls back to OPENAI_CHAT_MODEL)
    LLM_TEMPERATURE   optional  float in [0, 2], default 0.0
    LLM_API_BASE_URL  optional  default None (use the provider's default endpoint)
    PORT              optional  int in [1, 65535], default 7860
    HOST              optional  address the web app listens on, default 127.0.0.1
                                (local only; the Docker image sets 0.0.0.0)

Real environment variables take precedence over values in ``.env``.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_TEMPERATURE = 0.0
DEFAULT_PORT = 7860  # Gradio's default port
DEFAULT_HOST = "127.0.0.1"  # local only


class ConfigError(ValueError):
    """Raised when a required variable is missing or a value is invalid."""


@dataclass(frozen=True)
class Settings:
    # repr=False keeps the key out of logs and tracebacks.
    api_key: str = field(repr=False)
    model: str
    temperature: float = DEFAULT_TEMPERATURE
    api_base_url: str | None = None
    port: int = DEFAULT_PORT
    host: str = DEFAULT_HOST


def _get(*names: str) -> str | None:
    """Return the first non-blank value among ``names``, stripped."""
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _require(*names: str) -> str:
    value = _get(*names)
    if value is None:
        raise ConfigError(f"Missing required environment variable: {' or '.join(names)}")
    return value


def _parse_temperature(raw: str | None) -> float:
    if raw is None:
        return DEFAULT_TEMPERATURE
    try:
        value = float(raw)
    except ValueError:
        raise ConfigError(f"LLM_TEMPERATURE must be a number, got {raw!r}") from None
    if not 0.0 <= value <= 2.0:
        raise ConfigError(f"LLM_TEMPERATURE must be between 0 and 2, got {value}")
    return value


def _parse_port(raw: str | None) -> int:
    if raw is None:
        return DEFAULT_PORT
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"PORT must be an integer, got {raw!r}") from None
    if not 1 <= value <= 65535:
        raise ConfigError(f"PORT must be between 1 and 65535, got {value}")
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load ``.env`` (if present), validate, and return the cached settings."""
    load_dotenv(ENV_FILE, override=False)
    return Settings(
        api_key=_require("LLM_API_KEY", "OPENAI_API_KEY"),
        model=_require("LLM_MODEL", "OPENAI_CHAT_MODEL"),
        temperature=_parse_temperature(_get("LLM_TEMPERATURE")),
        api_base_url=_get("LLM_API_BASE_URL"),
        port=_parse_port(_get("PORT")),
        host=_get("HOST") or DEFAULT_HOST,
    )
