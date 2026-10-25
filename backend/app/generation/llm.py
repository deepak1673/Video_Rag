"""Provider-agnostic chat completion for answer generation.

Ollama uses its native /api/chat; Groq and any other OpenAI-compatible endpoint share
one client. Callers only see `generate(messages) -> str`.
"""

from typing import Protocol

import httpx

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import get_logger

log = get_logger("llm")

Messages = list[dict[str, str]]


class LLMClient(Protocol):
    def generate(self, messages: Messages, *, temperature: float = 0.2,
                 max_tokens: int = 512) -> str: ...


def _unavailable(reason: str) -> AppError:
    return AppError(503, "llm_unavailable", f"Answer generation is unavailable: {reason}")


class OllamaClient:
    def __init__(self, s: Settings):
        self._url = f"{s.llm_effective_base_url}/api/chat"
        self._model = s.llm_effective_model
        self._timeout = s.ollama_timeout_seconds

    def generate(self, messages: Messages, *, temperature: float = 0.2,
                 max_tokens: int = 512) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            r = httpx.post(self._url, json=payload, timeout=self._timeout)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            log.warning("ollama_failed", error=type(exc).__name__)
            raise _unavailable("local model did not respond") from exc


class OpenAICompatClient:
    def __init__(self, s: Settings):
        if not s.llm_api_key:
            raise _unavailable("LLM_API_KEY is not set")
        self._url = f"{s.llm_effective_base_url}/chat/completions"
        self._model = s.llm_effective_model
        self._timeout = s.llm_timeout_seconds
        self._headers = {"Authorization": f"Bearer {s.llm_api_key}"}

    def generate(self, messages: Messages, *, temperature: float = 0.2,
                 max_tokens: int = 512) -> str:
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = httpx.post(self._url, json=payload, headers=self._headers,
                           timeout=self._timeout)
            if r.status_code == 429:
                raise AppError(429, "llm_rate_limited",
                               "The language model is rate limited; try again shortly.")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except AppError:
            raise
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            # Never log the response body or headers: they may echo credentials.
            log.warning("llm_failed", error=type(exc).__name__)
            raise _unavailable("provider request failed") from exc


def get_llm() -> LLMClient:
    s = get_settings()
    return OllamaClient(s) if s.llm_is_local else OpenAICompatClient(s)
