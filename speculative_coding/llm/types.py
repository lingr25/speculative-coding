"""Foundational LLM types, protocols, constants, overflow detection, and stream event helpers."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

logger = logging.getLogger("speculative_coding.llm")

COOLDOWN_FAIL_THRESHOLD = 2
COOLDOWN_SECONDS = 90.0
MAX_AGENT_STEPS = 25
DEFAULT_AUTO_COMPACT_TOKENS = int(os.environ.get("SPEC_AUTO_COMPACT_TOKENS", "258000") or 258000)
MAX_CONTEXT_RETRIES = 2
MAX_LENGTH_CONTINUES = 2
REPEAT_REMINDER_AT = 3
REPEAT_STOP_AT = 5

_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "context window",
    "too many tokens",
    "token limit",
    "reduce the length",
    "prompt is too long",
    "request too large",
    "input is too long",
)


class ContextOverflowError(RuntimeError):
    """The provider rejected the prompt because the requested context was too large."""


def _is_context_overflow_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in _CONTEXT_OVERFLOW_MARKERS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    return status in (400, 413) and ("token" in text or "context" in text)


@dataclass
class ToolCallRequest:
    id: str
    name: str
    arguments: str


@dataclass
class LLMStreamResult:
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    provider_id: str = ""
    model: str = ""
    native_parts: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    raw_content: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


@dataclass
class AgentTurnResult:
    text: str
    images: list[str] = field(default_factory=list)
    provider_id: str = ""
    steps: int = 0
    files: list[dict[str, str]] = field(default_factory=list)
    reasoning: str = ""
    tool_records: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    return_value: Any = None


ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
StreamEventCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal pluggable LLM client interface for speculative agents and orchestrator."""

    async def complete_step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        preferred_provider_id: str | None = None,
        need_vision: bool = False,
        timeout_override: float | None = None,
        session_id: str | None = None,
        event_callback: StreamEventCallback | None = None,
        prefer_stream: bool = False,
        tools_tokens: int | None = None,
    ) -> LLMStreamResult: ...

    def compact_limit_for(self, preferred_provider_id: str | None = None, need_vision: bool = False) -> int: ...


async def _fire_event(cb: StreamEventCallback | None, payload: dict[str, Any]) -> None:
    if cb is None:
        return
    try:
        res = cb(payload)
        if asyncio.iscoroutine(res) or isinstance(res, Awaitable):
            await res
    except Exception as exc:
        logger.debug("StreamEventCallback error (ignored): %s", exc)
