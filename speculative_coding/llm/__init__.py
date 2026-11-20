"""Standalone LLM routing and transport package (OpenAI-compatible & Gemini)."""

from __future__ import annotations

from .context import (
    compact_in_loop_atomic,
    emergency_compact,
    env_token_budget,
    estimate_tokens,
    estimate_tools_tokens,
    repair_orphaned_tool_calls,
    validate_tool_arguments,
)
from .gemini import GeminiReply, build_request, complete_gemini_step, generate as generate_gemini, openai_messages, parse_reply
from .openai_transport import complete_non_stream, request_max_tokens, stream_once
from .router import ProviderRouter
from .thinking import TagStreamSplitter, extract_think_and_clean
from .types import (
    AgentTurnResult,
    COOLDOWN_FAIL_THRESHOLD,
    COOLDOWN_SECONDS,
    ContextOverflowError,
    DEFAULT_AUTO_COMPACT_TOKENS,
    LLMClient,
    LLMStreamResult,
    MAX_AGENT_STEPS,
    MAX_CONTEXT_RETRIES,
    MAX_LENGTH_CONTINUES,
    REPEAT_REMINDER_AT,
    REPEAT_STOP_AT,
    StreamEventCallback,
    ToolCallRequest,
    ToolExecutor,
)

__all__ = [
    "AgentTurnResult",
    "COOLDOWN_FAIL_THRESHOLD",
    "COOLDOWN_SECONDS",
    "ContextOverflowError",
    "DEFAULT_AUTO_COMPACT_TOKENS",
    "GeminiReply",
    "LLMClient",
    "LLMStreamResult",
    "MAX_AGENT_STEPS",
    "MAX_CONTEXT_RETRIES",
    "MAX_LENGTH_CONTINUES",
    "ProviderRouter",
    "REPEAT_REMINDER_AT",
    "REPEAT_STOP_AT",
    "StreamEventCallback",
    "TagStreamSplitter",
    "ToolCallRequest",
    "ToolExecutor",
    "build_request",
    "compact_in_loop_atomic",
    "complete_gemini_step",
    "complete_non_stream",
    "emergency_compact",
    "env_token_budget",
    "estimate_tokens",
    "estimate_tools_tokens",
    "extract_think_and_clean",
    "generate_gemini",
    "openai_messages",
    "parse_reply",
    "repair_orphaned_tool_calls",
    "request_max_tokens",
    "stream_once",
    "validate_tool_arguments",
]
