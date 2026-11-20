"""In-loop preflight compaction, overflow recovery, and step token telemetry."""

from __future__ import annotations

import logging
from typing import Any, Callable

from ..llm.context import compact_in_loop_atomic, emergency_compact, estimate_tokens
from ..llm.types import (
    ContextOverflowError,
    LLMStreamResult,
    MAX_CONTEXT_RETRIES,
    StreamEventCallback,
    _fire_event,
)

logger = logging.getLogger("speculative_coding.agent_loop.compaction")


async def run_preflight_compaction(
    messages: list[dict[str, Any]],
    *,
    tools_tokens: int,
    compact_limit: int,
    actual_prompt_tokens: int,
    step: int,
    event_callback: StreamEventCallback | None = None,
    estimate_tokens_fn: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
    compact_fn: Callable[..., tuple[list[dict[str, Any]], int]] = compact_in_loop_atomic,
    emergency_compact_fn: Callable[..., list[dict[str, Any]]] = emergency_compact,
) -> tuple[list[dict[str, Any]], int]:
    """Compact atomic tool groups before a provider call when approaching the context limit."""
    est_tokens = max(estimate_tokens_fn(messages) + tools_tokens, actual_prompt_tokens)
    if compact_limit > 0 and est_tokens >= compact_limit:
        compacted_msgs, evicted = compact_fn(messages, keep_recent_groups=6)
        if evicted > 0:
            after_tokens = estimate_tokens_fn(compacted_msgs)
            messages = compacted_msgs
            actual_prompt_tokens = 0
            logger.info(
                "[In-Loop Auto-Compact] compacted %d atomic tool groups (tokens: %d -> %d)",
                evicted,
                est_tokens,
                after_tokens,
            )
            await _fire_event(
                event_callback,
                {
                    "type": "compact_boundary",
                    "step": step,
                    "evicted_groups": evicted,
                    "before_tokens": est_tokens,
                    "after_tokens": after_tokens,
                },
            )

    post_est = estimate_tokens_fn(messages) + tools_tokens
    if compact_limit > 0 and post_est >= compact_limit:
        before_emergency = post_est
        messages = emergency_compact_fn(messages, keep_recent_groups=2)
        actual_prompt_tokens = 0
        post_est = estimate_tokens_fn(messages) + tools_tokens
        await _fire_event(
            event_callback,
            {
                "type": "compact_boundary",
                "step": step,
                "evicted_groups": 0,
                "reason": "preflight_overflow",
                "before_tokens": before_emergency,
                "after_tokens": post_est,
            },
        )

    used_tokens = max(post_est, actual_prompt_tokens)
    await _fire_event(
        event_callback,
        {
            "type": "context_usage",
            "used_tokens": used_tokens,
            "limit_tokens": compact_limit,
            "estimated": not bool(actual_prompt_tokens),
        },
    )
    return messages, actual_prompt_tokens


async def recover_from_overflow(
    messages: list[dict[str, Any]],
    *,
    exc: ContextOverflowError,
    overflow_retries: int,
    tools_tokens: int,
    step: int,
    event_callback: StreamEventCallback | None = None,
    estimate_tokens_fn: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
    emergency_compact_fn: Callable[..., list[dict[str, Any]]] = emergency_compact,
) -> tuple[list[dict[str, Any]], int]:
    """Emergency-compact context after a provider ContextOverflowError, up to MAX_CONTEXT_RETRIES."""
    if overflow_retries >= MAX_CONTEXT_RETRIES:
        logger.warning("Context overflow persisted after emergency compaction: %s", exc)
        raise exc
    overflow_retries += 1
    before_overflow = estimate_tokens_fn(messages) + tools_tokens
    messages = emergency_compact_fn(messages, keep_recent_groups=1)
    after_overflow = estimate_tokens_fn(messages) + tools_tokens
    await _fire_event(
        event_callback,
        {
            "type": "compact_boundary",
            "step": step,
            "evicted_groups": 0,
            "reason": "overflow_retry",
            "before_tokens": before_overflow,
            "after_tokens": after_overflow,
        },
    )
    return messages, overflow_retries


async def record_step_usage(
    step_res: LLMStreamResult,
    messages: list[dict[str, Any]],
    *,
    compact_limit: int,
    event_callback: StreamEventCallback | None = None,
    estimate_tokens_fn: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
) -> tuple[int, int, int]:
    """Compute actual/estimated step token usage and emit a step context_usage event."""
    actual_prompt_tokens = int(step_res.input_tokens or 0)
    step_in = step_res.input_tokens or estimate_tokens_fn(messages)
    step_out = step_res.output_tokens or max(1, int(len(step_res.content + step_res.reasoning) * 0.4))
    await _fire_event(
        event_callback,
        {
            "type": "context_usage",
            "used_tokens": step_in + step_out,
            "usage_kind": "step",
            "input_tokens": step_in,
            "output_tokens": step_out,
            "limit_tokens": compact_limit,
            "estimated": not bool(step_res.input_tokens and step_res.output_tokens),
        },
    )
    return actual_prompt_tokens, step_in, step_out
