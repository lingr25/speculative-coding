"""Budget stop evaluation, output length auto-continuation, fallback text precedence, and turn wrap-up."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Protocol

from ..llm.context import emergency_compact, estimate_tokens
from ..llm.types import (
    AgentTurnResult,
    ContextOverflowError,
    LLMStreamResult,
    MAX_LENGTH_CONTINUES,
    StreamEventCallback,
    _fire_event,
)

logger = logging.getLogger("speculative_coding.agent_loop.finalization")


class StepCompleter(Protocol):
    """Callable contract for executing a single LLM completion step."""

    async def __call__(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        preferred_provider_id: str | None = None,
        need_vision: bool = False,
        session_id: str | None = None,
        event_callback: StreamEventCallback | None = None,
        prefer_stream: bool = False,
    ) -> LLMStreamResult: ...


def check_budget_stop(
    *,
    step: int,
    started_at: float,
    max_steps: int | None,
    deadline_seconds: float | None,
    max_total_tokens: int | None,
    total_tokens: int,
) -> str:
    """Return budget stop reason ('max_steps', 'deadline', 'tokens') or empty string."""
    if max_steps is not None and step > max_steps:
        return "max_steps"
    if deadline_seconds is not None and time.monotonic() - started_at >= deadline_seconds:
        return "deadline"
    if max_total_tokens is not None and total_tokens >= max_total_tokens:
        return "tokens"
    return ""


async def maybe_continue_on_length(
    step_res: LLMStreamResult,
    messages: list[dict[str, Any]],
    *,
    length_continues: int,
    step: int,
    event_callback: StreamEventCallback | None = None,
) -> tuple[bool, int]:
    """Auto-continue truncated plain-text responses up to MAX_LENGTH_CONTINUES."""
    if step_res.finish_reason in ("length", "max_tokens"):
        if not step_res.tool_calls and length_continues < MAX_LENGTH_CONTINUES:
            length_continues += 1
            messages.append({"role": "assistant", "content": step_res.content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[System Continue] Your previous response was truncated at the token limit. "
                        "Continue directly from where you stopped without repeating."
                    ),
                }
            )
            await _fire_event(
                event_callback,
                {"type": "length_continue", "step": step, "attempt": length_continues},
            )
            return True, length_continues
        logger.warning(
            "Model output hit length limit (finish_reason=%s, tools=%d); ending step",
            step_res.finish_reason,
            len(step_res.tool_calls),
        )
        return False, length_continues
    return False, 0


def resolve_final_text(
    raw_text: str,
    *,
    budget_stop: str = "",
    collected_images: list[str] | None = None,
    collected_files: list[dict[str, str]] | None = None,
) -> str:
    """Apply deterministic generic fallback text precedence when model output is empty."""
    final_text = raw_text.strip()
    if budget_stop and not final_text:
        final_text = f"[Turn ended due to {budget_stop} without a final wrap-up; inspect tool records.]"
    return final_text


async def finalize_budget_turn(
    *,
    step_completer: StepCompleter,
    messages: list[dict[str, Any]],
    budget_stop: str,
    step: int,
    max_steps: int | None,
    preferred_provider_id: str | None,
    need_vision: bool,
    session_id: str,
    event_callback: StreamEventCallback | None,
    prefer_stream: bool,
    compact_limit: int,
    collected_images: list[str],
    collected_files: list[dict[str, str]],
    collected_reasoning: list[str],
    tool_records: list[dict[str, Any]],
    total_in_tokens: int,
    total_out_tokens: int,
    estimate_tokens_fn: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
    emergency_compact_fn: Callable[..., list[dict[str, Any]]] = emergency_compact,
    output_validator: Callable[[str], tuple[bool, Any, str]] | None = None,
    max_output_retries: int = 2,
) -> AgentTurnResult:
    """Emit budget_exhausted (if any), request a final tool-less wrap-up step, and build AgentTurnResult."""
    if budget_stop:
        await _fire_event(
            event_callback,
            {
                "type": "budget_exhausted",
                "reason": budget_stop,
                "steps": step,
                "input_tokens": total_in_tokens,
                "output_tokens": total_out_tokens,
            },
        )
        wrap_prompt = (
            f"[System] Turn ended due to {budget_stop}. Do not call any tools; immediately output the final JSON object requested by the system prompt."
            if output_validator is not None
            else (
                f"[System] Turn ended due to {budget_stop}. Wrap up concisely using existing evidence: "
                "state what succeeded, what remains, and any blockers without calling tools."
            )
        )
        messages.append({"role": "user", "content": wrap_prompt})

    async def _run_toolless_step() -> tuple[LLMStreamResult, int, int]:
        nonlocal messages
        try:
            res = await step_completer(
                messages=messages,
                tools=None,
                preferred_provider_id=preferred_provider_id,
                need_vision=need_vision,
                session_id=session_id,
                event_callback=event_callback,
                prefer_stream=prefer_stream,
            )
        except ContextOverflowError:
            messages = emergency_compact_fn(messages, keep_recent_groups=1)
            res = await step_completer(
                messages=messages,
                tools=None,
                preferred_provider_id=preferred_provider_id,
                need_vision=need_vision,
                session_id=session_id,
                event_callback=event_callback,
                prefer_stream=prefer_stream,
            )
        if res.reasoning:
            collected_reasoning.append(res.reasoning)
        s_in = res.input_tokens or estimate_tokens_fn(messages)
        s_out = res.output_tokens or max(1, int(len(res.content + res.reasoning) * 0.4))
        await _fire_event(
            event_callback,
            {
                "type": "context_usage",
                "usage_kind": "step",
                "input_tokens": s_in,
                "output_tokens": s_out,
                "used_tokens": s_in + s_out,
                "limit_tokens": compact_limit,
                "estimated": not bool(res.input_tokens and res.output_tokens),
            },
        )
        return res, s_in, s_out

    final_res, final_in, final_out = await _run_toolless_step()
    acc_in = total_in_tokens + final_in
    acc_out = total_out_tokens + final_out

    return_value: Any = None
    if output_validator is not None:
        retries = 0
        while True:
            cand_text = (final_res.content or "").strip()
            is_valid, parsed_value, retry_prompt = output_validator(cand_text)
            if is_valid:
                return_value = parsed_value
                break
            if retries >= max(0, max_output_retries):
                break
            retries += 1
            messages.append({"role": "assistant", "content": final_res.content or ""})
            messages.append(
                {
                    "role": "user",
                    "content": retry_prompt
                    or "Invalid output format. Please provide a valid response without calling tools.",
                }
            )
            final_res, r_in, r_out = await _run_toolless_step()
            acc_in += r_in
            acc_out += r_out

    final_text = resolve_final_text(
        final_res.content,
        budget_stop=budget_stop if return_value is None else "",
        collected_images=collected_images,
        collected_files=collected_files,
    )
    return AgentTurnResult(
        text=final_text,
        images=collected_images,
        files=collected_files,
        provider_id=final_res.provider_id,
        steps=max_steps if budget_stop == "max_steps" and max_steps is not None else step,
        reasoning="\n\n".join(collected_reasoning).strip(),
        tool_records=tool_records,
        input_tokens=acc_in,
        output_tokens=acc_out,
        return_value=return_value,
    )
