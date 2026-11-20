"""Provider-agnostic long-horizon AgentLoop runner."""

from __future__ import annotations

import inspect
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..llm.context import (
    compact_in_loop_atomic,
    emergency_compact,
    estimate_tokens,
    estimate_tools_tokens,
    repair_orphaned_tool_calls,
    validate_tool_arguments,
)
from ..llm.types import (
    AgentTurnResult,
    ContextOverflowError,
    DEFAULT_AUTO_COMPACT_TOKENS,
    StreamEventCallback,
    ToolExecutor,
    _fire_event,
)
from ..tool_result import safe_output_name
from .compaction import record_step_usage, recover_from_overflow, run_preflight_compaction
from .finalization import (
    StepCompleter,
    check_budget_stop,
    finalize_budget_turn,
    maybe_continue_on_length,
    resolve_final_text,
)
from .tool_handler import (
    ToolBatchTracker,
    execute_tool_calls,
    extract_tool_names,
    extract_tool_schemas,
)


def _completer_accepts_tools_tokens(step_completer: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(step_completer).parameters
    except (TypeError, ValueError):
        return False
    param = params.get("tools_tokens")
    if param is not None:
        return param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


class AgentLoop:
    """Generic multi-step tool loop decoupled from transport/provider routing."""

    def __init__(
        self,
        step_completer: StepCompleter,
        *,
        spill_root: Path | None = None,
        resolve_compact_limit: Callable[[str | None, bool], int] | None = None,
        estimate_tokens_fn: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
        estimate_tools_tokens_fn: Callable[[list[dict[str, Any]] | None], int] = estimate_tools_tokens,
        compact_fn: Callable[..., tuple[list[dict[str, Any]], int]] = compact_in_loop_atomic,
        emergency_compact_fn: Callable[..., list[dict[str, Any]]] = emergency_compact,
        repair_orphans_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] = repair_orphaned_tool_calls,
        validate_tool_args_fn: Callable[[str, dict[str, Any], dict[str, dict[str, Any]]], str] = validate_tool_arguments,
    ) -> None:
        self._step_completer = step_completer
        self._spill_root = spill_root
        self._resolve_compact_limit = resolve_compact_limit
        self._estimate_tokens = estimate_tokens_fn
        self._estimate_tools_tokens = estimate_tools_tokens_fn
        self._compact = compact_fn
        self._emergency_compact = emergency_compact_fn
        self._repair_orphans = repair_orphans_fn
        self._validate_tool_args = validate_tool_args_fn

    async def run(
        self,
        system_prompt: str,
        history: list[dict[str, Any]],
        user_content: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_executor: ToolExecutor,
        max_steps: int | None = None,
        preferred_provider_id: str | None = None,
        session_id: str | None = None,
        event_callback: StreamEventCallback | None = None,
        prefer_stream: bool = False,
        auto_compact_tokens: int | None = None,
        max_total_tokens: int | None = None,
        deadline_seconds: float | None = None,
        output_validator: Callable[[str], tuple[bool, Any, str]] | None = None,
        max_output_retries: int = 2,
    ) -> AgentTurnResult:
        """Execute a long-horizon turn with Preserved Thinking, atomic compaction, and budget wrap-up."""
        need_vision = isinstance(user_content, list) and any(
            isinstance(p, dict) and p.get("type") == "image_url" for p in user_content
        )
        safe_history = self._repair_orphans(list(history))
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(safe_history)
        messages.append({"role": "user", "content": user_content})

        tracker = ToolBatchTracker(need_vision=need_vision)
        collected_reasoning: list[str] = []
        total_in_tokens = 0
        total_out_tokens = 0
        last_provider = ""

        if session_id is None:
            session_id = uuid.uuid4().hex

        tools_tokens = self._estimate_tools_tokens(tools)
        step_extra_kw: dict[str, Any] = (
            {"tools_tokens": tools_tokens}
            if _completer_accepts_tools_tokens(self._step_completer)
            else {}
        )
        tool_schemas = extract_tool_schemas(tools)
        allowed_tool_names = extract_tool_names(tools)
        if auto_compact_tokens is not None:
            compact_limit = int(auto_compact_tokens)
        elif self._resolve_compact_limit is not None:
            compact_limit = self._resolve_compact_limit(preferred_provider_id, tracker.need_vision)
        else:
            compact_limit = DEFAULT_AUTO_COMPACT_TOKENS

        spill_root = self._spill_root
        if spill_root is None:
            spill_root = Path(tempfile.gettempdir()) / "spec_tool_outputs"
        turn_ref = f"{safe_output_name(session_id, 40)}_{uuid.uuid4().hex[:8]}"

        actual_prompt_tokens = 0
        length_continues = 0
        overflow_retries = 0
        output_retries = 0
        budget_stop = ""
        started_at = time.monotonic()

        step = 0

        def _build_result(text: str, return_value: Any = None) -> AgentTurnResult:
            return AgentTurnResult(
                text=text,
                images=tracker.collected_images,
                files=tracker.collected_files,
                provider_id=last_provider,
                steps=step,
                reasoning="\n\n".join(collected_reasoning).strip(),
                tool_records=tracker.tool_records,
                input_tokens=total_in_tokens,
                output_tokens=total_out_tokens,
                return_value=return_value,
            )

        while True:
            step += 1
            budget_stop = check_budget_stop(
                step=step,
                started_at=started_at,
                max_steps=max_steps,
                deadline_seconds=deadline_seconds,
                max_total_tokens=max_total_tokens,
                total_tokens=total_in_tokens + total_out_tokens,
            )
            if budget_stop:
                break

            messages, actual_prompt_tokens = await run_preflight_compaction(
                messages,
                tools_tokens=tools_tokens,
                compact_limit=compact_limit,
                actual_prompt_tokens=actual_prompt_tokens,
                step=step,
                event_callback=event_callback,
                estimate_tokens_fn=self._estimate_tokens,
                compact_fn=self._compact,
                emergency_compact_fn=self._emergency_compact,
            )
            await _fire_event(event_callback, {"type": "step_start", "step": step})
            try:
                step_res = await self._step_completer(
                    messages=messages,
                    tools=tools if tools else None,
                    preferred_provider_id=preferred_provider_id,
                    need_vision=tracker.need_vision,
                    session_id=session_id,
                    event_callback=event_callback,
                    prefer_stream=prefer_stream,
                    **step_extra_kw,
                )
            except ContextOverflowError as exc:
                messages, overflow_retries = await recover_from_overflow(
                    messages,
                    exc=exc,
                    overflow_retries=overflow_retries,
                    tools_tokens=tools_tokens,
                    step=step,
                    event_callback=event_callback,
                    estimate_tokens_fn=self._estimate_tokens,
                    emergency_compact_fn=self._emergency_compact,
                )
                actual_prompt_tokens = 0
                step -= 1
                continue

            last_provider = step_res.provider_id
            actual_prompt_tokens, step_in, step_out = await record_step_usage(
                step_res,
                messages,
                compact_limit=compact_limit,
                event_callback=event_callback,
                estimate_tokens_fn=self._estimate_tokens,
            )
            total_in_tokens += step_in
            total_out_tokens += step_out

            if step_res.reasoning:
                collected_reasoning.append(step_res.reasoning)

            should_continue, length_continues = await maybe_continue_on_length(
                step_res,
                messages,
                length_continues=length_continues,
                step=step,
                event_callback=event_callback,
            )
            if should_continue:
                step -= 1
                continue

            if not step_res.tool_calls:
                final_text = resolve_final_text(
                    step_res.content or "",
                    collected_images=tracker.collected_images,
                    collected_files=tracker.collected_files,
                )
                return_value: Any = None
                if output_validator is not None:
                    is_valid, parsed_value, retry_prompt = output_validator(final_text)
                    if not is_valid:
                        if output_retries < max(0, max_output_retries):
                            output_retries += 1
                            messages.append({"role": "assistant", "content": step_res.content or ""})
                            messages.append(
                                {
                                    "role": "user",
                                    "content": retry_prompt
                                    or "Invalid output format. Please provide a valid response.",
                                }
                            )
                            continue
                    else:
                        return_value = parsed_value
                return _build_result(final_text, return_value=return_value)

            await execute_tool_calls(
                step_res,
                messages,
                tracker,
                step=step,
                tool_executor=tool_executor,
                tool_schemas=tool_schemas,
                allowed_tool_names=allowed_tool_names,
                spill_root=spill_root,
                turn_ref=turn_ref,
                event_callback=event_callback,
                validate_tool_args_fn=self._validate_tool_args,
            )

            if tracker.terminated:
                final_text = resolve_final_text(
                    step_res.content or "",
                    collected_images=tracker.collected_images,
                    collected_files=tracker.collected_files,
                )
                return _build_result(final_text, return_value=tracker.return_value)

            if tracker.repeat_stop:
                budget_stop = budget_stop or "repeat"
                break

        return await finalize_budget_turn(
            step_completer=self._step_completer,
            messages=messages,
            budget_stop=budget_stop,
            step=step,
            max_steps=max_steps,
            preferred_provider_id=preferred_provider_id,
            need_vision=tracker.need_vision,
            session_id=session_id,
            event_callback=event_callback,
            prefer_stream=prefer_stream,
            compact_limit=compact_limit,
            collected_images=tracker.collected_images,
            collected_files=tracker.collected_files,
            collected_reasoning=collected_reasoning,
            tool_records=tracker.tool_records,
            total_in_tokens=total_in_tokens,
            total_out_tokens=total_out_tokens,
            estimate_tokens_fn=self._estimate_tokens,
            emergency_compact_fn=self._emergency_compact,
            output_validator=output_validator,
            max_output_retries=max_output_retries,
        )


async def run_agent_loop(
    step_completer: StepCompleter,
    system_prompt: str,
    history: list[dict[str, Any]],
    user_content: str | list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_executor: ToolExecutor,
    max_steps: int | None = None,
    preferred_provider_id: str | None = None,
    session_id: str | None = None,
    event_callback: StreamEventCallback | None = None,
    prefer_stream: bool = False,
    auto_compact_tokens: int | None = None,
    max_total_tokens: int | None = None,
    deadline_seconds: float | None = None,
    *,
    spill_root: Path | None = None,
    resolve_compact_limit: Callable[[str | None, bool], int] | None = None,
    output_validator: Callable[[str], tuple[bool, Any, str]] | None = None,
    max_output_retries: int = 2,
) -> AgentTurnResult:
    """Convenience functional entrypoint wrapping AgentLoop(step_completer, ...).run(...)."""
    loop = AgentLoop(
        step_completer,
        spill_root=spill_root,
        resolve_compact_limit=resolve_compact_limit,
    )
    return await loop.run(
        system_prompt=system_prompt,
        history=history,
        user_content=user_content,
        tools=tools,
        tool_executor=tool_executor,
        max_steps=max_steps,
        preferred_provider_id=preferred_provider_id,
        session_id=session_id,
        event_callback=event_callback,
        prefer_stream=prefer_stream,
        auto_compact_tokens=auto_compact_tokens,
        max_total_tokens=max_total_tokens,
        deadline_seconds=deadline_seconds,
        output_validator=output_validator,
        max_output_retries=max_output_retries,
    )
