"""OpenAI Chat Completions streaming, non-streaming, and max_tokens budgeting transport."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable

from openai import AsyncOpenAI

from ..config import ProviderEndpoint
from .context import estimate_tokens, estimate_tools_tokens
from .gemini import openai_messages
from .thinking import TagStreamSplitter, extract_think_and_clean
from .types import LLMStreamResult, StreamEventCallback, ToolCallRequest, _fire_event


def request_max_tokens(
    ep: ProviderEndpoint,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    tools_tokens: int | None = None,
    estimate_tokens_fn: Callable[[list[dict[str, Any]]], int] = estimate_tokens,
    estimate_tools_tokens_fn: Callable[[list[dict[str, Any]] | None], int] = estimate_tools_tokens,
) -> int | None:
    limit = int(getattr(ep, "max_output_tokens", 0) or 0)
    if limit <= 0 or getattr(ep, "omit_max_tokens", False):
        return None
    window = int(getattr(ep, "context_window", 0) or 0)
    if window <= 0:
        return limit
    tool_tok = int(tools_tokens) if tools_tokens is not None else estimate_tools_tokens_fn(tools)
    used = estimate_tokens_fn(messages) + tool_tok
    remaining = window - used - 512
    if remaining <= 0:
        return 0
    return min(limit, max(256, remaining))


async def complete_non_stream(
    ep: ProviderEndpoint,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout: float,
    event_callback: StreamEventCallback | None = None,
    max_output_tokens: int | None = None,
    *,
    client: Any = None,
    client_cls: Any = AsyncOpenAI,
) -> LLMStreamResult:
    """Non-streaming completion for OpenAI-compatible endpoints."""
    owned_client = None
    if client is None:
        owned_client = client_cls(
            api_key=api_key,
            base_url=ep.api_base,
            timeout=timeout,
            default_headers=getattr(ep, "custom_headers", None) or None,
        )
        active_client = owned_client
    else:
        active_client = client
    kwargs: dict[str, Any] = {
        "model": ep.model,
        "messages": openai_messages(messages),
        "stream": False,
    }
    if tools:
        kwargs["tools"] = tools
    if max_output_tokens and max_output_tokens > 0 and not getattr(ep, "omit_max_tokens", False):
        kwargs["max_tokens"] = int(max_output_tokens)

    try:
        resp = await active_client.chat.completions.create(**kwargs)
    finally:
        if owned_client is not None:
            await owned_client.close()

    if not getattr(resp, "choices", None):
        raise RuntimeError(f"Provider {ep.provider_id} returned empty choices in non-stream mode")

    msg = resp.choices[0].message
    raw_content = getattr(msg, "content", None) or ""
    explicit_reasoning = getattr(msg, "reasoning_content", None) or ""
    tag_reasoning, content = extract_think_and_clean(raw_content)
    reasoning = "\n\n".join(p for p in (explicit_reasoning.strip(), tag_reasoning) if p).strip()

    if reasoning:
        await _fire_event(event_callback, {"type": "reasoning_delta", "delta": reasoning})
    if content:
        await _fire_event(event_callback, {"type": "text_delta", "delta": content})

    tool_calls: list[ToolCallRequest] = []
    raw_tcs = getattr(msg, "tool_calls", None) or []
    for idx, tc in enumerate(raw_tcs):
        fn = getattr(tc, "function", None)
        if not fn or not getattr(fn, "name", None):
            continue
        call_id = getattr(tc, "id", None) or f"call_{idx}_{int(time.time() * 1000)}"
        tool_calls.append(
            ToolCallRequest(
                id=call_id,
                name=fn.name,
                arguments=getattr(fn, "arguments", None) or "{}",
            )
        )

    if not content and not tool_calls:
        raise RuntimeError(f"Provider {ep.provider_id} returned empty content and no tool_calls in non-stream mode")

    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

    return LLMStreamResult(
        content=content,
        tool_calls=tool_calls,
        provider_id=ep.provider_id,
        model=ep.model,
        reasoning=reasoning,
        raw_content=raw_content,
        input_tokens=in_tok,
        output_tokens=out_tok,
        finish_reason=str(getattr(resp.choices[0], "finish_reason", "") or ""),
    )


async def stream_once(
    ep: ProviderEndpoint,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout: float,
    event_callback: StreamEventCallback | None = None,
    max_output_tokens: int | None = None,
    *,
    client: Any = None,
    client_cls: Any = AsyncOpenAI,
) -> LLMStreamResult:
    """Streaming completion for OpenAI-compatible endpoints."""
    owned_client = None
    if client is None:
        owned_client = client_cls(
            api_key=api_key,
            base_url=ep.api_base,
            timeout=timeout,
            default_headers=getattr(ep, "custom_headers", None) or None,
        )
        active_client = owned_client
    else:
        active_client = client
    kwargs: dict[str, Any] = {
        "model": ep.model,
        "messages": openai_messages(messages),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        kwargs["tools"] = tools
    if max_output_tokens and max_output_tokens > 0 and not getattr(ep, "omit_max_tokens", False):
        kwargs["max_tokens"] = int(max_output_tokens)

    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    splitter = TagStreamSplitter()
    tc_acc: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None
    in_tok = 0
    out_tok = 0
    stream: Any = None

    try:
        try:
            stream = await active_client.chat.completions.create(**kwargs)
        except Exception as create_err:
            if "stream_options" in str(create_err).lower() or isinstance(create_err, TypeError):
                retry_kwargs = {k: v for k, v in kwargs.items() if k != "stream_options"}
                stream = await active_client.chat.completions.create(**retry_kwargs)
            else:
                raise
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                in_tok = int(getattr(usage, "prompt_tokens", 0) or in_tok)
                out_tok = int(getattr(usage, "completion_tokens", 0) or out_tok)
            if not chunk.choices:
                continue
            choice0 = chunk.choices[0]
            if getattr(choice0, "finish_reason", None):
                finish_reason = choice0.finish_reason
            delta = choice0.delta
            if delta is None:
                continue
            r_delta = getattr(delta, "reasoning_content", None)
            if r_delta:
                reasoning_parts.append(r_delta)
                await _fire_event(event_callback, {"type": "reasoning_delta", "delta": r_delta})
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                for kind, piece in splitter.feed(delta.content):
                    if piece:
                        if kind == "reasoning":
                            reasoning_parts.append(piece)
                            await _fire_event(event_callback, {"type": "reasoning_delta", "delta": piece})
                        else:
                            await _fire_event(event_callback, {"type": "text_delta", "delta": piece})
            if getattr(delta, "tool_calls", None):
                for tc in delta.tool_calls:
                    idx = tc.index if tc.index is not None else 0
                    slot = tc_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["arguments"] += tc.function.arguments
        for kind, piece in splitter.flush():
            if piece:
                if kind == "reasoning":
                    reasoning_parts.append(piece)
                    await _fire_event(event_callback, {"type": "reasoning_delta", "delta": piece})
                else:
                    await _fire_event(event_callback, {"type": "text_delta", "delta": piece})
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                close_res = stream.close()
                if inspect.isawaitable(close_res):
                    await close_res
            except Exception:
                pass
        if owned_client is not None:
            await owned_client.close()

    tool_calls: list[ToolCallRequest] = []
    for idx in sorted(tc_acc.keys()):
        item = tc_acc[idx]
        if not item["name"]:
            continue
        call_id = item["id"] or f"call_{idx}_{int(time.time() * 1000)}"
        tool_calls.append(
            ToolCallRequest(
                id=call_id,
                name=item["name"],
                arguments=item["arguments"] or "{}",
            )
        )

    raw_joined = "".join(content_parts)
    _, content = extract_think_and_clean(raw_joined)
    reasoning = "".join(reasoning_parts).strip()

    if finish_reason == "tool_calls" and not tool_calls:
        raise RuntimeError(
            f"Provider {ep.provider_id} returned finish_reason='tool_calls' with empty delta.tool_calls in stream mode"
        )
    if not content and not tool_calls:
        raise RuntimeError(f"Provider {ep.provider_id} returned empty content and no tool_calls in stream mode")

    return LLMStreamResult(
        content=content,
        tool_calls=tool_calls,
        provider_id=ep.provider_id,
        model=ep.model,
        reasoning=reasoning,
        raw_content=raw_joined,
        input_tokens=in_tok,
        output_tokens=out_tok,
        finish_reason=finish_reason or "",
    )
