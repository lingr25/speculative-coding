"""Gemini generateContent protocol translation, signed parts continuation, and step execution."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from urllib.parse import quote

import httpx

from ..config import ProviderEndpoint
from .thinking import extract_think_and_clean
from .types import LLMStreamResult, StreamEventCallback, ToolCallRequest, _fire_event

_NON_OPENAI_KEYS = frozenset({"message_ids", "turn_id"})


@dataclass
class GeminiReply:
    content: str
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    parts: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


def openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep opaque Gemini continuation metadata out of OpenAI requests."""
    return [
        {k: v for k, v in m.items() if not k.startswith("_gemini_") and k not in _NON_OPENAI_KEYS}
        for m in messages
    ]


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}] if content else []
    parts = []
    for item in content or []:
        if item.get("type") == "text":
            parts.append({"text": item.get("text") or ""})
        elif item.get("type") == "image_url":
            url = (item.get("image_url") or {}).get("url") or ""
            match = re.fullmatch(r"data:(image/(?:png|jpeg|webp|gif));base64,([A-Za-z0-9+/=\r\n]+)", url)
            if not match:
                raise ValueError("Gemini images must be prepared as base64 image data before routing")
            parts.append({"inlineData": {"mimeType": match[1], "data": match[2]}})
        else:
            raise ValueError("Unsupported message content type for Gemini")
    return parts


def build_request(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    contents: list[dict[str, Any]] = []
    system_parts = []
    calls: dict[str, tuple[str, str | None]] = {}

    def append(role: str, parts: list[dict[str, Any]]) -> None:
        if not parts:
            return
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            system_parts.extend(_parts(message.get("content")))
        elif role == "assistant":
            native = message.get("_gemini_parts") if message.get("_gemini_model") == model else None
            parts = copy.deepcopy(native) if native else _parts(message.get("content"))
            native_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            for idx, call in enumerate(message.get("tool_calls") or []):
                fn = call["function"]
                if native:
                    native_id = native_calls[idx].get("id") if idx < len(native_calls) else None
                else:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("Gemini tool arguments must be an object")
                    parts.append(
                        {
                            "functionCall": {"name": fn["name"], "args": args},
                            "thoughtSignature": "skip_thought_signature_validator",
                        }
                    )
                    native_id = None
                calls[call["id"]] = (fn["name"], native_id)
            append("model", parts)
        elif role == "tool":
            name, native_id = calls.get(message.get("tool_call_id"), (message.get("name"), None))
            if not name:
                raise ValueError("Gemini tool response has no matching function name")
            response: dict[str, Any] = {"name": name, "response": {"output": message.get("content") or ""}}
            if native_id:
                response["id"] = native_id
            append("user", [{"functionResponse": response}])
        elif role == "user":
            append("user", _parts(message.get("content")))
        else:
            raise ValueError("Unsupported message role for Gemini")

    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": system_parts}
    if tools:
        declarations = []
        for tool in tools:
            fn = tool["function"]
            declarations.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description") or "",
                    "parametersJsonSchema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
        body["tools"] = [{"functionDeclarations": declarations}]
    if max_output_tokens and max_output_tokens > 0:
        body["generationConfig"] = {"maxOutputTokens": int(max_output_tokens)}
    return body


def parse_reply(data: dict[str, Any]) -> GeminiReply:
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError("Gemini returned no candidates (empty or blocked response)")
    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = []
    thoughts = []
    calls = []
    for part in parts:
        if part.get("thought"):
            if "text" in part and part["text"]:
                thoughts.append(str(part["text"]))
            continue
        if "text" in part:
            text.append(part["text"])
        if "functionCall" in part:
            fc = part["functionCall"]
            args = fc.get("args") or {}
            if not fc.get("name") or not isinstance(args, dict):
                raise RuntimeError("Gemini returned a malformed function call")
            calls.append(
                {
                    "id": fc.get("id") or "call_" + uuid.uuid4().hex,
                    "name": fc["name"],
                    "arguments": json.dumps(args, ensure_ascii=False),
                }
            )
    if calls and candidate.get("finishReason") != "STOP":
        raise RuntimeError("Gemini returned an incomplete function call")
    if not "".join(text).strip() and not calls:
        raise RuntimeError("Gemini returned empty content and no function calls")

    usage = data.get("usageMetadata") or {}
    in_tok = int(usage.get("promptTokenCount") or 0)
    out_tok = int(usage.get("candidatesTokenCount") or 0) + int(usage.get("thoughtsTokenCount") or 0)

    return GeminiReply(
        "".join(text),
        calls,
        copy.deepcopy(parts),
        reasoning="".join(thoughts).strip(),
        input_tokens=in_tok,
        output_tokens=out_tok,
        finish_reason=str(candidate.get("finishReason") or ""),
    )


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    param = params.get(name)
    if param is not None:
        return param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


async def generate(
    ep: ProviderEndpoint,
    key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout: float,
    session_id: str | None = None,
    max_output_tokens: int | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> GeminiReply:
    body = build_request(messages, tools, ep.model, max_output_tokens=max_output_tokens)
    headers = {"User-Agent": "speculative-coding/0.1", **ep.custom_headers}
    headers["Authorization"] = "Bearer " + key
    headers["x-session-id"] = hashlib.sha256((session_id or uuid.uuid4().hex).encode()).hexdigest()
    url = ep.api_base.rstrip("/") + "/models/" + quote(ep.model, safe="") + ":generateContent"
    if client is not None:
        response = await client.post(url, headers=headers, json=body, timeout=timeout)
    else:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as temp_client:
            response = await temp_client.post(url, headers=headers, json=body)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini HTTP {response.status_code}")
    return parse_reply(response.json())


async def complete_gemini_step(
    ep: ProviderEndpoint,
    api_key: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    timeout: float,
    session_id: str | None = None,
    event_callback: StreamEventCallback | None = None,
    max_output_tokens: int | None = None,
    *,
    client: httpx.AsyncClient | None = None,
    generate_fn: Callable[..., Awaitable[GeminiReply]] = generate,
) -> LLMStreamResult:
    """Execute one Gemini native step with timeout guard, thinking extraction, and stream delta emission."""
    extra_kw: dict[str, Any] = {}
    if max_output_tokens is not None:
        extra_kw["max_output_tokens"] = max_output_tokens
    if client is not None and (generate_fn is generate or _accepts_kwarg(generate_fn, "client")):
        extra_kw["client"] = client
    reply = await asyncio.wait_for(
        generate_fn(
            ep,
            api_key,
            messages,
            tools,
            timeout,
            session_id,
            **extra_kw,
        ),
        timeout=timeout + 5,
    )
    tag_reasoning, clean_text = extract_think_and_clean(reply.content)
    full_reasoning = "\n\n".join(p for p in (reply.reasoning, tag_reasoning) if p).strip()
    if full_reasoning:
        await _fire_event(event_callback, {"type": "reasoning_delta", "delta": full_reasoning})
    if clean_text:
        await _fire_event(event_callback, {"type": "text_delta", "delta": clean_text})
    finish = str(getattr(reply, "finish_reason", "") or "")
    return LLMStreamResult(
        content=clean_text,
        tool_calls=[ToolCallRequest(**call) for call in reply.tool_calls],
        provider_id=ep.provider_id,
        model=ep.model,
        native_parts=reply.parts,
        reasoning=full_reasoning,
        raw_content=reply.content,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
        finish_reason="length" if finish == "MAX_TOKENS" else finish,
    )
