"""Multi-provider, multi-key OpenAI & Gemini router with cooldowns and strict pool isolation."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import weakref
from typing import Any, Callable

import httpx
from openai import AsyncOpenAI

from ..agent_loop import AgentLoop
from ..config import ProviderEndpoint, SpecConfig
from .context import (
    compact_in_loop_atomic,
    emergency_compact,
    estimate_tokens,
    estimate_tools_tokens,
    repair_orphaned_tool_calls,
    validate_tool_arguments,
)
from .gemini import complete_gemini_step, generate as generate_gemini
from .openai_transport import complete_non_stream, request_max_tokens, stream_once
from .types import (
    AgentTurnResult,
    COOLDOWN_FAIL_THRESHOLD,
    ContextOverflowError,
    DEFAULT_AUTO_COMPACT_TOKENS,
    LLMStreamResult,
    StreamEventCallback,
    ToolExecutor,
    _fire_event,
    _is_context_overflow_error,
)

logger = logging.getLogger("speculative_coding.llm.router")


def _make_loop_ref(loop: asyncio.AbstractEventLoop) -> Callable[[], asyncio.AbstractEventLoop | None]:
    try:
        return weakref.ref(loop)
    except TypeError:
        return lambda: loop


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


def _is_non_fallback_transport_error(err: Exception) -> bool:
    if _is_context_overflow_error(err):
        return True
    if isinstance(
        err,
        (
            asyncio.TimeoutError,
            TimeoutError,
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.NetworkError,
        ),
    ):
        return True
    status_code = getattr(err, "status_code", None) or getattr(
        getattr(err, "response", None), "status_code", None
    )
    if isinstance(status_code, int) and (status_code in (401, 403, 408, 429) or status_code >= 500):
        return True
    cls_name = type(err).__name__
    if cls_name in (
        "APITimeoutError",
        "APIConnectionError",
        "AuthenticationError",
        "PermissionDeniedError",
        "RateLimitError",
        "InternalServerError",
    ):
        return True
    low = str(err).lower()
    return any(
        marker in low
        for marker in (
            "timed out",
            "timeout",
            "rate limit",
            "ratelimit",
            "status code 401",
            "status code 403",
            "status code 429",
            "status code 500",
            "status code 502",
            "status code 503",
            "status code 504",
            "http 401",
            "http 403",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "429 too many requests",
            "401 unauthorized",
            "403 forbidden",
            "connection error",
            "connect error",
            "connection refused",
            "connection reset",
        )
    )


class ProviderRouter:
    """Multi-channel, multi-key OpenAI/Gemini router with exponential backoff cooldown and strict pool isolation."""

    def __init__(self, config: SpecConfig, *, pool: str = "speculative") -> None:
        if pool not in {"main", "speculative"}:
            raise ValueError("invalid provider pool")
        self.pool = pool
        self.tokens_used = 0
        self.config = config
        self.allowed_provider_ids: set[str] | None = None
        self._fail_counts: dict[str, int] = {}
        self._cooldown_strikes: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._openai_clients: dict[tuple[Any, ...], tuple[Callable[[], asyncio.AbstractEventLoop | None], Any]] = {}
        self._gemini_clients: dict[int, tuple[Callable[[], asyncio.AbstractEventLoop | None], httpx.AsyncClient]] = {}
        self.cooldown_enabled: bool = bool(getattr(config, "cooldown_enabled", True))
        self.fail_threshold: int = max(1, int(getattr(config, "cooldown_fail_threshold", COOLDOWN_FAIL_THRESHOLD)))
        self.base_cooldown: float = max(5.0, float(getattr(config, "cooldown_seconds", 180.0)))
        self.max_cooldown: float = max(self.base_cooldown, float(getattr(config, "max_cooldown_seconds", 1800.0)))
        self.spill_root = config.data_dir / "temp" / "tool_outputs"

    def _ensure_client_pools(self):
        openai_pool = getattr(self, "_openai_clients", None)
        if openai_pool is None:
            openai_pool = {}
            self._openai_clients = openai_pool
        gemini_pool = getattr(self, "_gemini_clients", None)
        if gemini_pool is None:
            gemini_pool = {}
            self._gemini_clients = gemini_pool
        return openai_pool, gemini_pool

    def _prune_dead_loop_clients(self) -> None:
        openai_pool, gemini_pool = self._ensure_client_pools()
        for key, (loop_ref, _) in list(openai_pool.items()):
            ref_loop = loop_ref()
            if ref_loop is None or ref_loop.is_closed():
                openai_pool.pop(key, None)
        for key, (loop_ref, _) in list(gemini_pool.items()):
            ref_loop = loop_ref()
            if ref_loop is None or ref_loop.is_closed():
                gemini_pool.pop(key, None)

    def _get_openai_client(self, ep: ProviderEndpoint, api_key: str, timeout: float) -> AsyncOpenAI:
        self._prune_dead_loop_clients()
        openai_pool, _ = self._ensure_client_pools()
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        raw_headers = getattr(ep, "custom_headers", None) or {}
        headers_key = tuple(sorted(raw_headers.items()))
        cache_key = (loop_id, ep.api_base, api_key, float(timeout), headers_key)
        entry = openai_pool.get(cache_key)
        if entry is not None:
            loop_ref, cached_client = entry
            if (
                loop_ref() is loop
                and not loop.is_closed()
                and getattr(cached_client, "is_closed", False) is not True
            ):
                return cached_client
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=ep.api_base,
            timeout=timeout,
            default_headers=raw_headers or None,
        )
        openai_pool[cache_key] = (_make_loop_ref(loop), client)
        return client

    def _get_gemini_client(self) -> httpx.AsyncClient:
        self._prune_dead_loop_clients()
        _, gemini_pool = self._ensure_client_pools()
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        entry = gemini_pool.get(loop_id)
        if entry is not None:
            loop_ref, cached_client = entry
            if (
                loop_ref() is loop
                and not loop.is_closed()
                and getattr(cached_client, "is_closed", False) is not True
            ):
                return cached_client
        client = httpx.AsyncClient(follow_redirects=False)
        gemini_pool[loop_id] = (_make_loop_ref(loop), client)
        return client

    async def aclose(self) -> None:
        self._prune_dead_loop_clients()
        openai_pool, gemini_pool = self._ensure_client_pools()
        loop = asyncio.get_running_loop()

        to_close_oai: list[Any] = []
        for key, (loop_ref, client) in list(openai_pool.items()):
            ref_loop = loop_ref()
            if ref_loop is None or ref_loop.is_closed():
                openai_pool.pop(key, None)
            elif ref_loop is loop:
                openai_pool.pop(key, None)
                to_close_oai.append(client)

        to_close_gem: list[httpx.AsyncClient] = []
        for key, (loop_ref, client) in list(gemini_pool.items()):
            ref_loop = loop_ref()
            if ref_loop is None or ref_loop.is_closed():
                gemini_pool.pop(key, None)
            elif ref_loop is loop:
                gemini_pool.pop(key, None)
                to_close_gem.append(client)

        for client in to_close_oai:
            close_fn = getattr(client, "close", None) or getattr(client, "aclose", None)
            if close_fn is not None:
                await close_fn()
        for client in to_close_gem:
            close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close_fn is not None:
                await close_fn()

    def _resolve_compact_limit(self, preferred_provider_id: str | None, need_vision: bool) -> int:
        if getattr(self, "config", None) is None:
            return DEFAULT_AUTO_COMPACT_TOKENS
        try:
            endpoints = self.candidate_chain(preferred_provider_id, need_vision=need_vision)
        except Exception:
            return DEFAULT_AUTO_COMPACT_TOKENS
        for endpoint in endpoints:
            window = int(getattr(endpoint, "context_window", 0) or 0)
            if window > 0:
                return max(16_000, int(window * 0.8))
        return DEFAULT_AUTO_COMPACT_TOKENS

    def compact_limit_for(self, preferred_provider_id: str | None = None, need_vision: bool = False) -> int:
        return self._resolve_compact_limit(preferred_provider_id, need_vision)

    def _is_cooling(self, pid: str) -> bool:
        if not self.cooldown_enabled:
            return False
        until = self._cooldown_until.get(pid, 0.0)
        if until <= 0:
            return False
        if time.monotonic() >= until:
            self._cooldown_until.pop(pid, None)
            self._fail_counts[pid] = 0
            return False
        return True

    def _mark_success(self, pid: str) -> None:
        self._fail_counts[pid] = 0
        self._cooldown_strikes[pid] = 0
        self._cooldown_until.pop(pid, None)

    def _mark_failure(self, pid: str) -> None:
        if not self.cooldown_enabled:
            return
        cnt = self._fail_counts.get(pid, 0) + 1
        self._fail_counts[pid] = cnt
        if cnt >= self.fail_threshold:
            strikes = self._cooldown_strikes.get(pid, 0)
            duration = min(self.base_cooldown * (2**strikes), self.max_cooldown)
            self._cooldown_strikes[pid] = strikes + 1
            self._cooldown_until[pid] = time.monotonic() + duration
            logger.warning(
                "Provider %s failed %d consecutive times; cooling for %.0fs (strike #%d)",
                pid,
                cnt,
                duration,
                strikes + 1,
            )

    def provider_status(self, pid: str) -> dict[str, Any]:
        cooling = self._is_cooling(pid)
        remaining = max(0.0, self._cooldown_until.get(pid, 0.0) - time.monotonic()) if cooling else 0.0
        return {
            "cooling": cooling,
            "cooldown_remaining": remaining,
            "fail_count": self._fail_counts.get(pid, 0),
            "strikes": self._cooldown_strikes.get(pid, 0),
        }

    def candidate_chain(self, preferred_id: str | None = None, need_vision: bool = False) -> list[ProviderEndpoint]:
        """Return ordered endpoints strictly filtered by pool (main vs speculative)."""
        order: list[str] = []
        if preferred_id and preferred_id in self.config.providers:
            order.append(preferred_id)
        if self.pool == "speculative":
            speculative_default = getattr(self.config, "speculative_provider_id", "")
            if speculative_default and speculative_default not in order:
                order.append(speculative_default)
        else:
            if self.config.default_provider_id and self.config.default_provider_id not in order:
                order.append(self.config.default_provider_id)
            for fid in self.config.fallback_provider_ids:
                if fid not in order:
                    order.append(fid)
        for pid in self.config.providers:
            if pid not in order:
                order.append(pid)

        healthy: list[ProviderEndpoint] = []
        cooling: list[ProviderEndpoint] = []
        for pid in order:
            ep = self.config.providers.get(pid)
            if not ep:
                continue
            # Strict bidirectional pool isolation: speculative never falls back to main,
            # and main never leaks into speculative.
            if getattr(ep, "pool", "speculative") != self.pool:
                continue
            if self.allowed_provider_ids is not None and pid not in self.allowed_provider_ids:
                continue
            if need_vision and "image" not in ep.modalities:
                continue
            if self._is_cooling(pid):
                cooling.append(ep)
            else:
                healthy.append(ep)
        return healthy if self.pool == "speculative" else healthy + cooling

    async def _complete_non_stream(
        self,
        ep: ProviderEndpoint,
        api_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        timeout: float,
        event_callback: StreamEventCallback | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMStreamResult:
        return await complete_non_stream(
            ep,
            api_key,
            messages,
            tools,
            timeout,
            event_callback=event_callback,
            max_output_tokens=max_output_tokens,
            client=self._get_openai_client(ep, api_key, timeout),
            client_cls=AsyncOpenAI,
        )

    async def _stream_once(
        self,
        ep: ProviderEndpoint,
        api_key: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        timeout: float,
        event_callback: StreamEventCallback | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMStreamResult:
        return await stream_once(
            ep,
            api_key,
            messages,
            tools,
            timeout,
            event_callback=event_callback,
            max_output_tokens=max_output_tokens,
            client=self._get_openai_client(ep, api_key, timeout),
            client_cls=AsyncOpenAI,
        )

    def _request_max_tokens(
        self,
        ep: ProviderEndpoint,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tools_tokens: int | None = None,
    ) -> int | None:
        return request_max_tokens(
            ep,
            messages,
            tools,
            tools_tokens=tools_tokens,
            estimate_tokens_fn=estimate_tokens,
            estimate_tools_tokens_fn=estimate_tools_tokens,
        )

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
    ) -> LLMStreamResult:
        endpoints = self.candidate_chain(preferred_provider_id, need_vision=need_vision)
        if not endpoints:
            raise RuntimeError(f"no available provider endpoints in pool '{self.pool}'")

        last_err: Exception | None = None
        saw_overflow = False
        emitted_any_deltas = False
        for ep in endpoints:
            ep_ok = False
            ep_hard_failure = False
            timeout = timeout_override or float(ep.timeout or 120)
            req_max_fn = self._request_max_tokens
            if (
                "_request_max_tokens" not in getattr(self, "__dict__", {})
                and getattr(type(self), "_request_max_tokens", None) is ProviderRouter._request_max_tokens
            ) or _accepts_kwarg(req_max_fn, "tools_tokens"):
                max_out = req_max_fn(ep, messages, tools, tools_tokens=tools_tokens)
            else:
                max_out = req_max_fn(ep, messages, tools)
            max_kw = {"max_output_tokens": max_out} if max_out is not None else {}
            for key_idx, api_key in enumerate(ep.keys):
                attempt_emitted_deltas = False

                async def _tracked_cb(evt: dict[str, Any]) -> None:
                    nonlocal attempt_emitted_deltas, emitted_any_deltas
                    if evt.get("type") in ("text_delta", "reasoning_delta") and evt.get("delta"):
                        if emitted_any_deltas and not attempt_emitted_deltas:
                            return
                        attempt_emitted_deltas = True
                        emitted_any_deltas = True
                    await _fire_event(event_callback, evt)

                active_cb: StreamEventCallback | None = (
                    None if (event_callback is None or emitted_any_deltas) else _tracked_cb
                )
                try:
                    if ep.api_type == "gemini_generate_content":
                        gem_kw = dict(max_kw)
                        if (
                            getattr(complete_gemini_step, "__module__", "") == "speculative_coding.llm.gemini"
                            or _accepts_kwarg(complete_gemini_step, "client")
                        ):
                            gem_kw["client"] = self._get_gemini_client()
                        res = await complete_gemini_step(
                            ep,
                            api_key,
                            messages,
                            tools,
                            timeout,
                            session_id=session_id,
                            event_callback=active_cb,
                            generate_fn=generate_gemini,
                            **gem_kw,
                        )
                        self._mark_success(ep.provider_id)
                        self.tokens_used += res.input_tokens + res.output_tokens
                        return res

                    extra_kw = {"event_callback": active_cb} if active_cb is not None else {}
                    if prefer_stream and getattr(ep, "prefer_stream", True):
                        try:
                            res = await asyncio.wait_for(
                                self._stream_once(ep, api_key, messages, tools, timeout, **max_kw, **extra_kw),
                                timeout=timeout + 5,
                            )
                        except Exception as stream_err:
                            if attempt_emitted_deltas or _is_non_fallback_transport_error(stream_err):
                                raise
                            fallback_kw = (
                                {}
                                if (attempt_emitted_deltas or emitted_any_deltas or active_cb is None)
                                else {"event_callback": active_cb}
                            )
                            res = await asyncio.wait_for(
                                self._complete_non_stream(
                                    ep, api_key, messages, tools, timeout, **max_kw, **fallback_kw
                                ),
                                timeout=timeout + 5,
                            )
                    else:
                        try:
                            res = await asyncio.wait_for(
                                self._complete_non_stream(ep, api_key, messages, tools, timeout, **max_kw, **extra_kw),
                                timeout=timeout + 5,
                            )
                        except Exception as non_stream_err:
                            if attempt_emitted_deltas or _is_non_fallback_transport_error(non_stream_err):
                                raise
                            fallback_kw = (
                                {}
                                if (attempt_emitted_deltas or emitted_any_deltas or active_cb is None)
                                else {"event_callback": active_cb}
                            )
                            res = await asyncio.wait_for(
                                self._stream_once(ep, api_key, messages, tools, timeout, **max_kw, **fallback_kw),
                                timeout=timeout + 5,
                            )
                    self._mark_success(ep.provider_id)
                    ep_ok = True
                    self.tokens_used += res.input_tokens + res.output_tokens
                    return res
                except Exception as exc:
                    last_err = exc
                    if _is_context_overflow_error(exc):
                        saw_overflow = True
                        logger.warning("Provider %s (key #%d) context overflow: %s", ep.provider_id, key_idx + 1, exc)
                        continue
                    ep_hard_failure = True
                    logger.warning("Provider %s (key #%d) failed: %s", ep.provider_id, key_idx + 1, exc)
            if not ep_ok and ep_hard_failure:
                self._mark_failure(ep.provider_id)

        if saw_overflow:
            raise ContextOverflowError(f"All providers failed with context overflow; last error: {last_err}")
        raise RuntimeError(f"All providers failed; last error: {last_err}")

    async def run_agent_loop(
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
    ) -> AgentTurnResult:
        loop = AgentLoop(
            self.complete_step,
            spill_root=getattr(self, "spill_root", None),
            resolve_compact_limit=self._resolve_compact_limit,
            estimate_tokens_fn=estimate_tokens,
            estimate_tools_tokens_fn=estimate_tools_tokens,
            compact_fn=compact_in_loop_atomic,
            emergency_compact_fn=emergency_compact,
            repair_orphans_fn=repair_orphaned_tool_calls,
            validate_tool_args_fn=validate_tool_arguments,
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
        )
