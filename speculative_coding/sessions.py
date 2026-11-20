"""Lightweight standalone session store, real-time EventBus, and TurnRunner for SpecAgent."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger("speculative_coding.sessions")


def new_trace_id() -> str:
    return f"tr_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def new_turn_id() -> str:
    return f"tn_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def iso_now(ts: float | None = None) -> str:
    t = ts if ts is not None else time.time()
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).isoformat()


class SessionEvent:
    def __init__(
        self,
        type: str = "",
        session_key: str = "",
        turn_id: str = "",
        trace_id: str = "",
        origin_channel: str = "web",
        user_id: str = "",
        payload: dict[str, Any] | None = None,
        event_id: str = "",
        timestamp: float = 0.0,
        seq: int = 0,
        *,
        event_type: str = "",
    ) -> None:
        self.type = type or event_type
        self.session_key = session_key
        self.turn_id = turn_id
        self.trace_id = trace_id
        self.origin_channel = origin_channel
        self.user_id = user_id
        self.payload = dict(payload) if payload is not None else {}
        self.event_id = event_id or f"ev_{uuid.uuid4().hex[:16]}"
        self.timestamp = timestamp or time.time()
        self.seq = seq

    @property
    def event_type(self) -> str:
        return self.type

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "seq": self.seq,
            "type": self.type,
            "event_type": self.type,
            "session_key": self.session_key,
            "turn_id": self.turn_id,
            "trace_id": self.trace_id,
            "origin_channel": self.origin_channel,
            "user_id": self.user_id,
            "timestamp": self.timestamp,
            "iso_time": iso_now(self.timestamp),
            "payload": self.payload,
        }


SessionEventCallback = Callable[[SessionEvent], Awaitable[None] | None]


class EventBus:
    """In-process pub/sub event bus scoped by session_key, user_id, or global listeners."""

    def __init__(self) -> None:
        self._seq: int = 0
        self._global_sinks: dict[str, SessionEventCallback] = {}
        self._session_sinks: dict[str, dict[str, SessionEventCallback]] = {}
        self._user_sinks: dict[str, dict[str, SessionEventCallback]] = {}

    def subscribe(
        self,
        sink: SessionEventCallback,
        *,
        session_key: str | None = None,
        user_id: str | None = None,
    ) -> Callable[[], None]:
        if session_key:
            sub_id = f"ssub_{uuid.uuid4().hex[:12]}"
            self._session_sinks.setdefault(session_key, {})[sub_id] = sink
        elif user_id:
            sub_id = f"usub_{uuid.uuid4().hex[:12]}"
            self._user_sinks.setdefault(str(user_id), {})[sub_id] = sink
        else:
            sub_id = f"gsub_{uuid.uuid4().hex[:12]}"
            self._global_sinks[sub_id] = sink
        return lambda: self.unsubscribe(sub_id)

    def unsubscribe(self, sub_id: str) -> None:
        self._global_sinks.pop(sub_id, None)
        for key, bucket in list(self._session_sinks.items()):
            bucket.pop(sub_id, None)
            if not bucket:
                self._session_sinks.pop(key, None)
        for key, bucket in list(self._user_sinks.items()):
            bucket.pop(sub_id, None)
            if not bucket:
                self._user_sinks.pop(key, None)

    async def _invoke_sink(self, sink: SessionEventCallback, event: SessionEvent) -> None:
        try:
            res = sink(event)
            if asyncio.iscoroutine(res) or isinstance(res, Awaitable):
                await res
        except Exception as exc:
            logger.debug("SessionEventSink error (isolated): %s", exc)

    async def emit(self, event: SessionEvent) -> SessionEvent:
        self._seq += 1
        if not event.seq:
            event.seq = self._seq
        targets: list[SessionEventCallback] = list(self._global_sinks.values())
        if event.session_key in self._session_sinks:
            for cb in self._session_sinks[event.session_key].values():
                if cb not in targets:
                    targets.append(cb)
        if event.user_id and str(event.user_id) in self._user_sinks:
            for cb in self._user_sinks[str(event.user_id)].values():
                if cb not in targets:
                    targets.append(cb)
        if targets:
            await asyncio.gather(*(self._invoke_sink(sink, event) for sink in targets), return_exceptions=True)
        return event


class SessionStore:
    """Minimal SQLite store for SpecAgent conversation turns and tool/usage events."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS spec_agent_sessions (
                    session_key TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS spec_agent_turns (
                    turn_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    origin_channel TEXT NOT NULL DEFAULT 'web',
                    sender_name TEXT NOT NULL DEFAULT '',
                    user_content TEXT NOT NULL,
                    assistant_content TEXT NOT NULL DEFAULT '',
                    assistant_reasoning TEXT NOT NULL DEFAULT '',
                    provider_id TEXT NOT NULL DEFAULT '',
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    steps INTEGER NOT NULL DEFAULT 0,
                    tool_calls_json TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'running',
                    created_at REAL NOT NULL,
                    completed_at REAL
                );
                CREATE INDEX IF NOT EXISTS idx_spec_agent_turns_session ON spec_agent_turns(session_key, created_at);
                CREATE TABLE IF NOT EXISTS spec_agent_events (
                    event_id TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_spec_agent_events_session ON spec_agent_events(session_key, created_at);
                """
            )

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def set_auto_title_if_default(self, *, session_key: str, owner_id: str, title: str) -> None:
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT title FROM spec_agent_sessions WHERE session_key=?", (session_key,)
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO spec_agent_sessions (session_key, owner_id, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                    (session_key, owner_id, title, now, now),
                )
            elif not row["title"]:
                conn.execute(
                    "UPDATE spec_agent_sessions SET title=?, updated_at=? WHERE session_key=?",
                    (title, now, session_key),
                )

    def start_turn(
        self,
        *,
        session_key: str,
        owner_id: str,
        sender_name: str,
        origin_channel: str,
        user_content: str,
        turn_id: str,
        trace_id: str,
    ) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO spec_agent_sessions (session_key, owner_id, title, created_at, updated_at)
                   VALUES (?, ?, '', ?, ?) ON CONFLICT(session_key) DO UPDATE SET updated_at=excluded.updated_at""",
                (session_key, owner_id, now, now),
            )
            conn.execute(
                """INSERT INTO spec_agent_turns
                   (turn_id, session_key, owner_id, trace_id, origin_channel, sender_name, user_content, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
                (turn_id, session_key, owner_id, trace_id, origin_channel, sender_name, user_content, now),
            )

    def append_event(self, event: SessionEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO spec_agent_events
                   (event_id, session_key, turn_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.session_key,
                    event.turn_id,
                    event.type,
                    json.dumps(event.payload, ensure_ascii=False, default=str),
                    event.timestamp,
                ),
            )

    def complete_turn(
        self,
        *,
        turn_id: str,
        assistant_content: str,
        assistant_reasoning: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_id: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        steps: int = 1,
        status: str = "completed",
    ) -> bool:
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                """UPDATE spec_agent_turns
                   SET assistant_content=?, assistant_reasoning=?, input_tokens=?, output_tokens=?,
                       provider_id=?, tool_calls_json=?, steps=?, status=?, completed_at=?
                   WHERE turn_id=? AND status='running'""",
                (
                    assistant_content,
                    assistant_reasoning,
                    int(input_tokens),
                    int(output_tokens),
                    provider_id,
                    json.dumps(tool_calls or [], ensure_ascii=False, default=str),
                    int(steps),
                    status,
                    now,
                    turn_id,
                ),
            )
            return bool(cur.rowcount)

    def list_turns(self, session_key: str, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM spec_agent_turns WHERE session_key=? ORDER BY created_at DESC LIMIT ?",
                (session_key, max(1, min(200, int(limit)))),
            ).fetchall()
        out = []
        for row in reversed(rows):
            item = dict(row)
            item["tool_calls"] = json.loads(item.pop("tool_calls_json") or "[]")
            out.append(item)
        return out

    def build_llm_messages(
        self,
        session_key: str,
        *,
        limit_turns: int = 40,
        preserve_thinking: bool = True,
    ) -> list[dict[str, Any]]:
        turns = self.list_turns(session_key, limit=limit_turns)
        messages: list[dict[str, Any]] = []
        for turn in turns:
            if turn["status"] == "running":
                continue
            messages.append({"role": "user", "content": turn["user_content"]})
            content = turn["assistant_content"] or ""
            reasoning = turn["assistant_reasoning"] or ""
            if content or reasoning:
                msg: dict[str, Any] = {"role": "assistant", "content": content}
                if preserve_thinking and reasoning:
                    msg["reasoning_content"] = reasoning
                messages.append(msg)
        return messages


def normalize_loop_event(event: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    kind = str(event.get("type") or "")
    payload = {key: value for key, value in event.items() if key != "type"}
    if kind in {"tool_call_started", "tool_call_result", "tool_call_error"}:
        payload.update(
            tool_call_id=payload.get("id", ""),
            tool_name=payload.get("name", ""),
            arguments=payload.get("args", {}),
            result_preview=payload.get("output_preview", ""),
        )
        if kind != "tool_call_started":
            kind = "tool_call_completed"
    elif kind == "compact_boundary":
        kind = "context_compacted"
    return kind, payload


class TurnRunner:
    """Bookkeeping wrapper bridging AgentLoop events to SessionStore and EventBus."""

    def __init__(
        self,
        *,
        session_store: SessionStore | None,
        event_bus: EventBus | None,
        session_key: str,
        owner_id: str,
        turn_id: str,
        trace_id: str,
        origin_channel: str = "web",
    ) -> None:
        self.store = session_store
        self.bus = event_bus
        self.session_key = session_key
        self.owner_id = owner_id
        self.turn_id = turn_id
        self.trace_id = trace_id
        self.origin_channel = origin_channel
        self.closed = False
        self.input_tokens = 0
        self.output_tokens = 0

    async def emit(self, kind: str, payload: dict[str, Any]) -> SessionEvent:
        event = SessionEvent(
            session_key=self.session_key,
            user_id=self.owner_id,
            origin_channel=self.origin_channel,
            trace_id=self.trace_id,
            turn_id=self.turn_id,
            event_type=kind,
            payload=payload,
        )
        if self.bus is not None:
            await self.bus.emit(event)
        return event

    async def begin(self, *, user_text: str, sender_name: str = "") -> None:
        if self.store is not None:
            self.store.start_turn(
                session_key=self.session_key,
                owner_id=self.owner_id,
                sender_name=sender_name,
                origin_channel=self.origin_channel,
                user_content=user_text,
                turn_id=self.turn_id,
                trace_id=self.trace_id,
            )
        await self.emit(
            "turn_started",
            {"user_text": user_text, "sender_name": sender_name, "origin_channel": self.origin_channel},
        )

    async def event_callback(self, event: dict[str, Any]) -> None:
        if self.closed:
            return
        kind, payload = normalize_loop_event(event)
        if kind == "context_usage" and payload.get("usage_kind") == "step":
            self.input_tokens += int(payload.get("input_tokens") or 0)
            self.output_tokens += int(payload.get("output_tokens") or 0)
        evt = await self.emit(kind, payload)
        if self.store is not None and kind in {
            "tool_call_started",
            "tool_call_completed",
            "context_compacted",
            "context_usage",
        }:
            try:
                self.store.append_event(evt)
            except Exception as exc:
                logger.warning("Failed to persist session event: %s", exc)

    async def complete(
        self,
        *,
        assistant_content: str,
        assistant_reasoning: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        provider_id: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        steps: int = 1,
        status: str = "completed",
    ) -> None:
        if self.closed:
            return
        self.closed = True
        if self.store is not None:
            changed = self.store.complete_turn(
                turn_id=self.turn_id,
                assistant_content=assistant_content,
                assistant_reasoning=assistant_reasoning,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_id=provider_id,
                tool_calls=tool_calls or [],
                steps=steps,
                status=status,
            )
            if changed is False:
                return
        await self.emit(
            "turn_complete",
            {
                "text": assistant_content,
                "reasoning": assistant_reasoning,
                "status": status,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tool_records": tool_calls or [],
            },
        )

    async def fail(self, error: Exception) -> None:
        if self.closed:
            return
        await self.complete(
            assistant_content=f"[Error: {error}]",
            status="error",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
        await self.emit("error", {"error": str(error)})

    async def cancel(self) -> None:
        await self.complete(
            assistant_content="",
            status="stopped",
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
        )
