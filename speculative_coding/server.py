"""Standalone aiohttp control-plane HTTP & WebSocket server for Speculative Coding."""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import mimetypes
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from aiohttp import WSMsgType, web

from .agent import SpecAgent
from .automation import SpecAutomation
from .config import SpecConfig
from .github import GitHubAppClient, SpecRepoRegistry
from .orchestrator import SpeculativeOrchestrator
from .protocols import AuthProvider, BearerTokenAuthProvider, NotificationSink, SingleOwnerAuthProvider
from .sandbox import RemoteWorkerSandbox, SpecSandboxBackend
from .sessions import EventBus, SessionStore

logger = logging.getLogger(__name__)

DEFAULT_UI_STATIC_DIR = Path(__file__).resolve().parent / "ui_static"
_WS_TICKET_TTL_SECONDS = 60.0
_WS_TICKET_MAX_PENDING = 256

_STATIC_MIME_TYPES: dict[str, tuple[str, str | None]] = {
    ".html": ("text/html", "utf-8"),
    ".htm": ("text/html", "utf-8"),
    ".css": ("text/css", "utf-8"),
    ".js": ("application/javascript", "utf-8"),
    ".mjs": ("application/javascript", "utf-8"),
    ".json": ("application/json", "utf-8"),
    ".svg": ("image/svg+xml", "utf-8"),
    ".txt": ("text/plain", "utf-8"),
    ".map": ("application/json", "utf-8"),
    ".png": ("image/png", None),
    ".jpg": ("image/jpeg", None),
    ".jpeg": ("image/jpeg", None),
    ".webp": ("image/webp", None),
    ".gif": ("image/gif", None),
    ".ico": ("image/x-icon", None),
    ".woff": ("font/woff", None),
    ".woff2": ("font/woff2", None),
}


def _is_loopback_host(host: str) -> bool:
    """Return True only when host resolves literally to a loopback interface."""
    cleaned = str(host or "").strip().lower()
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]
    if cleaned == "localhost":
        return True
    if not cleaned:
        return False
    try:
        return ipaddress.ip_address(cleaned).is_loopback
    except ValueError:
        return False


def validate_bind_auth(
    config: SpecConfig,
    host: str,
    *,
    allow_insecure_no_auth: bool = False,
    has_custom_auth: bool = False,
) -> None:
    """Fail closed when binding to a non-loopback address without API authentication."""
    if has_custom_auth or (config.api_token or "").strip():
        return
    if not _is_loopback_host(host) and not allow_insecure_no_auth:
        raise RuntimeError(
            f"Refusing to start Speculative Coding control plane on non-loopback host {host!r} without api_token. "
            "Configure `api_token` / `SPEC_API_TOKEN` (>= 16 chars), bind to a loopback host (127.0.0.1 / ::1 / localhost), "
            "or explicitly pass `--insecure-no-auth`."
        )


def _has_raw_traversal(raw_path: str) -> bool:
    """Detect encoded or unencoded traversal segments in the raw HTTP request path."""
    lower = raw_path.lower()
    if "%5c" in lower or "\\" in lower or "\x00" in lower or "%00" in lower:
        return True
    decoded = unquote(raw_path)
    if "\\" in decoded or "\x00" in decoded:
        return True
    parts = [p for p in decoded.split("/") if p]
    return any(part in {".", ".."} for part in parts)


def _resolve_ui_static_file(static_root: Path, rel_path: str) -> Path | None:
    """Safely resolve a relative path inside static_root, rejecting traversal and escaping symlinks."""
    if not rel_path or "\x00" in rel_path or "\\" in rel_path:
        return None
    cleaned = rel_path.lstrip("/")
    if not cleaned:
        return None
    raw = Path(cleaned)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        return None
    try:
        root_resolved = static_root.resolve(strict=True)
    except OSError:
        return None
    cur = root_resolved
    for part in raw.parts:
        cur = cur / part
        if cur.is_symlink():
            try:
                if not cur.resolve(strict=True).is_relative_to(root_resolved):
                    return None
            except OSError:
                return None
    candidate = root_resolved / raw
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        return None
    return resolved


def _static_file_response(file_path: Path, *, cache_control: str = "no-cache") -> web.Response:
    suffix = file_path.suffix.lower()
    if suffix in _STATIC_MIME_TYPES:
        content_type, charset = _STATIC_MIME_TYPES[suffix]
    else:
        guessed = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        content_type = guessed
        charset = "utf-8" if guessed.startswith("text/") else None
    headers = {
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": cache_control,
    }
    return web.Response(
        body=file_path.read_bytes(),
        content_type=content_type,
        charset=charset,
        headers=headers,
    )


@dataclass
class ControlPlane:
    app: web.Application
    config: SpecConfig
    engine: SpeculativeOrchestrator
    repo_registry: SpecRepoRegistry
    agent: SpecAgent
    automation: SpecAutomation
    session_store: SessionStore
    event_bus: EventBus


def create_control_plane(
    config: SpecConfig,
    *,
    backend: SpecSandboxBackend | None = None,
    auth: AuthProvider | None = None,
    notifier: NotificationSink | None = None,
    prefix: str = "/api/spec",
    host: str = "127.0.0.1",
    allow_insecure_no_auth: bool = False,
    allow_owner_header: bool = False,
    ui_static_dir: Path | None = None,
) -> ControlPlane:
    """Assemble and wire a standalone Speculative Coding control-plane web application."""
    if auth is None:
        token = (config.api_token or "").strip()
        if token:
            if len(token) < 16:
                logger.warning(
                    "SECURITY WARNING: Configured api_token is shorter than 16 characters; "
                    "BearerTokenAuthProvider will reject all authenticated requests until a >=16-char token is set."
                )
            auth = BearerTokenAuthProvider(
                token,
                owner_id=config.default_owner_id,
                allow_owner_header=allow_owner_header,
            )
        else:
            validate_bind_auth(config, host, allow_insecure_no_auth=allow_insecure_no_auth)
            logger.warning(
                "SECURITY WARNING: Starting Speculative Coding control plane on host=%s without api_token; "
                "SingleOwnerAuthProvider (no auth) is active. Configure SPEC_API_TOKEN or config.api_token for authenticated access.",
                host,
            )
            auth = SingleOwnerAuthProvider(owner_id=config.default_owner_id)

    session_store = SessionStore(config.sessions_db_path)
    event_bus = EventBus()
    engine = SpeculativeOrchestrator(config, backend=backend, auth=auth)
    github_client = GitHubAppClient(
        app_id=config.github_app_id,
        installation_id=config.github_installation_id,
        private_key_path=config.github_private_key_path,
    )
    repo_registry = SpecRepoRegistry(engine.store, github_client)
    engine.repo_registry = repo_registry
    agent = SpecAgent(
        config,
        engine,
        repo_registry=repo_registry,
        session_store=session_store,
        event_bus=event_bus,
    )
    engine.conversation_agent = agent
    automation = SpecAutomation(engine, agent, notifier=notifier)

    app = web.Application()
    engine.register(app, prefix=prefix)
    base = prefix.rstrip("/")
    static_root = Path(ui_static_dir) if ui_static_dir is not None else DEFAULT_UI_STATIC_DIR
    ws_tickets: dict[str, tuple[float, dict[str, Any]]] = {}

    def _prune_ws_tickets(now: float) -> None:
        expired = [k for k, (exp, _) in ws_tickets.items() if exp <= now]
        for k in expired:
            ws_tickets.pop(k, None)
        while len(ws_tickets) >= _WS_TICKET_MAX_PENDING:
            oldest = next(iter(ws_tickets))
            ws_tickets.pop(oldest, None)

    async def health_api(_request: web.Request) -> web.Response:
        worker = (
            engine.backend.status()
            if isinstance(engine.backend, RemoteWorkerSandbox)
            else {"connected": True}
        )
        return web.json_response({"ok": True, "worker_connected": bool(worker.get("connected"))})

    async def ui_index_api(_request: web.Request) -> web.Response:
        index_file = _resolve_ui_static_file(static_root, "index.html")
        if index_file is None:
            return web.Response(
                status=404,
                text="Speculative Coding Review UI is not available (ui_static/index.html missing).",
                content_type="text/plain",
                headers={"X-Content-Type-Options": "nosniff"},
            )
        return _static_file_response(index_file, cache_control="no-cache")

    async def ui_asset_api(request: web.Request) -> web.Response:
        if _has_raw_traversal(request.raw_path):
            raise web.HTTPNotFound()
        rel_path = request.match_info.get("path", "")
        if not rel_path or rel_path == "/":
            return await ui_index_api(request)
        resolved = _resolve_ui_static_file(static_root, rel_path)
        if resolved is None:
            raise web.HTTPNotFound()
        return _static_file_response(resolved, cache_control="no-cache")

    async def agent_ws_ticket_api(request: web.Request) -> web.Response:
        user = engine.auth.require_admin(request)
        now = time.monotonic()
        _prune_ws_tickets(now)
        ticket = secrets.token_urlsafe(32)
        ws_tickets[ticket] = (now + _WS_TICKET_TTL_SECONDS, dict(user))
        return web.json_response({"ok": True, "ticket": ticket, "expires_in": int(_WS_TICKET_TTL_SECONDS)})

    async def agent_turn_api(request: web.Request) -> web.Response:
        user = engine.auth.require_admin(request)
        if not user.get("is_admin"):
            return web.json_response({"ok": False, "error": "forbidden"}, status=403)
        owner = str(user.get("owner_id") or "")
        body = await request.json()
        session_key = str(body.get("session_key") or f"spec:{owner}:main")
        result = await agent.submit_turn(
            session_key=session_key,
            owner_id=owner,
            sender_name=str(user.get("name") or owner),
            user_text=str(body.get("text") or ""),
            title=body.get("title"),
            origin_channel="http",
        )
        status = 200 if result.get("ok") else 400
        return web.json_response(result, status=status)

    async def agent_turns_api(request: web.Request) -> web.Response:
        user = engine.auth.require_admin(request)
        if not user.get("is_admin"):
            raise web.HTTPForbidden(text="admin required")
        owner = str(user.get("owner_id") or "")
        session_key = request.match_info["session_key"]
        if session_key != f"spec:{owner}:main" and not session_key.startswith(f"spec:{owner}:"):
            raise web.HTTPForbidden(text="session owner mismatch")
        limit = max(1, min(200, int(request.query.get("limit", 50))))
        return web.json_response({"ok": True, "turns": session_store.list_turns(session_key, limit=limit)})

    async def agent_ws_api(request: web.Request) -> web.WebSocketResponse:
        ticket = request.query.get("ticket", "").strip()
        if ticket:
            entry = ws_tickets.pop(ticket, None)
            if entry is None or entry[0] < time.monotonic():
                raise web.HTTPUnauthorized(text="invalid or expired ws ticket")
            user = entry[1]
        else:
            user = engine.auth.require_admin(request)
        owner = str(user.get("owner_id") or "")
        is_admin = bool(user.get("is_admin"))
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(request)
        default_session = f"spec:{owner}:main"
        await ws.send_json({"type": "connected", "owner_id": owner, "session_key": default_session})

        async def _forward(event: Any) -> None:
            if not ws.closed:
                await ws.send_json(event.to_dict())

        unsubscribe = event_bus.subscribe(_forward, user_id=owner)
        active_tasks: set[asyncio.Task] = set()
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    await ws.send_json({"type": "error", "code": "invalid_json", "message": "Invalid JSON payload"})
                    continue
                kind = str(data.get("type") or "")
                if kind == "chat":
                    if not is_admin:
                        await ws.send_json({"type": "error", "code": "forbidden", "message": "Admin required"})
                        continue
                    session_key = str(data.get("session_key") or default_session)
                    text = str(data.get("text") or "")

                    async def _run_turn(skey: str, txt: str) -> None:
                        res = await agent.submit_turn(
                            session_key=skey,
                            owner_id=owner,
                            sender_name=str(user.get("name") or owner),
                            user_text=txt,
                            origin_channel="ws",
                        )
                        if not res.get("ok") and not ws.closed:
                            await ws.send_json({"type": "error", "code": res.get("error", "turn_failed")})

                    task = asyncio.create_task(_run_turn(session_key, text))
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
                elif kind == "stop":
                    session_key = str(data.get("session_key") or default_session)
                    cancelled = agent.cancel_session(session_key)
                    await ws.send_json({"type": "stopped", "session_key": session_key, "result": {"running": cancelled}})
        finally:
            unsubscribe()
            for task in list(active_tasks):
                task.cancel()
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)
        return ws

    app.router.add_get("/", ui_index_api)
    app.router.add_get("/ui", ui_index_api)
    app.router.add_get("/ui/", ui_index_api)
    app.router.add_get("/ui/{path:.*}", ui_asset_api)
    app.router.add_get("/health", health_api)
    app.router.add_post(base + "/agent/ws-ticket", agent_ws_ticket_api)
    app.router.add_post(base + "/agent/turn", agent_turn_api)
    app.router.add_get(base + "/agent/sessions/{session_key}/turns", agent_turns_api)
    app.router.add_get(base + "/agent/ws", agent_ws_api)

    return ControlPlane(
        app=app,
        config=config,
        engine=engine,
        repo_registry=repo_registry,
        agent=agent,
        automation=automation,
        session_store=session_store,
        event_bus=event_bus,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Speculative Coding standalone control-plane server")
    parser.add_argument("--config", default="", help="Path to JSON config file")
    parser.add_argument("--data-dir", default="", help="Data directory for SQLite stores")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--insecure-no-auth",
        action="store_true",
        help="Explicitly allow binding to a non-loopback host without api_token (UNSAFE)",
    )
    parser.add_argument(
        "--allow-owner-header",
        action="store_true",
        help="Trust client X-Spec-Owner header with BearerTokenAuthProvider (ONLY behind a trusted reverse proxy)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.config:
        config = SpecConfig.load_from_file(args.config)
    else:
        config = SpecConfig.from_env(data_dir=Path(args.data_dir) if args.data_dir else None)
    try:
        plane = create_control_plane(
            config,
            host=args.host,
            allow_insecure_no_auth=args.insecure_no_auth,
            allow_owner_header=args.allow_owner_header,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        raise SystemExit(str(exc)) from exc
    logger.info("Starting Speculative Coding control plane on http://%s:%d", args.host, args.port)
    web.run_app(plane.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
