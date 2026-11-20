"""Standalone aiohttp control-plane HTTP & WebSocket server for Speculative Coding."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
) -> ControlPlane:
    """Assemble and wire a standalone Speculative Coding control-plane web application."""
    if auth is None:
        if config.api_token:
            auth = BearerTokenAuthProvider(config.api_token, owner_id=config.default_owner_id)
        else:
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

    async def health_api(_request: web.Request) -> web.Response:
        worker = (
            engine.backend.status()
            if isinstance(engine.backend, RemoteWorkerSandbox)
            else {"connected": True}
        )
        return web.json_response({"ok": True, "worker_connected": bool(worker.get("connected"))})

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

    app.router.add_get("/health", health_api)
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.config:
        config = SpecConfig.load_from_file(args.config)
    else:
        config = SpecConfig.from_env(data_dir=Path(args.data_dir) if args.data_dir else None)
    plane = create_control_plane(config)
    logger.info("Starting Speculative Coding control plane on http://%s:%d", args.host, args.port)
    web.run_app(plane.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
