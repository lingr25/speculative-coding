"""Speculative agent persistence, conversation, and standalone server HTTP/WS contracts."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aiohttp.test_utils import TestClient, TestServer

from speculative_coding.agent import SpecAgent
from speculative_coding.config import SpecConfig
from speculative_coding.github import SpecRepoRegistry
from speculative_coding.llm.types import LLMStreamResult, ToolCallRequest
from speculative_coding.orchestrator import SpeculativeOrchestrator
from speculative_coding.protocols import BearerTokenAuthProvider
from speculative_coding.server import create_control_plane
from speculative_coding.sessions import EventBus, SessionStore
from speculative_coding.store import SpecStore


class SpecAgentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "spec-agent.db"
        self.store = SpecStore(self.db_path)
        self.project = self.store.new_project(
            "owner",
            {"name": "demo", "repo_source": "https://github.com/example/demo.git"},
        )

    def test_project_repo_metadata_and_legacy_upgrade(self):
        self.assertEqual(self.project["repo_id"], "")
        with self.store.conn() as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(spec_projects)")}
        self.assertTrue({"repo_id", "repo_full_name", "github_installation_id"} <= columns)

        legacy_path = Path(self.temp.name) / "legacy.db"
        with closing(sqlite3.connect(legacy_path)) as db:
            db.execute(
                "CREATE TABLE spec_projects (project_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, name TEXT NOT NULL, repo_source TEXT NOT NULL, target_branch TEXT NOT NULL DEFAULT 'master')"
            )
        upgraded = SpecStore(legacy_path)
        with upgraded.conn() as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(spec_projects)")}
        self.assertTrue({"repo_id", "repo_full_name", "github_installation_id"} <= columns)

    def test_repo_sync_is_owned_and_contains_no_credentials(self):
        repo = self.store.upsert_repo(
            "owner",
            {
                "repo_id": 42,
                "installation_id": 7,
                "full_name": "example/demo",
                "clone_url": "https://github.com/example/demo.git",
                "default_branch": "main",
                "description": "Demo",
                "private": True,
                "language": "Python",
                "topics": ["agent", "coding"],
            },
        )
        self.assertEqual(repo["full_name"], "example/demo")
        self.assertEqual(repo["topics_json"], ["agent", "coding"])
        self.assertEqual(self.store.rows("repos", "owner_id=?", ("owner",))[0]["repo_id"], "42")
        self.assertEqual(self.store.rows("repos", "owner_id=?", ("other",)), [])
        with self.store.conn() as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(spec_repos)")}
        self.assertNotIn("token", columns)
        self.assertNotIn("private_key", columns)

    def test_memory_supersedes_and_events_are_scoped(self):
        first = self.store.remember("owner", self.project["project_id"], "report_style", "detailed", source="user")
        second = self.store.remember("owner", self.project["project_id"], "report_style", "top-3", source="user")
        self.assertNotEqual(first, second)
        active = self.store.memories("owner", self.project["project_id"])
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["value"], "top-3")
        self.assertEqual(active[0]["supersedes"], first)
        self.assertEqual(self.store.memories("other", self.project["project_id"]), [])

        event_id = self.store.record_event(
            "owner",
            self.project["project_id"],
            "review",
            "Code review passed",
            run_id="run-1",
            cand_id="cand-1",
            head_sha="b" * 40,
            payload={"decision": "NOMINATE", "evidence": ["tests/test_x.py"]},
        )
        events = self.store.events("owner", cand_id="cand-1")
        self.assertEqual(events[0]["event_id"], event_id)
        self.assertEqual(events[0]["payload_json"]["decision"], "NOMINATE")
        self.assertEqual(self.store.events("other", cand_id="cand-1"), [])

    def test_automation_and_notification_idempotency_contracts(self):
        now = 1000.0
        self.store.insert(
            "automations",
            {
                "automation_id": "auto-1",
                "owner_id": "owner",
                "project_id": self.project["project_id"],
                "kind": "morning_brief",
                "hour": 8,
                "minute": 0,
                "timezone": "UTC",
                "always_notify": 0,
                "enabled": 1,
                "next_fire_at": now,
                "created_at": now,
                "updated_at": now,
            },
        )
        first = self.store.enqueue_notification("auto-1:1000", "owner", "spec:owner:main", "brief", "summary", now)
        second = self.store.enqueue_notification("auto-1:1000", "owner", "spec:owner:main", "brief", "summary", now + 1)
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(self.store.rows("notifications", "owner_id=?", ("owner",))), 1)


class SpecAgentConversationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        config = SpecConfig(data_dir=root)
        self.sessions = SessionStore(root / "sessions.db")
        self.bus = EventBus()
        self.backend = SimpleNamespace(call=AsyncMock(return_value={}))
        self.engine = SpeculativeOrchestrator(config, self.backend)
        self.project = self.engine.store.new_project(
            "owner",
            {"name": "demo", "repo_source": "https://github.com/example/demo.git", "gate_commands_json": ["python -m unittest"]},
        )
        self.started = []
        self.engine._spawn = lambda project: self.started.append(project["project_id"]) or True
        self.registry = SpecRepoRegistry(self.engine.store, None)
        self.agent = SpecAgent(
            config,
            self.engine,
            repo_registry=self.registry,
            session_store=self.sessions,
            event_bus=self.bus,
        )
        self.agent.provisioner.ensure_project = AsyncMock(
            return_value={"status": "ok", "project": self.project, "baseline_validated": True}
        )

    async def test_dialogue_delegate_uses_real_agent_loop_and_spec_pool(self):
        self.agent.router.complete_step = AsyncMock(
            side_effect=[
                LLMStreamResult(
                    tool_calls=[
                        ToolCallRequest("call-1", "spec_delegate", json.dumps({"goal": "find improvements", "run_now": True}))
                    ],
                    input_tokens=5,
                    output_tokens=3,
                ),
                LLMStreamResult(content="Delegated demo project and started the initial batch.", input_tokens=4, output_tokens=4),
            ]
        )
        result = await self.agent.submit_turn(
            session_key="spec:owner:main",
            owner_id="owner",
            sender_name="operator",
            user_text="Explore improvements for demo",
        )
        self.assertTrue(result["ok"], result)
        self.assertEqual(self.started, [self.project["project_id"]])
        self.assertEqual(self.agent.router.pool, "speculative")
        memories = self.engine.store.memories("owner", self.project["project_id"])
        self.assertEqual(next(item for item in memories if item["key"] == "goal")["value"], "find improvements")
        history = self.sessions.build_llm_messages("spec:owner:main")
        self.assertEqual(history[-1]["content"], "Delegated demo project and started the initial batch.")

    async def test_explain_binds_candidate_sha_and_evidence(self):
        now = 1000.0
        self.engine.store.insert(
            "runs",
            {"run_id": "run-1", "project_id": self.project["project_id"], "base_sha": "a" * 40, "status": "completed", "created_at": now},
        )
        self.engine.store.insert(
            "candidates",
            {
                "cand_id": "cand-1",
                "run_id": "run-1",
                "project_id": self.project["project_id"],
                "route_index": 0,
                "title": "Improvement",
                "route_plan": "Complete improvement",
                "status": "staged",
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "human_report_md": "Report",
                "diff_text": "+real",
                "created_at": now,
                "updated_at": now,
            },
        )
        self.engine.store.record_event(
            "owner",
            self.project["project_id"],
            "review",
            "NOMINATE",
            run_id="run-1",
            cand_id="cand-1",
            head_sha="b" * 40,
            payload={"decision": "NOMINATE"},
        )
        result = await self.agent.execute_tool("spec_explain", {"cand_id": "cand-1"}, {"owner_id": "owner", "is_admin": True})
        self.assertEqual(result["candidate"]["head_sha"], "b" * 40)
        self.assertEqual(result["candidate"]["diff_text"], "+real")
        self.assertEqual(result["events"][0]["payload_json"]["decision"], "NOMINATE")
        denied = await self.agent.execute_tool("spec_explain", {"cand_id": "cand-1"}, {"owner_id": "other", "is_admin": True})
        self.assertIn("error", denied)

    async def test_agent_injects_only_owned_engineering_memory(self):
        self.engine.store.remember("owner", self.project["project_id"], "report_style", "top-3-only", source="user")
        self.engine.store.remember("other", "", "report_style", "must-not-leak", source="user")
        self.agent.router.complete_step = AsyncMock(return_value=LLMStreamResult(content="OK"))
        result = await self.agent.submit_turn(
            session_key="spec:owner:main",
            owner_id="owner",
            sender_name="operator",
            user_text="continue",
        )
        self.assertTrue(result["ok"], result)
        system = self.agent.router.complete_step.await_args.kwargs["messages"][0]["content"]
        self.assertIn("top-3-only", system)
        self.assertNotIn("must-not-leak", system)

    async def test_control_tools_have_no_merge_or_arbitrary_repo_input(self):
        tools = self.agent.get_tools()
        names = {item["function"]["name"] for item in tools}
        self.assertNotIn("spec_merge", names)
        delegate = next(item for item in tools if item["function"]["name"] == "spec_delegate")
        self.assertNotIn("repo_source", delegate["function"]["parameters"]["properties"])
        result = await self.agent.execute_tool(
            "spec_candidate_action", {"cand_id": "x", "operation": "merge"}, {"owner_id": "owner", "is_admin": True}
        )
        self.assertIn("error", result)
        wrong_session = await self.agent.submit_turn(
            session_key="spec:other:main", owner_id="owner", sender_name="operator", user_text="test"
        )
        self.assertFalse(wrong_session["ok"])


class StandaloneServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.token = "a" * 32
        self.user = {"owner_id": "owner", "name": "operator", "is_admin": True}

        class SwitchingAuth:
            def __init__(self, test_case):
                self.tc = test_case

            def require_admin(self, request):
                return self.tc.user

        config = SpecConfig(data_dir=Path(self.temp.name))
        self.plane = create_control_plane(config, auth=SwitchingAuth(self))
        self.plane.agent.submit_turn = AsyncMock(return_value={"ok": True, "text": "done"})
        self.plane.agent.cancel_session = lambda session_key: 1
        self.client = TestClient(TestServer(self.plane.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_health_and_ws_chat_stop_and_admin_guard(self):
        resp = await self.client.get("/health")
        self.assertEqual(resp.status, 200)
        body = await resp.json()
        self.assertTrue(body["ok"])

        ws = await self.client.ws_connect("/api/spec/agent/ws")
        hello = await ws.receive_json()
        self.assertEqual(hello["type"], "connected")
        self.assertEqual(hello["session_key"], "spec:owner:main")
        await ws.send_json({"type": "chat", "text": "find improvements"})
        await asyncio.sleep(0.03)
        self.plane.agent.submit_turn.assert_awaited_once()
        self.assertEqual(self.plane.agent.submit_turn.await_args.kwargs["session_key"], "spec:owner:main")
        await ws.send_json({"type": "stop", "session_key": "spec:owner:main"})
        packet = await ws.receive_json()
        self.assertEqual(packet["type"], "stopped")
        self.assertEqual(packet["result"]["running"], 1)
        await ws.close()

        self.user = {"owner_id": "member", "name": "member", "is_admin": False}
        self.plane.agent.submit_turn.reset_mock()
        ws2 = await self.client.ws_connect("/api/spec/agent/ws")
        await ws2.receive_json()
        await ws2.send_json({"type": "chat", "text": "unauthorized"})
        err = await ws2.receive_json()
        self.assertEqual(err["type"], "error")
        self.assertEqual(err["code"], "forbidden")
        self.plane.agent.submit_turn.assert_not_awaited()
        await ws2.close()

    async def test_bearer_token_auth_enforces_token(self):
        cfg = SpecConfig(data_dir=Path(self.temp.name) / "bearer", api_token=self.token)
        plane = create_control_plane(cfg, auth=BearerTokenAuthProvider(self.token, owner_id="admin"))
        client = TestClient(TestServer(plane.app))
        await client.start_server()
        self.addAsyncCleanup(client.close)

        unauth = await client.get("/api/spec/status")
        self.assertEqual(unauth.status, 401)
        authed = await client.get("/api/spec/status", headers={"Authorization": "Bearer " + self.token})
        self.assertEqual(authed.status, 200)
        payload = await authed.json()
        self.assertTrue(payload["ok"])


if __name__ == "__main__":
    unittest.main()
