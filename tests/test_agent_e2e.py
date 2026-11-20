"""Full local speculative-agent journey with real Git, gates, tools and scheduling."""

from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from speculative_coding.agent import SpecAgent
from speculative_coding.automation import SpecAutomation
from speculative_coding.config import SpecConfig
from speculative_coding.github import SpecRepoRegistry
from speculative_coding.llm.types import LLMStreamResult, ToolCallRequest
from speculative_coding.orchestrator import SpeculativeOrchestrator
from speculative_coding.sandbox import LocalDirSandbox
from speculative_coding.sessions import EventBus, SessionStore


class SpecAgentEndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        (source / "requirements.txt").write_text("pytest\n")
        (source / "app.py").write_text("value = 1\n")
        (source / "tests").mkdir()
        (source / "tests" / "test_app.py").write_text("from app import value\n\ndef test_value():\n    assert value >= 1\n")
        subprocess.run(["git", "init", "-b", "master"], cwd=source, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial"],
            cwd=source,
            check=True,
            capture_output=True,
        )

        config = SpecConfig(data_dir=self.root, default_timezone="Asia/Shanghai")
        self.sessions = SessionStore(self.root / "sessions.db")
        self.bus = EventBus()
        self.notifier = SimpleNamespace(send_notification=AsyncMock(return_value=True))
        self.backend = LocalDirSandbox(self.root / "pool", unsafe_test_mode=True, min_free_gb=0)
        self.engine = SpeculativeOrchestrator(config, self.backend)
        self.project = self.engine.store.new_project(
            "owner",
            {"name": "local-demo", "repo_source": str(source), "repo_full_name": "example/local-demo"},
        )
        self.registry = SpecRepoRegistry(self.engine.store, None)
        self.agent = SpecAgent(
            config,
            self.engine,
            repo_registry=self.registry,
            session_store=self.sessions,
            event_bus=self.bus,
        )
        self.automation = SpecAutomation(self.engine, self.agent, notifier=self.notifier)
        self.addAsyncCleanup(self.engine.stop)

        async def coding_agent(project, role, instruction, context, **kwargs):
            if role == "recon":
                return {
                    "summary": "Improve app value",
                    "routes": [{"title": "Upgrade app value", "plan": "Upgrade value in app.py", "kind": "optimization"}],
                }
            if role == "coder":
                generation = kwargs["candidate"]["generation"]
                await self.backend.call(
                    "write",
                    project_id=project["project_id"],
                    cand_id=kwargs["candidate"]["cand_id"],
                    base_sha=context["base_sha"],
                    path="app.py",
                    content=f"value = {generation + 2}\n",
                )
                return {"summary": "implemented"}
            if "three-minute" in instruction:
                return {"report": "Real pytest passed; upgraded app value."}
            return {"decision": "NOMINATE", "feedback": "Real diff and pytest evidence verified"}

        self.engine.agent = coding_agent

    def scripted(self, *responses):
        self.agent.router.complete_step = AsyncMock(side_effect=list(responses))

    async def wait_for(self, predicate, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            await asyncio.sleep(0.1)
        self.fail("timed out waiting for speculative lifecycle")

    async def test_delegate_explain_steer_and_morning_brief(self):
        self.scripted(
            LLMStreamResult(
                tool_calls=[ToolCallRequest("delegate", "spec_delegate", json.dumps({"goal": "Upgrade value", "run_now": True}))],
                input_tokens=5,
                output_tokens=2,
            ),
            LLMStreamResult(content="Provisioned and started.", input_tokens=3, output_tokens=3),
        )
        result = await self.agent.submit_turn(
            session_key="spec:owner:main", owner_id="owner", sender_name="operator", user_text="Upgrade value"
        )
        self.assertTrue(result["ok"], result)
        candidate = await self.wait_for(
            lambda: next(
                (item for item in self.engine.store.rows("candidates") if item["status"] == "staged" and item["generation"] == 0),
                None,
            )
        )
        self.assertIn("+value = 2", candidate["diff_text"])

        explained = await self.agent.execute_tool(
            "spec_explain", {"cand_id": candidate["cand_id"]}, {"owner_id": "owner", "is_admin": True}
        )
        self.assertEqual(explained["candidate"]["head_sha"], candidate["head_sha"])
        self.assertIn("gate", [event["kind"] for event in explained["events"]])

        action = await self.agent.execute_tool(
            "spec_candidate_action",
            {"cand_id": candidate["cand_id"], "operation": "steer", "notes": "Upgrade value to 3"},
            {"owner_id": "owner", "is_admin": True},
        )
        self.assertTrue(action["ok"], action)
        revised = await self.wait_for(
            lambda: next(
                (item for item in self.engine.store.rows("candidates") if item["status"] == "staged" and item["generation"] == 1),
                None,
            )
        )
        self.assertIn("+value = 3", revised["diff_text"])
        self.assertNotEqual(revised["head_sha"], candidate["head_sha"])

        self.scripted(
            LLMStreamResult(
                tool_calls=[ToolCallRequest("list", "spec_candidates", json.dumps({"since_hours": 24}))],
                input_tokens=4,
                output_tokens=2,
            ),
            LLMStreamResult(
                tool_calls=[ToolCallRequest("explain", "spec_explain", json.dumps({"cand_id": revised["cand_id"]}))],
                input_tokens=4,
                output_tokens=2,
            ),
            LLMStreamResult(content="# Morning Review\nRecommended Upgrade app value; real pytest passed.", input_tokens=4, output_tokens=4),
        )
        automation = await self.automation.create("owner", self.project["project_id"], {"kind": "morning_brief", "hour": 8, "minute": 0})
        self.engine.store.update("automations", automation["automation_id"], next_fire_at=time.time() - 1)
        await self.automation.tick()
        notifications = self.engine.store.rows("notifications", "owner_id=?", ("owner",))
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["status"], "sent")
        self.assertIn("real pytest passed", notifications[0]["full_text"])
        self.notifier.send_notification.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
