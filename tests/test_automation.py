"""Persistent speculative automations and pluggable notification delivery tests."""

from __future__ import annotations

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
from speculative_coding.orchestrator import SpeculativeOrchestrator


class AutomationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        config = SpecConfig(data_dir=root, default_timezone="Asia/Shanghai")
        self.notifier = SimpleNamespace(send_notification=AsyncMock(return_value=True))
        self.engine = SpeculativeOrchestrator(config, SimpleNamespace())
        self.project = self.engine.store.new_project(
            "owner",
            {"name": "demo", "repo_source": "https://github.com/example/demo.git", "gate_commands_json": ["test"]},
        )
        self.agent = SpecAgent(config, self.engine, repo_registry=SpecRepoRegistry(self.engine.store, None))
        self.agent.submit_turn = AsyncMock(return_value={"ok": True, "text": "# Morning Review\nRecommended candidate 1."})
        self.automation = SpecAutomation(self.engine, self.agent, notifier=self.notifier)

    def add_candidate(self):
        now = time.time()
        self.engine.store.insert(
            "runs",
            {"run_id": "run", "project_id": self.project["project_id"], "base_sha": "a" * 40, "status": "completed", "created_at": now},
        )
        self.engine.store.insert(
            "candidates",
            {
                "cand_id": "cand",
                "run_id": "run",
                "project_id": self.project["project_id"],
                "route_index": 0,
                "title": "Candidate 1",
                "route_plan": "Implement feature",
                "status": "staged",
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "created_at": now,
                "updated_at": now,
            },
        )

    async def create_brief(self, **overrides):
        return await self.automation.create(
            "owner",
            self.project["project_id"],
            {"kind": "morning_brief", "hour": 8, "minute": 5, **overrides},
        )

    async def test_due_brief_runs_once_across_restart_and_uses_same_session(self):
        self.add_candidate()
        created = await self.create_brief()
        self.assertEqual(created["hour"], 8)
        self.engine.store.update("automations", created["automation_id"], next_fire_at=time.time() - 1)
        await self.automation.tick()
        self.automation2 = SpecAutomation(self.engine, self.agent, notifier=self.notifier)
        await self.automation2.tick()
        self.agent.submit_turn.assert_awaited_once()
        self.assertEqual(self.agent.submit_turn.await_args.kwargs["session_key"], "spec:owner:main")
        self.notifier.send_notification.assert_awaited_once()
        notifications = self.engine.store.rows("notifications", "owner_id=?", ("owner",))
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0]["status"], "sent")

    async def test_no_candidates_is_quiet(self):
        created = await self.create_brief()
        self.engine.store.update("automations", created["automation_id"], next_fire_at=time.time() - 1)
        await self.automation.tick()
        self.agent.submit_turn.assert_not_awaited()
        self.notifier.send_notification.assert_not_awaited()
        self.assertEqual(self.engine.store.rows("notifications"), [])
        self.assertEqual(self.engine.store.get("automations", created["automation_id"])["last_status"], "empty")

    async def test_delivery_failure_preserves_full_report_without_blind_retry(self):
        self.add_candidate()
        self.notifier.send_notification = AsyncMock(return_value=False)
        created = await self.create_brief()
        self.engine.store.update("automations", created["automation_id"], next_fire_at=time.time() - 1)
        await self.automation.tick()
        await self.automation.tick()
        self.agent.submit_turn.assert_awaited_once()
        notification = self.engine.store.rows("notifications")[0]
        self.assertEqual(notification["status"], "unknown")
        self.assertIn("Recommended candidate 1", notification["full_text"])

    async def test_run_batch_uses_existing_orchestrator(self):
        self.engine._spawn = lambda project: True
        created = await self.automation.create(
            "owner", self.project["project_id"], {"kind": "run_batch", "hour": 7, "minute": 30}
        )
        self.engine.store.update("automations", created["automation_id"], next_fire_at=time.time() - 1)
        await self.automation.tick()
        self.agent.submit_turn.assert_not_awaited()
        self.assertEqual(self.engine.store.get("automations", created["automation_id"])["last_status"], "ok")


if __name__ == "__main__":
    unittest.main()
