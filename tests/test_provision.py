"""Automatic project provisioning and baseline-gate contracts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from speculative_coding.config import SpecConfig
from speculative_coding.orchestrator import SpeculativeOrchestrator
from speculative_coding.provision import SpecProvisioner, infer_gate_commands


class InferenceTests(unittest.TestCase):
    def test_detects_python_and_nested_locked_node_projects(self):
        tree = "\n".join(
            [
                "requirements.txt",
                "tests/test_app.py",
                "web/package.json",
                "web/package-lock.json",
                "README.md",
            ]
        )
        files = {
            "requirements.txt": "aiohttp\n",
            "tests/test_app.py": "def test_ok(): assert True\n",
            "web/package.json": '{"scripts":{"test":"node --test","build":"vite build"}}',
            "web/package-lock.json": "{}",
        }
        commands = infer_gate_commands(tree, files)
        self.assertEqual(
            commands,
            [
                "python -m pytest tests -q --disable-warnings --tb=short",
                "npm --prefix web test",
                "npm --prefix web run build",
            ],
        )

    def test_never_infers_preview_or_unreliable_commands(self):
        self.assertEqual(infer_gate_commands("README.md\n", {"README.md": "npm start\npython accept.py"}), [])
        self.assertEqual(infer_gate_commands("web/package.json\n", {"web/package.json": '{"scripts":{"dev":"vite"}}'}), [])


class ProvisionBackend:
    def __init__(self, *, passed=True, with_manifest=True):
        self.passed = passed
        self.with_manifest = with_manifest
        self.calls = []

    async def call(self, op, **args):
        self.calls.append((op, args))
        if op == "prepare":
            return {"base_sha": "a" * 40, "tree": "requirements.txt\ntests/test_app.py", "recent_commits": "abc initial"}
        if op == "read":
            if not self.with_manifest:
                raise ValueError("not found")
            return {"content": "aiohttp\n" if args["path"] == "requirements.txt" else "def test_ok(): assert True\n"}
        if op == "hydrate":
            return {"path": "/sandbox/baseline"}
        if op == "gate":
            return {"passed": self.passed, "gates": [{"code": 0 if self.passed else 1, "output": "baseline"}]}
        if op == "drop":
            return {"dropped": True}
        raise AssertionError(op)


class ProvisionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        config = SpecConfig(data_dir=Path(self.temp.name))
        self.backend = ProvisionBackend()
        self.engine = SpeculativeOrchestrator(config, self.backend)
        self.provisioner = SpecProvisioner(self.engine)
        self.resolved = {
            "status": "ok",
            "project": None,
            "repo": {
                "repo_id": "42",
                "installation_id": "7",
                "full_name": "example/demo",
                "name": "demo",
                "clone_url": "https://github.com/example/demo.git",
                "default_branch": "main",
                "private": 0,
            },
        }

    async def test_auto_project_runs_baseline_and_cleans_up(self):
        result = await self.provisioner.ensure_project("owner", self.resolved, "find improvements")
        self.assertEqual(result["status"], "ok")
        project = result["project"]
        self.assertEqual(project["target_branch"], "main")
        self.assertEqual(project["repo_full_name"], "example/demo")
        self.assertEqual(project["gate_commands_json"], ["python -m pytest tests -q --disable-warnings --tb=short"])
        self.assertEqual(self.store_count(project["project_id"]), 0)
        operations = [op for op, _ in self.backend.calls]
        self.assertEqual(operations[-1], "drop")
        self.assertEqual(self.engine.store.rows("events", "owner_id=?", ("owner",))[-1]["kind"], "baseline")

    async def test_baseline_failure_does_not_create_candidates_or_bypass_retry(self):
        self.backend.passed = False
        result = await self.provisioner.ensure_project("owner", self.resolved, "find improvements")
        self.assertEqual(result["status"], "baseline_failed")
        self.assertEqual(self.store_count(result["project"]["project_id"]), 0)
        self.assertEqual(result["project"]["gate_commands_json"], [])
        resolved_failed = {**self.resolved, "project": result["project"]}
        retry = await self.provisioner.ensure_project("owner", resolved_failed, "find improvements")
        self.assertEqual(retry["status"], "baseline_failed")
        self.backend.passed = True
        recovered = await self.provisioner.ensure_project("owner", {**self.resolved, "project": retry["project"]}, "find improvements")
        self.assertEqual(recovered["status"], "ok")

    async def test_missing_manifest_requires_safe_setup(self):
        self.backend.with_manifest = False
        result = await self.provisioner.ensure_project("owner", self.resolved, "find improvements")
        self.assertEqual(result["status"], "gates_required")
        self.assertFalse(any(op == "gate" for op, _ in self.backend.calls))

    def store_count(self, project_id):
        return len(self.engine.store.rows("candidates", "project_id=?", (project_id,)))


if __name__ == "__main__":
    unittest.main()
