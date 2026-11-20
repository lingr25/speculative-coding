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

    def test_infers_pnpm_yarn_uv_go_and_cargo(self):
        self.assertEqual(
            infer_gate_commands(
                "pnpm-lock.yaml\npackage.json\n",
                {"package.json": '{"scripts":{"test":"vitest run","build":"vite build"}}'},
            ),
            ["pnpm test", "pnpm run build"],
        )
        self.assertEqual(
            infer_gate_commands(
                "web/pnpm-lock.yaml\nweb/package.json\n",
                {"web/package.json": '{"scripts":{"test":"vitest run","build":"vite build"}}'},
            ),
            ["pnpm --dir web test", "pnpm --dir web run build"],
        )
        self.assertEqual(
            infer_gate_commands(
                "yarn.lock\npackage.json\napp/package.json\n",
                {
                    "package.json": '{"scripts":{"test":"jest"}}',
                    "app/package.json": '{"scripts":{"build":"tsc"}}',
                },
            ),
            ["yarn test", "yarn --cwd app run build"],
        )
        self.assertEqual(
            infer_gate_commands(
                "pyproject.toml\nuv.lock\ntests/test_core.py\n",
                {"pyproject.toml": '[project]\nname="demo"\n'},
            ),
            ["uv run --offline --no-sync pytest tests -q --disable-warnings --tb=short"],
        )
        self.assertEqual(
            infer_gate_commands(
                "pyproject.toml\napp_test.py\n",
                {"pyproject.toml": '[project]\nname="demo"\n'},
            ),
            ["python -m pytest -q --disable-warnings --tb=short"],
        )
        self.assertEqual(
            infer_gate_commands("go.mod\nmain_test.go\n", {"go.mod": "module example.com/demo\n"}),
            ["go test ./..."],
        )
        self.assertEqual(
            infer_gate_commands("Cargo.toml\nsrc/lib.rs\n", {"Cargo.toml": '[package]\nname="demo"\n'}),
            ["cargo test"],
        )


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

    async def test_read_manifests_skips_test_files_and_does_not_crowd_out_subpackages(self):
        many_tests = [f"tests/test_{i:02d}.py" for i in range(25)]
        tree = "\n".join(
            [
                "requirements.txt",
                *many_tests,
                "packages/web/package.json",
                "packages/web/pnpm-lock.yaml",
            ]
        )

        class MultiPackageBackend:
            def __init__(self):
                self.read_paths: list[str] = []

            async def call(self, op, **args):
                if op == "prepare":
                    return {"base_sha": "b" * 40, "tree": tree, "recent_commits": "init"}
                if op == "read":
                    path = args["path"]
                    self.read_paths.append(path)
                    if path == "requirements.txt":
                        return {"content": "pytest\n"}
                    if path == "packages/web/package.json":
                        return {"content": '{"scripts":{"test":"vitest run","build":"vite build"}}'}
                    raise ValueError("unexpected read: " + path)
                if op == "hydrate":
                    return {"path": "/sandbox/baseline"}
                if op == "gate":
                    return {"passed": True, "gates": [{"code": 0, "output": "ok"}]}
                if op == "drop":
                    return {"dropped": True}
                raise AssertionError(op)

        backend = MultiPackageBackend()
        self.engine.backend = backend
        result = await self.provisioner.ensure_project("owner", self.resolved, "improve", validate_baseline=False)
        self.assertEqual(result["status"], "ok")
        self.assertFalse(any(p.startswith("tests/") for p in backend.read_paths), backend.read_paths)
        self.assertIn("packages/web/package.json", backend.read_paths)
        self.assertEqual(
            result["project"]["gate_commands_json"],
            [
                "python -m pytest tests -q --disable-warnings --tb=short",
                "pnpm --dir packages/web test",
                "pnpm --dir packages/web run build",
            ],
        )

    def store_count(self, project_id):
        return len(self.engine.store.rows("candidates", "project_id=?", (project_id,)))


if __name__ == "__main__":
    unittest.main()
