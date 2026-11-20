"""End-to-end speculative lifecycle, sandbox Git/bwrap boundaries, parallel tool agent, and standalone guards."""

from __future__ import annotations

import ast
import asyncio
import json
import sqlite3
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from speculative_coding.config import ProviderEndpoint, SpecConfig
from speculative_coding.llm.types import ContextOverflowError, LLMStreamResult, ToolCallRequest
from speculative_coding.orchestrator import SpecSleep, SpeculativeOrchestrator
from speculative_coding.sandbox import LocalDirSandbox, RemoteWorkerSandbox
from speculative_coding.tool_result import ToolResult
from speculative_coding.worker import Worker


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.hash = "one"

    async def call(self, op, **a):
        self.calls.append((op, a))
        if op == "status":
            return {}
        if op in {"prepare", "describe"}:
            return {"base_sha": "a" * 40, "tree": "app.py\nREADME.md", "recent_commits": "abc initial"}
        if op == "hydrate":
            return {"path": "/sandbox/candidate"}
        if op == "diff":
            return {"hash": self.hash, "diff": "+ real implementation"}
        if op == "gate":
            return {
                "passed": True,
                "gates": [{"code": 0, "output": "assertions passed"}],
                "preview": {"code": 0, "output": "interaction passed"},
                "artifacts": [],
            }
        if op == "dehydrate":
            return {"branch_ref": "refs/spec/" + a["cand_id"], "head_sha": "b" * 40}
        if op == "export":
            return {"patch": "+ real implementation", "base_sha": "a" * 40, "head_sha": "b" * 40}
        if op == "merge":
            return {"merged_sha": "c" * 40, "pushed": False}
        return {}


class Lifecycle(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = SpecConfig(data_dir=self.root)
        self.user = {"owner_id": "owner", "is_admin": True}
        self.auth = SimpleNamespace(require_admin=lambda request: self.user)
        self.backend = FakeBackend()
        self.engine = SpeculativeOrchestrator(self.config, self.backend, auth=self.auth)
        self.p = self.engine.store.new_project(
            "owner",
            {
                "name": "test",
                "repo_source": "https://github.com/example/repo.git",
                "gate_commands_json": ["python -m unittest"],
                "preview_command": "python accept.py",
            },
        )
        now = time.time()
        self.engine.store.insert(
            "runs",
            dict(run_id="run", project_id=self.p["project_id"], base_sha="a" * 40, status="executing", created_at=now),
        )
        self.engine.store.insert(
            "candidates",
            dict(
                cand_id="candidate",
                run_id="run",
                project_id=self.p["project_id"],
                route_index=0,
                title="Complete feature",
                route_plan="Implement complete feature",
                route_kind="feature_gameplay",
                branch_ref="refs/spec/candidate",
                status="coding",
                base_sha="a" * 40,
                created_at=now,
                updated_at=now,
            ),
        )
        self.c = lambda: self.engine.store.get("candidates", "candidate")

    async def client(self):
        app = web.Application()
        app.router.add_get("/c/{cand_id}", self.engine.candidate_api)
        app.router.add_post("/c/{cand_id}/{action}", self.engine.action_api)
        app.router.add_get("/preview/{cand_id}/{path:.*}", self.engine.preview_api)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def nominate(self):
        async def agent(p, role, instruction, context, **kw):
            if role == "coder":
                self.assertIn("tree", context)
                return {"summary": "done"}
            if "three-minute" in instruction:
                return {"report": "### What Changed\nFeature complete.\n### How It Works\nClean module.\n### Gates & Verification\nAll passed."}
            return {"decision": "NOMINATE", "feedback": "excellent based on runtime evidence"}

        self.engine.agent = agent
        await self.engine.drive(self.p, self.c())

    async def test_complete_dual_review_nomination_dehydration(self):
        await self.nominate()
        c = self.c()
        self.assertEqual(c["status"], "staged")
        self.assertEqual(c["active_worktree_path"], "")
        ops = [op for op, _ in self.backend.calls]
        self.assertLess(ops.index("gate"), len(ops) - 1 - ops[::-1].index("diff"))
        self.assertEqual(ops[-1], "dehydrate")
        self.assertEqual(self.backend.calls[-1][1]["expected_hash"], "one")
        self.assertGreater(c["expires_at"], time.time())

    async def test_nomination_records_queryable_gate_and_review_evidence(self):
        await self.nominate()
        events = self.engine.store.events("owner", cand_id="candidate")
        kinds = [event["kind"] for event in events]
        self.assertIn("gate", kinds)
        self.assertIn("review", kinds)
        self.assertIn("nominated", kinds)
        self.assertTrue(all(event["head_sha"] for event in events if event["kind"] == "nominated"))

    async def test_crash_after_worker_dehydration_replays_without_recoding(self):
        update = self.engine.store.update

        def crash(table, key, **values):
            if table == "candidates" and values.get("status") == "staged":
                raise asyncio.CancelledError()
            return update(table, key, **values)

        self.engine.store.update = crash
        with self.assertRaises(asyncio.CancelledError):
            await self.nominate()
        self.assertEqual(self.c()["status"], "dehydrating")
        self.assertTrue(self.c()["human_report_md"])
        prior = len(self.backend.calls)
        recovered = SpeculativeOrchestrator(self.config, self.backend, auth=self.auth)
        recovered.agent = AsyncMock(side_effect=AssertionError("must not recode"))
        await recovered.drive(self.p, recovered.store.get("candidates", "candidate"))
        self.assertEqual(recovered.store.get("candidates", "candidate")["status"], "staged")
        self.assertEqual([op for op, _ in self.backend.calls[prior:]], ["dehydrate"])

    async def test_manual_run_completion_crash_does_not_launch_second_batch(self):
        await self.nominate()
        self.assertEqual(self.p["enabled"], 0)
        engine = SpeculativeOrchestrator(self.config, self.backend, auth=self.auth)
        engine.agent = AsyncMock(side_effect=AssertionError("must not start second batch"))
        await engine.run_project(self.p)
        self.assertEqual(engine.store.get("runs", "run")["status"], "completed")
        self.assertEqual(len(engine.store.rows("runs")), 1)

    async def test_interrupted_manual_recon_is_retried_without_enabled_flag(self):
        with self.engine.store.conn() as db:
            db.execute("DELETE FROM spec_candidates")
        self.engine.store.update("runs", "run", status="recon")
        self.engine.agent = AsyncMock(return_value={"summary": "nothing valuable", "routes": []})
        await self.engine.run_project(self.p)
        runs = self.engine.store.rows("runs")
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(run["status"] == "empty" for run in runs))

    async def test_stagnation_tombstone_and_no_candidate_card(self):
        self.engine.agent = AsyncMock(side_effect=lambda *a, **k: {"decision": "REVISE", "feedback": "specific fix"})
        await self.engine.drive(self.p, self.c())
        self.assertEqual(self.c()["status"], "dropped")
        self.assertEqual(len(self.engine.store.rows("graveyard")), 1)
        self.assertFalse(any(op == "dehydrate" for op, _ in self.backend.calls))

    async def test_keep_freezes_ttl_and_owner_auth(self):
        await self.nominate()
        client = await self.client()
        response = await client.post("/c/candidate/keep", json={"action_id": "keep-1"})
        self.assertEqual(response.status, 200)
        self.assertEqual(self.c()["status"], "kept")
        self.assertIsNone(self.c()["expires_at"])
        response = await client.post("/c/candidate/keep", json={"action_id": "keep-1"})
        self.assertEqual(response.status, 200)
        self.user = {"owner_id": "other", "is_admin": True}
        response = await client.get("/c/candidate")
        self.assertEqual(response.status, 403)

    async def test_steer_idempotency_after_missing_response(self):
        await self.nominate()
        self.engine._spawn = lambda _: False
        client = await self.client()
        payload = {"action_id": "steer-1", "notes": "Improve keyboard navigation"}
        response = await client.post("/c/candidate/steer", json=payload)
        self.assertEqual(response.status, 200)
        with self.engine.store.conn() as db:
            db.execute("UPDATE spec_actions SET result_json=NULL WHERE action_id='steer-1'")
        response = await client.post("/c/candidate/steer", json=payload)
        self.assertEqual(response.status, 200)
        self.assertEqual(self.c()["generation"], 1)
        self.assertEqual(len(self.c()["steer_notes_json"]), 1)
        response = await client.post("/c/candidate/steer", json={**payload, "notes": "different"})
        self.assertEqual(response.status, 409)

    async def test_merge_requires_explicit_sha_and_model_has_no_capability(self):
        await self.nominate()
        client = await self.client()
        response = await client.post("/c/candidate/merge", json={"action_id": "merge-bad", "confirm": True})
        self.assertEqual(response.status, 409)
        self.assertFalse(any(op == "merge" for op, _ in self.backend.calls))
        payload = {
            "action_id": "merge-good",
            "confirm": True,
            "approved_target_sha": "a" * 40,
            "approved_head_sha": "b" * 40,
        }
        response = await client.post("/c/candidate/merge", json=payload)
        self.assertEqual(response.status, 200)
        response = await client.post("/c/candidate/merge", json=payload)
        self.assertEqual(response.status, 200)
        self.assertEqual(sum(op == "merge" for op, _ in self.backend.calls), 1)

    async def test_expiry_waits_for_remote_deletion(self):
        await self.nominate()
        self.engine.store.update("candidates", "candidate", expires_at=time.time() - 1)
        await self.engine.expire()
        self.assertEqual(self.c()["status"], "expired")
        self.assertEqual(self.c()["preview_meta_json"], {})
        self.assertEqual(self.c()["diff_text"], "")

    async def test_preview_direct_open_has_sandbox_csp(self):
        self.engine.store.update(
            "candidates",
            "candidate",
            preview_meta_json={"artifacts": [{"path": "index.html", "base64": "PGgxPkhlbGxvPC9oMT4="}]},
        )
        client = await self.client()
        response = await client.get("/preview/candidate/index.html")
        self.assertEqual(response.status, 200)
        self.assertIn("sandbox allow-scripts", response.headers["Content-Security-Policy"])
        self.assertIn("connect-src 'none'", response.headers["Content-Security-Policy"])

    async def test_pool_unavailability_sleeps_without_paid_fallback(self):
        with self.assertRaises(SpecSleep):
            await self.engine.llm(self.p, "coder", [])
        self.assertEqual(self.engine.router.pool, "speculative")


class SandboxGit(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._source_temp = tempfile.TemporaryDirectory()
        cls.source = Path(cls._source_temp.name) / "source"
        cls.source.mkdir()

        def git(*args):
            subprocess.run(["git", "-C", str(cls.source), *args], check=True, capture_output=True)

        git("init", "-b", "master")
        (cls.source / "app.py").write_text("alpha = 1\nbeta = 2\ngamma = 3\n", encoding="utf-8")
        (cls.source / "tests").mkdir()
        (cls.source / "tests" / "test_app.py").write_text("assert True\n", encoding="utf-8")
        git("add", ".")
        git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial")

    @classmethod
    def tearDownClass(cls):
        cls._source_temp.cleanup()
        super().tearDownClass()

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = LocalDirSandbox(self.root / "pool", unsafe_test_mode=True, min_free_gb=0)
        self.prepared = await self.backend.call("prepare", project_id="project", repo_source=str(self.source), target_branch="master")
        self.args = {"project_id": "project", "cand_id": "candidate", "base_sha": self.prepared["base_sha"]}
        await self.backend.call("hydrate", **self.args)

    async def test_isolated_sources_dehydrate_rehydrate_export_and_drop(self):
        await self.backend.call("write", **self.args, path="app.py", content="value=2\n")
        diff = await self.backend.call("diff", **self.args)
        self.assertIn("+value=2", diff["diff"])
        result = await self.backend.call("dehydrate", **self.args, expected_hash=diff["hash"])
        self.assertFalse(self.backend._wt("candidate").exists())
        export = await self.backend.call("export", **self.args)
        self.assertIn("+value=2", export["patch"])
        await self.backend.call("hydrate", **self.args, ref=result["branch_ref"])
        self.assertEqual((self.backend._wt("candidate") / "app.py").read_text(), "value=2\n")
        await self.backend.call("drop", **self.args)
        self.assertFalse(self.backend._wt("candidate").exists())

    async def test_dehydration_effect_without_journal_receipt_is_idempotent(self):
        await self.backend.call("write", **self.args, path="app.py", content="value=99")
        diff = await self.backend.call("diff", **self.args)
        result = await self.backend.call("dehydrate", **self.args, expected_hash=diff["hash"])
        replay = await self.backend.call("dehydrate", **self.args, expected_hash=diff["hash"])
        self.assertEqual(replay, result)
        merged = await self.backend.call(
            "merge",
            **self.args,
            target_branch="master",
            approved_target_sha=self.prepared["base_sha"],
            approved_head_sha=result["head_sha"],
        )
        replay = await self.backend.call(
            "merge",
            **self.args,
            target_branch="master",
            approved_target_sha=self.prepared["base_sha"],
            approved_head_sha=result["head_sha"],
        )
        self.assertEqual(replay, merged)

    async def test_bubblewrap_mount_contract_isolated_cache_git_and_dependency(self):
        wt = self.backend._wt("candidate")
        baseline = self.backend.root / "golden_bases" / "project" / "lock"
        (baseline / "node_modules").mkdir(parents=True)
        command = self.backend.sandbox_argv(["/bin/sh", "-lc", "npm test"], wt, baseline)
        self.assertIn("--unshare-all", command)
        self.assertIn("--die-with-parent", command)
        self.assertIn("--new-session", command)
        for source, target in [(str(wt / ".git"), "/workspace/.git"), (str(baseline / "node_modules"), "/workspace/node_modules")]:
            index = command.index(source)
            self.assertEqual(command[index - 1], "--ro-bind")
            self.assertEqual(command[index + 1], target)
        cache_index = command.index(str(self.backend.root / "shared_caches"))
        self.assertEqual(command[cache_index - 1], "--overlay-src")
        self.assertEqual(command[cache_index + 1 : cache_index + 3], ["--tmp-overlay", "/cache"])
        self.assertFalse(any("API_KEY" in item or "WORKER_TOKEN" in item for item in command))

    async def test_escape_protection_and_existing_test_mutation(self):
        for path in ("../escape", "/etc/passwd", ".git/config", ".env", "deploy/start.sh", "tests/test_app.py", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                await self.backend.call("write", **self.args, path=path, content="bad")
        with self.assertRaises(ValueError):
            self.backend._wt("../outside")
        await self.backend.call("write", **self.args, path="tests/test_new.py", content="assert True")

    async def test_tampered_gitdir_and_review_diff_hash_rejected(self):
        await self.backend.call("write", **self.args, path="app.py", content="value=3")
        with self.assertRaisesRegex(ValueError, "changed after review"):
            await self.backend.call("dehydrate", **self.args, expected_hash="wrong")
        wt = self.backend._wt("candidate")
        with (wt / ".git").open("r+") as pointer:
            pointer.write("gitdir: " + str(self.root / "source" / ".git"))
            pointer.truncate()
        with self.assertRaisesRegex(ValueError, "untrusted"):
            await self.backend.call("diff", **self.args)

    async def test_offline_remote_fails_promptly(self):
        with self.assertRaisesRegex(RuntimeError, "offline"):
            await asyncio.wait_for(RemoteWorkerSandbox().call("status"), 0.5)

    async def test_surgical_edit_grep_and_sliced_read(self):
        bare_grep = await self.backend.call("grep", project_id="project", base_sha=self.prepared["base_sha"], pattern="beta")
        self.assertEqual(bare_grep["matches"], [{"path": "app.py", "line": 2, "text": "beta = 2"}])
        bare_slice = await self.backend.call(
            "read", project_id="project", base_sha=self.prepared["base_sha"], path="app.py", start_line=2, end_line=3
        )
        self.assertEqual(bare_slice["content"], "2: beta = 2\n3: gamma = 3")
        self.assertEqual(bare_slice["total_lines"], 3)

        edited = await self.backend.call("edit", **self.args, path="app.py", target="beta = 2   ", replacement="beta = 20")
        self.assertEqual(edited["replacements"], 1)
        wt_grep = await self.backend.call("grep", **self.args, pattern="beta = 20")
        self.assertEqual(wt_grep["matches"], [{"path": "app.py", "line": 2, "text": "beta = 20"}])

        with self.assertRaisesRegex(ValueError, "existing tests are immutable"):
            await self.backend.call("edit", **self.args, path="tests/test_app.py", target="assert True", replacement="assert False")

    async def test_command_preserves_incremental_cache_without_polluting_diff(self):
        wt = self.backend._wt("candidate")
        await self.backend.call("edit", **self.args, path="app.py", target="alpha = 1", replacement="alpha = 10")
        await self.backend.call(
            "write", **self.args, path="tests/test_alpha.py", content="import app\ndef test_a():\n    assert app.alpha == 10\n"
        )
        res = await self.backend.call("command", **self.args, command="python -m pytest tests -q")
        self.assertEqual(res["code"], 0, res["output"])
        self.assertTrue((wt / ".pytest_cache").is_dir())
        diff = await self.backend.call("diff", **self.args)
        self.assertIn("+alpha = 10", diff["diff"])
        self.assertNotIn(".pytest_cache", diff["diff"])
        self.assertNotIn("__pycache__", diff["diff"])

        gate = await self.backend.call("gate", **self.args, commands=["python -m pytest tests -q"])
        self.assertTrue(gate["passed"], gate)
        self.assertFalse((wt / ".pytest_cache").exists())


class BaselineContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = LocalDirSandbox(Path(self.temp.name) / "pool", unsafe_test_mode=True, min_free_gb=0)
        self.wt = self.backend._wt("candidate")
        (self.wt / "frontend").mkdir(parents=True)
        (self.wt / "frontend/package-lock.json").write_text('{"lockfileVersion":3}')
        self.backend._git = AsyncMock(return_value="frontend/package-lock.json\0")
        self.args = {"project_id": "project", "cand_id": "candidate"}

    async def test_nested_lock_baseline_is_installed_once_and_mounted_readonly(self):
        installs = []

        async def install(argv, **kwargs):
            installs.append(argv)
            dependency = self.wt / "frontend/node_modules/pkg"
            dependency.mkdir(parents=True)
            (dependency / "index.js").write_text("module.exports=true")
            return {"code": 0, "output": ""}

        self.backend._exec = install
        baseline = await self.backend._baseline(self.args, self.wt)
        self.assertTrue((baseline / "frontend/node_modules/pkg/index.js").is_file())
        self.assertIn("cd /workspace/frontend", installs[0][-1])
        async with self.backend.baseline_lock:
            self.backend.baseline_leases[baseline.parent] -= 1
        again = await self.backend._baseline(self.args, self.wt)
        self.assertEqual(again, baseline)
        self.assertEqual(len(installs), 1)
        command = self.backend.sandbox_argv(["node", "test.js"], self.wt, again)
        mount = command.index(str(baseline / "frontend/node_modules"))
        self.assertEqual(command[mount - 1], "--ro-bind")
        self.assertEqual(command[mount + 1], "/workspace/frontend/node_modules")

    async def test_failed_baseline_install_cleans_partial_worktree_dependencies(self):
        self.backend._exec = AsyncMock(return_value={"code": 1, "output": "offline package missing"})
        with self.assertRaisesRegex(RuntimeError, "offline dependency baseline unavailable"):
            await self.backend._baseline(self.args, self.wt)
        self.backend._git.assert_any_await(self.wt, "clean", "-ffd", "-x")
        self.assertFalse(any((self.backend.root / "golden_bases/project").iterdir()))
        self.assertEqual(sum(self.backend.baseline_leases.values()), 0)

    async def test_auto_seed_deps_falls_back_to_controlled_network_with_ignore_scripts(self):
        backend = LocalDirSandbox(Path(self.temp.name) / "seed_pool", unsafe_test_mode=True, min_free_gb=0, auto_seed_deps=True)
        wt = backend._wt("candidate")
        wt.mkdir(parents=True)
        (wt / "requirements.txt").write_text("pytest==8.3.0\n", encoding="utf-8")
        (wt / "package-lock.json").write_text('{"lockfileVersion":3}', encoding="utf-8")
        backend._git = AsyncMock(return_value="requirements.txt\0package-lock.json\0")
        calls = []

        async def fake_exec(argv, **kwargs):
            calls.append((argv, kwargs))
            if not kwargs.get("allow_network"):
                return {"code": 1, "output": "cache miss"}
            (wt / ".venv" / "bin").mkdir(parents=True, exist_ok=True)
            (wt / "node_modules" / ".bin").mkdir(parents=True, exist_ok=True)
            return {"code": 0, "output": "seeded"}

        backend._exec = fake_exec
        baseline = await backend._baseline({"project_id": "project", "cand_id": "candidate"}, wt)
        self.assertIsNotNone(baseline)
        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0][1].get("allow_network", False))
        self.assertTrue(calls[1][1].get("allow_network"))
        self.assertTrue(calls[1][1].get("writable_cache"))
        bins = backend._baseline_bin_paths(baseline)
        self.assertIn("/workspace/.venv/bin", bins)
        self.assertIn("/workspace/node_modules/.bin", bins)

        seed_cmd = backend.sandbox_argv(["/bin/sh", "-lc", "npm ci --ignore-scripts"], wt, allow_network=True, writable_cache=True)
        self.assertNotIn("--unshare-all", seed_cmd)
        self.assertIn("--unshare-user", seed_cmd)
        gate_cmd = backend.sandbox_argv(["/bin/sh", "-lc", "pytest"], wt, baseline)
        self.assertIn("--unshare-all", gate_cmd)

    async def test_metrics_caches_immutable_baselines_and_dedicated_mount(self):
        ready_base = self.backend.root / "golden_bases" / "project" / "digest1"
        (ready_base / "node_modules" / "pkg").mkdir(parents=True)
        (ready_base / "node_modules" / "pkg" / "index.js").write_bytes(b"x" * 2048)
        (ready_base / "ready").write_text("1")
        self.backend._invalidate_disk_cache()
        first = self.backend.metrics()
        self.assertIn(ready_base, self.backend._immutable_sizes)
        self.assertGreater(first["disk_used_bytes"], 0)

        self.backend._dedicated_mount = True
        with patch("os.walk", side_effect=AssertionError("dedicated mount must use statvfs without os.walk")):
            fast = self.backend.metrics()
        self.assertGreater(fast["disk_limit_bytes"], 0)

    async def test_lru_cannot_evict_between_install_and_lease_return(self):
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def install(argv, **kwargs):
            entered.set()
            await finish.wait()
            (self.wt / "frontend/node_modules").mkdir()
            return {"code": 0, "output": ""}

        self.backend._exec = install
        self.backend.metrics = lambda: {"pressure": "high"}
        acquire = asyncio.create_task(self.backend._baseline(self.args, self.wt))
        await asyncio.wait_for(entered.wait(), 2)
        evict = asyncio.create_task(self.backend._admit())
        await asyncio.sleep(0)
        self.assertFalse(evict.done())
        finish.set()
        baseline = await acquire
        await evict
        self.assertTrue(baseline.exists())
        self.assertEqual(self.backend.baseline_leases[baseline.parent], 1)
        async with self.backend.baseline_lock:
            self.backend.baseline_leases[baseline.parent] -= 1
        await self.backend._admit()
        self.assertFalse(baseline.parent.exists())

    async def test_dependency_mount_rejects_escaping_symlink(self):
        baseline = self.backend.root / "golden_bases/project/lock"
        (baseline / "frontend/node_modules").mkdir(parents=True)
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        target = self.wt / "frontend/node_modules"
        try:
            target.symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest("directory symlinks unavailable on this Windows host")
        with self.assertRaisesRegex(ValueError, "escaping dependency mount"):
            self.backend.sandbox_argv(["node", "test.js"], self.wt, baseline)


class ParallelToolAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        (source / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
        (source / "README.md").write_text("Calculator module\n", encoding="utf-8")
        subprocess.run(["git", "init", "-b", "master"], cwd=source, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "init"],
            cwd=source,
            check=True,
            capture_output=True,
        )

        ep = ProviderEndpoint(
            provider_id="spec-fast",
            source_id="src",
            model="test-model",
            api_base="https://example.invalid/v1",
            keys=["k"],
            pool="speculative",
        )
        config = SpecConfig(
            data_dir=self.root,
            providers={"spec-fast": ep},
            speculative_provider_id="spec-fast",
        )
        self.backend = LocalDirSandbox(self.root / "pool", unsafe_test_mode=True, min_free_gb=0)
        self.engine = SpeculativeOrchestrator(config, self.backend)
        self.project = self.engine.store.new_project("owner", {"name": "calc", "repo_source": str(source)})
        self.prepared = await self.backend.call(
            "prepare", project_id=self.project["project_id"], repo_source=str(source), target_branch="master"
        )
        self.cand = {"cand_id": "cand1", "generation": 0}
        await self.backend.call("hydrate", project_id=self.project["project_id"], cand_id="cand1", base_sha=self.prepared["base_sha"])

    async def test_parallel_reads_surgical_edit_and_readonly_role_guard(self):
        router = self.engine._role(self.project, "coder")
        router.complete_step = AsyncMock(
            side_effect=[
                LLMStreamResult(
                    tool_calls=[
                        ToolCallRequest("c1", "spec_grep", json.dumps({"pattern": "def add"})),
                        ToolCallRequest("c2", "spec_read", json.dumps({"path": "calc.py", "start_line": 1, "end_line": 2})),
                    ],
                    input_tokens=20,
                    output_tokens=10,
                ),
                LLMStreamResult(
                    tool_calls=[
                        ToolCallRequest(
                            "c3", "spec_edit", json.dumps({"path": "calc.py", "target": "return a - b", "replacement": "return a + b"})
                        ),
                        ToolCallRequest("c4", "submit_result", json.dumps({"result": {"summary": "fixed addition"}})),
                    ],
                    input_tokens=30,
                    output_tokens=15,
                ),
            ]
        )
        res = await self.engine.agent(
            self.project,
            "coder",
            "Fix add()",
            {"base_sha": self.prepared["base_sha"], "tree": self.prepared["tree"]},
            run_id="",
            candidate=self.cand,
            allow_write=True,
        )
        self.assertEqual(res, {"summary": "fixed addition"})
        self.assertEqual(router.complete_step.await_count, 2)
        diff = await self.backend.call("diff", project_id=self.project["project_id"], cand_id="cand1", base_sha=self.prepared["base_sha"])
        self.assertIn("+    return a + b", diff["diff"])
        self.assertEqual(self.engine.store.get("projects", self.project["project_id"])["speculative_tokens"], 75)

        rev_router = self.engine._role(self.project, "reviewer")
        rev_router.complete_step = AsyncMock(
            side_effect=[
                LLMStreamResult(
                    tool_calls=[ToolCallRequest("r1", "spec_edit", json.dumps({"path": "calc.py", "target": "a + b", "replacement": "0"}))],
                    input_tokens=10,
                    output_tokens=5,
                ),
                LLMStreamResult(
                    tool_calls=[ToolCallRequest("r2", "submit_result", json.dumps({"result": {"decision": "NOMINATE", "feedback": "verified"}}))],
                    input_tokens=10,
                    output_tokens=5,
                ),
            ]
        )
        verdict = await self.engine.agent(
            self.project,
            "reviewer",
            "Review diff",
            {"base_sha": self.prepared["base_sha"], "diff": diff["diff"]},
            run_id="",
            candidate=self.cand,
            allow_write=False,
        )
        self.assertEqual(verdict["decision"], "NOMINATE")
        self.assertIn("return a + b", (self.backend._wt("cand1") / "calc.py").read_text(encoding="utf-8"))

    async def test_parallel_read_batching_runs_concurrently_in_agent(self):
        orig_call = self.backend.call
        in_flight = 0
        max_in_flight = 0
        both_entered = asyncio.Event()

        async def tracking_call(op, **kwargs):
            nonlocal in_flight, max_in_flight
            if op in {"grep", "read"}:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                if in_flight == 2:
                    both_entered.set()
                try:
                    await asyncio.wait_for(both_entered.wait(), timeout=1.0)
                    return await orig_call(op, **kwargs)
                finally:
                    in_flight -= 1
            return await orig_call(op, **kwargs)

        self.backend.call = tracking_call
        router = self.engine._role(self.project, "recon")
        router.complete_step = AsyncMock(
            side_effect=[
                LLMStreamResult(
                    tool_calls=[
                        ToolCallRequest("c1", "spec_grep", json.dumps({"pattern": "def add"})),
                        ToolCallRequest("c2", "spec_read", json.dumps({"path": "calc.py"})),
                    ],
                    input_tokens=10,
                    output_tokens=5,
                ),
                LLMStreamResult(
                    tool_calls=[ToolCallRequest("c3", "submit_result", json.dumps({"result": {"summary": "ok", "routes": []}}))],
                    input_tokens=15,
                    output_tokens=5,
                ),
            ]
        )
        res = await self.engine.agent(
            self.project,
            "recon",
            "Inspect repo",
            {"base_sha": self.prepared["base_sha"], "tree": self.prepared["tree"]},
            run_id="",
        )
        self.assertEqual(res, {"summary": "ok", "routes": []})
        self.assertEqual(max_in_flight, 2)
        self.assertEqual(router.complete_step.await_count, 2)

    async def test_submit_result_three_forms_and_invalid_retry(self):
        router = self.engine._role(self.project, "reviewer")
        cases = [
            ("dict_form", {"result": {"decision": "NOMINATE", "feedback": "dict"}}, {"decision": "NOMINATE", "feedback": "dict"}),
            ("string_form", {"result": '```json\n{"decision": "REVISE", "feedback": "str"}\n```'}, {"decision": "REVISE", "feedback": "str"}),
            ("raw_top_level_form", {"decision": "DROP", "feedback": "raw"}, {"decision": "DROP", "feedback": "raw"}),
        ]
        for label, payload, expected in cases:
            with self.subTest(form=label):
                router.complete_step = AsyncMock(
                    side_effect=[
                        LLMStreamResult(
                            tool_calls=[ToolCallRequest("s1", "submit_result", json.dumps(payload))],
                            input_tokens=10,
                            output_tokens=5,
                        )
                    ]
                )
                out = await self.engine.agent(
                    self.project,
                    "reviewer",
                    "Review",
                    {"base_sha": self.prepared["base_sha"]},
                    run_id="",
                    candidate=self.cand,
                )
                self.assertEqual(out, expected)
                self.assertEqual(router.complete_step.await_count, 1)

        router.complete_step = AsyncMock(
            side_effect=[
                LLMStreamResult(
                    tool_calls=[ToolCallRequest("bad", "submit_result", json.dumps({"result": "not-json"}))],
                    input_tokens=10,
                    output_tokens=4,
                ),
                LLMStreamResult(
                    tool_calls=[ToolCallRequest("good", "submit_result", json.dumps({"result": {"decision": "NOMINATE", "feedback": "fixed"}}))],
                    input_tokens=12,
                    output_tokens=6,
                ),
            ]
        )
        recovered = await self.engine.agent(
            self.project,
            "reviewer",
            "Review",
            {"base_sha": self.prepared["base_sha"]},
            run_id="",
            candidate=self.cand,
        )
        self.assertEqual(recovered, {"decision": "NOMINATE", "feedback": "fixed"})
        self.assertEqual(router.complete_step.await_count, 2)
        retry_msgs = router.complete_step.call_args_list[1].args[0]
        tool_msg = next(m for m in retry_msgs if m.get("role") == "tool")
        self.assertIn("requires `result` to be a JSON object", tool_msg["content"])

    async def test_agent_output_validator_retries_and_raises_on_exhaustion(self):
        router = self.engine._role(self.project, "reviewer")
        router.complete_step = AsyncMock(
            side_effect=[
                LLMStreamResult(content="thinking out loud", input_tokens=10, output_tokens=5),
                LLMStreamResult(content="still not json", input_tokens=15, output_tokens=5),
                LLMStreamResult(content='```json\n{"decision": "NOMINATE", "feedback": "ok"}\n```', input_tokens=20, output_tokens=6),
            ]
        )
        res = await self.engine.agent(
            self.project,
            "reviewer",
            "Review",
            {"base_sha": self.prepared["base_sha"]},
            run_id="",
            candidate=self.cand,
        )
        self.assertEqual(res, {"decision": "NOMINATE", "feedback": "ok"})
        self.assertEqual(router.complete_step.await_count, 3)

        router.complete_step = AsyncMock(
            side_effect=[LLMStreamResult(content=f"bad {i}", input_tokens=5, output_tokens=2) for i in range(3)]
        )
        with self.assertRaisesRegex(ValueError, "model must return valid JSON"):
            await self.engine.agent(
                self.project,
                "reviewer",
                "Review",
                {"base_sha": self.prepared["base_sha"]},
                run_id="",
                candidate=self.cand,
            )
        self.assertEqual(router.complete_step.await_count, 3)

    async def test_dispatch_sandbox_tool_returns_tool_result_and_guards_readonly(self):
        read_res = await self.engine._dispatch_sandbox_tool(
            "spec_read",
            {"path": "calc.py", "start_line": 1, "end_line": 2},
            p=self.project,
            base_sha=self.prepared["base_sha"],
            candidate=self.cand,
            allow_write=False,
        )
        self.assertIsInstance(read_res, ToolResult)
        self.assertFalse(read_res.terminate_loop)
        self.assertIn("def add", read_res.text)

        edit_blocked = await self.engine._dispatch_sandbox_tool(
            "spec_edit",
            {"path": "calc.py", "target": "a - b", "replacement": "a + b"},
            p=self.project,
            base_sha=self.prepared["base_sha"],
            candidate=self.cand,
            allow_write=False,
        )
        self.assertIsInstance(edit_blocked, ToolResult)
        self.assertFalse(edit_blocked.terminate_loop)
        self.assertIn("operation not permitted for this role", edit_blocked.text)

        submit_res = await self.engine._dispatch_sandbox_tool(
            "submit_result",
            {"result": {"summary": "ready"}},
            p=self.project,
            base_sha=self.prepared["base_sha"],
            candidate=self.cand,
            allow_write=False,
        )
        self.assertTrue(submit_res.terminate_loop)
        self.assertEqual(submit_res.return_value, {"summary": "ready"})

    async def test_agent_stopping_cancels_and_overflow_recovers(self):
        router = self.engine._role(self.project, "reviewer")
        router.complete_step = AsyncMock(
            side_effect=[
                ContextOverflowError("prompt is too long"),
                LLMStreamResult(
                    tool_calls=[ToolCallRequest("s1", "submit_result", json.dumps({"result": {"decision": "NOMINATE", "feedback": "recovered"}}))],
                    input_tokens=10,
                    output_tokens=5,
                ),
            ]
        )
        res = await self.engine.agent(
            self.project,
            "reviewer",
            "Review",
            {"base_sha": self.prepared["base_sha"]},
            run_id="",
            candidate=self.cand,
        )
        self.assertEqual(res["feedback"], "recovered")
        self.assertEqual(router.complete_step.await_count, 2)

        self.engine.stopping = True
        with self.assertRaises(asyncio.CancelledError):
            await self.engine.agent(
                self.project,
                "reviewer",
                "Review",
                {"base_sha": self.prepared["base_sha"]},
                run_id="",
                candidate=self.cand,
            )


class WorkerFailures(unittest.IsolatedAsyncioTestCase):
    async def test_journal_full_returns_error_instead_of_stranding_rpc(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Worker(Path(directory), "wss://example.invalid/ws", "x" * 32)
            messages = []

            async def send(message):
                messages.append(message)

            worker.ws = SimpleNamespace(closed=False, send_json=send)
            worker.maintain = Mock(side_effect=sqlite3.OperationalError("database or disk is full"))
            await worker.task({"id": "full-task", "op": "drop", "args": {"project_id": "p", "cand_id": "c"}})
            self.assertEqual(messages[0]["id"], "full-task")
            self.assertIn("persistence/execution error", messages[0]["error"])
            worker.db.close()


class StandaloneIsolationGuards(unittest.TestCase):
    def test_zero_xiao_core_imports_anywhere_in_package(self):
        pkg_root = Path(__file__).resolve().parent.parent / "speculative_coding"
        offenders: list[str] = []
        for py_file in sorted(pkg_root.rglob("*.py")):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "xiao_core" or alias.name.startswith("xiao_core."):
                            offenders.append(f"{py_file}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod == "xiao_core" or mod.startswith("xiao_core."):
                        offenders.append(f"{py_file}: from {mod} import ...")
        self.assertEqual(offenders, [], f"Forbidden xiao_core imports found: {offenders}")


if __name__ == "__main__":
    unittest.main()
