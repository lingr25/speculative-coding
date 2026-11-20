"""Independent acceptance checks for reconnect-safe worker side effects."""

from __future__ import annotations

import asyncio
import base64
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from speculative_coding.sandbox import LocalDirSandbox, RemoteWorkerSandbox, _is_test_path
from speculative_coding.worker import Worker


@contextmanager
def closing_db(path: Path):
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


class Socket:
    closed = False

    def __init__(self):
        self.messages = []

    async def send_json(self, value):
        self.messages.append(value)


class RecordingBackend:
    def __init__(self, *, block=False):
        self.calls = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not block:
            self.release.set()

    async def call(self, operation, **arguments):
        self.calls.append((operation, arguments))
        self.entered.set()
        await self.release.wait()
        return {"effect_count": len(self.calls)}


class WorkerReplayContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / "sandbox"

    def worker(self, backend):
        worker = Worker(self.root, "wss://worker.example.invalid/api/spec-worker/ws", "t" * 32)
        self.addCleanup(worker.db.close)
        worker.backend = backend
        worker.ws = Socket()
        return worker

    async def test_reconnect_replay_does_not_repeat_in_flight_mutation(self):
        backend = RecordingBackend(block=True)
        worker = self.worker(backend)
        request = {"id": "stable-mutation", "op": "dehydrate", "args": {"cand_id": "candidate"}}
        original_socket = worker.ws
        first = asyncio.create_task(worker.task(request))
        try:
            await asyncio.wait_for(backend.entered.wait(), 2)
            worker.ws = Socket()
            await worker.task(request)
            backend.release.set()
            await asyncio.wait_for(first, 2)
            await worker.task(request)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(original_socket.messages, [])
            self.assertEqual(len(worker.ws.messages), 2)
            self.assertEqual(worker.ws.messages[0], worker.ws.messages[1])
            self.assertEqual(worker.ws.messages[0]["result"]["effect_count"], 1)
        finally:
            backend.release.set()
            await asyncio.gather(first, return_exceptions=True)

    async def test_completed_result_survives_worker_restart(self):
        backend = RecordingBackend()
        first = self.worker(backend)
        request = {"id": "durable-ref", "op": "dehydrate", "args": {"cand_id": "candidate"}}
        await first.task(request)
        result = first.ws.messages[-1]
        first.db.close()
        restarted_backend = RecordingBackend()
        restarted = self.worker(restarted_backend)
        await restarted.task(request)
        self.assertEqual(restarted_backend.calls, [])
        self.assertEqual(restarted.ws.messages, [result])

    async def test_same_id_with_changed_payload_never_executes_new_command(self):
        backend = RecordingBackend()
        worker = self.worker(backend)
        await worker.task({"id": "once", "op": "command", "args": {"command": "first"}})
        await worker.task({"id": "once", "op": "command", "args": {"command": "different"}})
        self.assertEqual(len(backend.calls), 1)
        self.assertIn("mismatch", worker.ws.messages[-1]["error"])

    async def test_interrupted_mutation_is_reported_after_restart_without_reexecution(self):
        backend = RecordingBackend(block=True)
        first = self.worker(backend)
        request = {"id": "interrupted", "op": "merge", "args": {"cand_id": "candidate"}}
        task = asyncio.create_task(first.task(request))
        await asyncio.wait_for(backend.entered.wait(), 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        first.db.close()
        restarted_backend = RecordingBackend()
        restarted = self.worker(restarted_backend)
        await restarted.task(request)
        self.assertEqual(restarted_backend.calls, [])
        self.assertIn("restart", restarted.ws.messages[-1]["error"])

    async def test_worker_throttles_journal_maintenance_across_burst_mutations(self):
        backend = RecordingBackend()
        worker = self.worker(backend)
        maintain_calls = 0
        orig_maintain = worker.maintain

        def counted_maintain():
            nonlocal maintain_calls
            maintain_calls += 1
            orig_maintain()

        worker.maintain = counted_maintain
        for idx in range(6):
            await worker.task({"id": f"mut-{idx}", "op": "write", "args": {"path": f"f{idx}.py"}})
        self.assertEqual(maintain_calls, 1)


class ControlPlaneReplayContracts(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.journal = Path(self.directory.name) / "outbox.sqlite3"

    async def connection(self, backend):
        app = web.Application()
        app.router.add_get("/worker", backend.connect)
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        socket = await client.ws_connect("/worker")
        return client, socket

    async def test_live_reconnect_resends_same_request_and_resolves_original_caller(self):
        backend = RemoteWorkerSandbox(self.journal)
        client, socket = await self.connection(backend)
        task = asyncio.create_task(backend.call("gate", cand_id="candidate", task_id="stable-gate"))
        try:
            original = await socket.receive_json(timeout=2)
            await socket.close()
            replacement = await client.ws_connect("/worker")
            replay = await replacement.receive_json(timeout=2)
            self.assertEqual(replay, original)
            await replacement.send_json({"id": replay["id"], "result": {"passed": True}})
            self.assertEqual(await asyncio.wait_for(task, 2), {"passed": True})
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_controller_restart_recovers_completed_reply_from_durable_outbox(self):
        original_backend = RemoteWorkerSandbox(self.journal)
        _, socket = await self.connection(original_backend)
        arguments = {"cand_id": "candidate", "task_id": "surviving-gate"}
        original_call = asyncio.create_task(original_backend.call("gate", **arguments))
        request = await socket.receive_json(timeout=2)
        original_call.cancel()
        await asyncio.gather(original_call, return_exceptions=True)
        await socket.close()

        restarted_backend = RemoteWorkerSandbox(self.journal)
        _, reconnected = await self.connection(restarted_backend)
        self.assertEqual(await reconnected.receive_json(timeout=2), request)
        await reconnected.send_json({"id": request["id"], "result": {"passed": True, "evidence": "survived"}})
        self.assertEqual(await reconnected.receive_json(timeout=2), {"ack": request["id"]})
        result = await asyncio.wait_for(restarted_backend.call("gate", **arguments), 2)
        self.assertEqual(result, {"passed": True, "evidence": "survived"})

    async def test_ephemeral_fast_reads_bypass_sqlite_outbox_and_worker_journal(self):
        backend = RemoteWorkerSandbox(self.journal)
        _, socket = await self.connection(backend)
        worker_dir = Path(self.directory.name) / "worker_root"
        worker = Worker(worker_dir, "wss://worker.example.invalid/api/spec-worker/ws", "t" * 32)
        self.addCleanup(worker.db.close)

        class FastBackend:
            async def call(self, op, **kwargs):
                return {"op": op, "path": kwargs.get("path", "")}

        worker.backend = FastBackend()

        async def bridge():
            for _ in range(3):
                msg = await socket.receive_json(timeout=2)
                self.assertTrue(msg.get("ephemeral"))
                worker.ws = socket
                await worker.task(msg)

        pump = asyncio.create_task(bridge())
        results = await asyncio.gather(
            backend.call("read", project_id="p", base_sha="a" * 40, path="a.py"),
            backend.call("grep", project_id="p", base_sha="a" * 40, pattern="value"),
            backend.call("diff", project_id="p", cand_id="c", base_sha="a" * 40),
        )
        await pump
        self.assertEqual([r["op"] for r in results], ["read", "grep", "diff"])

        with closing_db(self.journal) as outbox_db:
            outbox_count = outbox_db.execute("SELECT COUNT(*) FROM spec_worker_tasks").fetchone()[0]
        worker_count = worker.db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        self.assertEqual(outbox_count, 0)
        self.assertEqual(worker_count, 0)

    async def test_outbox_budget_prunes_consumed_and_never_blocks_drop(self):
        backend = RemoteWorkerSandbox(self.journal, max_outbox_bytes=48 * 1024)
        self.assertEqual(backend.status()["outbox_db"], str(self.journal))
        _, socket = await self.connection(backend)

        # Fill the outbox DB with large queued rows so budget is exceeded.
        with closing_db(self.journal) as db:
            for idx in range(8):
                db.execute(
                    "INSERT INTO spec_worker_tasks VALUES (?,?,?,NULL,?,?)",
                    (f"full-{idx}", f"dig-{idx}", "x" * 16384, "queued", 1.0),
                )
            db.commit()

        # Non-drop mutation fails with budget exhausted when active queued rows exceed budget.
        with self.assertRaisesRegex(RuntimeError, "outbox budget exhausted"):
            await backend.call("gate", project_id="p", cand_id="c1", task_id="blocked-gate")

        # Drop MUST bypass outbox budget so cleanup/recovery never deadlocks.
        drop_task = asyncio.create_task(
            backend.call("drop", project_id="p", cand_id="c1", task_id="recovery-drop")
        )
        msg = await socket.receive_json(timeout=2)
        self.assertEqual(msg["op"], "drop")
        await socket.send_json({"id": msg["id"], "result": {"dropped": True}})
        self.assertEqual(await socket.receive_json(timeout=2), {"ack": msg["id"]})
        self.assertEqual(await asyncio.wait_for(drop_task, 2), {"dropped": True})

        # Once those large rows are marked consumed, non-drop call prunes them and succeeds.
        with closing_db(self.journal) as db:
            db.execute("UPDATE spec_worker_tasks SET status='consumed' WHERE task_id LIKE 'full-%'")
            db.commit()
        gate_task = asyncio.create_task(
            backend.call("gate", project_id="p", cand_id="c2", task_id="recovered-gate")
        )
        gate_msg = await socket.receive_json(timeout=2)
        self.assertEqual(gate_msg["op"], "gate")
        await socket.send_json({"id": gate_msg["id"], "result": {"passed": True}})
        self.assertEqual(await socket.receive_json(timeout=2), {"ack": gate_msg["id"]})
        self.assertEqual(await asyncio.wait_for(gate_task, 2), {"passed": True})


class FineGrainedBaselineLockTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = LocalDirSandbox(Path(self.directory.name) / "pool", unsafe_test_mode=True, min_free_gb=0)

    def _make_wt(self, cand_id: str, lock_content: str) -> Path:
        wt = self.backend._wt(cand_id)
        wt.mkdir(parents=True, exist_ok=True)
        (wt / "package-lock.json").write_text(lock_content, encoding="utf-8")
        return wt

    async def test_slow_install_does_not_block_other_projects_or_high_pressure_eviction(self):
        wt_a = self._make_wt("candA", '{"a":1}')
        wt_b = self._make_wt("candB", '{"b":1}')

        # Pre-seed an idle baseline for projC that high-pressure _admit() should evict immediately.
        idle_parent = self.backend.root / "golden_bases" / "projC"
        idle_base = idle_parent / "deadbeef"
        idle_base.mkdir(parents=True)
        (idle_base / "ready").write_text("1", encoding="utf-8")

        self.backend._git = AsyncMock(return_value="package-lock.json\0")
        a_started = asyncio.Event()
        a_release = asyncio.Event()

        async def fake_exec(argv, **kwargs):
            cwd = kwargs["cwd"]
            if cwd == wt_a:
                a_started.set()
                await a_release.wait()
                (wt_a / "node_modules").mkdir(exist_ok=True)
                return {"code": 0, "output": "ok-a"}
            (wt_b / "node_modules").mkdir(exist_ok=True)
            return {"code": 0, "output": "ok-b"}

        self.backend._exec = fake_exec
        self.backend.metrics = lambda: {"pressure": "high"}

        task_a = asyncio.create_task(self.backend._baseline({"project_id": "projA", "cand_id": "candA"}, wt_a))
        await asyncio.wait_for(a_started.wait(), 2)

        # While projA is installing, projB must NOT block on global baseline_lock.
        base_b = await asyncio.wait_for(
            self.backend._baseline({"project_id": "projB", "cand_id": "candB"}, wt_b), timeout=1.0
        )
        self.assertTrue((base_b / "ready").is_file())
        self.assertEqual(self.backend.baseline_leases[base_b.parent], 1)

        # High-pressure _admit() must also complete without waiting for projA,
        # evicting idle projC while preserving leased projA and projB.
        await asyncio.wait_for(self.backend._admit(), timeout=1.0)
        self.assertFalse(idle_parent.exists())
        self.assertTrue((self.backend.root / "golden_bases" / "projA").exists())
        self.assertTrue(base_b.exists())

        # Releasing projB's lease while projA is still installing must not block.
        async def release_b():
            async with self.backend.baseline_lock:
                self.backend.baseline_leases[base_b.parent] -= 1

        await asyncio.wait_for(release_b(), timeout=1.0)

        a_release.set()
        base_a = await asyncio.wait_for(task_a, 2)
        self.assertTrue((base_a / "ready").is_file())
        self.assertEqual(self.backend.baseline_leases[base_a.parent], 1)

    async def test_same_project_digest_deduplicates_concurrent_install(self):
        wt_1 = self._make_wt("cand1", '{"same":1}')
        wt_2 = self._make_wt("cand2", '{"same":1}')
        self.backend._git = AsyncMock(return_value="package-lock.json\0")
        install_calls = 0
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_exec(argv, **kwargs):
            nonlocal install_calls
            install_calls += 1
            started.set()
            await release.wait()
            (kwargs["cwd"] / "node_modules").mkdir(exist_ok=True)
            return {"code": 0, "output": "ok"}

        self.backend._exec = fake_exec
        t1 = asyncio.create_task(self.backend._baseline({"project_id": "projX", "cand_id": "cand1"}, wt_1))
        await asyncio.wait_for(started.wait(), 2)
        t2 = asyncio.create_task(self.backend._baseline({"project_id": "projX", "cand_id": "cand2"}, wt_2))
        await asyncio.sleep(0.05)
        self.assertFalse(t2.done())
        release.set()
        b1, b2 = await asyncio.wait_for(asyncio.gather(t1, t2), 2)
        self.assertEqual(b1, b2)
        self.assertEqual(install_calls, 1)
        self.assertEqual(self.backend.baseline_leases[b1.parent], 2)


class SandboxPolicyExportAndPrepareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self._git_src("init", "-b", "master")
        (self.source / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
        (self.source / "tests").mkdir()
        (self.source / "tests" / "test_app.py").write_text(
            "import app\n\ndef test_value():\n    assert app.value() == 1\n", encoding="utf-8"
        )
        (self.source / "conftest.py").write_text("# root conftest\n", encoding="utf-8")
        (self.source / "src").mkdir()
        (self.source / "src" / "Button.test.tsx").write_text("test('ok', () => {})\n", encoding="utf-8")
        (self.source / "pkg").mkdir()
        (self.source / "pkg" / "handler_test.go").write_text("package pkg\n", encoding="utf-8")
        self._git_src("add", ".")
        self._git_src("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "initial")

    def _git_src(self, *args):
        subprocess.run(["git", "-C", str(self.source), *args], check=True, capture_output=True)

    def test_is_test_path_multilanguage_classification(self):
        for rel in (
            "tests/test_app.py",
            "tests/helpers.py",
            "pkg/tests/util.py",
            "test_root.py",
            "app_test.py",
            "conftest.py",
            "sub/conftest.py",
            "pytest.ini",
            "src/__tests__/App.tsx",
            "src/Button.test.tsx",
            "src/Button.spec.ts",
            "pkg/handler_test.go",
            "src/FooTest.java",
            "src/BarTests.cs",
            "src/BazTest.rs",
            "src/test/App.java",
        ):
            with self.subTest(expected_test=rel):
                self.assertTrue(_is_test_path(rel))

        for rel in (
            "app.py",
            "src/contest.py",
            "src/latest.ts",
            "src/TestUtils.java",
            "src/test_utils/helper.py",
            "README.md",
            "pyproject.toml",
        ):
            with self.subTest(expected_non_test=rel):
                self.assertFalse(_is_test_path(rel))

    async def test_allow_test_edits_policy_and_baseline_test_regression(self):
        backend = LocalDirSandbox(self.root / "pool", unsafe_test_mode=True, min_free_gb=0)
        prepared = await backend.call("prepare", project_id="proj", repo_source=str(self.source), target_branch="master")
        args = {"project_id": "proj", "cand_id": "cand1", "base_sha": prepared["base_sha"]}
        await backend.call("hydrate", **args)

        # Strict default: existing tests across languages are immutable.
        for path in ("tests/test_app.py", "conftest.py", "src/Button.test.tsx", "pkg/handler_test.go"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "existing tests are immutable"):
                await backend.call("write", **args, path=path, content="modified")

        # Opt-in allow_test_edits=True still blocks protected paths.
        for protected in (".git/config", ".env", ".env.local", "deploy/run.sh", ".github/workflows/ci.yml", "AGENTS.md"):
            with self.subTest(protected=protected), self.assertRaisesRegex(ValueError, "protected or escaping"):
                await backend.call("write", **args, path=protected, content="bad", allow_test_edits=True)

        # Modify app.py (breaking original expectation 1 -> 2) and update existing test to assert 2.
        await backend.call("edit", **args, path="app.py", target="return 1", replacement="return 2")
        await backend.call(
            "edit",
            **args,
            path="tests/test_app.py",
            target="assert app.value() == 1",
            replacement="assert app.value() == 2",
            allow_test_edits=True,
        )

        # Default diff (allow_test_edits=False) still rejects modified existing tests.
        with self.assertRaisesRegex(ValueError, "protected paths/tests"):
            await backend.call("diff", **args)

        # Opt-in diff surfaces modified_existing_tests cleanly.
        diff = await backend.call("diff", **args, allow_test_edits=True)
        self.assertEqual(diff["modified_existing_tests"], ["tests/test_app.py"])

        # Gate runs baseline_test_regression (fails against original test) AND candidate gate (passes).
        gate = await backend.call(
            "gate", **args, commands=["python -m pytest tests -q"], allow_test_edits=True
        )
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["modified_existing_tests"], ["tests/test_app.py"])
        reg = gate["baseline_test_regression"]
        self.assertTrue(reg["checked"])
        self.assertFalse(reg["passed"])
        self.assertEqual(reg["modified_existing_tests"], ["tests/test_app.py"])

        # Candidate's edited test is preserved after gate and dehydrates cleanly.
        wt = backend._wt("cand1")
        self.assertIn("assert app.value() == 2", (wt / "tests" / "test_app.py").read_text(encoding="utf-8"))
        dehydrated = await backend.call("dehydrate", **args, expected_hash=diff["hash"], allow_test_edits=True)
        self.assertTrue(dehydrated["head_sha"])

    async def test_failed_gate_preserves_incremental_cache_but_cleans_preview(self):
        backend = LocalDirSandbox(self.root / "pool", unsafe_test_mode=True, min_free_gb=0)
        prepared = await backend.call("prepare", project_id="proj", repo_source=str(self.source), target_branch="master")
        args = {"project_id": "proj", "cand_id": "cand_fail", "base_sha": prepared["base_sha"]}
        await backend.call("hydrate", **args)
        wt = backend._wt("cand_fail")

        await backend.call("edit", **args, path="app.py", target="return 1", replacement="return 999")
        (wt / ".spec-preview").mkdir()
        (wt / ".spec-preview" / "shot.png").write_bytes(b"fake")

        failed_gate = await backend.call("gate", **args, commands=["python -m pytest tests -q"])
        self.assertFalse(failed_gate["passed"])
        self.assertTrue((wt / ".pytest_cache").is_dir())
        self.assertFalse((wt / ".spec-preview").exists())

        # Fix the code; passing gate performs full clean.
        await backend.call("edit", **args, path="app.py", target="return 999", replacement="return 1")
        passed_gate = await backend.call("gate", **args, commands=["python -m pytest tests -q"])
        self.assertTrue(passed_gate["passed"])
        self.assertFalse((wt / ".pytest_cache").exists())

    async def test_portable_export_bundle_and_prepare_divergence_metadata(self):
        backend = LocalDirSandbox(self.root / "pool", unsafe_test_mode=True, min_free_gb=0)
        prepared = await backend.call("prepare", project_id="proj", repo_source=str(self.source), target_branch="master")
        self.assertFalse(prepared["diverged_from_upstream"])
        self.assertEqual(prepared["upstream_sha"], prepared["base_sha"])
        self.assertEqual(prepared["ahead_of_upstream"], 0)
        self.assertEqual(prepared["behind_upstream"], 0)

        args = {"project_id": "proj", "cand_id": "cand_exp", "base_sha": prepared["base_sha"]}
        await backend.call("hydrate", **args)
        await backend.call("edit", **args, path="app.py", target="return 1", replacement="return 42")
        diff = await backend.call("diff", **args)
        dehydrated = await backend.call("dehydrate", **args, expected_hash=diff["hash"])

        # Export includes valid git bundle + apply instructions.
        exported = await backend.call("export", **args)
        self.assertIn("+    return 42", exported["patch"])
        self.assertTrue(exported["bundle_base64"])
        self.assertEqual(exported["bundle_filename"], "cand_exp.bundle")
        self.assertEqual(exported["branch_ref"], "refs/spec/cand_exp")
        self.assertIn("git bundle verify", exported["apply_instructions"]["bundle"])
        self.assertIn("git apply", exported["apply_instructions"]["patch"])

        # Verify bundle can be verified and fetched into an independent clone of source!
        bundle_file = self.root / "cand_exp.bundle"
        bundle_file.write_bytes(base64.b64decode(exported["bundle_base64"]))
        local_clone = self.root / "local_clone"
        subprocess.run(["git", "clone", str(self.source), str(local_clone)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(local_clone), "bundle", "verify", str(bundle_file)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(local_clone), "fetch", str(bundle_file), f"{exported['branch_ref']}:refs/heads/spec/cand_exp"],
            check=True,
            capture_output=True,
        )
        fetched_sha = subprocess.run(
            ["git", "-C", str(local_clone), "rev-parse", "refs/heads/spec/cand_exp"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.assertEqual(fetched_sha, dehydrated["head_sha"])

        # include_bundle=False omits bundle_base64; export_bundle op always includes it.
        patch_only = await backend.call("export", **args, include_bundle=False)
        self.assertNotIn("bundle_base64", patch_only)
        bundle_alias = await backend.call("export_bundle", **args)
        self.assertEqual(bundle_alias["bundle_base64"], exported["bundle_base64"])

        # Local merge causes divergence from upstream.
        merged = await backend.call(
            "merge",
            **args,
            target_branch="master",
            approved_target_sha=prepared["base_sha"],
            approved_head_sha=dehydrated["head_sha"],
        )
        after_merge = await backend.call("prepare", project_id="proj", repo_source=str(self.source), target_branch="master")
        self.assertTrue(after_merge["diverged_from_upstream"])
        self.assertEqual(after_merge["base_sha"], merged["merged_sha"])
        self.assertEqual(after_merge["upstream_sha"], prepared["base_sha"])
        self.assertGreaterEqual(after_merge["ahead_of_upstream"], 1)
        self.assertEqual(after_merge["behind_upstream"], 0)

        # Upstream advances with a new commit; preserve_local reports both ahead & behind.
        (self.source / "README.md").write_text("upstream change\n", encoding="utf-8")
        self._git_src("add", "README.md")
        self._git_src("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "upstream commit")
        upstream_tip = subprocess.run(
            ["git", "-C", str(self.source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()

        diverged = await backend.call("prepare", project_id="proj", repo_source=str(self.source), target_branch="master")
        self.assertTrue(diverged["diverged_from_upstream"])
        self.assertEqual(diverged["base_sha"], merged["merged_sha"])
        self.assertEqual(diverged["upstream_sha"], upstream_tip)
        self.assertGreaterEqual(diverged["ahead_of_upstream"], 1)
        self.assertEqual(diverged["behind_upstream"], 1)

        # Explicit sync_strategy="reset_to_upstream" resumes upstream sync cleanly.
        synced = await backend.call(
            "prepare",
            project_id="proj",
            repo_source=str(self.source),
            target_branch="master",
            sync_strategy="reset_to_upstream",
        )
        self.assertFalse(synced["diverged_from_upstream"])
        self.assertEqual(synced["base_sha"], upstream_tip)
        self.assertEqual(synced["upstream_sha"], upstream_tip)
        self.assertEqual(synced["ahead_of_upstream"], 0)
        self.assertEqual(synced["behind_upstream"], 0)


if __name__ == "__main__":
    unittest.main()
