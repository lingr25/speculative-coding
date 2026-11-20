"""Independent acceptance checks for reconnect-safe worker side effects."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from speculative_coding.sandbox import RemoteWorkerSandbox
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


if __name__ == "__main__":
    unittest.main()
