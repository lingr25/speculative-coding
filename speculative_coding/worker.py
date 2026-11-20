"""Reverse WSS sandbox execution worker daemon with zero long-lived provider credentials."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path

from aiohttp import ClientSession, WSMsgType

from .sandbox import FAST_READ_OPS, LocalDirSandbox

log = logging.getLogger(__name__)

VALID_WORKER_OPS = frozenset(
    {
        "prepare",
        "hydrate",
        "read",
        "grep",
        "write",
        "edit",
        "diff",
        "command",
        "dehydrate",
        "drop",
        "export",
        "export_bundle",
        "merge",
        "status",
        "gate",
        "describe",
        "task_result",
    }
)
MAINTAIN_INTERVAL_SECONDS = 300.0


class Worker:
    def __init__(self, root: Path | str, server: str, token: str):
        if not server.startswith("wss://"):
            raise ValueError("TLS WSS is required")
        self.backend = LocalDirSandbox(Path(root))
        self.server, self.token = server, token
        self.db = sqlite3.connect(str(Path(root) / "task-journal.sqlite3"))
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, digest TEXT NOT NULL, response TEXT, created REAL NOT NULL)"
        )
        # A daemon restart kills all children via systemd KillMode=control-group.
        self.db.execute(
            "UPDATE tasks SET response=? WHERE response IS NULL",
            (json.dumps({"error": "worker restarted; retry operation with a new attempt ID"}),),
        )
        self.db.execute("PRAGMA max_page_count=131072")  # 512MiB hard journal ceiling
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(tasks)")}
        if "op" not in columns:
            self.db.execute("ALTER TABLE tasks ADD COLUMN op TEXT NOT NULL DEFAULT ''")
        if "acked" not in columns:
            self.db.execute("ALTER TABLE tasks ADD COLUMN acked INTEGER NOT NULL DEFAULT 0")
        self.db.commit()
        self._last_maintain = 0.0
        self.running: dict[str, asyncio.Task] = {}
        self.ws = None

    async def task(self, data: dict):
        try:
            await self._execute_task(data)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Journal full/corrupt/unwritable must fail visibly, never strand an RPC.
            if self.ws and not self.ws.closed:
                reply = {"id": data.get("id"), "error": "worker task persistence/execution error: " + str(exc)}
                if data.get("ephemeral"):
                    reply["ephemeral"] = True
                await self.ws.send_json(reply)

    def maintain(self):
        self.db.execute("DELETE FROM tasks WHERE response IS NOT NULL AND created<?", (time.time() - 7 * 86400,))
        self.db.execute(
            "DELETE FROM tasks WHERE acked=1 AND op IN ('read','grep','diff','describe','status','export','export_bundle')"
        )
        self.db.execute(
            "UPDATE tasks SET response=? WHERE acked=1 AND op IN ('command','gate') AND length(response)>65536",
            (json.dumps({"error": "acknowledged task result evicted; retry with a new attempt ID"}),),
        )
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(PASSIVE)")
        self._last_maintain = time.time()

    def _maybe_maintain(self):
        if time.time() - self._last_maintain >= MAINTAIN_INTERVAL_SECONDS:
            self.maintain()

    async def _execute_task(self, data: dict):
        tid = data["id"]
        op = data.get("op")
        if op == "task_result":
            row = self.db.execute("SELECT response FROM tasks WHERE id=?", (data["args"]["lookup_id"],)).fetchone()
            result = {"found": bool(row and row[0]), "response": json.loads(row[0]) if row and row[0] else None}
            if self.ws and not self.ws.closed:
                await self.ws.send_json({"id": tid, "result": result})
            return

        # Fast-path ephemeral read-only operations skip SQLite journal I/O completely.
        if data.get("ephemeral") and op in FAST_READ_OPS:
            try:
                result = await self.backend.call(op, **data["args"])
                response = {"id": tid, "ephemeral": True, "result": result}
            except Exception as exc:
                response = {"id": tid, "ephemeral": True, "error": str(exc)}
            if self.ws and not self.ws.closed:
                await self.ws.send_json(response)
            return

        digest = hashlib.sha256(json.dumps({"op": op, "args": data["args"]}, sort_keys=True).encode()).hexdigest()
        row = self.db.execute("SELECT digest,response FROM tasks WHERE id=?", (tid,)).fetchone()
        can_reconcile = (
            op == "dehydrate"
            and bool(data["args"].get("expected_hash"))
            or op == "merge"
            and {"project_id", "cand_id", "target_branch", "approved_target_sha", "approved_head_sha"}
            <= data["args"].keys()
        )
        if row and row[0] == digest and row[1] and json.loads(row[1]).get("error") and can_reconcile:
            self.db.execute("DELETE FROM tasks WHERE id=?", (tid,))
            self.db.commit()
            row = None
        if row:
            if row[0] != digest:
                response = {"id": tid, "error": "task ID payload mismatch"}
            elif row[1] is not None:
                response = {"id": tid, **json.loads(row[1])}
            else:
                return
        else:
            self._maybe_maintain()
            self.db.execute(
                "INSERT INTO tasks (id,digest,response,created,op) VALUES (?,?,NULL,?,?)",
                (tid, digest, time.time(), op),
            )
            self.db.commit()
            try:
                result = await self.backend.call(op, **data["args"])
                response = {"id": tid, "result": result}
            except Exception as exc:
                response = {"id": tid, "error": str(exc)}
            self.db.execute(
                "UPDATE tasks SET response=? WHERE id=?",
                (json.dumps({k: v for k, v in response.items() if k != "id"}), tid),
            )
            self.db.commit()
        if self.ws and not self.ws.closed:
            await self.ws.send_json(response)

    async def heartbeats(self, ws):
        while not ws.closed:
            try:
                self._maybe_maintain()
            except sqlite3.Error:
                log.exception("worker journal maintenance failed")
            await ws.send_json({"type": "heartbeat", "metrics": await asyncio.to_thread(self.backend.metrics)})
            await asyncio.sleep(15)

    async def run(self):
        delay = 1
        try:
            async with ClientSession() as session:
                while True:
                    try:
                        async with session.ws_connect(
                            self.server,
                            headers={"Authorization": "Bearer " + self.token},
                            heartbeat=20,
                            max_msg_size=32 * 1024 * 1024,
                        ) as ws:
                            self.ws = ws
                            delay = 1
                            heartbeat = asyncio.create_task(self.heartbeats(ws))
                            try:
                                async for msg in ws:
                                    if msg.type != WSMsgType.TEXT:
                                        continue
                                    data = json.loads(msg.data)
                                    if "ack" in data:
                                        cur = self.db.execute(
                                            "UPDATE tasks SET acked=1 WHERE id=? AND acked=0",
                                            (str(data["ack"]),),
                                        )
                                        if cur.rowcount:
                                            self.db.commit()
                                        continue
                                    tid = data.get("id")
                                    if not isinstance(tid, str) or len(tid) > 128 or data.get("op") not in VALID_WORKER_OPS:
                                        continue
                                    if tid not in self.running:
                                        task = asyncio.create_task(self.task(data))
                                        self.running[tid] = task
                                        task.add_done_callback(lambda _, key=tid: self.running.pop(key, None))
                            finally:
                                heartbeat.cancel()
                                await asyncio.gather(heartbeat, return_exceptions=True)
                    except (OSError, asyncio.TimeoutError, Exception) as exc:
                        log.warning("spec worker reconnect: %s", type(exc).__name__)
                    finally:
                        self.ws = None
                    try:
                        self._maybe_maintain()
                    except sqlite3.Error:
                        log.exception("worker journal maintenance failed")
                    await asyncio.sleep(delay)
                    delay = min(5, delay * 2)
        finally:
            await self.backend.shutdown()
            for task in self.running.values():
                task.cancel()
            await asyncio.gather(*self.running.values(), return_exceptions=True)
            self.db.close()


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Speculative Coding isolated sandbox execution worker")
    parser.add_argument("--root", default=os.environ.get("SPEC_SANDBOX_ROOT", "/opt/spec_sandbox"))
    parser.add_argument(
        "--server",
        default=os.environ.get("SPEC_WORKER_SERVER", ""),
        help="Control-plane WSS endpoint (e.g. wss://control-plane.example.com/api/spec-worker/ws)",
    )
    args = parser.parse_args(argv)
    if not args.server:
        parser.error("--server or SPEC_WORKER_SERVER is required (must start with wss://)")
    token = os.environ.pop("SPEC_WORKER_TOKEN", "").strip()
    if len(token) < 32:
        parser.error("SPEC_WORKER_TOKEN must contain at least 32 characters")
    logging.basicConfig(level=logging.INFO)
    asyncio.run(Worker(args.root, args.server, token).run())


if __name__ == "__main__":
    main()
