"""Independent speculative sandbox protocol; no provider credentials cross this boundary."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import signal
import socket
import sqlite3
import subprocess
import time
from abc import ABC, abstractmethod
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiohttp import WSMsgType, web

FAST_READ_OPS = frozenset({"read", "grep", "diff", "describe", "status"})

_ID = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
_PROTECTED = {".git", ".env", ".ssh", "deploy", ".github", "AGENTS.md"}
_CACHE_NOISE_DIRS = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".cache",
        "node_modules",
        ".venv",
        ".spec-preview",
    }
)
_CACHE_NOISE_FILES = frozenset({".coverage"})
_CACHE_NOISE_SUFFIXES = (".pyc", ".pyo", ".spec-tmp")


def _is_cache_noise(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if not parts:
        return True
    if any(part in _CACHE_NOISE_DIRS for part in parts):
        return True
    name = parts[-1]
    return name in _CACHE_NOISE_FILES or name.endswith(_CACHE_NOISE_SUFFIXES)


class SpecSandboxBackend(ABC):
    @abstractmethod
    async def call(self, operation: str, **arguments: Any) -> dict: ...


class RemoteWorkerSandbox(SpecSandboxBackend):
    """Reconnect resends stable IDs. Pure read-only queries bypass SQLite outbox overhead."""

    def __init__(self, journal_path: Path | str | None = None):
        self.journal_path = Path(journal_path) if journal_path else None
        self._consume_ticks = 0
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self._journal() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.execute("PRAGMA synchronous=NORMAL")
                db.execute(
                    "CREATE TABLE IF NOT EXISTS spec_worker_tasks ("
                    "task_id TEXT PRIMARY KEY, digest TEXT NOT NULL, payload_json TEXT NOT NULL, "
                    "response_json TEXT, status TEXT NOT NULL, created_at REAL NOT NULL)"
                )
                db.execute("CREATE INDEX IF NOT EXISTS idx_spec_worker_digest ON spec_worker_tasks(digest,status)")
        self.ws = None
        self.pending: dict[str, tuple[dict, asyncio.Future]] = {}
        self.metrics: dict = {}
        self.last_seen = 0.0

    @contextmanager
    def _journal(self):
        db = sqlite3.connect(str(self.journal_path), timeout=30)
        db.execute("PRAGMA synchronous=NORMAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def status(self):
        return {
            "connected": bool(self.ws and not self.ws.closed and (not self.last_seen or time.time() - self.last_seen < 60)),
            "last_seen": self.last_seen,
            "metrics": self.metrics,
            "pending_tasks": len(self.pending),
        }

    async def call(self, operation: str, **arguments: Any) -> dict:
        if not self.ws or self.ws.closed or (self.last_seen and time.time() - self.last_seen > 60):
            raise RuntimeError("worker offline; retry after reconnect")
        import uuid

        explicit_id = arguments.pop("task_id", None)
        ephemeral = operation in FAST_READ_OPS and explicit_id is None
        if ephemeral:
            task_id = uuid.uuid4().hex
            payload = {"id": task_id, "op": operation, "args": arguments, "ephemeral": True}
            future = asyncio.get_running_loop().create_future()
            self.pending[task_id] = (payload, future)
            try:
                if self.ws and not self.ws.closed:
                    await self.ws.send_json(payload)
                return await asyncio.wait_for(asyncio.shield(future), 120)
            finally:
                self.pending.pop(task_id, None)

        digest = hashlib.sha256(json.dumps({"op": operation, "args": arguments}, sort_keys=True).encode()).hexdigest()
        task_id = explicit_id or uuid.uuid4().hex
        replay = None
        sensitive = operation == "prepare" and bool(arguments.get("clone_token"))
        if self.journal_path and not sensitive:
            with self._journal() as db:
                if explicit_id:
                    row = db.execute(
                        "SELECT task_id,response_json,status,digest FROM spec_worker_tasks WHERE task_id=?",
                        (explicit_id,),
                    ).fetchone()
                    if row and row[3] != digest:
                        raise ValueError("task ID payload mismatch")
                else:
                    row = db.execute(
                        "SELECT task_id,response_json,status,digest FROM spec_worker_tasks "
                        "WHERE digest=? AND status IN ('queued','complete') ORDER BY created_at LIMIT 1",
                        (digest,),
                    ).fetchone()
                if row and row[2] in {"queued", "complete"}:
                    task_id = row[0]
                    if row[1]:
                        replay = json.loads(row[1])
                else:
                    payload = {"id": task_id, "op": operation, "args": arguments}
                    raw_payload = json.dumps(payload, ensure_ascii=False)
                    page_count = db.execute("PRAGMA page_count").fetchone()[0]
                    page_size = db.execute("PRAGMA page_size").fetchone()[0]
                    free_pages = db.execute("PRAGMA freelist_count").fetchone()[0]
                    used = max(0, (page_count - free_pages) * page_size)
                    if used + len(raw_payload.encode()) >= 256 * 1024**2:
                        raise RuntimeError("control-plane outbox budget exhausted")
                    db.execute(
                        "INSERT INTO spec_worker_tasks VALUES (?,?,?,NULL,?,?) "
                        "ON CONFLICT(task_id) DO UPDATE SET payload_json=excluded.payload_json,response_json=NULL,status='queued'",
                        (task_id, digest, raw_payload, "queued", time.time()),
                    )
        payload = {"id": task_id, "op": operation, "args": arguments}
        if replay is not None:
            self._consume(task_id)
            if replay.get("error"):
                raise RuntimeError(replay["error"])
            return replay.get("result", {})
        if task_id in self.pending:
            return await asyncio.shield(self.pending[task_id][1])
        future = asyncio.get_running_loop().create_future()
        self.pending[task_id] = (payload, future)
        try:
            if self.ws and not self.ws.closed:
                await self.ws.send_json(payload)
            result = await asyncio.wait_for(asyncio.shield(future), 3600 if operation in {"command", "gate"} else 120)
            self._consume(task_id)
            return result
        except BaseException:
            if future.done():
                self._consume(task_id)
            raise
        finally:
            self.pending.pop(task_id, None)

    async def reconcile_candidate(self, cand_id: str):
        if not self.journal_path:
            return
        with self._journal() as db:
            rows = db.execute(
                "SELECT payload_json FROM spec_worker_tasks WHERE status IN ('queued','complete') ORDER BY created_at"
            ).fetchall()
        for (raw,) in rows:
            payload = json.loads(raw)
            if payload.get("args", {}).get("cand_id") == cand_id:
                await self.call(payload["op"], **payload["args"], task_id=payload["id"])

    def _consume(self, task_id: str):
        if self.journal_path:
            self._consume_ticks += 1
            with self._journal() as db:
                db.execute(
                    "UPDATE spec_worker_tasks SET status='consumed',payload_json='{}',response_json=NULL WHERE task_id=?",
                    (task_id,),
                )
                if self._consume_ticks % 32 == 0:
                    db.execute(
                        "DELETE FROM spec_worker_tasks WHERE status='consumed' AND created_at<?",
                        (time.time() - 86400,),
                    )

    async def connect(self, request: web.Request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=32 * 1024 * 1024)
        await ws.prepare(request)
        old, self.ws = self.ws, ws
        if old and not old.closed:
            await old.close(code=1000, message=b"worker replaced")
        replayed = set()
        for payload, _ in list(self.pending.values()):
            await ws.send_json(payload)
            replayed.add(payload["id"])
        if self.journal_path:
            with self._journal() as db:
                rows = db.execute("SELECT task_id,payload_json FROM spec_worker_tasks WHERE status='queued'").fetchall()
            for task_id, raw in rows:
                if task_id not in replayed:
                    await ws.send_json(json.loads(raw))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                self.last_seen = time.time()
                if data.get("type") == "heartbeat":
                    self.metrics = data.get("metrics", {})
                    continue
                task_id = data.get("id")
                is_ephemeral = bool(data.get("ephemeral"))
                if self.journal_path and not is_ephemeral:
                    response = {k: v for k, v in data.items() if k not in {"id", "ephemeral"}}
                    with self._journal() as db:
                        db.execute(
                            "UPDATE spec_worker_tasks SET response_json=?,status='complete' WHERE task_id=? AND status='queued'",
                            (json.dumps(response, ensure_ascii=False), task_id),
                        )
                item = self.pending.get(task_id)
                if item and not item[1].done():
                    if data.get("error"):
                        item[1].set_exception(RuntimeError(data["error"]))
                    else:
                        item[1].set_result(data.get("result", {}))
                if not is_ephemeral:
                    await ws.send_json({"ack": task_id})
        finally:
            if self.ws is ws:
                self.ws = None
        return ws


class LocalDirSandbox(SpecSandboxBackend):
    """Worker-side sandbox implementation. unsafe_test_mode is never enabled by the production daemon."""

    def __init__(
        self,
        root: Path | str,
        *,
        unsafe_test_mode: bool = False,
        max_disk_gb: float = 30,
        min_free_gb: float = 8,
        worktree_concurrency: int = 3,
        build_concurrency: int = 2,
        auto_seed_deps: bool | None = None,
    ):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.unsafe_test_mode = unsafe_test_mode
        if auto_seed_deps is not None:
            self.auto_seed_deps = bool(auto_seed_deps)
        elif unsafe_test_mode:
            raw_env = os.environ.get("SPEC_AUTO_SEED_DEPS", "").strip().lower()
            self.auto_seed_deps = raw_env in {"1", "true", "yes"}
        else:
            raw_env = os.environ.get("SPEC_AUTO_SEED_DEPS", "1").strip().lower()
            self.auto_seed_deps = raw_env not in {"0", "false", "no"}
        self.max_bytes = int(max_disk_gb * 1024**3)
        self.min_free_bytes = int(min_free_gb * 1024**3)
        self.wt_limit = worktree_concurrency
        self.build_slots = asyncio.Semaphore(build_concurrency)
        self.lock = asyncio.Lock()
        self._repo_locks: dict[str, asyncio.Lock] = {}
        self._wt_locks: dict[str, asyncio.Lock] = {}
        for name in ("repos", "worktrees", "shared_caches", "golden_bases"):
            (self.root / name).mkdir(exist_ok=True)
        self._processes: set[asyncio.subprocess.Process] = set()
        self.baseline_leases: Counter = Counter()
        self.baseline_lock = asyncio.Lock()
        self.high_pressure_slot = asyncio.Lock()
        self.project_build_slots: dict[tuple[str, int], asyncio.Semaphore] = {}
        self._dedicated_mount = os.path.ismount(self.root)
        self._immutable_sizes: dict[Path, int] = {}
        self._cached_used: tuple[float, int] | None = None
        self._drops_since_gc: Counter[str] = Counter()

    def _invalidate_disk_cache(self):
        self._cached_used = None

    def _repo_lock(self, project_id: str) -> asyncio.Lock:
        pid = self._id(project_id)
        return self._repo_locks.setdefault(pid, asyncio.Lock())

    def _wt_lock(self, cand_id: str) -> asyncio.Lock:
        cid = self._id(cand_id)
        return self._wt_locks.setdefault(cid, asyncio.Lock())

    def _id(self, value) -> str:
        if not _ID.fullmatch(str(value)):
            raise ValueError("invalid sandbox identifier")
        return str(value)

    def _repo(self, project_id) -> Path:
        path = self.root / "repos" / (self._id(project_id) + ".git")
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("escaping repository")
        return path

    def _wt(self, cand_id) -> Path:
        path = self.root / "worktrees" / self._id(cand_id)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("escaping worktree")
        return path

    def _file(self, wt: Path, name: str, *, write: bool = False) -> Path:
        raw = Path(name)
        if (
            not raw.parts
            or raw.is_absolute()
            or ".." in raw.parts
            or any(p in _PROTECTED or p.startswith(".env") for p in raw.parts)
        ):
            raise ValueError("protected or escaping path")
        path = wt / raw
        if path.is_symlink() or not path.resolve().is_relative_to(wt.resolve()):
            raise ValueError("symlink/escaping path")
        if write and (raw.parts[0] == "tests" or raw.name.startswith("test_")):
            # Tests may be added, but never changed/deleted to make gates pass.
            if path.exists():
                raise ValueError("existing tests are immutable")
        return path

    def sandbox_argv(self, argv, cwd: Path, baseline: Path | None = None, *, allow_network: bool = False, writable_cache: bool = False):
        if allow_network:
            # Controlled dependency-seeding sandbox: network is permitted ONLY for package
            # acquisition with lifecycle scripts disabled; host filesystem remains isolated.
            command = [
                "bwrap",
                "--unshare-user",
                "--unshare-ipc",
                "--unshare-pid",
                "--unshare-uts",
                "--unshare-cgroup-try",
                "--die-with-parent",
                "--new-session",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
            ]
            ro_dirs = (
                "/usr",
                "/bin",
                "/lib",
                "/lib64",
                "/etc/alternatives",
                "/etc/resolv.conf",
                "/etc/hosts",
                "/etc/ssl",
                "/etc/ca-certificates",
                "/etc/pki",
            )
        else:
            command = [
                "bwrap",
                "--unshare-all",
                "--die-with-parent",
                "--new-session",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
            ]
            ro_dirs = ("/usr", "/bin", "/lib", "/lib64", "/etc/alternatives")
        for directory in ro_dirs:
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        command += ["--bind", str(cwd), "/workspace"]
        if writable_cache:
            command += ["--bind", str(self.root / "shared_caches"), "/cache"]
        else:
            command += ["--overlay-src", str(self.root / "shared_caches"), "--tmp-overlay", "/cache"]
        if (cwd / ".git").is_file():
            command += ["--ro-bind", str(cwd / ".git"), "/workspace/.git"]
        # Golden dependency directories are never writable by candidate shell commands.
        if baseline:
            for directory, dirs, _ in os.walk(baseline, followlinks=False):
                for name in list(dirs):
                    if name not in {"node_modules", ".venv"}:
                        continue
                    source = Path(directory) / name
                    if source.is_symlink():
                        raise ValueError("dependency baseline root cannot be a symlink")
                    relative = source.relative_to(baseline)
                    target = cwd / relative
                    if target.is_symlink() or not target.resolve().is_relative_to(cwd.resolve()):
                        raise ValueError("escaping dependency mount")
                    target.mkdir(parents=True, exist_ok=True)
                    command += ["--ro-bind", str(source), "/workspace/" + relative.as_posix()]
                    dirs.remove(name)
        return command + ["--chdir", "/workspace", "--", *argv]

    @staticmethod
    def _baseline_bin_paths(baseline: Path | None) -> list[str]:
        if not baseline or not baseline.is_dir():
            return []
        bins: list[str] = []
        for directory, dirs, _ in os.walk(baseline, followlinks=False):
            for name in list(dirs):
                if name not in {"node_modules", ".venv"}:
                    continue
                relative = (Path(directory) / name).relative_to(baseline).as_posix()
                sub = "bin" if name == ".venv" else ".bin"
                bins.append(f"/workspace/{relative}/{sub}")
                dirs.remove(name)
        return bins

    @contextmanager
    def _github_askpass(self, token: str):
        helper = self.root / f".git-askpass-{os.getpid()}-{time.time_ns()}.sh"
        helper.write_text(
            '#!/bin/sh\ncase "$1" in\n*Username*) printf "%s\\n" "$SPEC_GIT_USERNAME" ;;\n'
            '*Password*) printf "%s\\n" "$SPEC_GIT_PASSWORD" ;;\n*) exit 1 ;;\nesac\n',
            encoding="utf-8",
        )
        helper.chmod(0o700)
        try:
            yield {
                "env_extra": {
                    "GIT_ASKPASS": str(helper),
                    "SPEC_GIT_USERNAME": "x-access-token",
                    "SPEC_GIT_PASSWORD": str(token),
                },
                "redact": (str(token),),
            }
        finally:
            helper.unlink(missing_ok=True)

    @staticmethod
    async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
        if os.name == "nt":
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/PID",
                    str(proc.pid),
                    "/T",
                    "/F",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await killer.wait()
            except Exception:
                pass
        elif os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass

    async def _exec(
        self,
        argv,
        *,
        cwd=None,
        timeout=300,
        sandbox=False,
        baseline=None,
        allow_network=False,
        writable_cache=False,
        output_limit=None,
        env_extra=None,
        redact=(),
    ):
        base_path = os.environ.get("PATH", "/usr/bin:/bin")
        extra_bins = self._baseline_bin_paths(baseline) if sandbox and not self.unsafe_test_mode else []
        full_path = ":".join([*extra_bins, base_path]) if extra_bins else base_path
        env = {
            "PATH": full_path,
            "LANG": "C.UTF-8",
            "HOME": "/tmp",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "PNPM_HOME": "/tmp/pnpm",
            "npm_config_cache": "/cache/npm",
            "UV_CACHE_DIR": "/cache/uv",
            "UV_LINK_MODE": "copy",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if os.name == "nt":
            for name in ("SYSTEMROOT", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA"):
                if os.environ.get(name):
                    env[name] = os.environ[name]
        env.update(env_extra or {})
        if sandbox and not self.unsafe_test_mode:
            if os.name != "posix" or not shutil.which("bwrap"):
                raise RuntimeError("Linux bubblewrap is required; refusing unsandboxed execution")
            argv, cwd = (
                self.sandbox_argv(argv, cwd, baseline, allow_network=allow_network, writable_cache=writable_cache),
                self.root,
            )

        def child_setup():
            os.nice(10)
            try:
                Path("/proc/self/oom_score_adj").write_text("800")
            except OSError:
                pass

        spawn_opts: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True, "preexec_fn": child_setup}
        )
        proc = await asyncio.create_subprocess_exec(
            *map(str, argv),
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            **spawn_opts,
        )
        self._processes.add(proc)
        chunks, size = [], 0

        async def drain():
            nonlocal size
            while True:
                chunk = await proc.stdout.read(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                while output_limit and size > output_limit and len(chunks) > 1:
                    size -= len(chunks.pop(0))

        reader = asyncio.create_task(drain())

        async def watchdog():
            while proc.returncode is None:
                await asyncio.sleep(3)
                metrics = await asyncio.to_thread(self.metrics)
                if metrics["pressure"] == "blocked":
                    await self._kill_process_tree(proc)
                    raise RuntimeError("sandbox disk backpressure watchdog killed command")

        monitor = asyncio.create_task(watchdog()) if sandbox else None
        try:
            await asyncio.wait_for(proc.wait(), timeout)
        finally:
            if proc.returncode is None:
                await self._kill_process_tree(proc)
            elif os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await proc.wait()
            try:
                await asyncio.wait_for(reader, timeout=10.0)
            except asyncio.TimeoutError:
                await self._kill_process_tree(proc)
                reader.cancel()
                await asyncio.gather(reader, return_exceptions=True)
            self._processes.discard(proc)
            if monitor:
                monitor.cancel()
                watched = await asyncio.gather(monitor, return_exceptions=True)
                if watched and isinstance(watched[0], RuntimeError):
                    raise watched[0]
        output = b"".join(chunks).decode(errors="replace")
        for secret in redact:
            if secret:
                output = output.replace(str(secret), "[redacted]")
        return {"code": proc.returncode, "output": output}

    async def _git(self, repo: Path, *args, **kw):
        if repo.parent == self.root / "worktrees":
            pointer = repo / ".git"
            if not pointer.is_file() or pointer.is_symlink():
                raise ValueError("invalid worktree git pointer")
            raw = pointer.read_text().strip()
            expected = (self.root / "repos").resolve()
            gitdir = Path(raw.removeprefix("gitdir: ").strip()).resolve()
            if not raw.startswith("gitdir: ") or not gitdir.is_relative_to(expected) or gitdir.parent.name != "worktrees":
                raise ValueError("untrusted worktree git pointer")
            backlink = (gitdir / "gitdir").read_text().strip()
            if Path(backlink).resolve() != pointer.resolve():
                raise ValueError("worktree backlink mismatch")
        result = await self._exec(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "protocol.file.allow=" + ("always" if self.unsafe_test_mode else "never"),
                "-c",
                "http.followRedirects=false",
                "-C",
                str(repo),
                *args,
            ],
            **kw,
        )
        if result["code"]:
            raise RuntimeError(result["output"])
        return result["output"].strip()

    @staticmethod
    def _scan_dir_bytes(target: Path, seen: set[tuple[int, int]]) -> int:
        subtotal = 0
        for directory, _, files in os.walk(target, followlinks=False):
            for name in files:
                try:
                    st = (Path(directory) / name).lstat()
                    key = (st.st_dev, st.st_ino)
                    if key not in seen:
                        subtotal += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
                        seen.add(key)
                except OSError:
                    pass
        return subtotal

    def _compute_used_bytes(self) -> int:
        if self._dedicated_mount:
            return shutil.disk_usage(self.root).used
        now = time.monotonic()
        cached = self._cached_used
        if cached is not None and (now - cached[0]) < 5.0:
            return cached[1]
        total = 0
        seen: set[tuple[int, int]] = set()
        golden_root = self.root / "golden_bases"
        for directory, dirs, files in os.walk(self.root, followlinks=False):
            current = Path(directory)
            if current.parent == golden_root:
                for sub in list(dirs):
                    base_dir = current / sub
                    if base_dir in self._immutable_sizes:
                        total += self._immutable_sizes[base_dir]
                        dirs.remove(sub)
                    elif (base_dir / "ready").is_file() and not base_dir.is_symlink():
                        size = self._scan_dir_bytes(base_dir, seen)
                        self._immutable_sizes[base_dir] = size
                        total += size
                        dirs.remove(sub)
            for name in files:
                try:
                    st = (current / name).lstat()
                    key = (st.st_dev, st.st_ino)
                    if key not in seen:
                        total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
                        seen.add(key)
                except OSError:
                    pass
        self._cached_used = (now, total)
        return total

    def metrics(self):
        pool_usage = shutil.disk_usage(self.root)
        pool_free = pool_usage.free
        total = self._compute_used_bytes()
        free = shutil.disk_usage(self.root.parent).free
        memory = {}
        if Path("/proc/meminfo").exists():
            memory = {
                line.split(":")[0]: int(line.split()[1]) * 1024
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith(("MemTotal:", "MemAvailable:"))
            }
        try:
            wt_count = len(list((self.root / "worktrees").iterdir()))
        except OSError:
            wt_count = 0
        return {
            "disk_used_bytes": total,
            "disk_limit_bytes": self.max_bytes,
            "system_free_bytes": free,
            "min_free_bytes": self.min_free_bytes,
            "memory": memory,
            "worktrees": wt_count,
            "pressure": (
                "blocked"
                if total >= self.max_bytes or pool_free < 256 * 1024**2 or free < self.min_free_bytes
                else "high"
                if total >= self.max_bytes * 0.75
                else "normal"
            ),
        }

    async def _admit(self):
        m = await asyncio.to_thread(self.metrics)
        if m["pressure"] == "blocked":
            raise RuntimeError("sandbox disk backpressure; retry after dehydration")
        if m["pressure"] == "high":
            async with self.baseline_lock:
                for path in sorted((self.root / "golden_bases").iterdir(), key=lambda p: p.stat().st_mtime):
                    if path.is_dir() and not path.is_symlink() and self.baseline_leases[path] == 0:
                        for sub in list(self._immutable_sizes):
                            if sub.parent == path:
                                self._immutable_sizes.pop(sub, None)
                        await asyncio.to_thread(shutil.rmtree, path)
                        self.baseline_leases.pop(path, None)
                        self._invalidate_disk_cache()
        return m

    async def call(self, operation: str, **a: Any) -> dict:
        if operation == "status":
            return await asyncio.to_thread(self.metrics)
        if operation in {"command", "gate"}:
            limit = min(2, max(1, int(a.get("build_concurrency", 2))))
            slots = self.project_build_slots.setdefault((a["project_id"], limit), asyncio.Semaphore(limit))
            async with self.build_slots, slots:
                m = await self._admit()
                if m["pressure"] == "high":
                    async with self.high_pressure_slot:
                        return await (self._gate(a) if operation == "gate" else self._command(a))
                return await (self._gate(a) if operation == "gate" else self._command(a))
        # Read-only inspections run concurrently without serializing behind git fetch/gc.
        if operation in {"describe", "read", "grep", "export"}:
            return await self._operation(operation, a)
        if operation in {"write", "edit", "diff"}:
            async with self._wt_lock(a["cand_id"]):
                return await self._operation(operation, a)
        if "project_id" in a:
            async with self._repo_lock(a["project_id"]):
                if "cand_id" in a and operation in {"dehydrate", "drop"}:
                    async with self._wt_lock(a["cand_id"]):
                        return await self._operation(operation, a)
                return await self._operation(operation, a)
        async with self.lock:
            return await self._operation(operation, a)

    @staticmethod
    def _command_argv(command):
        return ["/bin/sh", "-lc", str(command)] if os.name == "posix" else ["cmd.exe", "/d", "/s", "/c", str(command)]

    async def _command(self, a):
        wt = self._wt(a["cand_id"])
        if not wt.is_dir():
            raise ValueError("candidate is not hydrated")
        baseline = await self._baseline(a, wt)
        lease = baseline.parent if baseline else None
        try:
            # Preserve incremental test/compiler caches across iterative coding steps;
            # full cleanup runs after final gate verification before dehydration.
            return await self._exec(
                self._command_argv(a["command"]),
                cwd=wt,
                timeout=min(1800, max(1, int(a.get("timeout", 600)))),
                sandbox=True,
                baseline=baseline,
                output_limit=2 * 1024 * 1024,
            )
        finally:
            self._invalidate_disk_cache()
            if lease:
                async with self.baseline_lock:
                    self.baseline_leases[lease] -= 1

    async def _clean(self, wt: Path):
        await self._git(wt, "clean", "-ffd", "-x")
        self._invalidate_disk_cache()

    @staticmethod
    def _install_plan(package_root: Path) -> tuple[str, str]:
        """Return (offline_cmd, controlled_online_cmd) with lifecycle scripts disabled."""
        if (package_root / "pnpm-lock.yaml").exists():
            return (
                "pnpm install --offline --frozen-lockfile --ignore-scripts --store-dir /cache/pnpm-store",
                "pnpm install --frozen-lockfile --ignore-scripts --store-dir /cache/pnpm-store",
            )
        if (package_root / "package-lock.json").exists():
            return (
                "npm ci --offline --ignore-scripts --cache /cache/npm --no-audit --no-fund",
                "npm ci --ignore-scripts --cache /cache/npm --no-audit --no-fund",
            )
        if (package_root / "uv.lock").exists():
            return (
                "uv sync --offline --frozen --cache-dir /cache/uv",
                "uv sync --frozen --cache-dir /cache/uv",
            )
        if (package_root / "requirements.txt").exists():
            return (
                "uv venv .venv && uv pip install --offline --python .venv/bin/python --cache-dir /cache/uv -r requirements.txt",
                "uv venv .venv && uv pip install --python .venv/bin/python --cache-dir /cache/uv -r requirements.txt",
            )
        return (
            "uv venv .venv && uv pip install --offline --python .venv/bin/python --cache-dir /cache/uv .",
            "uv venv .venv && uv pip install --python .venv/bin/python --cache-dir /cache/uv .",
        )

    async def _baseline(self, a, wt: Path):
        """One immutable dependency baseline per project/manifest digest."""
        tracked = [name for name in (await self._git(wt, "ls-files", "-z")).split("\0") if name]
        lock_names = {"package-lock.json", "pnpm-lock.yaml", "uv.lock"}
        include_py_manifests = self.auto_seed_deps or not self.unsafe_test_mode
        if include_py_manifests:
            manifest_names = lock_names | {"requirements.txt", "pyproject.toml"}
        else:
            manifest_names = lock_names

        by_dir: dict[Path, list[Path]] = {}
        for name in tracked:
            base_name = Path(name).name
            if base_name not in manifest_names:
                continue
            path = self._file(wt, name)
            if path.is_file():
                by_dir.setdefault(path.parent, []).append(path)

        locks: list[Path] = []
        for _parent_dir, items in by_dir.items():
            names_here = {p.name for p in items}
            if "uv.lock" in names_here:
                items = [p for p in items if p.name not in {"requirements.txt", "pyproject.toml"}]
            elif "requirements.txt" in names_here and "pyproject.toml" in names_here:
                items = [p for p in items if p.name != "pyproject.toml"]
            locks.extend(items)

        if not locks:
            return None
        digest = hashlib.sha256(
            b"".join(str(p.relative_to(wt)).encode() + b"\0" + p.read_bytes() for p in sorted(locks))
        ).hexdigest()
        parent = self.root / "golden_bases" / self._id(a["project_id"])
        baseline = parent / digest
        async with self.baseline_lock:
            if (baseline / "ready").exists():
                parent.touch()
                self.baseline_leases[parent] += 1
                return baseline
            parent.mkdir(exist_ok=True)
            baseline.mkdir(exist_ok=True)
            self.baseline_leases[parent] += 1
            try:
                roots = {p.parent for p in locks}
                for package_root in sorted(roots):
                    relative = package_root.relative_to(wt)
                    offline_cmd, online_cmd = self._install_plan(package_root)
                    ws_dir = shlex.quote("/workspace/" + relative.as_posix())
                    result = await self._exec(
                        ["/bin/sh", "-lc", f"cd {ws_dir} && {offline_cmd}"],
                        cwd=wt,
                        sandbox=True,
                        timeout=900,
                        output_limit=2 * 1024 * 1024,
                    )
                    if result["code"] and self.auto_seed_deps:
                        for partial_dir in ("node_modules", ".venv"):
                            target_partial = package_root / partial_dir
                            if target_partial.exists() and not target_partial.is_symlink():
                                await asyncio.to_thread(shutil.rmtree, target_partial, True)
                        result = await self._exec(
                            ["/bin/sh", "-lc", f"cd {ws_dir} && {online_cmd}"],
                            cwd=wt,
                            sandbox=True,
                            allow_network=True,
                            writable_cache=True,
                            timeout=900,
                            output_limit=2 * 1024 * 1024,
                        )
                    if result["code"]:
                        raise RuntimeError("offline dependency baseline unavailable; seed shared caches: " + result["output"])
                    for name in ("node_modules", ".venv"):
                        dependency = package_root / name
                        if dependency.is_dir() and not dependency.is_symlink():
                            destination = baseline / relative / name
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            await asyncio.to_thread(shutil.move, str(dependency), str(destination))
                (baseline / "ready").write_text(str(time.time()))
                parent.touch()
                self._invalidate_disk_cache()
            except BaseException:
                self._immutable_sizes.pop(baseline, None)
                await asyncio.to_thread(shutil.rmtree, baseline, True)
                await self._clean(wt)
                raise
            finally:
                self.baseline_leases[parent] -= 1
            self.baseline_leases[parent] += 1
        return baseline

    async def _gate(self, a):
        wt = self._wt(a["cand_id"])
        if not wt.is_dir():
            raise ValueError("candidate is not hydrated")
        async with self._wt_lock(a["cand_id"]):
            await self._operation("diff", a)
        baseline = await self._baseline(a, wt)
        lease = baseline.parent if baseline else None
        evidence = []
        try:
            for command in a.get("commands", []):
                result = await self._exec(
                    self._command_argv(command),
                    cwd=wt,
                    sandbox=True,
                    baseline=baseline,
                    timeout=900,
                    output_limit=2 * 1024 * 1024,
                )
                evidence.append({"command": command, **result})
                if result["code"]:
                    return {"passed": False, "gates": evidence}
            preview = None
            if a.get("preview_command"):
                preview = await self._exec(
                    self._command_argv(a["preview_command"]),
                    cwd=wt,
                    sandbox=True,
                    baseline=baseline,
                    timeout=300,
                    output_limit=2 * 1024 * 1024,
                )
                if preview["code"]:
                    return {"passed": False, "gates": evidence, "preview": preview}
            artifacts = []
            folder = wt / ".spec-preview"
            size = 0
            if folder.is_dir() and not folder.is_symlink():
                for path in sorted(folder.rglob("*")):
                    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(folder.resolve()):
                        continue
                    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".html", ".css", ".js", ".json", ".svg"}:
                        continue
                    size += path.stat().st_size
                    if size >= 13 * 1024**2:
                        raise RuntimeError("preview artifact budget exceeded")
                    artifacts.append(
                        {"path": str(path.relative_to(folder)), "base64": base64.b64encode(path.read_bytes()).decode()}
                    )
            return {"passed": True, "gates": evidence, "preview": preview, "artifacts": artifacts}
        finally:
            if lease:
                async with self.baseline_lock:
                    self.baseline_leases[lease] -= 1
            await self._clean(wt)

    @staticmethod
    def _apply_edit(original: str, target: str, replacement: str, *, replace_all: bool = False) -> tuple[str, int]:
        if not target:
            raise ValueError("edit target must be non-empty")
        exact_count = original.count(target)
        if exact_count == 1 or (exact_count > 1 and replace_all):
            updated = original.replace(target, replacement) if replace_all else original.replace(target, replacement, 1)
            return updated, (exact_count if replace_all else 1)
        if exact_count > 1:
            raise ValueError(f"target matched {exact_count} locations; include more surrounding context or set replace_all=true")

        newline = "\r\n" if "\r\n" in original else "\n"
        orig_lines = original.splitlines()
        target_lines = [ln.rstrip() for ln in target.splitlines()]
        if not target_lines:
            raise ValueError("target not found in file")
        win = len(target_lines)
        hits = [
            idx
            for idx in range(len(orig_lines) - win + 1)
            if [ln.rstrip() for ln in orig_lines[idx : idx + win]] == target_lines
        ]
        if not hits:
            raise ValueError("target not found in file; read the target lines first")
        if len(hits) > 1 and not replace_all:
            raise ValueError(f"target matched {len(hits)} locations; include more surrounding context")
        repl_lines = replacement.splitlines()
        out_lines: list[str] = []
        cursor = 0
        applied = 0
        for idx in hits:
            if idx < cursor:
                continue
            out_lines.extend(orig_lines[cursor:idx])
            out_lines.extend(repl_lines)
            cursor = idx + win
            applied += 1
            if not replace_all:
                break
        out_lines.extend(orig_lines[cursor:])
        trailing = newline if original.endswith(("\n", "\r\n")) else ""
        return newline.join(out_lines) + trailing, applied

    def _grep_worktree(self, wt: Path, pattern: str, subpath: str, *, regex: bool, max_results: int) -> dict:
        if not pattern:
            raise ValueError("grep pattern must be non-empty")
        matcher = re.compile(pattern) if regex else None
        search_root = self._file(wt, subpath) if subpath else wt
        files_to_scan: list[Path] = []
        if search_root.is_file():
            files_to_scan.append(search_root)
        elif search_root.is_dir():
            for directory, dirs, files in os.walk(search_root, followlinks=False):
                dirs[:] = sorted(d for d in dirs if d not in _CACHE_NOISE_DIRS and not d.startswith(".env"))
                for name in sorted(files):
                    if name in _CACHE_NOISE_FILES or name.endswith(_CACHE_NOISE_SUFFIXES) or name.startswith(".env"):
                        continue
                    candidate = Path(directory) / name
                    if not candidate.is_symlink():
                        files_to_scan.append(candidate)

        matches: list[dict[str, Any]] = []
        truncated = False
        for file_path in files_to_scan:
            rel = file_path.relative_to(wt).as_posix()
            if any(part in _PROTECTED for part in Path(rel).parts):
                continue
            try:
                if file_path.stat().st_size > 2 * 1024 * 1024:
                    continue
                raw = file_path.read_bytes()
                if b"\x00" in raw[:4096]:
                    continue
                text = raw.decode("utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                hit = bool(matcher.search(line)) if matcher is not None else (pattern in line)
                if hit:
                    if len(matches) >= max_results:
                        return {"matches": matches, "truncated": True}
                    matches.append({"path": rel, "line": lineno, "text": line[:240]})
        return {"matches": matches, "truncated": truncated}

    async def _grep_repo(self, repo: Path, ref: str, pattern: str, subpath: str, *, regex: bool, max_results: int) -> dict:
        if not pattern:
            raise ValueError("grep pattern must be non-empty")
        if not re.fullmatch(r"[0-9a-f]{40,64}", ref):
            raise ValueError("invalid SHA")
        if subpath:
            self._file(self.root / "absent", subpath)
        flag = "-E" if regex else "-F"
        cmd = ["git", "-c", "core.hooksPath=/dev/null", "-C", str(repo), "grep", "-n", "-I", flag, "-e", pattern, ref]
        if subpath:
            cmd.extend(["--", subpath])
        res = await self._exec(cmd, output_limit=1024 * 1024)
        if res["code"] == 1 and not res["output"].strip():
            return {"matches": [], "truncated": False}
        if res["code"] not in (0, 1):
            raise RuntimeError(res["output"])
        prefix = ref + ":"
        matches: list[dict[str, Any]] = []
        truncated = False
        for row in res["output"].splitlines():
            body = row[len(prefix) :] if row.startswith(prefix) else row
            parts = body.split(":", 2)
            if len(parts) < 3:
                continue
            rel_path, line_str, snippet = parts
            if any(part in _PROTECTED or part.startswith(".env") for part in Path(rel_path).parts):
                continue
            if len(matches) >= max_results:
                truncated = True
                break
            try:
                lineno = int(line_str)
            except ValueError:
                lineno = 1
            matches.append({"path": rel_path, "line": lineno, "text": snippet[:240]})
        return {"matches": matches, "truncated": truncated}

    @staticmethod
    def _slice_read(text: str, start_line: Any, end_line: Any) -> dict[str, Any]:
        if start_line is None and end_line is None:
            return {"content": text}
        lines = text.splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        end = min(total, int(end_line if end_line is not None else start + 399))
        if start > total:
            return {"content": "", "total_lines": total, "start_line": start, "end_line": total}
        numbered = "\n".join(f"{idx}: {lines[idx - 1]}" for idx in range(start, end + 1))
        return {"content": numbered, "total_lines": total, "start_line": start, "end_line": end}

    async def _stage_candidate_changes(self, wt: Path) -> None:
        await self._git(wt, "add", "-u")
        untracked = (await self._git(wt, "ls-files", "-z", "--others", "--exclude-standard")).split("\0")
        fresh = [
            rel
            for rel in untracked
            if rel and not _is_cache_noise(rel) and not any(p in _PROTECTED or p.startswith(".env") for p in Path(rel).parts)
        ]
        if fresh:
            for idx in range(0, len(fresh), 50):
                await self._git(wt, "add", "--", *fresh[idx : idx + 50])

    async def _operation(self, op: str, a: dict) -> dict:
        repo = self._repo(a["project_id"])
        wt = self._wt(a["cand_id"]) if "cand_id" in a else None
        if op == "prepare":
            from contextlib import nullcontext

            source = str(a["repo_source"])
            token = str(a.get("clone_token") or "")
            url = urlparse(source)
            if not self.unsafe_test_mode:
                allowed = set(os.environ.get("SPEC_REPO_HOSTS", "github.com,gitlab.com,gitee.com").split(","))
                if url.hostname not in allowed:
                    raise ValueError("repository host is not operator-allowlisted")
                addresses = await asyncio.to_thread(socket.getaddrinfo, url.hostname, url.port or 443, type=socket.SOCK_STREAM)
                if any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
                    raise ValueError("repository must resolve to public IPs")
            if not self.unsafe_test_mode and (
                url.scheme != "https"
                or url.username
                or url.password
                or url.query
                or url.fragment
                or not url.hostname
                or url.hostname in {"localhost", "127.0.0.1", "169.254.169.254"}
            ):
                raise ValueError("only credential-free HTTPS repository URLs are allowed")
            with (self._github_askpass(token) if token else nullcontext({"env_extra": {}, "redact": ()})) as auth:
                await self._admit()
                fresh_clone = not repo.exists()
                if fresh_clone:
                    result = await self._exec(
                        [
                            "git",
                            "-c",
                            "core.hooksPath=/dev/null",
                            "-c",
                            "http.followRedirects=false",
                            "clone",
                            "--bare",
                            "--",
                            source,
                            str(repo),
                        ],
                        **auth,
                    )
                    if result["code"]:
                        raise RuntimeError(result["output"])
                branch = str(a.get("target_branch", "master"))
                if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._/-]*", branch) or ".." in branch or "//" in branch or branch.endswith(("/", ".lock", ".")):
                    await self._git(repo, "check-ref-format", "--branch", branch)
                target = "refs/heads/" + branch
                tracking = "refs/remotes/origin/" + branch
                if fresh_clone:
                    sha = await self._git(repo, "rev-parse", target)
                    _, tree, logs = await asyncio.gather(
                        self._git(repo, "update-ref", tracking, sha),
                        self._git(repo, "ls-tree", "-r", "--name-only", sha),
                        self._git(repo, "log", "-12", "--oneline", sha),
                    )
                else:
                    current = await self._git(repo, "rev-parse", target)
                    previous = await self._exec(["git", "-C", str(repo), "rev-parse", "--verify", tracking])
                    previous_sha = previous["output"].strip() if previous["code"] == 0 else current
                    await self._git(repo, "fetch", "--prune", "origin", "+refs/heads/*:refs/remotes/origin/*", **auth)
                    remote = await self._git(repo, "rev-parse", tracking)
                    if current == previous_sha and remote != current:
                        await self._git(repo, "update-ref", target, remote, current)
                        sha = remote
                    else:
                        sha = await self._git(repo, "rev-parse", target)
                    tree, logs = await asyncio.gather(
                        self._git(repo, "ls-tree", "-r", "--name-only", sha),
                        self._git(repo, "log", "-12", "--oneline", sha),
                    )
                self._invalidate_disk_cache()
                return {"base_sha": sha, "tree": tree, "recent_commits": logs}
        if not repo.is_dir():
            raise ValueError("unknown project repository")
        if op == "describe":
            sha = str(a["base_sha"])
            if not re.fullmatch(r"[0-9a-f]{40,64}", sha):
                raise ValueError("invalid SHA")
            tree, recent_commits = await asyncio.gather(
                self._git(repo, "ls-tree", "-r", "--name-only", sha),
                self._git(repo, "log", "-12", "--oneline", sha),
            )
            return {"tree": tree, "recent_commits": recent_commits}
        if op == "hydrate":
            await self._admit()
            if wt.exists():
                return {"path": str(wt)}
            if len(list((self.root / "worktrees").iterdir())) >= self.wt_limit:
                raise RuntimeError("worktree slots exhausted")
            ref = a.get("ref") or a["base_sha"]
            if not re.fullmatch(r"[0-9a-f]{40,64}|refs/spec/[A-Za-z0-9_-]+", ref):
                raise ValueError("invalid source ref")
            await self._git(repo, "worktree", "add", "--detach", str(wt), ref)
            self._invalidate_disk_cache()
            return {"path": str(wt)}
        if op == "read":
            start_line = a.get("start_line")
            end_line = a.get("end_line")
            if wt is not None and wt.exists():
                path = self._file(wt, a["path"])
                text = await asyncio.to_thread(path.read_text, encoding="utf-8")
                return self._slice_read(text, start_line, end_line)
            path_str = str(a["path"])
            self._file(self.root / "absent", path_str)
            ref = a.get("base_sha", "HEAD")
            if not re.fullmatch(r"[0-9a-f]{40,64}", ref):
                raise ValueError("invalid SHA")
            text = await self._git(repo, "show", ref + ":" + path_str)
            return self._slice_read(text, start_line, end_line)
        if op == "grep":
            max_results = max(1, min(200, int(a.get("max_results", 80))))
            pattern = str(a.get("pattern") or "")
            subpath = str(a.get("path") or "").strip()
            regex = bool(a.get("regex", False))
            if wt is not None and wt.exists():
                return await asyncio.to_thread(
                    self._grep_worktree, wt, pattern, subpath, regex=regex, max_results=max_results
                )
            return await self._grep_repo(
                repo, str(a.get("base_sha", "HEAD")), pattern, subpath, regex=regex, max_results=max_results
            )
        if op == "write":
            if not wt or not wt.is_dir():
                raise ValueError("candidate is not hydrated")
            path = self._file(wt, a["path"], write=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".spec-tmp")
            temp.write_text(a["content"], encoding="utf-8")
            temp.replace(path)
            await self._git(wt, "add", "--", a["path"])
            self._invalidate_disk_cache()
            return {"written": a["path"]}
        if op == "edit":
            if not wt or not wt.is_dir():
                raise ValueError("candidate is not hydrated")
            path = self._file(wt, a["path"], write=True)
            if not path.is_file():
                raise ValueError("file does not exist; use write to create new files")
            original = path.read_text(encoding="utf-8")
            updated, count = self._apply_edit(
                original,
                str(a.get("target") or ""),
                str(a.get("replacement") or ""),
                replace_all=bool(a.get("replace_all", False)),
            )
            temp = path.with_name(path.name + ".spec-tmp")
            temp.write_text(updated, encoding="utf-8")
            temp.replace(path)
            await self._git(wt, "add", "--", a["path"])
            self._invalidate_disk_cache()
            return {"edited": a["path"], "replacements": count}
        if op == "diff":
            await self._stage_candidate_changes(wt)
            diff, names = await asyncio.gather(
                self._git(wt, "diff", "--cached", "--binary", a["base_sha"]),
                self._git(wt, "diff", "--cached", "--name-status", a["base_sha"]),
            )
            for line in names.splitlines():
                action, name = line.split("\t", 1)
                raw = Path(name)
                if any(p in _PROTECTED or p.startswith(".env") for p in raw.parts) or (
                    action != "A" and (raw.parts[0] == "tests" or raw.name.startswith("test_"))
                ):
                    raise ValueError("candidate changed protected paths/tests")
            return {"diff": diff, "hash": hashlib.sha256(diff.encode()).hexdigest()}
        if op == "dehydrate":
            ref = "refs/spec/" + a["cand_id"]
            existing = await self._exec(["git", "-C", str(repo), "rev-parse", "--verify", ref])
            if existing["code"] == 0 and a.get("expected_hash"):
                previous = await self._git(repo, "diff", "--binary", a["base_sha"], ref)
                if hashlib.sha256(previous.encode()).hexdigest() == a["expected_hash"]:
                    if wt.exists():
                        checked = await self.call_unlocked_diff(a)
                        if checked["hash"] != a["expected_hash"]:
                            raise ValueError("diff changed after review")
                        await self._remove(repo, wt)
                    return {"branch_ref": ref, "head_sha": existing["output"].strip()}
            checked = await self.call_unlocked_diff(a)
            if a.get("expected_hash") and checked["hash"] != a["expected_hash"]:
                raise ValueError("diff changed after review")
            await self._clean(wt)
            await self._git(
                wt,
                "-c",
                "user.name=Speculative Worker",
                "-c",
                "user.email=spec@localhost",
                "commit",
                "--allow-empty",
                "-m",
                "Speculative candidate " + a["cand_id"],
            )
            sha = await self._git(wt, "rev-parse", "HEAD")
            await self._git(repo, "update-ref", ref, sha)
            await self._remove(repo, wt)
            return {"branch_ref": ref, "head_sha": sha}
        if op == "drop":
            if wt and wt.exists():
                await self._remove(repo, wt)
            await self._git(repo, "update-ref", "-d", "refs/spec/" + self._id(a["cand_id"]))
            await self._git(repo, "worktree", "prune")
            pid = self._id(a["project_id"])
            self._drops_since_gc[pid] += 1
            if self._drops_since_gc[pid] >= 20:
                self._drops_since_gc[pid] = 0
                await self._git(repo, "gc", "--auto")
            self._wt_locks.pop(self._id(a["cand_id"]), None)
            self._invalidate_disk_cache()
            return {"dropped": True}
        if op == "export":
            ref = "refs/spec/" + self._id(a["cand_id"])
            patch = await self._git(repo, "diff", "--binary", a["base_sha"], ref)
            sha = await self._git(repo, "rev-parse", ref)
            return {"patch": patch, "head_sha": sha, "base_sha": a["base_sha"]}
        if op == "merge":
            branch = str(a["target_branch"])
            await self._git(repo, "check-ref-format", "--branch", branch)
            target = "refs/heads/" + branch
            current = await self._git(repo, "rev-parse", target)
            if current != a["approved_target_sha"]:
                parents = await self._git(repo, "show", "-s", "--format=%P", current)
                if parents.split() == [a["approved_target_sha"], a["approved_head_sha"]]:
                    return {"merged_sha": current, "target_branch": branch, "pushed": False}
                raise ValueError("target changed; export for review again")
            ref = "refs/spec/" + self._id(a["cand_id"])
            candidate = await self._git(repo, "rev-parse", ref)
            if candidate != a["approved_head_sha"]:
                raise ValueError("candidate changed since approval")
            result = await self._git(repo, "merge-tree", "--write-tree", current, candidate)
            tree = result.splitlines()[0]
            commit = await self._git(
                repo,
                "-c",
                "user.name=Speculative Human Reviewer",
                "-c",
                "user.email=reviewer@localhost",
                "commit-tree",
                tree,
                "-p",
                current,
                "-p",
                candidate,
                "-m",
                "Human approved speculative candidate " + a["cand_id"],
            )
            await self._git(repo, "update-ref", target, commit, current)
            return {"merged_sha": commit, "target_branch": branch, "pushed": False}
        raise ValueError("unsupported sandbox operation")

    async def call_unlocked_diff(self, a):
        return await self._operation("diff", a)

    async def _remove(self, repo: Path, wt: Path):
        await self._git(repo, "worktree", "remove", "--force", str(wt))
        self._invalidate_disk_cache()

    async def shutdown(self):
        for proc in list(self._processes):
            await self._kill_process_tree(proc)
