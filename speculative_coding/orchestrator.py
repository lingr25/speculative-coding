"""Asynchronous autonomous speculative coding control plane. No local build subprocesses."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import mimetypes
import os
import time
import uuid
from typing import Any
from urllib.parse import urlparse

from aiohttp import web

from .agent_loop import AgentLoop
from .config import SpecConfig
from .llm.router import ProviderRouter
from .llm.types import ContextOverflowError
from .protocols import AuthProvider, SingleOwnerAuthProvider
from .sandbox import RemoteWorkerSandbox, SpecSandboxBackend
from .store import JSON_FIELDS, PROJECT_DEFAULTS, SpecStore
from .tool_result import ToolResult, bound_tool_output

log = logging.getLogger(__name__)
ACTIVE = {"coding", "building", "reviewing", "inspecting", "revising", "dehydrating"}
FINAL = {"staged", "kept", "merged", "expired", "dropped"}


def _tool_def(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


def _spec_agent_tools(allow_write: bool) -> list[dict[str, Any]]:
    tools = [
        _tool_def(
            "spec_grep",
            "Fast substring or regex search across repository files. Prefer this over reading many files blindly; batch multiple searches/reads in one turn.",
            {
                "pattern": {"type": "string", "description": "Substring or regular expression to search for."},
                "path": {"type": "string", "description": "Optional relative directory or file path filter."},
                "regex": {"type": "boolean", "description": "Set true for regular expression matching."},
                "max_results": {"type": "integer", "description": "Maximum matches to return (1-200, default 80)."},
            },
            ["pattern"],
        ),
        _tool_def(
            "spec_read",
            "Read a repository file. Pass start_line/end_line for targeted slices of large files, and batch independent reads in parallel.",
            {
                "path": {"type": "string", "description": "Relative path to read."},
                "start_line": {"type": "integer", "description": "Optional 1-based start line."},
                "end_line": {"type": "integer", "description": "Optional 1-based inclusive end line."},
            },
            ["path"],
        ),
    ]
    if allow_write:
        tools.extend(
            [
                _tool_def(
                    "spec_edit",
                    "Apply a targeted surgical replacement in an existing file without rewriting the whole file. Always read target lines first.",
                    {
                        "path": {"type": "string", "description": "Relative file path to modify."},
                        "target": {
                            "type": "string",
                            "description": "Exact existing snippet to replace (include surrounding lines for uniqueness).",
                        },
                        "replacement": {"type": "string", "description": "Replacement snippet."},
                        "replace_all": {"type": "boolean", "description": "Replace every occurrence if true (default false)."},
                    },
                    ["path", "target", "replacement"],
                ),
                _tool_def(
                    "spec_write",
                    "Create a new file or overwrite an entire file when creating or replacing most of its contents.",
                    {
                        "path": {"type": "string", "description": "Relative file path to write."},
                        "content": {"type": "string", "description": "Complete UTF-8 file contents."},
                    },
                    ["path", "content"],
                ),
                _tool_def(
                    "spec_command",
                    "Run isolated test/lint/build shell commands inside the candidate sandbox (incremental caches are preserved across steps).",
                    {
                        "command": {"type": "string", "description": "Shell command to run in /workspace."},
                        "timeout": {"type": "integer", "description": "Timeout in seconds (1-1800, default 300)."},
                    },
                    ["command"],
                ),
            ]
        )
    tools.append(
        _tool_def(
            "submit_result",
            "Finish your task and submit the required structured JSON object result.",
            {"result": {"description": "Final structured JSON object requested by the system prompt."}},
            [],
        )
    )
    return tools


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        parts = text.split("\n", 1)
        text = parts[1].rsplit("```", 1)[0].strip() if len(parts) > 1 else text.strip("`").strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("model must return a JSON object")
    if parsed.get("action") == "finish" and isinstance(parsed.get("result"), dict):
        return parsed["result"]
    return parsed


def _parse_submit_result(tool_args: dict[str, Any]) -> dict[str, Any] | None:
    res_payload = tool_args.get("result")
    if isinstance(res_payload, dict):
        return res_payload
    if isinstance(res_payload, str):
        try:
            return _parse_json_object(res_payload)
        except ValueError:
            return None
    if tool_args and "result" not in tool_args:
        return tool_args
    return None


def _validate_json_output(raw_text: str) -> tuple[bool, Any, str]:
    try:
        return True, _parse_json_object(raw_text), ""
    except ValueError:
        return (
            False,
            None,
            "Invalid output: call `submit_result` with the required `result` JSON object, or output a valid JSON object directly.",
        )


def _compact_context_view(context: dict[str, Any]) -> dict[str, Any]:
    view = dict(context)
    tree = view.get("tree")
    if isinstance(tree, str):
        lines = tree.splitlines()
        if len(lines) > 500:
            view["tree"] = (
                "\n".join(lines[:500]) + f"\n... ({len(lines) - 500} more files; use spec_grep to locate targets)"
            )
    return view


class SpeculativeOrchestrator:
    def __init__(
        self,
        config: SpecConfig,
        backend: SpecSandboxBackend | None = None,
        *,
        store: SpecStore | None = None,
        auth: AuthProvider | None = None,
        router: ProviderRouter | None = None,
    ):
        self.config = config
        self.store = store or SpecStore(config.db_path)
        self.backend = backend or RemoteWorkerSandbox(config.db_path)
        self.auth = auth or SingleOwnerAuthProvider(owner_id=config.default_owner_id)
        self.router = router or ProviderRouter(config, pool="speculative")
        self.repo_registry = None
        self.conversation_agent = None
        self.automation = None
        self.tasks: dict[str, asyncio.Task] = {}
        self.stopping = False
        self.supervisor: asyncio.Task | None = None
        self.action_locks: dict[str, asyncio.Lock] = {}
        self.llm_slots = asyncio.Semaphore(8)
        self.role_routers: dict[tuple[str, tuple[str, ...]], ProviderRouter] = {}
        self.project_llm_slots: dict[tuple[str, int], asyncio.Semaphore] = {}

    def _event(self, p, kind, summary, *, run_id="", cand_id="", head_sha="", payload=None):
        try:
            return self.store.record_event(
                p["owner_id"],
                p["project_id"],
                kind,
                summary,
                run_id=run_id,
                cand_id=cand_id,
                head_sha=head_sha,
                payload=payload,
            )
        except Exception:
            log.exception("spec evidence event failed: %s", kind)
            return ""

    async def prepare_project(self, p: dict):
        token = ""
        if self.repo_registry is not None and p.get("repo_id"):
            repos = self.store.rows("repos", "owner_id=? AND repo_id=?", (p["owner_id"], str(p["repo_id"])))
            if repos:
                token = await self.repo_registry.clone_token(repos[0])
        args = dict(project_id=p["project_id"], repo_source=p["repo_source"], target_branch=p["target_branch"])
        if token:
            args["clone_token"] = token
        return await self.backend.call("prepare", **args)

    async def start(self):
        if self.automation is not None:
            await self.automation.start()
        if not self.supervisor or self.supervisor.done():
            self.stopping = False
            self.supervisor = asyncio.create_task(self.supervise(), name="spec-supervisor")

    async def stop(self):
        self.stopping = True
        if self.automation is not None:
            await self.automation.stop()
        tasks = [*self.tasks.values()]
        if self.supervisor:
            tasks.append(self.supervisor)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _spawn(self, project: dict) -> bool:
        pid = project["project_id"]
        if pid in self.tasks and not self.tasks[pid].done():
            return False
        task = asyncio.create_task(self.run_project(project), name="spec-" + pid)
        self.tasks[pid] = task
        return True

    async def _resume_project(self, p: dict):
        current = self.tasks.get(p["project_id"])
        if current and not current.done():
            await asyncio.shield(current)
        if not self.stopping:
            self._spawn(self.store.get("projects", p["project_id"]))

    async def supervise(self):
        while not self.stopping:
            try:
                await self.expire()
                connected = not isinstance(self.backend, RemoteWorkerSandbox) or self.backend.status()["connected"]
                if connected:
                    now = time.time()
                    for project in reversed(self.store.rows("projects")):
                        active = self.store.rows(
                            "candidates",
                            "project_id=? AND status IN ('coding','building','reviewing','inspecting','revising','dehydrating')",
                            (project["project_id"],),
                        )
                        unfinished = self.store.rows(
                            "runs", "project_id=? AND status IN ('recon','executing')", (project["project_id"],)
                        )
                        if (project["wake_at"] or 0) <= now and (active or unfinished or project["enabled"]):
                            self._spawn(project)
            except Exception:
                log.exception("spec supervisor tick failed")
            await asyncio.sleep(10)

    def _role(self, p: dict, role: str) -> ProviderRouter:
        ids = p.get(role + "_provider_ids", [])
        key = (role, tuple(ids))
        if key not in self.role_routers:
            router = ProviderRouter(self.config, pool="speculative")
            router.allowed_provider_ids = set(ids) if ids else None
            # Cooling is independent from main and shared across speculative roles.
            for attr in ("_fail_counts", "_cooldown_strikes", "_cooldown_until"):
                setattr(router, attr, getattr(self.router, attr))
            self.role_routers[key] = router
        return self.role_routers[key]

    async def _complete_slotted(self, p, role, messages, *, tools=None, run_id="", cand_id=""):
        router = self._role(p, role)
        endpoints = router.candidate_chain()
        if not endpoints:
            raise SpecSleep("speculative providers unavailable or cooling")
        project_slots = self.project_llm_slots.setdefault(
            (p["project_id"], p["llm_concurrency"]),
            asyncio.Semaphore(p["llm_concurrency"]),
        )
        async with self.llm_slots, project_slots:
            try:
                result = await router.complete_step(
                    messages,
                    tools=tools,
                    preferred_provider_id=endpoints[0].provider_id,
                    prefer_stream=True,
                )
            except ContextOverflowError:
                raise
            except Exception as exc:
                raise SpecSleep("speculative inference paused: " + str(exc)) from exc
        tokens = result.input_tokens + result.output_tokens
        self.store.charge(p["project_id"], run_id, cand_id, tokens)
        return result

    async def llm(self, p, role, messages, run_id="", cand_id=""):
        result = await self._complete_slotted(p, role, messages, tools=None, run_id=run_id, cand_id=cand_id)
        try:
            return _parse_json_object(result.content)
        except ValueError as exc:
            raise ValueError("model must return valid JSON") from exc

    async def _dispatch_sandbox_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        *,
        p: dict,
        base_sha: str,
        candidate: dict | None,
        allow_write: bool,
    ) -> ToolResult:
        if name == "submit_result":
            submitted = _parse_submit_result(tool_args)
            if submitted is not None:
                return ToolResult(
                    text=json.dumps({"ok": True}, ensure_ascii=False),
                    terminate_loop=True,
                    return_value=submitted,
                )
            return ToolResult(
                text=json.dumps(
                    {"error": "`submit_result` requires `result` to be a JSON object"},
                    ensure_ascii=False,
                )
            )
        base = {"project_id": p["project_id"], "base_sha": base_sha}
        if candidate:
            base["cand_id"] = candidate["cand_id"]
        try:
            if name == "spec_grep":
                outcome = await self.backend.call(
                    "grep",
                    **base,
                    pattern=str(tool_args.get("pattern") or ""),
                    path=str(tool_args.get("path") or ""),
                    regex=bool(tool_args.get("regex", False)),
                    max_results=int(tool_args.get("max_results", 80)),
                )
            elif name == "spec_read":
                read_kw: dict[str, Any] = {**base, "path": str(tool_args["path"])}
                if tool_args.get("start_line") is not None:
                    read_kw["start_line"] = int(tool_args["start_line"])
                if tool_args.get("end_line") is not None:
                    read_kw["end_line"] = int(tool_args["end_line"])
                outcome = await self.backend.call("read", **read_kw)
            elif name == "spec_edit":
                if not allow_write:
                    outcome = {"error": "operation not permitted for this role"}
                else:
                    outcome = await self.backend.call(
                        "edit",
                        **base,
                        path=str(tool_args["path"]),
                        target=str(tool_args["target"]),
                        replacement=str(tool_args["replacement"]),
                        replace_all=bool(tool_args.get("replace_all", False)),
                    )
            elif name == "spec_write":
                if not allow_write:
                    outcome = {"error": "operation not permitted for this role"}
                else:
                    outcome = await self.backend.call(
                        "write",
                        **base,
                        path=str(tool_args["path"]),
                        content=str(tool_args["content"]),
                    )
            elif name == "spec_command":
                if not allow_write:
                    outcome = {"error": "operation not permitted for this role"}
                else:
                    outcome = await self.backend.call(
                        "command",
                        **base,
                        command=str(tool_args["command"]),
                        timeout=int(tool_args.get("timeout", 300)),
                    )
            else:
                outcome = {"error": f"unknown tool: {name}"}
        except Exception as exc:
            outcome = {"error": str(exc)}
        raw_text = json.dumps(outcome, ensure_ascii=False)
        bounded_text, _ = bound_tool_output(raw_text, spill_path=None)
        return ToolResult(text=bounded_text)

    async def agent(self, p, role, instruction, context, *, run_id, candidate=None, allow_write=False):
        """Parallel tool-calling speculative agent with surgical edits, grep and auto-compaction.

        Model has explicit grep/read/edit/write/command capabilities in the isolated candidate only.
        No control-plane tools, credentials, refs, TTL actions or merge capability exist here.
        """
        tools = _spec_agent_tools(allow_write)
        router = self._role(p, role)
        cand_id = candidate["cand_id"] if candidate else ""
        base_sha = str(context.get("base_sha") or "")
        role_rules = (
            "\n\nTooling & Delivery Rules:\n"
            "- Batch independent `spec_grep` and `spec_read` calls in parallel within a single turn.\n"
            "- Use `spec_grep` and `spec_read(start_line, end_line)` to inspect code efficiently instead of reading huge files blindly.\n"
            + (
                "- Prefer `spec_edit` for targeted code changes instead of rewriting entire files; use `spec_write` only for new or small files.\n"
                "- Never modify deploy, .git, .env, AGENTS.md or existing tests (adding new test files is allowed). No merge/push capability.\n"
                if allow_write
                else "- Read-only role: only `spec_grep`, `spec_read`, and `submit_result` are permitted.\n"
            )
            + "- When finished, call `submit_result(result={...})` with the exact JSON object structure requested above."
        )

        async def _complete_step(*, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **_kw: Any):
            if self.stopping:
                raise asyncio.CancelledError()
            return await self._complete_slotted(
                p,
                role,
                messages,
                tools=tools,
                run_id=run_id,
                cand_id=cand_id,
            )

        async def _execute_tool(name: str, tool_args: dict[str, Any]) -> ToolResult:
            return await self._dispatch_sandbox_tool(
                name,
                tool_args,
                p=p,
                base_sha=base_sha,
                candidate=candidate,
                allow_write=allow_write,
            )

        loop = AgentLoop(
            _complete_step,
            spill_root=getattr(router, "spill_root", None),
            resolve_compact_limit=router.compact_limit_for,
        )
        turn_res = await loop.run(
            system_prompt=instruction + role_rules,
            history=[],
            user_content=json.dumps(_compact_context_view(context), ensure_ascii=False),
            tools=tools,
            tool_executor=_execute_tool,
            prefer_stream=True,
            output_validator=_validate_json_output,
            max_output_retries=2,
        )
        if turn_res.return_value is not None:
            return turn_res.return_value
        try:
            return _parse_json_object(turn_res.text)
        except ValueError as exc:
            raise ValueError("model must return valid JSON") from exc

    async def run_project(self, p: dict):
        pid = p["project_id"]
        self.store.update("projects", pid, status="running")
        try:
            active = self.store.rows(
                "candidates",
                "project_id=? AND status IN ('coding','building','reviewing','inspecting','revising','dehydrating')",
                (pid,),
            )
            if active:
                await self.run_candidates(p, active)
                for run_id in {c["run_id"] for c in active}:
                    pending = self.store.rows(
                        "candidates",
                        "run_id=? AND status IN ('coding','building','reviewing','inspecting','revising','dehydrating')",
                        (run_id,),
                    )
                    if not pending:
                        self.store.update("runs", run_id, status="completed", completed_at=time.time())
                return
            unfinished = self.store.rows("runs", "project_id=? AND status IN ('recon','executing')", (pid,))
            completed_prior = False
            for prior in unfinished:
                prior_candidates = self.store.rows("candidates", "run_id=?", (prior["run_id"],))
                self.store.update(
                    "runs",
                    prior["run_id"],
                    status="completed" if prior_candidates else "empty",
                    completed_at=time.time(),
                )
                completed_prior = completed_prior or bool(prior_candidates)
            if completed_prior:
                return
            run_id = "sr_" + uuid.uuid4().hex
            self.store.insert("runs", dict(run_id=run_id, project_id=pid, status="recon", created_at=time.time()))
            prepared = await self.prepare_project(p)
            base = prepared["base_sha"]
            self.store.update("runs", run_id, base_sha=base)
            self.store.update("projects", pid, last_head_sha=base)
            grave = self.store.rows("graveyard", "project_id=?", (pid,))
            memories = self.store.memories(p["owner_id"], pid)
            self._event(p, "run_started", "Speculative batch started", run_id=run_id, head_sha=base)
            result = await self.agent(
                p,
                "recon",
                f'Deeply inspect the repository, recent work and graveyard. Never repeat rejected ideas. Find only high-value directions. Submit result {{"summary":"...","routes":[{{"title":"...","plan":"...","kind":"optimization|feature_gameplay|architecture","competition_group":""}}]}} with zero to {p["routes_per_batch"]} routes. Empty routes means quiet sleep.',
                {**prepared, "graveyard": grave, "memory": memories},
                run_id=run_id,
            )
            routes = result.get("routes", [])
            if not isinstance(routes, list):
                raise ValueError("invalid routes")
            routes = routes[: p["routes_per_batch"]]
            self.store.update(
                "runs",
                run_id,
                recon_summary=str(result.get("summary", "")),
                routes_json=routes,
                status="executing" if routes else "empty",
                completed_at=None if routes else time.time(),
            )
            candidates = []
            for index, route in enumerate(routes):
                if not route.get("title") or not route.get("plan"):
                    continue
                if any(route["title"] == g["scope_hint"] or route["plan"] == g["idea"] for g in grave):
                    continue
                now = time.time()
                cid = "sc_" + uuid.uuid4().hex
                cand = dict(
                    cand_id=cid,
                    run_id=run_id,
                    project_id=pid,
                    route_index=index,
                    title=route["title"],
                    route_plan=route["plan"],
                    route_kind=route.get("kind", "optimization"),
                    competition_group=route.get("competition_group", ""),
                    branch_ref="refs/spec/" + cid,
                    status="coding",
                    base_sha=base,
                    created_at=now,
                    updated_at=now,
                )
                self.store.insert("candidates", cand)
                candidates.append(self.store.get("candidates", cid))
            await self.run_candidates(p, candidates)
            self.store.update("runs", run_id, status="completed" if candidates else "empty", completed_at=time.time())
        except SpecSleep as exc:
            self.store.update("projects", pid, status="sleeping", wake_at=time.time() + 60)
            log.info("spec project %s sleeping: %s", pid, exc)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.store.update("projects", pid, status="error", wake_at=time.time() + 300)
            log.exception("spec project %s failed", pid)
        finally:
            current = self.store.get("projects", pid)
            if current and current["status"] == "running":
                self.store.update("projects", pid, status="idle", wake_at=time.time() + 300)

    async def run_candidates(self, p: dict, candidates: list[dict]):
        slots = asyncio.Semaphore(min(4, p["wt_concurrency"]))

        async def run(candidate):
            async with slots:
                await self.drive(p, candidate)

        results = await asyncio.gather(*(run(c) for c in candidates), return_exceptions=True)
        for result in results:
            if isinstance(result, SpecSleep):
                raise result
            if isinstance(result, BaseException):
                log.error("spec candidate execution error: %s", result)

    async def drop(self, c: dict, reason: str, source: str = "gate", status: str = "dropped"):
        await self.backend.call("drop", project_id=c["project_id"], cand_id=c["cand_id"])
        self.store.grave(c, reason, source)
        self.store.update(
            "candidates",
            c["cand_id"],
            status=status,
            drop_reason=reason,
            active_worktree_path="",
            preview_meta_json={},
            diff_text="",
            human_report_md="",
            expires_at=None,
        )
        p = self.store.get("projects", c["project_id"])
        self._event(
            p,
            "drop",
            f"{status}: {str(reason)[:300]}",
            run_id=c["run_id"],
            cand_id=c["cand_id"],
            head_sha=c.get("head_sha") or c["base_sha"],
            payload={"source": source, "reason": str(reason)[:4000]},
        )
        if source == "human_drop":
            self.store.remember(
                p["owner_id"],
                p["project_id"],
                f"decision:{c['cand_id']}:{c['generation']}",
                f"Dropped: {str(reason)[:1000]}",
                kind="decision",
                source="human",
            )

    async def drive(self, p: dict, c: dict):
        cid = c["cand_id"]
        run_id = c["run_id"]
        args = {"project_id": p["project_id"], "cand_id": cid, "base_sha": c["base_sha"]}
        try:
            if c["status"] == "dehydrating":
                artifact = await self.backend.call(
                    "dehydrate", **args, expected_hash=c["last_diff_hash"], task_id=f'{cid}_dehydrate_{c["generation"]}'
                )
                self.store.update(
                    "candidates",
                    cid,
                    status="staged",
                    branch_ref=artifact["branch_ref"],
                    head_sha=artifact["head_sha"],
                    active_worktree_path="",
                    expires_at=time.time() + p["ttl_hours"] * 3600,
                )
                return
            if isinstance(self.backend, RemoteWorkerSandbox):
                await self.backend.reconcile_candidate(cid)
            prepared = await self.backend.call("describe", project_id=p["project_id"], base_sha=c["base_sha"])
            hydrated = await self.backend.call("hydrate", **args, ref=c["branch_ref"] if c["head_sha"] else c["base_sha"])
            self.store.update("candidates", cid, active_worktree_path=hydrated["path"])
            while True:
                c = self.store.get("candidates", cid)
                if c["status"] not in ACTIVE:
                    return
                if c["review_rounds"] >= p["max_review_rounds"]:
                    await self.drop(c, "maximum review rounds reached", "code_reviewer")
                    return
                round_no = c["review_rounds"] + 1
                self.store.update("candidates", cid, status="coding" if round_no == 1 else "revising")
                await self.agent(
                    p,
                    "coder",
                    'Implement the route completely. Inspect all relevant files. Fix concrete feedback. Submit result {"summary":"..."} when ready for strict gates. No deploying and no test deletion.',
                    {**prepared, "base_sha": c["base_sha"], "route": c["route_plan"], "feedback": c["feedback"], "steer": c["steer_notes_json"]},
                    run_id=run_id,
                    candidate=c,
                    allow_write=True,
                )
                diff = await self.backend.call("diff", **args)
                stagnant = c["stagnant_rounds"] + 1 if diff["hash"] == c["last_diff_hash"] else 0
                self.store.update(
                    "candidates",
                    cid,
                    review_rounds=round_no,
                    last_diff_hash=diff["hash"],
                    stagnant_rounds=stagnant,
                    status="building",
                )
                if not diff["diff"] or stagnant >= 2:
                    await self.drop(c, "empty or stagnant diff for two rounds", "code_reviewer")
                    return
                commands = list(p["gate_commands_json"])
                if not commands:
                    await self.drop(c, "no deterministic gates configured", "gate")
                    return
                gate = await self.backend.call(
                    "gate",
                    **args,
                    commands=commands,
                    preview_command=p["preview_command"],
                    build_concurrency=p["build_concurrency"],
                )
                gate_evidence = {
                    "passed": bool(gate.get("passed")),
                    "commands": commands,
                    "gates": [
                        {
                            "command": item.get("command", ""),
                            "code": item.get("code"),
                            "output": str(item.get("output") or "")[:4000],
                        }
                        for item in gate.get("gates", [])
                    ],
                    "preview": (
                        {k: v for k, v in (gate.get("preview") or {}).items() if k != "artifacts"}
                        if isinstance(gate.get("preview"), dict)
                        else gate.get("preview")
                    ),
                }
                self._event(
                    p,
                    "gate",
                    "Deterministic gates passed" if gate.get("passed") else "Deterministic gates failed",
                    run_id=run_id,
                    cand_id=cid,
                    head_sha=c.get("head_sha") or c["base_sha"],
                    payload=gate_evidence,
                )
                if not gate.get("passed"):
                    self.store.update(
                        "candidates",
                        cid,
                        status="revising",
                        feedback="Deterministic gate failed: " + json.dumps(gate, ensure_ascii=False),
                    )
                    continue
                diff = await self.backend.call("diff", **args)
                self.store.update("candidates", cid, status="reviewing", last_diff_hash=diff["hash"])
                context = {"base_sha": c["base_sha"], "route": c["route_plan"], "diff": diff["diff"], "gate_evidence": gate}
                review = await self.agent(
                    p,
                    "reviewer",
                    'Strict independent code reviewer. Only excellent, correct, maintainable complete code can be NOMINATE. Submit result {"decision":"DROP|REVISE|NOMINATE","feedback":"specific evidence and changes"}. Do not reward scaffolding or tests weakened to pass.',
                    context,
                    run_id=run_id,
                    candidate=c,
                )
                verdicts = [review]
                if c["route_kind"] == "feature_gameplay":
                    self.store.update("candidates", cid, status="inspecting")
                    if not p["preview_command"] or not gate.get("preview"):
                        verdicts.append(
                            {
                                "decision": "REVISE",
                                "feedback": "feature requires configured executable interaction/acceptance preview command and captured evidence",
                            }
                        )
                    else:
                        inspect = await self.agent(
                            p,
                            "inspector",
                            'Judge gameplay/completeness against actual runtime interaction evidence and code. Reject hollow buttons, incomplete workflows, mediocre design. Submit result {"decision":"DROP|REVISE|NOMINATE","feedback":"specific evidence"}. Excellent code AND excellent playability required.',
                            context,
                            run_id=run_id,
                            candidate=c,
                        )
                        verdicts.append(inspect)
                self._event(
                    p,
                    "review",
                    " / ".join(str(v.get("decision") or "UNKNOWN") for v in verdicts),
                    run_id=run_id,
                    cand_id=cid,
                    head_sha=c.get("head_sha") or c["base_sha"],
                    payload={
                        "round": round_no,
                        "verdicts": [
                            {"decision": v.get("decision"), "feedback": str(v.get("feedback") or "")[:12000]}
                            for v in verdicts
                        ],
                    },
                )
                if any(v.get("decision") == "DROP" for v in verdicts):
                    await self.drop(c, json.dumps(verdicts, ensure_ascii=False), "code_reviewer")
                    return
                if not all(v.get("decision") == "NOMINATE" for v in verdicts):
                    self.store.update("candidates", cid, status="revising", feedback=json.dumps(verdicts, ensure_ascii=False))
                    continue
                report = await self.agent(
                    p,
                    "reviewer",
                    'Write a concise human report (three-minute read, <= 400 words/chars, no raw code diffs) with three Markdown sections: "### What Changed" (summary and impact), "### How It Works" (key files/modules and mechanism), and "### Gates & Verification" (whether deterministic gates passed, commands/exit codes, measured results, and remaining risks). Submit result {"report":"markdown"}.',
                    context,
                    run_id=run_id,
                    candidate=c,
                )
                report_md = str(report.get("report", ""))
                if not report_md:
                    raise ValueError("missing nomination report")
                payload_bytes = len(
                    json.dumps({"report": report_md, "diff": diff["diff"], "preview": gate}, ensure_ascii=False).encode()
                )
                if payload_bytes >= 20 * 1024**2:
                    raise SpecSleep("preview/report package exceeds 20MiB")
                with self.store.conn() as db:
                    used = db.execute(
                        "SELECT COALESCE(SUM(length(CAST(human_report_md AS BLOB))+length(CAST(diff_text AS BLOB))+length(CAST(preview_meta_json AS BLOB))),0) FROM spec_candidates"
                    ).fetchone()[0]
                if used + payload_bytes >= 512 * 1024**2:
                    raise SpecSleep("control-plane artifact budget exhausted")
                self.store.update(
                    "candidates",
                    cid,
                    status="dehydrating",
                    human_report_md=report_md,
                    diff_text=diff["diff"],
                    preview_meta_json=gate,
                    last_diff_hash=diff["hash"],
                )
                artifact = await self.backend.call(
                    "dehydrate", **args, expected_hash=diff["hash"], task_id=f'{cid}_dehydrate_{c["generation"]}'
                )
                self.store.update(
                    "candidates",
                    cid,
                    status="staged",
                    branch_ref=artifact["branch_ref"],
                    head_sha=artifact["head_sha"],
                    active_worktree_path="",
                    human_report_md=report_md,
                    diff_text=diff["diff"],
                    preview_meta_json=gate,
                    expires_at=time.time() + p["ttl_hours"] * 3600,
                )
                self._event(
                    p,
                    "nominated",
                    "Candidate nominated and awaiting human review",
                    run_id=run_id,
                    cand_id=cid,
                    head_sha=artifact["head_sha"],
                    payload={"report": report_md[:12000], "diff_hash": diff["hash"], "gate": gate_evidence},
                )
                return
        except SpecSleep:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if isinstance(self.backend, RemoteWorkerSandbox) and not self.backend.status()["connected"]:
                raise SpecSleep("worker offline") from exc
            if any(
                word in str(exc).lower()
                for word in (
                    "backpressure",
                    "slots exhausted",
                    "restarted",
                    "timeout",
                    "disk",
                    "budget",
                    "dependency baseline",
                    "seed shared caches",
                )
            ):
                raise SpecSleep(str(exc)) from exc
            await self.drop(self.store.get("candidates", cid), str(exc), "gate")

    async def expire(self):
        with self.store.conn() as db:
            expired = [
                dict(row)
                for row in db.execute(
                    "SELECT cand_id FROM spec_candidates WHERE status='staged' AND expires_at IS NOT NULL AND expires_at<=? LIMIT 100",
                    (time.time(),),
                )
            ]
        for c in expired:
            async with self.action_locks.setdefault(c["cand_id"], asyncio.Lock()):
                fresh = self.store.get("candidates", c["cand_id"])
                if fresh["status"] == "staged" and fresh["expires_at"] and fresh["expires_at"] <= time.time():
                    await self.drop(fresh, "TTL expired", "ttl", status="expired")

    def validate_project(self, data: dict, *, partial: bool = False) -> dict:
        fields = {"name", "repo_source", *PROJECT_DEFAULTS.keys()} - {
            "status",
            "speculative_tokens",
            "wake_at",
            "last_head_sha",
            "hot_cache_last_used",
        }
        data = {k: v for k, v in data.items() if k in fields}
        for key in JSON_FIELDS & data.keys():
            if isinstance(data[key], str):
                data[key] = json.loads(data[key])
            if not isinstance(data[key], list):
                raise ValueError(key + " must be an array")
        if data.get("sandbox_mode", "remote_worker") != "remote_worker":
            raise ValueError("production control plane only permits remote_worker")
        if "repo_source" in data or not partial:
            url = urlparse(str(data.get("repo_source", "")))
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("public credential-free HTTPS repository required")
        if not partial and not str(data.get("name", "")).strip():
            raise ValueError("name required")
        for key, minimum, maximum in [
            ("routes_per_batch", 0, 12),
            ("llm_concurrency", 1, 8),
            ("wt_concurrency", 1, 4),
            ("build_concurrency", 1, 2),
            ("max_review_rounds", 1, 20),
            ("ttl_hours", 1, 8760),
        ]:
            if key in data:
                data[key] = int(data[key])
                if not minimum <= data[key] <= maximum:
                    raise ValueError("invalid " + key)
        for key in ("recon_provider_ids", "coder_provider_ids", "reviewer_provider_ids", "inspector_provider_ids"):
            for pid in data.get(key, []):
                ep = self.config.providers.get(pid)
                if not ep:
                    raise ValueError("unknown role provider: " + str(pid))
        for command in data.get("gate_commands_json", []):
            if not isinstance(command, str) or not command.strip():
                raise ValueError("gate command must be nonempty string")
        data["enabled"] = (
            int(bool(data.get("enabled", False))) if "enabled" in data else 0 if not partial else data.get("enabled", None)
        )
        return {k: v for k, v in data.items() if v is not None}

    @staticmethod
    def _owner_id(user: dict[str, Any]) -> str:
        return str(user.get("owner_id") or "")

    def _owned(self, request: web.Request, table: str, key: str):
        user = self.auth.require_admin(request)
        item = self.store.get(table, key)
        if not item:
            raise web.HTTPNotFound()
        owner = item if table == "projects" else self.store.get("projects", item["project_id"])
        if owner["owner_id"] != self._owner_id(user):
            raise web.HTTPForbidden(text="project owner required")
        return user, item

    async def status_api(self, request: web.Request):
        user = self.auth.require_admin(request)
        owner = self._owner_id(user)
        projects = self.store.rows("projects", "owner_id=?", (owner,))
        worker = (
            self.backend.status()
            if isinstance(self.backend, RemoteWorkerSandbox)
            else {"connected": True, "metrics": await self.backend.call("status")}
        )
        return web.json_response(
            {
                "ok": True,
                "worker": worker,
                "projects": projects,
                "tokens_used": sum(p["speculative_tokens"] for p in projects),
                "agent_session_key": f"spec:{owner}:main",
                "github_configured": bool(
                    self.repo_registry and self.repo_registry.client and self.repo_registry.client.configured
                ),
            }
        )

    async def projects_api(self, request: web.Request):
        user = self.auth.require_admin(request)
        owner = self._owner_id(user)
        if request.method == "GET":
            return web.json_response({"ok": True, "projects": self.store.rows("projects", "owner_id=?", (owner,))})
        try:
            data = self.validate_project(await request.json())
            project = self.store.new_project(owner, data)
            return web.json_response({"ok": True, "project": project})
        except (ValueError, TypeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def project_api(self, request: web.Request):
        _, p = self._owned(request, "projects", request.match_info["project_id"])
        try:
            data = self.validate_project(await request.json(), partial=True)
            if p["project_id"] in self.tasks and not self.tasks[p["project_id"]].done():
                raise ValueError("pause changes until current batch finishes")
            self.store.update("projects", p["project_id"], **data)
            return web.json_response({"ok": True, "project": self.store.get("projects", p["project_id"])})
        except (ValueError, TypeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

    async def run_api(self, request: web.Request):
        _, p = self._owned(request, "projects", request.match_info["project_id"])
        return web.json_response({"ok": True, "started": self._spawn(p)})

    async def candidates_api(self, request: web.Request):
        user = self.auth.require_admin(request)
        owner = self._owner_id(user)
        limit = max(1, min(100, int(request.query.get("limit", 50))))
        offset = max(0, int(request.query.get("offset", 0)))
        with self.store.conn() as db:
            candidates = [
                dict(row)
                for row in db.execute(
                    "SELECT c.cand_id,c.project_id,c.run_id,c.title,c.route_kind,c.status,c.review_rounds,c.tokens_used,c.expires_at,c.created_at,c.updated_at,c.branch_ref,c.base_sha,c.head_sha "
                    "FROM spec_candidates c JOIN spec_projects p ON p.project_id=c.project_id "
                    "WHERE p.owner_id=? AND (?='' OR c.project_id=?) AND c.status NOT IN ('dropped','expired') "
                    "ORDER BY c.created_at DESC LIMIT ? OFFSET ?",
                    (owner, request.query.get("project_id", ""), request.query.get("project_id", ""), limit, offset),
                )
            ]
        return web.json_response({"ok": True, "candidates": candidates, "limit": limit, "offset": offset})

    async def candidate_api(self, request: web.Request):
        _, c = self._owned(request, "candidates", request.match_info["cand_id"])
        return web.json_response({"ok": True, "candidate": c})

    async def perform_action(self, user: dict[str, Any], c: dict, operation: str, body: dict):
        actor = self._owner_id(user)
        action_id = str(body.get("action_id", ""))
        if not action_id or len(action_id) > 128:
            raise ValueError("action_id required")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        async with self.action_locks.setdefault(c["cand_id"], asyncio.Lock()):
            with self.store.conn() as db:
                row = db.execute("SELECT * FROM spec_actions WHERE action_id=?", (action_id,)).fetchone()
                if row:
                    if (
                        row["actor"] != actor
                        or row["cand_id"] != c["cand_id"]
                        or row["operation"] != operation
                        or row["payload_hash"] != digest
                    ):
                        raise ValueError("action ID mismatch")
                    if row["result_json"]:
                        return json.loads(row["result_json"])
                else:
                    db.execute(
                        "INSERT INTO spec_actions VALUES (?,?,?,?,?,NULL,?)",
                        (action_id, c["cand_id"], actor, operation, digest, time.time()),
                    )
            c = self.store.get("candidates", c["cand_id"])
            p = self.store.get("projects", c["project_id"])
            args = {"project_id": c["project_id"], "cand_id": c["cand_id"], "base_sha": c["base_sha"], "task_id": "human_" + action_id}
            recovered_steer = operation == "steer" and any(
                note.get("action_id") == action_id for note in c["steer_notes_json"]
            )
            recovered_merge = operation == "merge" and c["status"] == "merged" and row is not None
            if (
                c["status"] not in {"staged", "kept"}
                and not recovered_steer
                and not recovered_merge
                and not (operation == "drop" and c["status"] in {"dropped", "expired"})
            ):
                raise ValueError("candidate is not available for this action")
            result: dict[str, Any] = {"ok": True}
            notes = ""
            if operation == "keep":
                self.store.update("candidates", c["cand_id"], status="kept", expires_at=None)
            elif operation == "drop":
                if c["status"] not in {"dropped", "expired"}:
                    await self.drop(c, str(body.get("reason", "human drop")), "human_drop")
            elif operation == "steer" and not recovered_steer:
                notes = str(body.get("notes", "")).strip()
                if not notes:
                    raise ValueError("steer notes required")
                self.store.update(
                    "candidates",
                    c["cand_id"],
                    status="revising",
                    expires_at=None,
                    review_rounds=0,
                    stagnant_rounds=0,
                    generation=c["generation"] + 1,
                    feedback=notes,
                    steer_notes_json=c["steer_notes_json"]
                    + [{"notes": notes, "actor": actor, "at": time.time(), "action_id": action_id}],
                )
                asyncio.create_task(self._resume_project(p))
            elif operation == "export":
                result.update(await self.backend.call("export", **args))
            elif operation == "merge":
                if (
                    body.get("confirm") is not True
                    or body.get("approved_head_sha") != c["head_sha"]
                    or not body.get("approved_target_sha")
                ):
                    raise ValueError("explicit human confirmation with reviewed candidate/target SHA required")
                if body.get("target_branch", p["target_branch"]) != p["target_branch"]:
                    raise ValueError("target branch differs from project")
                result.update(
                    await self.backend.call(
                        "merge",
                        **args,
                        target_branch=p["target_branch"],
                        approved_head_sha=c["head_sha"],
                        approved_target_sha=body["approved_target_sha"],
                    )
                )
                self.store.update("candidates", c["cand_id"], status="merged", expires_at=None)
            if operation != "drop":
                fresh = self.store.get("candidates", c["cand_id"])
                self._event(
                    p,
                    "human_" + operation,
                    f"Human action: {operation}",
                    run_id=c["run_id"],
                    cand_id=c["cand_id"],
                    head_sha=fresh.get("head_sha") or c["base_sha"],
                    payload={"notes": str(body.get("notes") or "")[:4000], "action_id": action_id},
                )
            if operation == "keep":
                self.store.remember(
                    p["owner_id"],
                    p["project_id"],
                    f"decision:{c['cand_id']}:{c['generation']}",
                    "Kept candidate",
                    kind="decision",
                    source="human",
                )
            elif operation == "steer" and not recovered_steer:
                self.store.remember(
                    p["owner_id"],
                    p["project_id"],
                    f"decision:{c['cand_id']}:{c['generation']}",
                    f"Steered: {notes[:1000]}",
                    kind="decision",
                    source="human",
                )
            with self.store.conn() as db:
                db.execute(
                    "UPDATE spec_actions SET result_json=? WHERE action_id=?",
                    (json.dumps(result, ensure_ascii=False), action_id),
                )
            return result

    async def action_api(self, request: web.Request):
        user, c = self._owned(request, "candidates", request.match_info["cand_id"])
        operation = request.match_info["action"]
        if operation not in {"keep", "steer", "drop", "export", "merge"}:
            raise web.HTTPNotFound()
        try:
            result = await self.perform_action(user, c, operation, await request.json())
            return web.json_response(result)
        except (ValueError, RuntimeError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=409)

    async def preview_api(self, request: web.Request):
        _, c = self._owned(request, "candidates", request.match_info["cand_id"])
        path = request.match_info["path"]
        artifact = next((a for a in c["preview_meta_json"].get("artifacts", []) if a["path"] == path), None)
        if not artifact:
            raise web.HTTPNotFound()
        headers = {
            "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'",
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        }
        return web.Response(
            body=base64.b64decode(artifact["base64"], validate=True),
            content_type=mimetypes.guess_type(path)[0] or "application/octet-stream",
            headers=headers,
        )

    async def worker_api(self, request: web.Request):
        expected = (self.config.worker_token or os.environ.get("SPEC_WORKER_TOKEN", "")).strip()
        auth = request.headers.get("Authorization", "")
        if len(expected) < 32 or not hmac.compare_digest(auth, "Bearer " + expected):
            raise web.HTTPUnauthorized()
        if not isinstance(self.backend, RemoteWorkerSandbox):
            raise web.HTTPServiceUnavailable()
        return await self.backend.connect(request)

    def register(self, app: web.Application, *, prefix: str = "/api/spec"):
        base = prefix.rstrip("/")
        app.router.add_get(base + "-worker/ws", self.worker_api)
        app.router.add_get(base + "/status", self.status_api)
        app.router.add_get(base + "/projects", self.projects_api)
        app.router.add_post(base + "/projects", self.projects_api)
        app.router.add_patch(base + "/projects/{project_id}", self.project_api)
        app.router.add_post(base + "/projects/{project_id}/run", self.run_api)
        app.router.add_get(base + "/candidates", self.candidates_api)
        app.router.add_get(base + "/candidates/{cand_id}", self.candidate_api)
        app.router.add_get(base + "/candidates/{cand_id}/preview/{path:.*}", self.preview_api)
        app.router.add_post(base + "/candidates/{cand_id}/{action}", self.action_api)
        app.on_startup.append(lambda _: self.start())
        app.on_cleanup.append(lambda _: self.stop())


class SpecSleep(RuntimeError):
    pass
