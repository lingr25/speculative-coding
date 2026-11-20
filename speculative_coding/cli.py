"""Unified CLI for Speculative Coding (serve, worker, run, status, candidates, export)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from aiohttp import web

from .config import SpecConfig
from .orchestrator import SpeculativeOrchestrator
from .sandbox import LocalDirSandbox
from .server import create_control_plane
from .store import SpecStore
from .worker import main as worker_main


def _load_config(args: argparse.Namespace) -> SpecConfig:
    if getattr(args, "config", ""):
        return SpecConfig.load_from_file(args.config)
    data_dir = Path(args.data_dir) if getattr(args, "data_dir", "") else None
    return SpecConfig.from_env(data_dir=data_dir)


def _cmd_serve(args: argparse.Namespace) -> None:
    config = _load_config(args)
    plane = create_control_plane(config)
    web.run_app(plane.app, host=args.host, port=args.port)


def _cmd_status(args: argparse.Namespace) -> None:
    config = _load_config(args)
    store = SpecStore(config.db_path)
    projects = store.rows("projects")
    candidates = store.rows("candidates")
    summary = {
        "db_path": str(config.db_path),
        "projects": len(projects),
        "tokens_used": sum(p["speculative_tokens"] for p in projects),
        "candidates": {
            "total": len(candidates),
            "staged": sum(1 for c in candidates if c["status"] == "staged"),
            "kept": sum(1 for c in candidates if c["status"] == "kept"),
            "merged": sum(1 for c in candidates if c["status"] == "merged"),
            "active": sum(
                1
                for c in candidates
                if c["status"] in {"coding", "building", "reviewing", "inspecting", "revising", "dehydrating"}
            ),
        },
        "project_list": [
            {
                "project_id": p["project_id"],
                "name": p["name"],
                "status": p["status"],
                "enabled": bool(p["enabled"]),
                "tokens": p["speculative_tokens"],
            }
            for p in projects
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cmd_candidates(args: argparse.Namespace) -> None:
    config = _load_config(args)
    store = SpecStore(config.db_path)
    where = "project_id=?" if args.project_id else ""
    qargs = (args.project_id,) if args.project_id else ()
    rows = store.rows("candidates", where, qargs)
    if not args.all:
        rows = [r for r in rows if r["status"] not in {"dropped", "expired"}]
    brief = [
        {
            "cand_id": c["cand_id"],
            "project_id": c["project_id"],
            "title": c["title"],
            "status": c["status"],
            "route_kind": c["route_kind"],
            "head_sha": c["head_sha"],
            "review_rounds": c["review_rounds"],
            "tokens_used": c["tokens_used"],
        }
        for c in rows[: args.limit]
    ]
    print(json.dumps(brief, ensure_ascii=False, indent=2))


def _cmd_export(args: argparse.Namespace) -> None:
    config = _load_config(args)
    store = SpecStore(config.db_path)
    cand = store.get("candidates", args.cand_id)
    if not cand:
        raise SystemExit(f"Candidate not found: {args.cand_id}")
    if args.sandbox_root:
        backend = LocalDirSandbox(Path(args.sandbox_root), unsafe_test_mode=True, min_free_gb=0)
        exported = asyncio.run(
            backend.call(
                "export",
                project_id=cand["project_id"],
                cand_id=cand["cand_id"],
                base_sha=cand["base_sha"],
            )
        )
        print(exported["patch"])
    else:
        print(cand["diff_text"])


def _cmd_run(args: argparse.Namespace) -> None:
    if not args.allow_local_sandbox:
        raise SystemExit(
            "Local single-batch execution requires --allow-local-sandbox. "
            "For production isolation, deploy the remote bubblewrap worker via `spec-coding worker`."
        )
    config = _load_config(args)
    sandbox_root = Path(args.sandbox_root or (config.data_dir / "local_sandbox"))
    backend = LocalDirSandbox(sandbox_root, unsafe_test_mode=True, min_free_gb=0)
    engine = SpeculativeOrchestrator(config, backend=backend)
    project = engine.store.new_project(
        config.default_owner_id,
        {
            "name": args.name or Path(args.repo).name or "local-project",
            "repo_source": args.repo,
            "target_branch": args.branch,
            "gate_commands_json": list(args.gate or []),
            "preview_command": args.preview or "",
            "routes_per_batch": args.routes,
        },
    )

    async def _go():
        try:
            await engine.run_project(project)
        finally:
            await engine.stop()
            await backend.shutdown()

    asyncio.run(_go())
    candidates = engine.store.rows("candidates", "project_id=?", (project["project_id"],))
    print(
        json.dumps(
            {
                "project_id": project["project_id"],
                "candidates": [
                    {"cand_id": c["cand_id"], "title": c["title"], "status": c["status"], "head_sha": c["head_sha"]}
                    for c in candidates
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="spec-coding", description="Speculative Coding CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the standalone HTTP/WS control-plane server")
    p_serve.add_argument("--config", default="", help="Path to JSON config file")
    p_serve.add_argument("--data-dir", default="", help="Data directory")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8765)
    p_serve.set_defaults(func=_cmd_serve)

    p_worker = sub.add_parser("worker", help="Run the reverse-WSS sandbox worker daemon")
    p_worker.add_argument("worker_args", nargs=argparse.REMAINDER)

    p_status = sub.add_parser("status", help="Inspect control-plane store status")
    p_status.add_argument("--config", default="")
    p_status.add_argument("--data-dir", default="")
    p_status.set_defaults(func=_cmd_status)

    p_cands = sub.add_parser("candidates", help="List speculative candidates")
    p_cands.add_argument("--config", default="")
    p_cands.add_argument("--data-dir", default="")
    p_cands.add_argument("--project-id", default="")
    p_cands.add_argument("--all", action="store_true", help="Include dropped/expired candidates")
    p_cands.add_argument("--limit", type=int, default=20)
    p_cands.set_defaults(func=_cmd_candidates)

    p_export = sub.add_parser("export", help="Print patch diff for a candidate")
    p_export.add_argument("cand_id")
    p_export.add_argument("--config", default="")
    p_export.add_argument("--data-dir", default="")
    p_export.add_argument("--sandbox-root", default="", help="Optional local sandbox root to export directly from git refs")
    p_export.set_defaults(func=_cmd_export)

    p_run = sub.add_parser("run", help="Run one local speculative batch (development/testing)")
    p_run.add_argument("--config", default="")
    p_run.add_argument("--data-dir", default="")
    p_run.add_argument("--repo", required=True, help="Git repository path or URL")
    p_run.add_argument("--branch", default="main")
    p_run.add_argument("--name", default="")
    p_run.add_argument("--gate", action="append", default=[], help="Deterministic gate command (repeatable)")
    p_run.add_argument("--preview", default="")
    p_run.add_argument("--routes", type=int, default=2)
    p_run.add_argument("--sandbox-root", default="")
    p_run.add_argument("--allow-local-sandbox", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "worker":
        worker_main(args.worker_args)
        return
    args.func(args)


if __name__ == "__main__":
    main()
