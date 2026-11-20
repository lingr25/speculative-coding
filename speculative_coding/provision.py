"""Conservative, worker-side project provisioning for speculative delegation."""

from __future__ import annotations

import json
import uuid
from pathlib import PurePosixPath


def infer_gate_commands(tree: str, files: dict[str, str]) -> list[str]:
    paths = set(line.strip() for line in tree.splitlines() if line.strip()) | set(files)
    commands: list[str] = []

    python_project = any(
        marker in files for marker in ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg", "Pipfile")
    ) or ("uv.lock" in paths and ("pyproject.toml" in files or "uv.lock" in files))
    python_tests_in_dir = any(
        path.startswith("tests/")
        and path.endswith(".py")
        and (PurePosixPath(path).name.startswith("test_") or PurePosixPath(path).name.endswith("_test.py"))
        for path in paths
    )
    python_tests_anywhere = python_tests_in_dir or any(
        path.endswith(".py")
        and (PurePosixPath(path).name.startswith("test_") or PurePosixPath(path).name.endswith("_test.py"))
        for path in paths
    )
    if python_project and python_tests_anywhere:
        target = "tests " if python_tests_in_dir else ""
        if "uv.lock" in paths:
            commands.append(f"uv run --offline --no-sync pytest {target}-q --disable-warnings --tb=short")
        else:
            commands.append(f"python -m pytest {target}-q --disable-warnings --tb=short")

    if "go.mod" in files:
        commands.append("go test ./...")

    if "Cargo.toml" in files:
        commands.append("cargo test")

    package_files = sorted(
        (path for path in paths if path == "package.json" or path.endswith("/package.json")),
        key=lambda p: (p.count("/"), p),
    )
    for package_path in package_files[:6]:
        raw = files.get(package_path)
        if not raw:
            continue
        try:
            package = json.loads(raw)
        except (ValueError, TypeError):
            continue
        scripts = package.get("scripts") or {}
        directory = str(PurePosixPath(package_path).parent)
        prefix = "" if directory == "." else directory + "/"

        manager = ""
        for lock_name, mgr in (
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("package-lock.json", "npm"),
        ):
            if prefix + lock_name in paths:
                manager = mgr
                break
        if not manager:
            for lock_name, mgr in (
                ("pnpm-lock.yaml", "pnpm"),
                ("yarn.lock", "yarn"),
                ("package-lock.json", "npm"),
            ):
                if lock_name in paths:
                    manager = mgr
                    break
        if not manager:
            continue

        for script in ("test", "build"):
            if script not in scripts:
                continue
            if manager == "pnpm":
                if directory == ".":
                    command = "pnpm test" if script == "test" else "pnpm run build"
                else:
                    command = f"pnpm --dir {directory} test" if script == "test" else f"pnpm --dir {directory} run build"
            elif manager == "yarn":
                if directory == ".":
                    command = "yarn test" if script == "test" else "yarn run build"
                else:
                    command = f"yarn --cwd {directory} test" if script == "test" else f"yarn --cwd {directory} run build"
            else:
                if directory == ".":
                    command = "npm test" if script == "test" else "npm run build"
                else:
                    command = (
                        f"npm --prefix {directory} test"
                        if script == "test"
                        else f"npm --prefix {directory} run build"
                    )
            commands.append(command)
    return commands[:6]


class SpecProvisioner:
    def __init__(self, engine):
        self.engine = engine
        self.store = engine.store

    async def _read_manifests(self, project: dict, prepared: dict) -> dict[str, str]:
        tree_lines = [line.strip() for line in (prepared.get("tree") or "").splitlines() if line.strip()]
        tree_set = set(tree_lines)
        root_manifests = (
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "Pipfile",
            "go.mod",
            "Cargo.toml",
        )
        wanted: list[str] = [name for name in root_manifests if not tree_set or name in tree_set]
        wanted.extend(
            sorted(
                (path for path in tree_set if path == "package.json" or path.endswith("/package.json")),
                key=lambda p: (p.count("/"), p),
            )
        )
        files: dict[str, str] = {}
        for path in wanted[:20]:
            try:
                result = await self.engine.backend.call(
                    "read", project_id=project["project_id"], base_sha=prepared["base_sha"], path=path
                )
                files[path] = str(result.get("content") or "")
            except (ValueError, RuntimeError, OSError):
                continue
        return files

    async def ensure_project(self, owner: str, resolved: dict, goal: str, *, validate_baseline: bool = True) -> dict:
        if resolved.get("status") != "ok":
            return resolved
        project = resolved.get("project")
        repo = resolved["repo"]
        if project is None:
            project = self.store.new_project(
                owner,
                {
                    "name": repo.get("name") or repo["full_name"].rsplit("/", 1)[-1],
                    "repo_source": repo["clone_url"],
                    "target_branch": repo.get("default_branch") or "main",
                    "repo_id": str(repo.get("repo_id") or ""),
                    "repo_full_name": repo.get("full_name") or "",
                    "github_installation_id": str(repo.get("installation_id") or ""),
                    "enabled": 0,
                },
            )
        if project.get("gate_commands_json"):
            return {"status": "ok", "project": project, "provisioned": False}

        prepared = await self.engine.prepare_project(project)
        files = await self._read_manifests(project, prepared)
        commands = infer_gate_commands(prepared.get("tree") or "", files)
        if not commands:
            return {"status": "gates_required", "project": project, "base_sha": prepared["base_sha"]}
        if not validate_baseline:
            self.store.update(
                "projects", project["project_id"], gate_commands_json=commands, last_head_sha=prepared["base_sha"]
            )
            project = self.store.get("projects", project["project_id"])
            return {"status": "ok", "project": project, "provisioned": True, "baseline_validated": False}

        cand_id = "b" + uuid.uuid4().hex
        args = {"project_id": project["project_id"], "cand_id": cand_id, "base_sha": prepared["base_sha"]}
        gate: dict = {}
        try:
            await self.engine.backend.call("hydrate", **args, ref=prepared["base_sha"])
            gate = await self.engine.backend.call(
                "gate",
                **args,
                commands=commands,
                preview_command="",
                build_concurrency=project["build_concurrency"],
            )
        finally:
            await self.engine.backend.call("drop", **args)
        evidence = {
            "commands": commands,
            "gates": [
                {"code": item.get("code"), "output": str(item.get("output") or "")[:4000]}
                for item in gate.get("gates", [])
            ],
        }
        self.store.record_event(
            owner,
            project["project_id"],
            "baseline",
            "Baseline gate passed" if gate.get("passed") else "Baseline gate failed",
            head_sha=prepared["base_sha"],
            payload=evidence,
        )
        if not gate.get("passed"):
            return {"status": "baseline_failed", "project": project, "gate": evidence}
        self.store.update(
            "projects", project["project_id"], gate_commands_json=commands, last_head_sha=prepared["base_sha"]
        )
        project = self.store.get("projects", project["project_id"])
        return {"status": "ok", "project": project, "provisioned": True, "baseline_validated": True}
