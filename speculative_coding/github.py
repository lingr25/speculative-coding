"""GitHub App repository discovery with short-lived, repository-scoped credentials."""

from __future__ import annotations

import base64
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15

from .store import SpecStore


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _expires_at(value: str) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return time.time() + 300


class GitHubAppClient:
    def __init__(
        self,
        *,
        app_id: str = "",
        installation_id: str = "",
        private_key_path: str = "",
    ) -> None:
        self.app_id = (app_id or os.environ.get("SPEC_GITHUB_APP_ID", "")).strip()
        self.installation_id = (
            installation_id or os.environ.get("SPEC_GITHUB_APP_INSTALLATION_ID", "")
        ).strip()
        self.private_key_path = (
            private_key_path or os.environ.get("SPEC_GITHUB_APP_PRIVATE_KEY_PATH", "")
        ).strip()
        self._key: RSA.RsaKey | None = None
        self._tokens: dict[str, tuple[float, str]] = {}

    @property
    def configured(self) -> bool:
        return bool(self.app_id and self.installation_id and self.private_key_path)

    def _app_jwt(self, *, now: float | None = None) -> str:
        if not self.configured:
            raise RuntimeError("GitHub App is not configured")
        if self._key is None:
            self._key = RSA.import_key(Path(self.private_key_path).read_bytes())
        issued = int((time.time() if now is None else now) - 60)
        header = _b64url(json.dumps({"alg": "RS256", "typ": "JWT"}, separators=(",", ":")).encode())
        payload = _b64url(
            json.dumps({"iat": issued, "exp": issued + 540, "iss": self.app_id}, separators=(",", ":")).encode()
        )
        signed = f"{header}.{payload}".encode()
        signature = pkcs1_15.new(self._key).sign(SHA256.new(signed))
        return f"{header}.{payload}.{_b64url(signature)}"

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str = "",
        json_body: dict | None = None,
    ) -> tuple[int, Any]:
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = "Bearer " + token
        timeout = aiohttp.ClientTimeout(total=20)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, "https://api.github.com" + path, headers=headers, json=json_body) as response:
                raw = await response.content.read(4 * 1024 * 1024 + 1)
                if len(raw) > 4 * 1024 * 1024:
                    raise RuntimeError("GitHub API response exceeded 4MiB")
                try:
                    data = json.loads(raw) if raw else {}
                except ValueError as exc:
                    raise RuntimeError("GitHub API returned invalid JSON") from exc
                if response.status >= 400:
                    message = (
                        str((data or {}).get("message") or "request failed")
                        if isinstance(data, dict)
                        else "request failed"
                    )
                    if token:
                        message = message.replace(token, "[redacted]")
                    raise RuntimeError(f"GitHub API {response.status}: {message[:300]}")
                return response.status, data

    async def _installation_token(self, *, repository_id: int | None = None) -> str:
        cache_key = str(repository_id or "installation")
        cached = self._tokens.get(cache_key)
        if cached and cached[0] > time.time() + 60:
            return cached[1]
        body: dict[str, Any] = {"permissions": {"contents": "read"}}
        if repository_id is not None:
            body["repository_ids"] = [int(repository_id)]
        _, data = await self._request(
            "POST",
            f"/app/installations/{self.installation_id}/access_tokens",
            token=self._app_jwt(),
            json_body=body,
        )
        token = str(data.get("token") or "")
        if not token:
            raise RuntimeError("GitHub App returned an empty installation token")
        self._tokens[cache_key] = (_expires_at(str(data.get("expires_at") or "")), token)
        return token

    async def clone_token(self, repo: dict) -> str:
        return await self._installation_token(repository_id=int(repo["repo_id"]))

    async def list_repositories(self) -> list[dict]:
        token = await self._installation_token()
        repositories: list[dict] = []
        for page in range(1, 11):
            _, data = await self._request("GET", f"/installation/repositories?per_page=100&page={page}", token=token)
            batch = list(data.get("repositories") or [])
            repositories.extend(batch)
            if len(batch) < 100:
                break
        else:
            raise RuntimeError("GitHub repository pagination exceeded 1000 entries")
        return repositories


class SpecRepoRegistry:
    def __init__(self, store: SpecStore, client: GitHubAppClient | None):
        self.store = store
        self.client = client

    async def sync(self, owner: str) -> list[dict]:
        if self.client is None or not self.client.configured:
            return self.store.rows("repos", "owner_id=?", (str(owner),))
        installation_id = self.client.installation_id
        discovered = await self.client.list_repositories()
        for raw in discovered:
            self.store.upsert_repo(
                owner,
                {
                    "repo_id": raw["id"],
                    "installation_id": installation_id,
                    "full_name": raw["full_name"],
                    "name": raw.get("name") or "",
                    "description": raw.get("description") or "",
                    "clone_url": raw["clone_url"],
                    "default_branch": raw.get("default_branch") or "main",
                    "private": bool(raw.get("private")),
                    "language": raw.get("language") or "",
                    "topics": raw.get("topics") or [],
                },
            )
        self.store.prune_repos(owner, installation_id, [raw["id"] for raw in discovered])
        return self.store.rows("repos", "owner_id=?", (str(owner),))

    @staticmethod
    def _from_project(project: dict) -> dict:
        return {
            "repo_id": project.get("repo_id") or "",
            "installation_id": project.get("github_installation_id") or "",
            "full_name": project.get("repo_full_name") or project["name"],
            "name": project["name"],
            "description": "",
            "clone_url": project["repo_source"],
            "default_branch": project["target_branch"],
            "private": 0,
            "language": "",
            "topics_json": [],
        }

    @staticmethod
    def _linked_project(projects: list[dict], repo: dict) -> dict | None:
        for project in projects:
            if repo.get("repo_id") and project.get("repo_id") == repo["repo_id"]:
                return project
            if project.get("repo_full_name") and project["repo_full_name"].casefold() == repo["full_name"].casefold():
                return project
            if (
                project["repo_source"].rstrip("/").removesuffix(".git").casefold()
                == repo["clone_url"].rstrip("/").removesuffix(".git").casefold()
            ):
                return project
        return None

    async def resolve(self, owner: str, hint: str = "") -> dict:
        owner = str(owner)
        projects = self.store.rows("projects", "owner_id=?", (owner,))
        repos = self.store.rows("repos", "owner_id=?", (owner,))
        query = str(hint or "").strip().casefold()

        def project_score(project: dict) -> int:
            names = [project["name"], project.get("repo_full_name") or "", project["repo_source"]]
            normalized = [name.casefold() for name in names if name]
            if query in normalized:
                return 3
            if query and any(query in name for name in normalized):
                return 1
            return 0

        if query:
            exact_projects = [p for p in projects if project_score(p) == 3]
            if len(exact_projects) == 1:
                project = exact_projects[0]
                return {"status": "ok", "project": project, "repo": self._from_project(project)}
        elif len(projects) == 1:
            project = projects[0]
            return {"status": "ok", "project": project, "repo": self._from_project(project)}

        matches: list[tuple[int, dict]] = []
        for repo in repos:
            names = [repo["full_name"], repo["name"], *(repo.get("topics_json") or [])]
            normalized = [str(name).casefold() for name in names if name]
            score = 3 if query in normalized else 1 if query and any(query in name for name in normalized) else 0
            if not query and len(repos) == 1:
                score = 1
            if score:
                matches.append((score, repo))
        if matches:
            best = max(score for score, _ in matches)
            candidates = [repo for score, repo in matches if score == best]
            if len(candidates) == 1:
                repo = candidates[0]
                return {"status": "ok", "repo": repo, "project": self._linked_project(projects, repo)}

        if query:
            fuzzy_projects = [p for p in projects if project_score(p) > 0]
            if len(fuzzy_projects) == 1 and not matches:
                project = fuzzy_projects[0]
                return {"status": "ok", "project": project, "repo": self._from_project(project)}
            candidates = [*fuzzy_projects, *[repo for _, repo in matches]]
        else:
            candidates = [*projects, *repos]
        if not candidates:
            return {"status": "repo_not_found", "configured": bool(self.client and self.client.configured)}
        return {
            "status": "repo_ambiguous",
            "candidates": [
                {
                    "name": item.get("name") or item.get("full_name"),
                    "full_name": item.get("full_name") or item.get("repo_full_name") or "",
                }
                for item in candidates[:8]
            ],
        }

    async def clone_token(self, repo: dict) -> str:
        if not repo.get("private"):
            return ""
        if self.client is None or not self.client.configured or not repo.get("repo_id"):
            raise RuntimeError("private repository requires a configured GitHub App")
        return await self.client.clone_token(repo)
