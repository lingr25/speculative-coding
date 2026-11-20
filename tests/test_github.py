"""GitHub repository discovery and credential-boundary tests."""

from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15

from speculative_coding.github import GitHubAppClient, SpecRepoRegistry
from speculative_coding.sandbox import LocalDirSandbox, RemoteWorkerSandbox
from speculative_coding.store import SpecStore


class GitHubAppTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.key = RSA.generate(2048)
        cls.key_bytes = cls.key.export_key()

    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.key_path = self.root / "github-app-test-key.pem"
        self.key_path.write_bytes(self.key_bytes)
        self.env = {
            "SPEC_GITHUB_APP_ID": "123",
            "SPEC_GITHUB_APP_INSTALLATION_ID": "456",
            "SPEC_GITHUB_APP_PRIVATE_KEY_PATH": str(self.key_path),
        }

    def client(self):
        with patch.dict("os.environ", self.env, clear=False):
            return GitHubAppClient()

    async def test_jwt_is_rs256_and_short_lived(self):
        client = self.client()
        token = client._app_jwt(now=1_000_000)
        header_raw, payload_raw, signature_raw = token.split(".")
        header = json.loads(base64.urlsafe_b64decode(header_raw + "=="))
        payload = json.loads(base64.urlsafe_b64decode(payload_raw + "=="))
        self.assertEqual(header, {"alg": "RS256", "typ": "JWT"})
        self.assertEqual(payload["iss"], "123")
        self.assertLessEqual(payload["exp"] - payload["iat"], 600)
        pkcs1_15.new(self.key.publickey()).verify(
            SHA256.new(f"{header_raw}.{payload_raw}".encode()),
            base64.urlsafe_b64decode(signature_raw + "=="),
        )

    async def test_sync_and_repo_scoped_clone_token(self):
        client = self.client()
        client._request = AsyncMock(
            side_effect=[
                (200, {"token": "installation-token", "expires_at": "2030-01-01T00:00:00Z"}),
                (
                    200,
                    {
                        "repositories": [
                            {
                                "id": 42,
                                "name": "demo",
                                "full_name": "example/demo",
                                "description": "Demo repo",
                                "clone_url": "https://github.com/example/demo.git",
                                "default_branch": "main",
                                "private": True,
                                "language": "Python",
                                "topics": ["agent"],
                            }
                        ]
                    },
                ),
                (200, {"token": "one-repo-token", "expires_at": "2030-01-01T00:00:00Z"}),
            ]
        )
        store = SpecStore(self.root / "repos.db")
        store.upsert_repo(
            "owner",
            {
                "repo_id": 99,
                "installation_id": "456",
                "full_name": "example/revoked",
                "clone_url": "https://github.com/example/revoked.git",
            },
        )
        registry = SpecRepoRegistry(store, client)
        repos = await registry.sync("owner")
        self.assertEqual([repo["full_name"] for repo in repos], ["example/demo"])
        token = await client.clone_token(store.rows("repos")[0])
        self.assertEqual(token, "one-repo-token")
        token_call = client._request.await_args_list[-1]
        self.assertEqual(token_call.kwargs["json_body"]["repository_ids"], [42])
        self.assertEqual(token_call.kwargs["json_body"]["permissions"], {"contents": "read"})

    async def test_resolution_uses_existing_projects_and_asks_on_ambiguity(self):
        store = SpecStore(self.root / "resolve.db")
        store.new_project(
            "owner",
            {
                "name": "Demo Workbench",
                "repo_source": "https://github.com/example/demo.git",
                "repo_full_name": "example/demo",
            },
        )
        registry = SpecRepoRegistry(store, None)
        resolved = await registry.resolve("owner", "Demo")
        self.assertEqual(resolved["status"], "ok")
        self.assertEqual(resolved["project"]["name"], "Demo Workbench")
        for repo_id, full_name in enumerate(("example/alpha", "example/beta"), start=1):
            store.upsert_repo(
                "owner",
                {
                    "repo_id": repo_id,
                    "installation_id": 1,
                    "full_name": full_name,
                    "clone_url": f"https://github.com/{full_name}.git",
                },
            )
        ambiguous = await registry.resolve("owner", "example")
        self.assertEqual(ambiguous["status"], "repo_ambiguous")
        self.assertGreaterEqual(len(ambiguous["candidates"]), 2)


class CredentialBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_sensitive_prepare_is_not_written_to_control_journal(self):
        with tempfile.TemporaryDirectory() as directory:
            remote = RemoteWorkerSandbox(Path(directory) / "control.db")
            sent = []

            class WS:
                closed = False

                async def send_json(self, payload):
                    sent.append(payload)

            remote.ws = WS()
            task = asyncio.create_task(
                remote.call(
                    "prepare",
                    project_id="p",
                    repo_source="https://github.com/example/private.git",
                    clone_token="test-secret",
                )
            )
            await asyncio.sleep(0.02)
            self.assertEqual(sent[0]["args"]["clone_token"], "test-secret")
            with remote._journal() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM spec_worker_tasks").fetchone()[0], 0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def test_askpass_helper_contains_no_secret_and_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalDirSandbox(Path(directory) / "pool", unsafe_test_mode=True, min_free_gb=0)
            with backend._github_askpass("test-secret") as auth:
                helper = Path(auth["env_extra"]["GIT_ASKPASS"])
                self.assertTrue(helper.is_file())
                self.assertNotIn("test-secret", helper.read_text())
                self.assertEqual(auth["env_extra"]["SPEC_GIT_PASSWORD"], "test-secret")
                self.assertIn("test-secret", auth["redact"])
            self.assertFalse(helper.exists())

    async def test_exec_redacts_secret_and_adds_auth_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = LocalDirSandbox(Path(directory) / "pool", unsafe_test_mode=True, min_free_gb=0)
            result = await backend._exec(
                ["python", "-c", "import os; print(os.environ['MARKER']); print('token=test-secret')"],
                env_extra={"MARKER": "present"},
                redact=("test-secret",),
            )
            self.assertIn("present", result["output"])
            self.assertNotIn("test-secret", result["output"])
            self.assertIn("[redacted]", result["output"])


if __name__ == "__main__":
    unittest.main()
