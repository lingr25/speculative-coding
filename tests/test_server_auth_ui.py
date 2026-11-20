"""Unit and integration tests for fail-closed server auth, header hardening, WS tickets, CLI, and UI static serving."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientResponseError, web
from aiohttp.test_utils import TestClient, TestServer

from speculative_coding import cli, server
from speculative_coding.config import SpecConfig
from speculative_coding.protocols import BearerTokenAuthProvider, SingleOwnerAuthProvider
from speculative_coding.server import (
    _is_loopback_host,
    _resolve_ui_static_file,
    create_control_plane,
    validate_bind_auth,
)
from speculative_coding.store import SpecStore


class LoopbackAndFailClosedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_is_loopback_host_classification(self) -> None:
        for host in ("127.0.0.1", "127.0.0.2", "::1", "[::1]", "localhost", "LOCALHOST", " localhost "):
            self.assertTrue(_is_loopback_host(host), f"expected loopback for {host!r}")
        for host in ("0.0.0.0", "::", "[::]", "192.168.1.10", "10.0.0.5", "example.com", ""):
            self.assertFalse(_is_loopback_host(host), f"expected non-loopback for {host!r}")

    def test_non_loopback_without_token_fails_closed(self) -> None:
        cfg = SpecConfig(data_dir=self.root / "failclosed", api_token="")
        with self.assertRaisesRegex(RuntimeError, "Refusing to start.*non-loopback"):
            validate_bind_auth(cfg, "0.0.0.0")
        with self.assertRaisesRegex(RuntimeError, "Refusing to start.*non-loopback"):
            create_control_plane(cfg, host="0.0.0.0")
        with self.assertRaisesRegex(RuntimeError, "Refusing to start.*non-loopback"):
            create_control_plane(cfg, host="::")

    def test_non_loopback_with_token_or_insecure_override_allowed(self) -> None:
        cfg_tok = SpecConfig(data_dir=self.root / "with_token", api_token="s" * 32)
        plane = create_control_plane(cfg_tok, host="0.0.0.0")
        self.assertIsInstance(plane.engine.auth, BearerTokenAuthProvider)
        self.assertFalse(plane.engine.auth.allow_owner_header)

        cfg_no_tok = SpecConfig(data_dir=self.root / "insecure", api_token="")
        with self.assertLogs("speculative_coding.server", level="WARNING") as logs:
            plane2 = create_control_plane(cfg_no_tok, host="0.0.0.0", allow_insecure_no_auth=True)
        self.assertIsInstance(plane2.engine.auth, SingleOwnerAuthProvider)
        self.assertTrue(any("SingleOwnerAuthProvider" in line for line in logs.output))

    def test_loopback_without_token_logs_prominent_warning(self) -> None:
        cfg = SpecConfig(data_dir=self.root / "loopback", api_token="")
        with self.assertLogs("speculative_coding.server", level="WARNING") as logs:
            plane = create_control_plane(cfg, host="127.0.0.1")
        self.assertIsInstance(plane.engine.auth, SingleOwnerAuthProvider)
        self.assertTrue(any("SECURITY WARNING" in line and "SingleOwnerAuthProvider" in line for line in logs.output))

    def test_server_and_cli_main_fail_closed_on_non_loopback(self) -> None:
        data_dir = str(self.root / "cli_fail")
        with patch.dict(os.environ, {"SPEC_API_TOKEN": ""}, clear=False):
            with patch("speculative_coding.server.web.run_app") as run_app:
                with self.assertRaises(SystemExit) as ctx:
                    server.main(["--data-dir", data_dir, "--host", "0.0.0.0", "--port", "8765"])
                self.assertIn("non-loopback", str(ctx.exception))
                run_app.assert_not_called()

            with patch("speculative_coding.cli.web.run_app") as cli_run_app:
                with self.assertRaises(SystemExit) as ctx:
                    cli.main(["serve", "--data-dir", data_dir, "--host", "0.0.0.0", "--port", "8765"])
                self.assertIn("non-loopback", str(ctx.exception))
                cli_run_app.assert_not_called()

            with patch("speculative_coding.cli.web.run_app") as cli_run_app:
                with self.assertLogs("speculative_coding.server", level="WARNING"):
                    cli.main(["serve", "--data-dir", data_dir, "--host", "0.0.0.0", "--insecure-no-auth"])
                cli_run_app.assert_called_once()


class BearerHeaderHardeningTests(unittest.TestCase):
    def test_owner_header_ignored_by_default_and_honored_when_opted_in(self) -> None:
        token = "t" * 32
        req = SimpleNamespace(
            headers={"Authorization": "Bearer " + token, "X-Spec-Owner": "impersonated-victim"}
        )
        strict_auth = BearerTokenAuthProvider(token, owner_id="admin")
        identity = strict_auth.require_admin(req)
        self.assertEqual(identity["owner_id"], "admin")
        self.assertTrue(identity["is_admin"])

        proxy_auth = BearerTokenAuthProvider(token, owner_id="admin", allow_owner_header=True)
        proxy_identity = proxy_auth.require_admin(req)
        self.assertEqual(proxy_identity["owner_id"], "impersonated-victim")

        req_no_owner = SimpleNamespace(headers={"Authorization": "Bearer " + token})
        self.assertEqual(proxy_auth.require_admin(req_no_owner)["owner_id"], "admin")

    def test_short_or_wrong_token_rejected(self) -> None:
        short_auth = BearerTokenAuthProvider("too-short", owner_id="admin")
        with self.assertRaises(web.HTTPUnauthorized):
            short_auth.require_admin(SimpleNamespace(headers={"Authorization": "Bearer too-short"}))

        good_auth = BearerTokenAuthProvider("a" * 32, owner_id="admin")
        with self.assertRaises(web.HTTPUnauthorized):
            good_auth.require_admin(SimpleNamespace(headers={"Authorization": "Bearer " + "b" * 32}))


class ServerAuthAndStaticUITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.token = "k" * 32

        self.secret_file = self.root / "secret.txt"
        self.secret_file.write_text("TOP_SECRET_DO_NOT_LEAK", encoding="utf-8")

        self.ui_dir = self.root / "ui_static"
        (self.ui_dir / "assets").mkdir(parents=True)
        (self.ui_dir / "index.html").write_text("<!doctype html><html><body>Spec UI</body></html>", encoding="utf-8")
        (self.ui_dir / "app.js").write_text("console.log('spec ui');", encoding="utf-8")
        (self.ui_dir / "assets" / "styles.css").write_text("body { color: #111; }", encoding="utf-8")

        cfg = SpecConfig(
            data_dir=self.root / "data",
            api_token=self.token,
            default_owner_id="admin",
        )
        self.plane = create_control_plane(cfg, host="0.0.0.0", ui_static_dir=self.ui_dir)
        self.client = TestClient(TestServer(self.plane.app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_ui_shell_and_assets_served_without_auth_while_api_requires_bearer(self) -> None:
        for path in ("/", "/ui", "/ui/"):
            resp = await self.client.get(path)
            self.assertEqual(resp.status, 200, path)
            self.assertIn("text/html", resp.headers["Content-Type"])
            self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
            self.assertIn("Spec UI", await resp.text())

        js_resp = await self.client.get("/ui/app.js")
        self.assertEqual(js_resp.status, 200)
        self.assertIn("javascript", js_resp.headers["Content-Type"])
        self.assertIn("console.log", await js_resp.text())

        css_resp = await self.client.get("/ui/assets/styles.css")
        self.assertEqual(css_resp.status, 200)
        self.assertIn("text/css", css_resp.headers["Content-Type"])
        self.assertIn("color: #111", await css_resp.text())

        unauth_api = await self.client.get("/api/spec/status")
        self.assertEqual(unauth_api.status, 401)

        authed_api = await self.client.get(
            "/api/spec/status",
            headers={"Authorization": "Bearer " + self.token, "X-Spec-Owner": "spoofed-user"},
        )
        self.assertEqual(authed_api.status, 200)
        body = await authed_api.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["agent_session_key"], "spec:admin:main")

    async def test_static_ui_blocks_path_traversal(self) -> None:
        for bad_rel in ("../secret.txt", "assets/../../secret.txt", "/etc/passwd", "a\\..\\secret.txt", "app.js\x00.txt"):
            self.assertIsNone(_resolve_ui_static_file(self.ui_dir, bad_rel), bad_rel)

        # Also test symlink escaping static_root when the platform permits symlinks.
        escape_link = self.ui_dir / "escape.txt"
        try:
            escape_link.symlink_to(self.secret_file)
        except OSError:
            escape_link = None
        if escape_link is not None:
            self.assertIsNone(_resolve_ui_static_file(self.ui_dir, "escape.txt"))

        for bad_url in (
            "/ui/%2e%2e/secret.txt",
            "/ui/assets/%2e%2e/%2e%2e/secret.txt",
            "/ui/..%5csecret.txt",
            "/ui/missing.js",
        ):
            resp = await self.client.get(bad_url)
            self.assertIn(resp.status, (400, 404), bad_url)
            self.assertNotIn("TOP_SECRET", await resp.text())

    async def test_ws_ticket_single_use_and_auth_enforcement(self) -> None:
        unauth = await self.client.post("/api/spec/agent/ws-ticket")
        self.assertEqual(unauth.status, 401)

        ticket_resp = await self.client.post(
            "/api/spec/agent/ws-ticket",
            headers={"Authorization": "Bearer " + self.token},
        )
        self.assertEqual(ticket_resp.status, 200)
        ticket_data = await ticket_resp.json()
        self.assertTrue(ticket_data["ok"])
        ticket = ticket_data["ticket"]
        self.assertTrue(ticket)

        # Connect using ticket without Authorization header (browser WS flow)
        ws = await self.client.ws_connect(f"/api/spec/agent/ws?ticket={ticket}")
        hello = await ws.receive_json()
        self.assertEqual(hello["type"], "connected")
        self.assertEqual(hello["owner_id"], "admin")
        await ws.close()

        # Replaying the same ticket must fail closed with 401
        with self.assertRaises(ClientResponseError) as ctx:
            await self.client.ws_connect(f"/api/spec/agent/ws?ticket={ticket}")
        self.assertEqual(ctx.exception.status, 401)

        # Bogus ticket fails closed with 401
        with self.assertRaises(ClientResponseError) as ctx2:
            await self.client.ws_connect("/api/spec/agent/ws?ticket=bogus-ticket")
        self.assertEqual(ctx2.exception.status, 401)

        # Direct Bearer header on WS still works
        ws_bearer = await self.client.ws_connect(
            "/api/spec/agent/ws",
            headers={"Authorization": "Bearer " + self.token},
        )
        hello2 = await ws_bearer.receive_json()
        self.assertEqual(hello2["owner_id"], "admin")
        await ws_bearer.close()

    async def test_missing_ui_static_returns_clean_404(self) -> None:
        cfg = SpecConfig(data_dir=self.root / "data_empty_ui", api_token=self.token)
        plane = create_control_plane(cfg, ui_static_dir=self.root / "nonexistent_ui")
        client = TestClient(TestServer(plane.app))
        await client.start_server()
        self.addAsyncCleanup(client.close)

        resp = await client.get("/")
        self.assertEqual(resp.status, 404)
        self.assertIn("ui_static/index.html missing", await resp.text())


class CliExportBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.store = SpecStore(self.data_dir / "speculative.db")
        proj = self.store.new_project("local", {"name": "demo", "repo_source": "https://github.com/example/demo.git"})
        self.store.insert(
            "runs",
            {"run_id": "run1", "project_id": proj["project_id"], "base_sha": "a" * 40, "status": "completed", "created_at": 1.0},
        )
        self.store.insert(
            "candidates",
            {
                "cand_id": "cand1",
                "run_id": "run1",
                "project_id": proj["project_id"],
                "route_index": 0,
                "title": "Demo",
                "route_plan": "Plan",
                "route_kind": "optimization",
                "branch_ref": "refs/spec/cand1",
                "status": "staged",
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "diff_text": "+hello\n",
                "created_at": 1.0,
                "updated_at": 1.0,
            },
        )

    def test_cli_export_bundle_writes_decoded_bundle(self) -> None:
        sandbox_root = self.root / "sandbox"
        sandbox_root.mkdir()
        bundle_out = self.root / "out" / "cand1.bundle"
        raw_bundle = b"# v2 git bundle\nfake-bundle-bytes"
        fake_result = {
            "patch": "+hello\n",
            "head_sha": "b" * 40,
            "base_sha": "a" * 40,
            "branch_ref": "refs/spec/cand1",
            "bundle_base64": base64.b64encode(raw_bundle).decode("ascii"),
            "bundle_filename": "cand1.bundle",
            "apply_instructions": {"bundle": "git fetch cand1.bundle refs/spec/cand1"},
        }
        buf = io.StringIO()
        with patch("speculative_coding.cli.LocalDirSandbox.call", new=AsyncMock(return_value=fake_result)) as mock_call:
            with redirect_stdout(buf):
                cli.main(
                    [
                        "export",
                        "cand1",
                        "--data-dir",
                        str(self.data_dir),
                        "--sandbox-root",
                        str(sandbox_root),
                        "--bundle",
                        str(bundle_out),
                    ]
                )
        self.assertTrue(bundle_out.is_file())
        self.assertEqual(bundle_out.read_bytes(), raw_bundle)
        payload = json.loads(buf.getvalue())
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["head_sha"], "b" * 40)
        mock_call.assert_awaited_once()
        self.assertTrue(mock_call.await_args.kwargs["include_bundle"])

    def test_cli_export_bundle_without_sandbox_root_exits_cleanly(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            cli.main(
                [
                    "export",
                    "cand1",
                    "--data-dir",
                    str(self.data_dir),
                    "--bundle",
                    str(self.root / "out.bundle"),
                ]
            )
        self.assertIn("--sandbox-root", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
