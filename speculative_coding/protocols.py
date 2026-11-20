"""Pluggable protocols and default adapters for auth, notifications, and LLM clients."""

from __future__ import annotations

import hmac
import logging
from typing import Any, Protocol, runtime_checkable

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)


@runtime_checkable
class AuthProvider(Protocol):
    """Authenticates control-plane HTTP/WS requests and returns an owner identity dict."""

    def require_admin(self, request: web.Request) -> dict[str, Any]:
        """Return a dict containing at least {'owner_id': str, 'is_admin': bool} or raise HTTPUnauthorized/HTTPForbidden."""
        ...


class SingleOwnerAuthProvider:
    """Simple local/dev auth provider that authenticates all requests as a single admin owner."""

    def __init__(self, owner_id: str = "local", display_name: str = "operator") -> None:
        self.owner_id = str(owner_id)
        self.display_name = str(display_name)

    def require_admin(self, request: web.Request) -> dict[str, Any]:
        return {
            "owner_id": self.owner_id,
            "name": self.display_name,
            "is_admin": True,
        }


class BearerTokenAuthProvider:
    """Constant-time Bearer token auth provider for standalone production HTTP servers."""

    def __init__(self, token: str, *, owner_id: str = "admin", display_name: str = "admin") -> None:
        self.token = token.strip()
        self.owner_id = str(owner_id)
        self.display_name = str(display_name)

    def require_admin(self, request: web.Request) -> dict[str, Any]:
        if len(self.token) < 16:
            raise web.HTTPUnauthorized(text="server API token is not configured")
        header = request.headers.get("Authorization", "")
        if not hmac.compare_digest(header, "Bearer " + self.token):
            raise web.HTTPUnauthorized(text="invalid bearer token")
        owner = request.headers.get("X-Spec-Owner", "").strip() or self.owner_id
        return {
            "owner_id": owner,
            "name": self.display_name,
            "is_admin": True,
        }


@runtime_checkable
class NotificationSink(Protocol):
    """Delivers scheduled review summaries (e.g. morning brief) to an external channel."""

    async def send_notification(self, owner_id: str, summary: str, *, full_text: str = "") -> bool:
        """Deliver notification; return True if acknowledged, False otherwise."""
        ...


class NullNotificationSink:
    """No-op notification sink that logs summaries locally."""

    async def send_notification(self, owner_id: str, summary: str, *, full_text: str = "") -> bool:
        logger.info("Spec notification for %s: %s", owner_id, summary[:200])
        return True


class WebhookNotificationSink:
    """Generic HTTP JSON webhook notification sink."""

    def __init__(self, url: str, *, headers: dict[str, str] | None = None, timeout: float = 15.0) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.timeout = timeout

    async def send_notification(self, owner_id: str, summary: str, *, full_text: str = "") -> bool:
        payload = {"owner_id": owner_id, "summary": summary, "full_text": full_text}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=payload, headers=self.headers) as resp:
                return 200 <= resp.status < 300
