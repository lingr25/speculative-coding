"""Durable scheduled speculative runs and morning reviews with pluggable notification delivery."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from .agent import SpecAgent
from .orchestrator import SpeculativeOrchestrator
from .protocols import NotificationSink, NullNotificationSink

log = logging.getLogger(__name__)


class SpecAutomation:
    def __init__(
        self,
        engine: SpeculativeOrchestrator,
        agent: SpecAgent,
        *,
        notifier: NotificationSink | None = None,
        default_timezone: str | None = None,
    ):
        self.engine = engine
        self.agent = agent
        self.store = engine.store
        self.notifier = notifier or NullNotificationSink()
        self.default_timezone = default_timezone or engine.config.default_timezone or "UTC"
        self.engine.automation = self
        self._task: asyncio.Task | None = None
        self._tick_lock = asyncio.Lock()
        self._stopping = False

    @staticmethod
    def _next_fire(hour: int, minute: int, weekdays: list[int], timezone: str, after: float | None = None) -> float:
        zone = ZoneInfo(timezone)
        now = datetime.fromtimestamp(time.time() if after is None else after, zone)
        candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate.timestamp() <= now.timestamp():
            candidate = datetime.fromtimestamp(candidate.timestamp() + 86400, zone)
        allowed = set(weekdays)
        for _ in range(8):
            if not allowed or candidate.weekday() in allowed:
                return candidate.timestamp()
            candidate = datetime.fromtimestamp(candidate.timestamp() + 86400, zone)
        raise ValueError("no matching weekday")

    async def create(self, owner: str, project_id: str, args: dict) -> dict:
        kind = str(args.get("kind") or "")
        if kind not in {"run_batch", "morning_brief"}:
            return {"error": "invalid_automation_kind"}
        hour = int(args.get("hour", 8))
        minute = int(args.get("minute", 0))
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            return {"error": "invalid_schedule_time"}
        weekdays = sorted({int(day) for day in args.get("weekdays", [])})
        if any(day < 0 or day > 6 for day in weekdays):
            return {"error": "invalid_weekdays"}
        timezone = str(args.get("timezone") or self.default_timezone or "UTC").strip()
        try:
            ZoneInfo(timezone)
        except Exception:
            return {"error": "invalid_timezone"}
        now = time.time()
        automation_id = "sa_" + uuid.uuid4().hex
        next_fire = self._next_fire(hour, minute, weekdays, timezone)
        self.store.insert(
            "automations",
            {
                "automation_id": automation_id,
                "owner_id": str(owner),
                "project_id": str(project_id or ""),
                "kind": kind,
                "hour": hour,
                "minute": minute,
                "weekdays_json": weekdays,
                "timezone": timezone,
                "always_notify": int(bool(args.get("always_notify"))),
                "enabled": 1,
                "next_fire_at": next_fire,
                "created_at": now,
                "updated_at": now,
            },
        )
        return {
            "ok": True,
            "automation_id": automation_id,
            "kind": kind,
            "hour": hour,
            "minute": minute,
            "weekdays": weekdays,
            "timezone": timezone,
            "next_fire_at": next_fire,
        }

    async def cancel(self, owner: str, automation_id: str) -> dict:
        automation = self.store.get("automations", automation_id)
        if not automation or automation["owner_id"] != str(owner):
            return {"error": "automation_not_found"}
        self.store.update("automations", automation_id, enabled=0)
        return {"ok": True, "automation_id": automation_id, "enabled": False}

    async def start(self):
        if self._task is None or self._task.done():
            self._stopping = False
            self._task = asyncio.create_task(self._loop(), name="spec-automations")

    async def stop(self):
        self._stopping = True
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _loop(self):
        while not self._stopping:
            try:
                await self.tick()
            except Exception:
                log.exception("spec automation tick failed")
            await asyncio.sleep(20)

    def _claim(self, automation: dict, scheduled: float) -> bool:
        next_fire = self._next_fire(
            automation["hour"],
            automation["minute"],
            automation["weekdays_json"],
            automation["timezone"],
            max(time.time(), scheduled + 60),
        )
        with self.store.conn() as db:
            changed = db.execute(
                "UPDATE spec_automations SET next_fire_at=?,last_fire_at=?,last_status='running',updated_at=? WHERE automation_id=? AND enabled=1 AND next_fire_at=?",
                (next_fire, scheduled, time.time(), automation["automation_id"], scheduled),
            )
            return bool(changed.rowcount)

    async def tick(self, *, now: float | None = None):
        async with self._tick_lock:
            current = time.time() if now is None else now
            due = self.store.rows("automations", "enabled=1 AND next_fire_at<=?", (current,))
            for automation in due:
                scheduled = float(automation["next_fire_at"])
                if not self._claim(automation, scheduled):
                    continue
                try:
                    status = await self._execute(automation, scheduled)
                except Exception:
                    log.exception("spec automation %s failed", automation["automation_id"])
                    status = "error"
                self.store.update("automations", automation["automation_id"], last_status=status)

    async def _execute(self, automation: dict, scheduled: float) -> str:
        owner = automation["owner_id"]
        project_id = automation["project_id"]
        project = None
        if project_id:
            project = self.store.get("projects", project_id)
            if not project or project["owner_id"] != owner:
                return "error"
        if automation["kind"] == "run_batch":
            if not project_id or project is None:
                return "error"
            self.engine._spawn(project)
            return "ok"

        candidates = self.agent._candidates(owner, {"project_id": project_id, "since_hours": 24, "limit": 10})
        if not candidates.get("candidates") and not automation["always_notify"]:
            return "empty"
        scope = f"project {project_id}" if project_id else "all speculative projects"
        prompt = (
            f"Run the daily review briefing for {scope}. First call spec_candidates for the last 24 hours, "
            "then call spec_explain on candidates worth highlighting to verify head_sha, gate, and review evidence. "
            "Recommend at most three items, summarizing what changed, how it works, and verified/unverified gate status without dumping raw diffs. "
            "Output a concise Markdown brief; never merge, push, or deploy."
        )
        result = await self.agent.submit_turn(
            session_key=f"spec:{owner}:main",
            owner_id=owner,
            sender_name="SpecAutomation",
            user_text=prompt,
            title="Daily Speculative Brief",
            origin_channel="automation",
        )
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "morning review failed"))
        full_text = str(result.get("text") or "").strip()
        if not full_text:
            raise RuntimeError("morning review returned an empty report")
        summary = "[Speculative Coding Brief]\n" + full_text[:430]
        dedup_key = f"{automation['automation_id']}:{int(scheduled)}"
        inserted = self.store.enqueue_notification(
            dedup_key,
            owner,
            f"spec:{owner}:main",
            full_text,
            summary,
            scheduled,
        )
        if not inserted:
            return "ok"
        notification = self.store.rows("notifications", "dedup_key=?", (dedup_key,))[0]
        self.store.update("notifications", notification["notification_id"], status="persisted")
        try:
            delivered = await self.notifier.send_notification(owner, summary, full_text=full_text)
            self.store.update(
                "notifications",
                notification["notification_id"],
                status="sent" if delivered else "unknown",
                delivery_error="" if delivered else "notification delivery was not acknowledged",
            )
        except Exception as exc:
            self.store.update(
                "notifications",
                notification["notification_id"],
                status="unknown",
                delivery_error=str(exc)[:500],
            )
        return "ok"
