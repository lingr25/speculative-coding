"""Durable SQLite metadata store for Speculative Coding."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid

PROJECT_DEFAULTS = dict(
    target_branch="master",
    sandbox_mode="remote_worker",
    recon_provider_ids=[],
    coder_provider_ids=[],
    reviewer_provider_ids=[],
    inspector_provider_ids=[],
    gate_commands_json=[],
    preview_command="",
    routes_per_batch=3,
    llm_concurrency=6,
    wt_concurrency=3,
    build_concurrency=2,
    max_review_rounds=5,
    ttl_hours=72,
    enabled=0,
    status="idle",
    speculative_tokens=0,
    wake_at=None,
    last_head_sha="",
    hot_cache_last_used=None,
    repo_id="",
    repo_full_name="",
    github_installation_id="",
)
JSON_FIELDS = {
    "recon_provider_ids",
    "coder_provider_ids",
    "reviewer_provider_ids",
    "inspector_provider_ids",
    "gate_commands_json",
    "routes_json",
    "steer_notes_json",
    "preview_meta_json",
    "topics_json",
    "payload_json",
    "weekdays_json",
}
TABLE_KEYS = {
    "projects": "project_id",
    "runs": "run_id",
    "candidates": "cand_id",
    "graveyard": "grave_id",
    "actions": "action_id",
    "repos": "repo_key",
    "memories": "memory_id",
    "events": "event_id",
    "automations": "automation_id",
    "notifications": "notification_id",
}


class SpecStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            c.executescript(
                """
CREATE TABLE IF NOT EXISTS spec_projects (
 project_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, name TEXT NOT NULL, repo_source TEXT NOT NULL,
 target_branch TEXT NOT NULL DEFAULT 'master', sandbox_mode TEXT NOT NULL DEFAULT 'remote_worker',
 recon_provider_ids TEXT NOT NULL DEFAULT '[]', coder_provider_ids TEXT NOT NULL DEFAULT '[]',
 reviewer_provider_ids TEXT NOT NULL DEFAULT '[]', inspector_provider_ids TEXT NOT NULL DEFAULT '[]',
 gate_commands_json TEXT NOT NULL DEFAULT '[]', preview_command TEXT NOT NULL DEFAULT '',
 routes_per_batch INTEGER NOT NULL DEFAULT 3, llm_concurrency INTEGER NOT NULL DEFAULT 6,
 wt_concurrency INTEGER NOT NULL DEFAULT 3, build_concurrency INTEGER NOT NULL DEFAULT 2,
 max_review_rounds INTEGER NOT NULL DEFAULT 5, ttl_hours INTEGER NOT NULL DEFAULT 72,
 enabled INTEGER NOT NULL DEFAULT 0, hot_cache_last_used REAL, status TEXT NOT NULL DEFAULT 'idle',
 speculative_tokens INTEGER NOT NULL DEFAULT 0, wake_at REAL, last_head_sha TEXT NOT NULL DEFAULT '',
 created_at REAL NOT NULL, updated_at REAL NOT NULL,
 repo_id TEXT NOT NULL DEFAULT '', repo_full_name TEXT NOT NULL DEFAULT '',
 github_installation_id TEXT NOT NULL DEFAULT '');
CREATE TABLE IF NOT EXISTS spec_runs (
 run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, base_sha TEXT NOT NULL DEFAULT '',
 recon_summary TEXT NOT NULL DEFAULT '', routes_json TEXT NOT NULL DEFAULT '[]',
 status TEXT NOT NULL, tokens_used INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, completed_at REAL);
CREATE TABLE IF NOT EXISTS spec_candidates (
 cand_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, project_id TEXT NOT NULL, route_index INTEGER NOT NULL,
 route_kind TEXT NOT NULL DEFAULT 'optimization', competition_group TEXT NOT NULL DEFAULT '',
 title TEXT NOT NULL, route_plan TEXT NOT NULL, steer_notes_json TEXT NOT NULL DEFAULT '[]',
 branch_ref TEXT NOT NULL DEFAULT '', active_worktree_path TEXT NOT NULL DEFAULT '',
 status TEXT NOT NULL, review_rounds INTEGER NOT NULL DEFAULT 0, last_diff_hash TEXT NOT NULL DEFAULT '',
 human_report_md TEXT NOT NULL DEFAULT '', preview_meta_json TEXT NOT NULL DEFAULT '{}',
 drop_reason TEXT NOT NULL DEFAULT '', tokens_used INTEGER NOT NULL DEFAULT 0, expires_at REAL,
 created_at REAL NOT NULL, updated_at REAL NOT NULL, diff_text TEXT NOT NULL DEFAULT '',
 base_sha TEXT NOT NULL, head_sha TEXT NOT NULL DEFAULT '', feedback TEXT NOT NULL DEFAULT '',
 stagnant_rounds INTEGER NOT NULL DEFAULT 0, generation INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS idx_spec_cand_expiry ON spec_candidates(status,expires_at);
CREATE TABLE IF NOT EXISTS spec_graveyard (
 grave_id TEXT PRIMARY KEY, project_id TEXT NOT NULL, scope_hint TEXT NOT NULL, idea TEXT NOT NULL,
 source TEXT NOT NULL, reason TEXT NOT NULL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS spec_actions (
 action_id TEXT PRIMARY KEY, cand_id TEXT NOT NULL, actor TEXT NOT NULL, operation TEXT NOT NULL,
 payload_hash TEXT NOT NULL, result_json TEXT, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS spec_repos (
 repo_key TEXT PRIMARY KEY, owner_id TEXT NOT NULL, repo_id TEXT NOT NULL, installation_id TEXT NOT NULL,
 full_name TEXT NOT NULL, name TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', clone_url TEXT NOT NULL,
 default_branch TEXT NOT NULL DEFAULT 'main', private INTEGER NOT NULL DEFAULT 0,
 language TEXT NOT NULL DEFAULT '', topics_json TEXT NOT NULL DEFAULT '[]',
 synced_at REAL NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
 UNIQUE(owner_id,installation_id,repo_id));
CREATE INDEX IF NOT EXISTS idx_spec_repos_owner ON spec_repos(owner_id,full_name);
CREATE TABLE IF NOT EXISTS spec_memories (
 memory_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, project_id TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL,
 key TEXT NOT NULL, value TEXT NOT NULL, source TEXT NOT NULL, source_event TEXT NOT NULL DEFAULT '',
 supersedes TEXT NOT NULL DEFAULT '', superseded_by TEXT NOT NULL DEFAULT '', active INTEGER NOT NULL DEFAULT 1,
 created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_spec_memories_scope ON spec_memories(owner_id,project_id,key,active);
CREATE TABLE IF NOT EXISTS spec_events (
 event_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, project_id TEXT NOT NULL, run_id TEXT NOT NULL DEFAULT '',
 cand_id TEXT NOT NULL DEFAULT '', head_sha TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL,
 summary TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_spec_events_candidate ON spec_events(owner_id,cand_id,created_at);
CREATE TABLE IF NOT EXISTS spec_automations (
 automation_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, project_id TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL,
 hour INTEGER NOT NULL DEFAULT 8, minute INTEGER NOT NULL DEFAULT 0, weekdays_json TEXT NOT NULL DEFAULT '[]',
 timezone TEXT NOT NULL DEFAULT 'UTC', always_notify INTEGER NOT NULL DEFAULT 0,
 enabled INTEGER NOT NULL DEFAULT 1, next_fire_at REAL NOT NULL, last_fire_at REAL,
 last_status TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_spec_automations_due ON spec_automations(enabled,next_fire_at);
CREATE TABLE IF NOT EXISTS spec_notifications (
 notification_id TEXT PRIMARY KEY, dedup_key TEXT NOT NULL UNIQUE, owner_id TEXT NOT NULL,
 session_key TEXT NOT NULL, full_text TEXT NOT NULL, summary TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
 delivery_error TEXT NOT NULL DEFAULT '', scheduled_fire_at REAL NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_spec_notifications_status ON spec_notifications(status,created_at);
"""
            )
            columns = {row[1] for row in c.execute("PRAGMA table_info(spec_projects)")}
            for name in ("repo_id", "repo_full_name", "github_installation_id"):
                if name not in columns:
                    c.execute(f"ALTER TABLE spec_projects ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")

    @contextmanager
    def conn(self):
        c = sqlite3.connect(str(self.path), timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        c.execute("PRAGMA synchronous=NORMAL")
        try:
            with c:
                yield c
        finally:
            c.close()

    def decode(self, row):
        if row is None:
            return None
        d = dict(row)
        for k in JSON_FIELDS & d.keys():
            d[k] = json.loads(d[k])
        return d

    def rows(self, table, where="", args=()):
        assert table in TABLE_KEYS
        with self.conn() as c:
            return [
                self.decode(r)
                for r in c.execute(
                    "SELECT * FROM spec_" + table + (" WHERE " + where if where else "") + " ORDER BY created_at DESC",
                    args,
                )
            ]

    def get(self, table, key):
        rows = self.rows(table, TABLE_KEYS[table] + "=?", (key,))
        return rows[0] if rows else None

    def insert(self, table, d):
        assert table in TABLE_KEYS
        values = {k: json.dumps(v, ensure_ascii=False) if k in JSON_FIELDS else v for k, v in d.items()}
        with self.conn() as c:
            c.execute(
                "INSERT INTO spec_"
                + table
                + " ("
                + ",".join(values)
                + ") VALUES ("
                + ",".join("?" for _ in values)
                + ")",
                tuple(values.values()),
            )

    def update(self, table, key, **values):
        assert table in TABLE_KEYS
        col = TABLE_KEYS[table]
        if table not in {"runs", "graveyard", "actions", "events"}:
            values["updated_at"] = time.time()
        values = {k: json.dumps(v, ensure_ascii=False) if k in JSON_FIELDS else v for k, v in values.items()}
        with self.conn() as c:
            c.execute(
                "UPDATE spec_" + table + " SET " + ",".join(k + "=?" for k in values) + " WHERE " + col + "=?",
                (*values.values(), key),
            )

    def new_project(self, owner, data):
        now = time.time()
        pid = "sp_" + uuid.uuid4().hex
        project = {
            **PROJECT_DEFAULTS,
            **data,
            "project_id": pid,
            "owner_id": str(owner),
            "created_at": now,
            "updated_at": now,
        }
        self.insert("projects", project)
        return self.get("projects", pid)

    def upsert_repo(self, owner, data):
        now = time.time()
        repo_id = str(data["repo_id"])
        installation_id = str(data["installation_id"])
        full_name = str(data["full_name"]).strip()
        clone_url = str(data["clone_url"]).strip()
        if not full_name or not clone_url.startswith("https://") or any(ch in clone_url for ch in ("@", "?", "#")):
            raise ValueError("invalid credential-free repository metadata")
        key = f"{owner}:{installation_id}:{repo_id}"
        values = dict(
            repo_key=key,
            owner_id=str(owner),
            repo_id=repo_id,
            installation_id=installation_id,
            full_name=full_name,
            name=str(data.get("name") or full_name.rsplit("/", 1)[-1]),
            description=str(data.get("description") or ""),
            clone_url=clone_url,
            default_branch=str(data.get("default_branch") or "main"),
            private=int(bool(data.get("private"))),
            language=str(data.get("language") or ""),
            topics_json=list(data.get("topics") or []),
            synced_at=float(data.get("synced_at") or now),
            created_at=now,
            updated_at=now,
        )
        with self.conn() as c:
            c.execute(
                """INSERT INTO spec_repos (
                repo_key,owner_id,repo_id,installation_id,full_name,name,description,clone_url,default_branch,
                private,language,topics_json,synced_at,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(repo_key) DO UPDATE SET full_name=excluded.full_name,name=excluded.name,
                description=excluded.description,clone_url=excluded.clone_url,default_branch=excluded.default_branch,
                private=excluded.private,language=excluded.language,topics_json=excluded.topics_json,
                synced_at=excluded.synced_at,updated_at=excluded.updated_at""",
                (
                    key,
                    str(owner),
                    repo_id,
                    installation_id,
                    values["full_name"],
                    values["name"],
                    values["description"],
                    clone_url,
                    values["default_branch"],
                    values["private"],
                    values["language"],
                    json.dumps(values["topics_json"], ensure_ascii=False),
                    values["synced_at"],
                    now,
                    now,
                ),
            )
        return self.get("repos", key)

    def prune_repos(self, owner, installation_id, keep_repo_ids):
        keep = [str(repo_id) for repo_id in keep_repo_ids]
        with self.conn() as c:
            if keep:
                placeholders = ",".join("?" for _ in keep)
                c.execute(
                    f"DELETE FROM spec_repos WHERE owner_id=? AND installation_id=? AND repo_id NOT IN ({placeholders})",
                    (str(owner), str(installation_id), *keep),
                )
            else:
                c.execute(
                    "DELETE FROM spec_repos WHERE owner_id=? AND installation_id=?",
                    (str(owner), str(installation_id)),
                )

    def remember(self, owner, project_id, key, value, *, kind="preference", source="user", source_event=""):
        key = str(key).strip()
        value = str(value).strip()
        if not key or not value:
            raise ValueError("memory key and value required")
        if len(key) > 100 or len(value) > 4000:
            raise ValueError("memory entry too large")
        now = time.time()
        memory_id = "sm_" + uuid.uuid4().hex
        with self.conn() as c:
            previous = c.execute(
                """SELECT memory_id FROM spec_memories
                WHERE owner_id=? AND project_id=? AND key=? AND active=1
                ORDER BY created_at DESC LIMIT 1""",
                (str(owner), str(project_id or ""), key),
            ).fetchone()
            supersedes = previous["memory_id"] if previous else ""
            c.execute(
                """INSERT INTO spec_memories (
                memory_id,owner_id,project_id,kind,key,value,source,source_event,supersedes,active,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,1,?,?)""",
                (
                    memory_id,
                    str(owner),
                    str(project_id or ""),
                    str(kind),
                    key,
                    value,
                    str(source),
                    str(source_event),
                    supersedes,
                    now,
                    now,
                ),
            )
            if previous:
                c.execute(
                    "UPDATE spec_memories SET active=0,superseded_by=?,updated_at=? WHERE memory_id=?",
                    (memory_id, now, supersedes),
                )
        return memory_id

    def memories(self, owner, project_id=None, *, include_history=False):
        where = "owner_id=?"
        args = [str(owner)]
        if project_id is not None:
            where += " AND project_id IN (?,?)"
            args.extend([str(project_id), ""])
        if not include_history:
            where += " AND active=1"
        return self.rows("memories", where, tuple(args))

    def record_event(self, owner, project_id, kind, summary, *, run_id="", cand_id="", head_sha="", payload=None):
        event_id = "se_" + uuid.uuid4().hex
        raw = json.dumps(payload or {}, ensure_ascii=False, default=str)
        if len(raw) > 65536:
            raise ValueError("event payload exceeds 64KiB")
        with self.conn() as c:
            c.execute(
                """INSERT INTO spec_events (
                event_id,owner_id,project_id,run_id,cand_id,head_sha,kind,summary,payload_json,created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    event_id,
                    str(owner),
                    str(project_id),
                    str(run_id),
                    str(cand_id),
                    str(head_sha),
                    str(kind),
                    str(summary)[:4000],
                    raw,
                    time.time(),
                ),
            )
        return event_id

    def events(self, owner, *, project_id="", cand_id="", limit=100):
        where = "owner_id=?"
        args: list = [str(owner)]
        if project_id:
            where += " AND project_id=?"
            args.append(str(project_id))
        if cand_id:
            where += " AND cand_id=?"
            args.append(str(cand_id))
        args.append(max(1, min(500, int(limit))))
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM spec_events WHERE " + where + " ORDER BY created_at ASC LIMIT ?",
                args,
            ).fetchall()
            return [self.decode(row) for row in rows]

    def enqueue_notification(self, dedup_key, owner, session_key, full_text, summary, scheduled_fire_at):
        now = time.time()
        notification_id = "sn_" + uuid.uuid4().hex
        with self.conn() as c:
            try:
                c.execute(
                    """INSERT INTO spec_notifications (
                    notification_id,dedup_key,owner_id,session_key,full_text,summary,status,scheduled_fire_at,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,'pending',?,?,?)""",
                    (
                        notification_id,
                        str(dedup_key),
                        str(owner),
                        str(session_key),
                        str(full_text),
                        str(summary),
                        float(scheduled_fire_at),
                        now,
                        now,
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def grave(self, candidate, reason, source):
        gid = candidate["cand_id"] + "_" + str(candidate["generation"])
        with self.conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO spec_graveyard VALUES (?,?,?,?,?,?,?)",
                (gid, candidate["project_id"], candidate["title"], candidate["route_plan"], source, reason, time.time()),
            )

    def charge(self, project_id, run_id, cand_id, tokens):
        with self.conn() as c:
            c.execute(
                "UPDATE spec_projects SET speculative_tokens=speculative_tokens+? WHERE project_id=?",
                (tokens, project_id),
            )
            if run_id:
                c.execute("UPDATE spec_runs SET tokens_used=tokens_used+? WHERE run_id=?", (tokens, run_id))
            if cand_id:
                c.execute("UPDATE spec_candidates SET tokens_used=tokens_used+? WHERE cand_id=?", (tokens, cand_id))
