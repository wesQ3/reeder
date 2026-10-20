"""SQLite-backed operational store for Reeder jobs, tasks, and workers.

The database is the source of truth for queue state. Filesystem directories
under ``paths["artifacts"]`` and ``paths["logs"]`` hold large payloads; this
module owns their layout and the atomic state transitions around them.

All public methods create their own short-lived SQLite connection. That keeps
the FastAPI thread pool safe without sharing a connection across threads, and
lets the web process and orchestrator share the same database file.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

WRITE_STATES = ("claimed", "running", "uploading")
TASK_STATES = (
    "queued",
    "claimed",
    "running",
    "uploading",
    "completed",
    "failed",
    "expired",
    "canceled",
)


class StoreError(Exception):
    """Base error for store operations."""


class NotFoundError(StoreError):
    pass


class ConflictError(StoreError):
    """Lease mismatch, stale attempt, or another idempotency conflict."""


class ValidationError(StoreError):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 0,
    title           TEXT,
    source_url      TEXT,
    request_json    TEXT NOT NULL,
    pipeline        TEXT NOT NULL,
    error_json      TEXT,
    completed_json  TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id                   TEXT PRIMARY KEY,
    job_id               TEXT NOT NULL REFERENCES jobs(id),
    type                 TEXT NOT NULL,
    sequence             INTEGER NOT NULL,
    status               TEXT NOT NULL,
    executor             TEXT NOT NULL,
    required_caps_json   TEXT NOT NULL,
    depends_on_json      TEXT NOT NULL,
    payload_json         TEXT NOT NULL,
    attempt              INTEGER NOT NULL DEFAULT 0,
    max_attempts         INTEGER NOT NULL DEFAULT 2,
    lease_owner          TEXT,
    lease_token          TEXT,
    lease_expires_at     TEXT,
    last_log_seq         INTEGER NOT NULL DEFAULT 0,
    created_at           TEXT NOT NULL,
    started_at           TEXT,
    finished_at          TEXT,
    error_json           TEXT,
    result_json          TEXT,
    UNIQUE(job_id, sequence)
);

CREATE TABLE IF NOT EXISTS workers (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    capabilities_json TEXT NOT NULL,
    inventory_json    TEXT,
    state             TEXT NOT NULL,
    current_task_id   TEXT,
    last_seen_at      TEXT,
    last_status_json  TEXT
);

CREATE TABLE IF NOT EXISTS task_events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    job_id    TEXT,
    task_id   TEXT,
    worker_id TEXT,
    level     TEXT NOT NULL,
    event     TEXT NOT NULL,
    message   TEXT,
    data_json TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id           TEXT PRIMARY KEY,
    job_id       TEXT NOT NULL,
    task_id      TEXT NOT NULL,
    attempt      INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    filename     TEXT NOT NULL,
    content_type TEXT,
    size         INTEGER NOT NULL,
    sha256       TEXT,
    path         TEXT NOT NULL,
    created_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tasks_claim
    ON tasks(status, executor, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_job
    ON tasks(job_id, sequence);
CREATE INDEX IF NOT EXISTS idx_events_job
    ON task_events(job_id, id);
CREATE INDEX IF NOT EXISTS idx_events_task
    ON task_events(task_id, id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task
    ON artifacts(task_id, attempt);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def utc_after(seconds: float) -> str:
    return (
        datetime.now(timezone.utc) + timedelta(seconds=seconds)
    ).isoformat(timespec="seconds")


def slugify(text: str, max_len: int = 50) -> str:
    text = re.sub(r"https?://", "", text or "")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text)
    return text[:max_len].strip("-") or "job"


def new_task_id() -> str:
    # ULID-like enough for sorting/debugging; only uniqueness is required.
    return f"task_{uuid.uuid4().hex}"


def new_artifact_id() -> str:
    return f"art_{uuid.uuid4().hex}"


def parse_json(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def safe_filename(name: str) -> str:
    """Return a single safe path component for an artifact filename."""
    name = Path(str(name or "")).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    if not name:
        raise ValidationError("Invalid artifact filename")
    return name[:180]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


PIPELINES = {
    "url": ("fetch_extract", "tts", "finalize_audio"),
    "text": ("tts", "finalize_audio"),
}


class Store:
    def __init__(
        self,
        db_path: Path,
        artifacts_dir: Path,
        logs_dir: Path,
        max_artifact_bytes: int = 4 * 1024 * 1024 * 1024,
    ):
        self.db_path = Path(db_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.logs_dir = Path(logs_dir)
        self.max_artifact_bytes = int(max_artifact_bytes)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.init_db()

    # ------------------------------------------------------------------
    # connections / schema
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return dict(row)

    # ------------------------------------------------------------------
    # events / logs
    # ------------------------------------------------------------------
    def add_event(
        self,
        event: str,
        *,
        job_id: str | None = None,
        task_id: str | None = None,
        worker_id: str | None = None,
        level: str = "info",
        message: str | None = None,
        data: Any = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        sql = (
            "INSERT INTO task_events(ts, job_id, task_id, worker_id, level, event,"
            " message, data_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        args = (
            utcnow(),
            job_id,
            task_id,
            worker_id,
            level,
            event,
            message,
            json.dumps(data) if data is not None else None,
        )
        if conn is not None:
            conn.execute(sql, args)
        else:
            with self.connect() as own:
                own.execute(sql, args)
                own.commit()

    def _decode_event(self, row: dict) -> dict:
        row["data"] = parse_json(row.pop("data_json", None), None)
        return row

    def log_path(self, job_id: str, task_id: str, attempt: int) -> Path:
        return self.logs_dir / job_id / task_id / f"attempt-{attempt}.log"

    def append_logs(
        self,
        task_id: str,
        *,
        attempt: int,
        entries: Iterable[dict],
        worker_id: str | None = None,
        lease_token: str | None = None,
    ) -> dict:
        """Append idempotent log entries for one task attempt.

        Each entry must carry a monotonically increasing integer ``seq``. The
        host ignores entries at/below the stored high-water mark. Returns the
        current high-water mark and the number of accepted lines.
        """
        with self.connect() as conn:
            task = self._require_task(conn, task_id)
            if lease_token is not None:
                self._check_lease(task, attempt, lease_token, worker_id)
            entries = list(entries)
            normalized = []
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValidationError("Log entries must be objects")
                try:
                    seq = int(entry.get("seq"))
                except (TypeError, ValueError):
                    raise ValidationError("Log entry seq must be an integer")
                normalized.append(
                    {
                        "seq": seq,
                        "ts": entry.get("ts") or utcnow(),
                        "level": entry.get("level") or "info",
                        "message": str(entry.get("message", "")),
                        "data": entry.get("data"),
                    }
                )
            normalized.sort(key=lambda item: item["seq"])

            last_seq = int(task["last_log_seq"])
            accepted = [item for item in normalized if item["seq"] > last_seq]
            if accepted:
                path = self.log_path(task["job_id"], task_id, attempt)
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    for item in accepted:
                        record = {
                            "ts": item["ts"],
                            "seq": item["seq"],
                            "level": item["level"],
                            "message": item["message"],
                        }
                        if item["data"] is not None:
                            record["data"] = item["data"]
                        f.write(json.dumps(record) + "\n")
                last_seq = accepted[-1]["seq"]
                conn.execute(
                    "UPDATE tasks SET last_log_seq=? WHERE id=?",
                    (last_seq, task_id),
                )
                conn.commit()
            return {"last_seq": last_seq, "accepted": len(accepted)}

    def read_log(
        self, job_id: str, task_id: str, attempt: int, tail: int | None = None
    ) -> str:
        path = self.log_path(job_id, task_id, attempt)
        if not path.exists():
            return ""
        if tail and tail > 0:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-tail:])
        return path.read_text(encoding="utf-8", errors="replace")

    def job_logs(self, job_id: str, tail: int = 200) -> str:
        tasks = self.get_job_tasks(job_id)
        sections = []
        for task in tasks:
            attempts = range(1, max(1, task["attempt"]) + 1)
            for attempt in attempts:
                text = self.read_log(job_id, task["id"], attempt)
                if text:
                    sections.append(
                        f"=== {task['type']} {task['id']} attempt {attempt} ===\n{text}"
                    )
        if not sections:
            return ""
        combined = "\n".join(sections).splitlines(keepends=True)
        return "".join(combined[-tail:])

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------
    def create_job(self, request: dict, *, pipeline: str | None = None) -> dict:
        if not isinstance(request, dict):
            raise ValidationError("Job request must be an object")
        job_type = request.get("type", "url")
        if job_type not in PIPELINES:
            raise ValidationError(f"Unsupported job type: {job_type}")
        if job_type == "url" and not request.get("url"):
            raise ValidationError("URL is required")
        if job_type == "text" and not request.get("text"):
            raise ValidationError("Text is required")

        stamp = int(datetime.now(timezone.utc).timestamp())
        title = request.get("title") or request.get("url") or "text"
        base_id = f"{stamp}-{slugify(str(title))}"
        pipeline_name = pipeline or job_type

        with self.connect() as conn:
            job_id = base_id
            suffix = 2
            while conn.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                job_id = f"{base_id}-{suffix}"
                suffix += 1
            now = utcnow()
            conn.execute(
                "INSERT INTO jobs(id, created_at, updated_at, status, priority,"
                " title, source_url, request_json, pipeline)"
                " VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?)",
                (
                    job_id,
                    now,
                    now,
                    int(request.get("priority", 0)),
                    request.get("title"),
                    request.get("url"),
                    json.dumps(request),
                    pipeline_name,
                ),
            )
            conn.commit()
            self.add_event(
                "job_created",
                job_id=job_id,
                message=f"Created {pipeline_name} job",
                data={"request": request},
                conn=conn,
            )
            self._materialize_initial_task(conn, job_id, request, pipeline_name)
            conn.commit()
            return self.get_job(job_id)

    def _materialize_initial_task(
        self,
        conn: sqlite3.Connection,
        job_id: str,
        request: dict,
        pipeline: str,
    ) -> None:
        steps = PIPELINES.get(pipeline, ())
        if not steps:
            return
        first = steps[0]
        if first == "fetch_extract":
            self._insert_task(
                conn,
                job_id=job_id,
                task_type="fetch_extract",
                executor="host",
                required_caps=[],
                depends_on=[],
                payload={"url": request.get("url")},
                sequence=1,
            )
        # Text jobs have no materialized task yet: the orchestrator prepares
        # the tokenizer-dependent input artifact before creating the tts task.

    def get_job(self, job_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"Job not found: {job_id}")
            job = self._decode_job(dict(row))
            job["tasks"] = [
                self._decode_task(dict(r))
                for r in conn.execute(
                    "SELECT * FROM tasks WHERE job_id=? ORDER BY sequence", (job_id,)
                ).fetchall()
            ]
            job["artifacts"] = [
                self._decode_artifact(dict(r))
                for r in conn.execute(
                    "SELECT * FROM artifacts WHERE job_id=? ORDER BY created_at",
                    (job_id,),
                ).fetchall()
            ]
            job["events"] = [
                self._decode_event(dict(r))
                for r in conn.execute(
                    "SELECT * FROM task_events WHERE job_id=? ORDER BY id", (job_id,)
                ).fetchall()
            ]
            return job

    def list_jobs(self, limit: int = 50, status: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 500))
        sql = "SELECT * FROM jobs"
        args: list[Any] = []
        if status:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        with self.connect() as conn:
            return [
                self._decode_job(dict(r))
                for r in conn.execute(sql, args).fetchall()
            ]

    def get_events(self, job_id: str, limit: int = 500) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM task_events WHERE job_id=? ORDER BY id DESC LIMIT ?",
                (job_id, max(1, min(int(limit), 5000))),
            ).fetchall()
        return [self._decode_event(dict(r)) for r in reversed(rows)]

    def update_job_status(
        self,
        job_id: str,
        status: str,
        *,
        error: Any = None,
        completed: Any = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        def apply(c: sqlite3.Connection):
            c.execute(
                "UPDATE jobs SET status=?, updated_at=?, error_json=?,"
                " completed_json=COALESCE(?, completed_json) WHERE id=?",
                (
                    status,
                    utcnow(),
                    json.dumps(error) if error is not None else None,
                    json.dumps(completed) if completed is not None else None,
                    job_id,
                ),
            )

        if conn is not None:
            apply(conn)
        else:
            with self.connect() as own:
                apply(own)
                own.commit()

    def _decode_job(self, row: dict) -> dict:
        row["request"] = parse_json(row.pop("request_json", None), {})
        row["error"] = parse_json(row.pop("error_json", None), None)
        row["completed"] = parse_json(row.pop("completed_json", None), None)
        return row

    # ------------------------------------------------------------------
    # tasks
    # ------------------------------------------------------------------
    def _insert_task(
        self,
        conn: sqlite3.Connection,
        *,
        job_id: str,
        task_type: str,
        executor: str,
        required_caps: list[str],
        depends_on: list[str],
        payload: dict,
        sequence: int | None = None,
        max_attempts: int = 2,
    ) -> dict:
        if sequence is None:
            row = conn.execute(
                "SELECT COALESCE(MAX(sequence), 0) AS seq FROM tasks WHERE job_id=?",
                (job_id,),
            ).fetchone()
            sequence = int(row["seq"]) + 1
        task_id = new_task_id()
        conn.execute(
            "INSERT INTO tasks(id, job_id, type, sequence, status, executor,"
            " required_caps_json, depends_on_json, payload_json, attempt,"
            " max_attempts, created_at) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?,"
            " ?, 0, ?, ?)",
            (
                task_id,
                job_id,
                task_type,
                sequence,
                executor,
                json.dumps(required_caps),
                json.dumps(depends_on),
                json.dumps(payload),
                max_attempts,
                utcnow(),
            ),
        )
        self.add_event(
            "task_queued",
            job_id=job_id,
            task_id=task_id,
            message=f"Queued {task_type}",
            data={"executor": executor, "sequence": sequence},
            conn=conn,
        )
        return self._require_task(conn, task_id)

    def create_task(
        self,
        job_id: str,
        task_type: str,
        *,
        executor: str,
        required_caps: list[str] | None = None,
        depends_on: list[str] | None = None,
        payload: dict | None = None,
        sequence: int | None = None,
        max_attempts: int = 2,
    ) -> dict:
        with self.connect() as conn:
            if not conn.execute("SELECT 1 FROM jobs WHERE id=?", (job_id,)).fetchone():
                raise NotFoundError(f"Job not found: {job_id}")
            task = self._insert_task(
                conn,
                job_id=job_id,
                task_type=task_type,
                executor=executor,
                required_caps=required_caps or [],
                depends_on=depends_on or [],
                payload=payload or {},
                sequence=sequence,
                max_attempts=max_attempts,
            )
            conn.commit()
            return self._decode_task(dict(task))

    def update_task_payload(self, task_id: str, payload: dict) -> dict:
        with self.connect() as conn:
            self._require_task(conn, task_id)
            conn.execute(
                "UPDATE tasks SET payload_json=? WHERE id=?",
                (json.dumps(payload), task_id),
            )
            conn.commit()
            return self._decode_task(dict(self._require_task(conn, task_id)))

    def get_task(self, task_id: str) -> dict:
        with self.connect() as conn:
            return self._decode_task(dict(self._require_task(conn, task_id)))

    def get_job_tasks(self, job_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tasks WHERE job_id=? ORDER BY sequence", (job_id,)
            ).fetchall()
        return [self._decode_task(dict(r)) for r in rows]

    def task_by_type(self, job_id: str, task_type: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE job_id=? AND type=?"
                " ORDER BY sequence DESC LIMIT 1",
                (job_id, task_type),
            ).fetchone()
        return self._decode_task(dict(row)) if row else None

    def _require_task(self, conn: sqlite3.Connection, task_id: str) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Task not found: {task_id}")
        return row

    def _decode_task(self, row: dict) -> dict:
        row["required_caps"] = parse_json(row.pop("required_caps_json", None), [])
        row["depends_on"] = parse_json(row.pop("depends_on_json", None), [])
        row["payload"] = parse_json(row.pop("payload_json", None), {})
        row["error"] = parse_json(row.pop("error_json", None), None)
        row["result"] = parse_json(row.pop("result_json", None), None)
        return row

    def _dependencies_complete(
        self, conn: sqlite3.Connection, depends_on: list[str]
    ) -> bool:
        for dep_id in depends_on:
            row = conn.execute(
                "SELECT status FROM tasks WHERE id=?", (dep_id,)
            ).fetchone()
            if row is None or row["status"] != "completed":
                return False
        return True

    def claim_task(
        self,
        worker_id: str,
        *,
        capabilities: Iterable[str] = (),
        executor: str = "worker",
        lease_seconds: int = 180,
        job_id: str | None = None,
    ) -> dict | None:
        """Atomically claim one eligible task, or return None.

        Capability matching uses the worker's advertised capability set. An
        empty task requirement is always eligible.
        """
        caps = set(capabilities)
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                sql = (
                    "SELECT tasks.*, jobs.priority AS job_priority FROM tasks"
                    " JOIN jobs ON jobs.id=tasks.job_id"
                    " WHERE tasks.status='queued' AND tasks.executor=?"
                )
                args: list[Any] = [executor]
                if job_id:
                    sql += " AND tasks.job_id=?"
                    args.append(job_id)
                sql += (
                    " ORDER BY jobs.priority DESC, tasks.created_at ASC,"
                    " tasks.sequence ASC LIMIT 100"
                )
                rows = conn.execute(sql, args).fetchall()
                selected = None
                for row in rows:
                    required = set(parse_json(row["required_caps_json"], []))
                    if not required.issubset(caps):
                        continue
                    deps = parse_json(row["depends_on_json"], [])
                    if not self._dependencies_complete(conn, deps):
                        continue
                    selected = row
                    break
                if selected is None:
                    conn.rollback()
                    return None

                attempt = int(selected["attempt"]) + 1
                token = uuid.uuid4().hex
                expires = utc_after(lease_seconds)
                now = utcnow()
                updated = conn.execute(
                    "UPDATE tasks SET status='claimed', attempt=?, lease_owner=?,"
                    " lease_token=?, lease_expires_at=?, started_at=?,"
                    " last_log_seq=0 WHERE id=? AND status='queued'",
                    (attempt, worker_id, token, expires, now, selected["id"]),
                )
                if updated.rowcount != 1:
                    conn.rollback()
                    return None
                conn.execute(
                    "UPDATE jobs SET status='running', updated_at=?"
                    " WHERE id=? AND status='queued'",
                    (now, selected["job_id"]),
                )
                self.add_event(
                    "task_claimed",
                    job_id=selected["job_id"],
                    task_id=selected["id"],
                    worker_id=worker_id,
                    message=f"Claimed {selected['type']} attempt {attempt}",
                    data={"lease_expires_at": expires},
                    conn=conn,
                )
                conn.commit()
                row = self._require_task(conn, selected["id"])
                task = self._decode_task(dict(row))
                task["lease_token"] = token
                task["heartbeat_interval_seconds"] = max(
                    5, min(30, int(lease_seconds) // 3)
                )
                return task
            except Exception:
                conn.rollback()
                raise

    def claim_next_host_task(self, lease_seconds: int = 300) -> dict | None:
        return self.claim_task(
            "host", capabilities=[], executor="host", lease_seconds=lease_seconds
        )

    def _check_lease(
        self,
        task: sqlite3.Row | dict,
        attempt: int,
        lease_token: str,
        worker_id: str | None = None,
    ) -> None:
        if int(task["attempt"]) != int(attempt):
            raise ConflictError(
                f"Stale attempt: task is at {task['attempt']}, got {attempt}"
            )
        if task["lease_token"] != lease_token:
            raise ConflictError("Stale lease token")
        if worker_id is not None and task["lease_owner"] != worker_id:
            raise ConflictError("Task is owned by another worker")
        if task["status"] not in WRITE_STATES:
            raise ConflictError(f"Task is not writable in state {task['status']}")
        expires = task["lease_expires_at"]
        if not expires or expires < utcnow():
            raise ConflictError("Lease has expired")

    def heartbeat_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        state: str | None = None,
        progress: dict | None = None,
        message: str | None = None,
        metrics: dict | None = None,
        lease_seconds: int = 180,
    ) -> dict:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._require_task(conn, task_id)
                self._check_lease(task, attempt, lease_token, worker_id)
                expires = utc_after(lease_seconds)
                new_state = state if state in WRITE_STATES else task["status"]
                data = {
                    "progress": progress,
                    "metrics": metrics,
                    "message": message,
                }
                conn.execute(
                    "UPDATE tasks SET status=?, lease_expires_at=? WHERE id=?",
                    (new_state, expires, task_id),
                )
                conn.execute(
                    "UPDATE workers SET state='busy', current_task_id=?,"
                    " last_seen_at=?, last_status_json=? WHERE id=?",
                    (
                        task_id,
                        utcnow(),
                        json.dumps(data),
                        worker_id,
                    ),
                )
                self.add_event(
                    "task_progress",
                    job_id=task["job_id"],
                    task_id=task_id,
                    worker_id=worker_id,
                    message=message,
                    data=data,
                    conn=conn,
                )
                cancel_requested = self._is_cancel_requested(conn, task)
                conn.commit()
                return {
                    "task_id": task_id,
                    "attempt": attempt,
                    "lease_expires_at": expires,
                    "cancel_requested": cancel_requested,
                    "heartbeat_interval_seconds": max(
                        5, min(30, int(lease_seconds) // 3)
                    ),
                }
            except Exception:
                conn.rollback()
                raise

    def _is_cancel_requested(self, conn: sqlite3.Connection, task) -> bool:
        job = conn.execute(
            "SELECT status FROM jobs WHERE id=?", (task["job_id"],)
        ).fetchone()
        return bool(job and job["status"] == "canceled")

    def complete_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        result: dict | None = None,
    ) -> dict:
        """Mark a task completed. Safe to call twice with the same attempt."""
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._require_task(conn, task_id)
                if task["status"] == "completed":
                    if int(task["attempt"]) == int(attempt):
                        conn.commit()
                        return self._decode_task(dict(task))
                    raise ConflictError("Task already completed by another attempt")
                self._check_lease(task, attempt, lease_token, worker_id)
                now = utcnow()
                conn.execute(
                    "UPDATE tasks SET status='completed', finished_at=?,"
                    " result_json=?, lease_expires_at=NULL WHERE id=?",
                    (now, json.dumps(result or {}), task_id),
                )
                conn.execute(
                    "UPDATE jobs SET updated_at=? WHERE id=?",
                    (now, task["job_id"]),
                )
                conn.execute(
                    "UPDATE workers SET state='idle', current_task_id=NULL,"
                    " last_seen_at=? WHERE id=?",
                    (now, worker_id),
                )
                self.add_event(
                    "task_completed",
                    job_id=task["job_id"],
                    task_id=task_id,
                    worker_id=worker_id,
                    message=f"Completed {task['type']}",
                    data=result or {},
                    conn=conn,
                )
                conn.commit()
                return self._decode_task(dict(self._require_task(conn, task_id)))
            except Exception:
                conn.rollback()
                raise

    def fail_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        error: str | dict,
        retryable: bool = True,
    ) -> dict:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._require_task(conn, task_id)
                if task["status"] == "failed" and int(task["attempt"]) == int(attempt):
                    conn.commit()
                    return self._decode_task(dict(task))
                self._check_lease(task, attempt, lease_token, worker_id)
                self._finish_task_failure(
                    conn, task, error, retryable=retryable, worker_id=worker_id
                )
                conn.commit()
                return self._decode_task(dict(self._require_task(conn, task_id)))
            except Exception:
                conn.rollback()
                raise

    def release_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        reason: str | None = None,
    ) -> dict:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._require_task(conn, task_id)
                self._check_lease(task, attempt, lease_token, worker_id)
                conn.execute(
                    "UPDATE tasks SET status='queued', attempt=attempt-1,"
                    " lease_owner=NULL, lease_token=NULL, lease_expires_at=NULL,"
                    " last_log_seq=0 WHERE id=?",
                    (task_id,),
                )
                self.add_event(
                    "task_released",
                    job_id=task["job_id"],
                    task_id=task_id,
                    worker_id=worker_id,
                    message=reason or "Released by worker",
                    conn=conn,
                )
                conn.commit()
                return self._decode_task(dict(self._require_task(conn, task_id)))
            except Exception:
                conn.rollback()
                raise

    def _finish_task_failure(
        self,
        conn: sqlite3.Connection,
        task: sqlite3.Row,
        error: Any,
        *,
        retryable: bool,
        worker_id: str | None = None,
    ) -> None:
        attempt = int(task["attempt"])
        max_attempts = int(task["max_attempts"])
        error_data = error if isinstance(error, dict) else {"message": str(error)}
        now = utcnow()
        if retryable and attempt < max_attempts:
            conn.execute(
                "UPDATE tasks SET status='queued', lease_owner=NULL,"
                " lease_token=NULL, lease_expires_at=NULL, last_log_seq=0,"
                " error_json=? WHERE id=?",
                (json.dumps(error_data), task["id"]),
            )
            conn.execute(
                "UPDATE jobs SET updated_at=? WHERE id=?", (now, task["job_id"])
            )
            self.add_event(
                "task_retry",
                job_id=task["job_id"],
                task_id=task["id"],
                worker_id=worker_id,
                level="warning",
                message=f"Attempt {attempt} failed; requeued",
                data=error_data,
                conn=conn,
            )
        else:
            conn.execute(
                "UPDATE tasks SET status='failed', finished_at=?, lease_expires_at=NULL,"
                " error_json=? WHERE id=?",
                (now, json.dumps(error_data), task["id"]),
            )
            conn.execute(
                "UPDATE jobs SET status='failed', updated_at=?, error_json=?"
                " WHERE id=? AND status!='canceled'",
                (now, json.dumps(error_data), task["job_id"]),
            )
            self.add_event(
                "task_failed",
                job_id=task["job_id"],
                task_id=task["id"],
                worker_id=worker_id,
                level="error",
                message=f"Failed {task['type']}",
                data=error_data,
                conn=conn,
            )

    def reap_expired(self) -> list[dict]:
        """Expire stale leases according to retry policy."""
        now = utcnow()
        reaped = []
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                rows = conn.execute(
                    "SELECT * FROM tasks WHERE status IN ('claimed','running','uploading')"
                    " AND lease_expires_at IS NOT NULL AND lease_expires_at < ?",
                    (now,),
                ).fetchall()
                for task in rows:
                    conn.execute(
                        "UPDATE tasks SET lease_owner=NULL, lease_token=NULL,"
                        " lease_expires_at=NULL WHERE id=?",
                        (task["id"],),
                    )
                    self.add_event(
                        "task_expired",
                        job_id=task["job_id"],
                        task_id=task["id"],
                        worker_id=task["lease_owner"],
                        level="warning",
                        message=f"Lease expired at attempt {task['attempt']}",
                        conn=conn,
                    )
                    self._finish_task_failure(
                        conn,
                        task,
                        {"message": "Lease expired before completion"},
                        retryable=True,
                        worker_id=task["lease_owner"],
                    )
                    reaped.append(self._decode_task(dict(self._require_task(conn, task["id"]))))
                conn.commit()
                return reaped
            except Exception:
                conn.rollback()
                raise

    def cancel_job(self, job_id: str) -> dict:
        with self.connect() as conn:
            job = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None:
                raise NotFoundError(f"Job not found: {job_id}")
            conn.execute(
                "UPDATE jobs SET status='canceled', updated_at=? WHERE id=?",
                (utcnow(), job_id),
            )
            conn.execute(
                "UPDATE tasks SET status='canceled', finished_at=?,"
                " lease_expires_at=NULL WHERE job_id=? AND status='queued'",
                (utcnow(), job_id),
            )
            self.add_event(
                "job_canceled", job_id=job_id, message="Canceled by operator", conn=conn
            )
            conn.commit()
        return self.get_job(job_id)

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------
    def artifact_dir(self, job_id: str, task_id: str, attempt: int) -> Path:
        return self.artifacts_dir / job_id / task_id / f"attempt-{attempt}"

    def _decode_artifact(self, row: dict) -> dict:
        row["url"] = f"/api/v1/artifacts/{row['id']}/content"
        return row

    def get_artifact(self, artifact_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM artifacts WHERE id=?", (artifact_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Artifact not found: {artifact_id}")
            return self._decode_artifact(dict(row))

    def list_task_artifacts(self, task_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM artifacts WHERE task_id=? ORDER BY created_at",
                (task_id,),
            ).fetchall()
        return [self._decode_artifact(dict(r)) for r in rows]

    def init_artifact(
        self,
        task_id: str,
        *,
        worker_id: str,
        attempt: int,
        lease_token: str,
        filename: str,
        size: int,
        sha256: str | None = None,
        content_type: str | None = None,
        kind: str = "other",
    ) -> dict:
        filename = safe_filename(filename)
        try:
            size = int(size)
        except (TypeError, ValueError):
            raise ValidationError("Artifact size must be an integer")
        if size < 0 or size > self.max_artifact_bytes:
            raise ValidationError(
                f"Artifact size out of range (max {self.max_artifact_bytes} bytes)"
            )

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                task = self._require_task(conn, task_id)
                self._check_lease(task, attempt, lease_token, worker_id)
                existing = conn.execute(
                    "SELECT * FROM artifacts WHERE task_id=? AND attempt=?"
                    " AND filename=? AND kind=?",
                    (task_id, attempt, filename, kind),
                ).fetchone()
                if existing:
                    conn.commit()
                    item = self._decode_artifact(dict(existing))
                    existing_final = self.artifacts_dir / item["path"]
                    existing_part = self.artifacts_dir / (item["path"] + ".part")
                    if existing_final.exists():
                        item["uploaded_size"] = existing_final.stat().st_size
                        item["offset"] = int(item["size"])
                    else:
                        item["uploaded_size"] = (
                            existing_part.stat().st_size if existing_part.exists() else 0
                        )
                        item["offset"] = item["uploaded_size"]
                    return item

                artifact_id = new_artifact_id()
                rel = (
                    Path(task["job_id"])
                    / task_id
                    / f"attempt-{attempt}"
                    / f"{artifact_id}-{filename}"
                )
                part = self.artifacts_dir / (str(rel) + ".part")
                part.parent.mkdir(parents=True, exist_ok=True)
                part.touch()
                conn.execute(
                    "INSERT INTO artifacts(id, job_id, task_id, attempt, kind,"
                    " filename, content_type, size, sha256, path, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        artifact_id,
                        task["job_id"],
                        task_id,
                        attempt,
                        kind,
                        filename,
                        content_type,
                        size,
                        sha256,
                        str(rel),
                        utcnow(),
                    ),
                )
                self.add_event(
                    "artifact_started",
                    job_id=task["job_id"],
                    task_id=task_id,
                    worker_id=worker_id,
                    message=f"Uploading {filename}",
                    data={"artifact_id": artifact_id, "size": size},
                    conn=conn,
                )
                conn.commit()
                item = self.get_artifact(artifact_id)
                item["offset"] = 0
                item["uploaded_size"] = part.stat().st_size
                return item
            except Exception:
                conn.rollback()
                raise

    def upload_artifact(
        self, artifact_id: str, *, offset: int, data: bytes
    ) -> dict:
        artifact = self.get_artifact(artifact_id)
        part = self.artifacts_dir / (artifact["path"] + ".part")
        try:
            offset = int(offset)
        except (TypeError, ValueError):
            raise ValidationError("Offset must be an integer")
        if offset < 0:
            raise ValidationError("Offset must be non-negative")
        part.parent.mkdir(parents=True, exist_ok=True)
        current = part.stat().st_size if part.exists() else 0
        if offset > current:
            raise ConflictError(
                f"Offset {offset} is beyond uploaded size {current}"
            )
        if offset < current:
            # A retry of already-written bytes is safe only if it matches.
            with open(part, "rb") as f:
                f.seek(offset)
                existing = f.read(len(data))
            if existing != data:
                raise ConflictError("Retried chunk differs from stored bytes")
        else:
            with open(part, "ab") as f:
                f.write(data)
        new_size = part.stat().st_size
        if new_size > int(artifact["size"]):
            raise ValidationError("Artifact exceeds declared size")
        return {"next_offset": new_size, "uploaded_size": new_size}

    def complete_artifact(self, artifact_id: str) -> dict:
        artifact = self.get_artifact(artifact_id)
        part = self.artifacts_dir / (artifact["path"] + ".part")
        final = self.artifacts_dir / artifact["path"]
        if final.exists():
            return self.get_artifact(artifact_id)
        if not part.exists():
            raise NotFoundError("Artifact upload was not started")
        actual_size = part.stat().st_size
        if actual_size != int(artifact["size"]):
            raise ValidationError(
                f"Size mismatch: expected {artifact['size']}, got {actual_size}"
            )
        actual_hash = _sha256_file(part)
        if artifact["sha256"] and artifact["sha256"] != actual_hash:
            raise ValidationError(
                f"SHA-256 mismatch: expected {artifact['sha256']}, got {actual_hash}"
            )
        os.replace(part, final)
        with self.connect() as conn:
            conn.execute(
                "UPDATE artifacts SET size=?, sha256=? WHERE id=?",
                (actual_size, actual_hash, artifact_id),
            )
            conn.commit()
        self.add_event(
            "artifact_completed",
            job_id=artifact["job_id"],
            task_id=artifact["task_id"],
            message=f"Uploaded {artifact['filename']}",
            data={"artifact_id": artifact_id, "size": actual_size, "sha256": actual_hash},
        )
        return self.get_artifact(artifact_id)

    def create_artifact_bytes(
        self,
        task_id: str,
        *,
        filename: str,
        data: bytes,
        kind: str = "other",
        content_type: str | None = None,
        result_name: str | None = None,
    ) -> dict:
        """Create a completed host-side artifact directly."""
        filename = safe_filename(filename)
        task = self.get_task(task_id)
        attempt = max(1, int(task["attempt"]))
        artifact_id = new_artifact_id()
        rel = (
            Path(task["job_id"])
            / task_id
            / f"attempt-{attempt}"
            / f"{artifact_id}-{filename}"
        )
        path = self.artifacts_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO artifacts(id, job_id, task_id, attempt, kind, filename,"
                " content_type, size, sha256, path, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    task["job_id"],
                    task_id,
                    attempt,
                    kind,
                    filename,
                    content_type,
                    len(data),
                    digest,
                    str(rel),
                    utcnow(),
                ),
            )
            conn.commit()
        self.add_event(
            "artifact_created",
            job_id=task["job_id"],
            task_id=task_id,
            message=f"Created {filename}",
            data={"artifact_id": artifact_id, "size": len(data), "sha256": digest},
        )
        return self.get_artifact(artifact_id)

    def artifact_content_path(self, artifact_id: str) -> Path:
        artifact = self.get_artifact(artifact_id)
        path = (self.artifacts_dir / artifact["path"]).resolve()
        root = self.artifacts_dir.resolve()
        if root not in path.parents and path != root:
            raise ValidationError("Artifact path escapes artifact root")
        if not path.exists():
            raise NotFoundError("Artifact content is not available")
        return path

    # ------------------------------------------------------------------
    # workers / status
    # ------------------------------------------------------------------
    def register_worker(
        self,
        *,
        worker_id: str,
        name: str,
        capabilities: Iterable[str],
        inventory: dict | None = None,
        version: str | None = None,
    ) -> dict:
        now = utcnow()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT id FROM workers WHERE id=?", (worker_id,)
            ).fetchone()
            status = {"version": version, "registered_at": now}
            if existing:
                conn.execute(
                    "UPDATE workers SET name=?, capabilities_json=?,"
                    " inventory_json=?, state='idle', last_seen_at=?,"
                    " last_status_json=? WHERE id=?",
                    (
                        name,
                        json.dumps(sorted(set(capabilities))),
                        json.dumps(inventory or {}),
                        now,
                        json.dumps(status),
                        worker_id,
                    ),
                )
            else:
                conn.execute(
                    "INSERT INTO workers(id, name, capabilities_json, inventory_json,"
                    " state, current_task_id, last_seen_at, last_status_json)"
                    " VALUES (?, ?, ?, ?, 'idle', NULL, ?, ?)",
                    (
                        worker_id,
                        name,
                        json.dumps(sorted(set(capabilities))),
                        json.dumps(inventory or {}),
                        now,
                        json.dumps(status),
                    ),
                )
            conn.commit()
        self.add_event(
            "worker_registered",
            worker_id=worker_id,
            message=f"Worker {name} registered",
            data={"capabilities": sorted(set(capabilities)), "inventory": inventory},
        )
        return self.get_worker(worker_id) or {}

    def get_worker(self, worker_id: str) -> dict | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE id=?", (worker_id,)
            ).fetchone()
        return self._decode_worker(dict(row)) if row else None

    def worker_heartbeat(
        self,
        *,
        worker_id: str,
        state: str = "idle",
        current_task_id: str | None = None,
        status: dict | None = None,
    ) -> dict:
        now = utcnow()
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM workers WHERE id=?", (worker_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"Worker not registered: {worker_id}")
            conn.execute(
                "UPDATE workers SET state=?, current_task_id=?, last_seen_at=?,"
                " last_status_json=? WHERE id=?",
                (
                    state,
                    current_task_id,
                    now,
                    json.dumps(status or {}),
                    worker_id,
                ),
            )
            conn.commit()
        return self.get_worker(worker_id) or {}

    def list_workers(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM workers ORDER BY last_seen_at DESC"
            ).fetchall()
        return [self._decode_worker(dict(r)) for r in rows]

    def _decode_worker(self, row: dict) -> dict:
        row["capabilities"] = parse_json(row.pop("capabilities_json", None), [])
        row["inventory"] = parse_json(row.pop("inventory_json", None), {})
        row["status"] = parse_json(row.pop("last_status_json", None), {})
        last_seen = row.get("last_seen_at")
        online = False
        if last_seen:
            try:
                seen = datetime.fromisoformat(last_seen)
                online = (
                    datetime.now(timezone.utc) - seen
                ).total_seconds() < 60
            except ValueError:
                online = False
        row["online"] = online
        return row

    def status(self) -> dict:
        with self.connect() as conn:
            counts = {
                row["status"]: row["n"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
                ).fetchall()
            }
            task_counts = {
                row["status"]: row["n"]
                for row in conn.execute(
                    "SELECT status, COUNT(*) AS n FROM tasks GROUP BY status"
                ).fetchall()
            }
            active = conn.execute(
                "SELECT tasks.*, jobs.title FROM tasks JOIN jobs ON jobs.id=tasks.job_id"
                " WHERE tasks.status IN ('claimed','running','uploading')"
                " ORDER BY tasks.started_at DESC LIMIT 1"
            ).fetchone()
        workers = self.list_workers()
        return {
            "jobs": counts,
            "tasks": task_counts,
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "completed": counts.get("completed", 0),
            "failed": counts.get("failed", 0),
            "workers": workers,
            "online_workers": [w for w in workers if w["online"]],
            "active_task": self._decode_task(dict(active)) if active else None,
            "updated_at": utcnow(),
        }

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def task_input_artifact(self, task_id: str) -> dict:
        task = self.get_task(task_id)
        artifact_id = task.get("payload", {}).get("input_artifact")
        if not artifact_id:
            raise NotFoundError("Task has no input artifact")
        return self.get_artifact(artifact_id)

    def wait_for_task(
        self,
        worker_id: str,
        *,
        capabilities: Iterable[str],
        lease_seconds: int,
        wait_seconds: int,
        executor: str = "worker",
    ) -> dict | None:
        deadline = time.monotonic() + max(0, wait_seconds)
        while True:
            task = self.claim_task(
                worker_id,
                capabilities=capabilities,
                executor=executor,
                lease_seconds=lease_seconds,
            )
            if task is not None:
                return task
            if time.monotonic() >= deadline:
                return None
            time.sleep(min(1.0, max(0.1, deadline - time.monotonic())))


def store_from_config(config: dict | None = None) -> Store:
    """Build a Store from a Reeder config dictionary."""
    if config is None:
        from reeder.config import load_config

        config = load_config()
    from reeder.config import get_paths

    paths = get_paths(config)
    worker_api = config.get("jobs", {}).get("worker_api", {})
    max_artifact_bytes = int(worker_api.get("max_artifact_mb", 4096)) * 1024 * 1024
    return Store(
        paths["db"],
        paths["artifacts"],
        paths["logs"],
        max_artifact_bytes=max_artifact_bytes,
    )
