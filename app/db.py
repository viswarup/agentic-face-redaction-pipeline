"""Database access and atomic idempotency management using SQLite WAL mode."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator, Optional, Tuple

from app.config import config


def get_db_path(db_path: Optional[str | Path] = None) -> Path:
    """Resolve database path, falling back to configured default."""
    if db_path is None:
        target = config.JOBS_DB_PATH
    else:
        target = Path(db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@contextmanager
def get_connection(db_path: Optional[str | Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for SQLite connections with WAL mode enabled."""
    path = get_db_path(db_path)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[str | Path] = None) -> None:
    """Initialize SQLite database with schema and indexes."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                idempotency_key TEXT UNIQUE NOT NULL,
                event_id TEXT NOT NULL,
                file_sha256 TEXT NOT NULL,
                input_path TEXT,
                output_path TEXT,
                status TEXT NOT NULL,
                failure_reason TEXT,
                flagged_frames_json TEXT DEFAULT '[]',
                telemetry_json TEXT DEFAULT '{}',
                duplicate_request_count INTEGER DEFAULT 0,
                processing_fps REAL DEFAULT 0.0,
                reviewer_notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_idempotency_key ON jobs(idempotency_key);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);"
        )


def compute_idempotency_key(event_id: str, file_sha256: str) -> str:
    """Compute deterministic idempotency key from event_id and file_sha256."""
    raw = f"{event_id}:{file_sha256}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_or_get_job(
    event_id: str,
    file_sha256: str,
    input_path: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> Tuple[dict[str, Any], bool]:
    """
    Atomically retrieve an existing job or insert a new job in a thread-safe / multi-process safe manner.

    Returns:
        (job_dict, is_new):
            is_new is True if a new job row was created.
            is_new is False if an existing job was found and duplicate_request_count incremented.
    """
    init_db(db_path)
    idempotency_key = compute_idempotency_key(event_id, file_sha256)
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        # Use BEGIN IMMEDIATE to prevent concurrent write collisions
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
        existing = cursor.fetchone()

        if existing:
            cursor.execute(
                """
                UPDATE jobs
                SET duplicate_request_count = duplicate_request_count + 1,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (now_iso, idempotency_key),
            )
            cursor.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
            row = cursor.fetchone()
            conn.commit()
            return dict(row), False

        job_id = f"job_{uuid.uuid4().hex[:12]}"
        try:
            cursor.execute(
                """
                INSERT INTO jobs (
                    job_id, idempotency_key, event_id, file_sha256,
                    input_path, output_path, status, failure_reason,
                    flagged_frames_json, telemetry_json, duplicate_request_count,
                    processing_fps, reviewer_notes, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    idempotency_key,
                    event_id,
                    file_sha256,
                    input_path,
                    None,
                    "QUEUED",
                    None,
                    "[]",
                    "{}",
                    0,
                    0.0,
                    None,
                    now_iso,
                    now_iso,
                ),
            )
            cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
            new_row = cursor.fetchone()
            conn.commit()
            return dict(new_row), True
        except sqlite3.IntegrityError:
            # Handle collision if two concurrent transactions raced
            cursor.execute(
                """
                UPDATE jobs
                SET duplicate_request_count = duplicate_request_count + 1,
                    updated_at = ?
                WHERE idempotency_key = ?
                """,
                (now_iso, idempotency_key),
            )
            cursor.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,))
            existing_after_race = cursor.fetchone()
            conn.commit()
            return dict(existing_after_race), False


def get_job(job_id: str, db_path: Optional[str | Path] = None) -> Optional[dict[str, Any]]:
    """Retrieve a job by job_id."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def list_jobs(db_path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """List all jobs ordered by created_at descending."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        return [dict(row) for row in cursor.fetchall()]


def json_default(obj: Any) -> Any:
    """Fallback JSON serializer handling numpy scalars, arrays, and Path objects."""
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def safe_json_dumps(obj: Any) -> str:
    """Serialize object to JSON string handling all numpy types."""
    return json.dumps(obj, default=json_default)


def update_job_status(
    job_id: str,
    status: Optional[str] = None,
    output_path: Optional[str] = None,
    failure_reason: Optional[str] = None,
    flagged_frames: Optional[list[Any]] = None,
    telemetry: Optional[dict[str, Any]] = None,
    processing_fps: Optional[float] = None,
    db_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """Update processing status and telemetry of an existing job."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None

        current = dict(row)
        new_status = status if status is not None else current.get("status", "QUEUED")
        new_output = output_path if output_path is not None else current.get("output_path")
        new_failure = failure_reason if failure_reason is not None else current.get("failure_reason")
        new_frames = (
            safe_json_dumps(flagged_frames)
            if flagged_frames is not None
            else current.get("flagged_frames_json", "[]")
        )
        new_telemetry = (
            safe_json_dumps(telemetry)
            if telemetry is not None
            else current.get("telemetry_json", "{}")
        )
        new_fps = processing_fps if processing_fps is not None else current.get("processing_fps", 0.0)

        cursor.execute(
            """
            UPDATE jobs
            SET status = ?,
                output_path = ?,
                failure_reason = ?,
                flagged_frames_json = ?,
                telemetry_json = ?,
                processing_fps = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (
                status,
                new_output,
                new_failure,
                new_frames,
                new_telemetry,
                new_fps,
                now_iso,
                job_id,
            ),
        )
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        updated = cursor.fetchone()
        conn.commit()
        return dict(updated) if updated else None


def update_review_action(
    job_id: str,
    status: str,
    reviewer_notes: Optional[str] = None,
    db_path: Optional[str | Path] = None,
) -> Optional[dict[str, Any]]:
    """Update job status following human review."""
    init_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cursor.fetchone()
        if not row:
            return None

        cursor.execute(
            """
            UPDATE jobs
            SET status = ?,
                reviewer_notes = ?,
                updated_at = ?
            WHERE job_id = ?
            """,
            (status, reviewer_notes, now_iso, job_id),
        )
        cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        updated = cursor.fetchone()
        conn.commit()
        return dict(updated) if updated else None
