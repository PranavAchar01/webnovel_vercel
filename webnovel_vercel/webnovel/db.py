"""Database helpers for the web novel service.

The service stores metadata about novels, chapters and background jobs in
an SQLite database. Each helper function opens its own connection on
demand using the standard ``sqlite3`` module and ensures that
connections are closed as soon as possible. SQLite is thread safe for
readers, but writes require a single writer at a time; therefore
operations that modify the database should be kept lightweight.

The schema is defined in ``init_db()``. Novel and chapter identifiers
are stored as TEXT and expected to be globally unique (e.g. UUIDs).
Status fields use simple text values such as ``new``, ``extracting``,
``ready`` or ``error``. See the top level documentation for details.
"""

from __future__ import annotations

import sqlite3
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DB_PATH = os.environ.get("WEBNOVEL_DB", str(Path("/mnt/data/webnovel.db")))


def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set to dict-like.

    SQLite connections are not inherently thread safe when used across
    multiple threads. Each function opens and closes its own connection
    to avoid cross‑thread sharing. ``row_factory`` is configured so
    that rows behave like dictionaries keyed by column names.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialise the SQLite database and create tables if they do not exist.

    This function is idempotent: it can be called repeatedly without
    harming existing data. It creates three tables: ``novels``,
    ``chapters`` and ``jobs``, according to the schema described in the
    specification. The function also creates simple indices to
    accelerate common lookups.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Create novels table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS novels (
            id TEXT PRIMARY KEY,
            title TEXT,
            author TEXT,
            source_url TEXT UNIQUE,
            description TEXT,
            tags TEXT,
            num_chapters INTEGER,
            cover_path TEXT,
            status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    # Create chapters table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chapters (
            id TEXT PRIMARY KEY,
            novel_id TEXT,
            idx INTEGER,
            title TEXT,
            source_url TEXT,
            text_path TEXT,
            audio_path TEXT,
            duration_seconds INTEGER,
            tts_provider TEXT,
            status TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(novel_id) REFERENCES novels(id)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chapters_novel_id ON chapters(novel_id)")
    # Create jobs table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            type TEXT,
            novel_id TEXT,
            payload TEXT,
            status TEXT,
            progress INTEGER,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def insert_novel(novel: Dict[str, Any]) -> None:
    """Insert or update a novel row.

    The provided ``novel`` dictionary must contain at least an ``id``.
    Any keys not present will default to ``None``. If a row with the
    same ``id`` already exists the function performs an ``UPDATE``
    instead of an ``INSERT``. This behaviour simplifies upserts.
    """
    fields = [
        "id",
        "title",
        "author",
        "source_url",
        "description",
        "tags",
        "num_chapters",
        "cover_path",
        "status",
    ]
    values = [novel.get(f) for f in fields]
    conn = get_connection()
    cur = conn.cursor()
    # Try to insert; if fails due to primary key constraint then update
    try:
        cur.execute(
            """
            INSERT INTO novels(id, title, author, source_url, description,
                               tags, num_chapters, cover_path, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
    except sqlite3.IntegrityError:
        set_clause = ", ".join([f"{field} = ?" for field in fields[1:]])
        cur.execute(
            f"UPDATE novels SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values[1:] + [novel["id"]],
        )
    conn.commit()
    conn.close()


def insert_chapter(chapter: Dict[str, Any]) -> None:
    """Insert or update a chapter row.

    The ``chapter`` dict must contain ``id`` and ``novel_id``. If a row
    already exists for the given ``id``, the record is updated. The
    ``idx`` field is named ``idx`` (instead of ``index``) because
    ``index`` is a reserved keyword in SQL.
    """
    fields = [
        "id",
        "novel_id",
        "idx",
        "title",
        "source_url",
        "text_path",
        "audio_path",
        "duration_seconds",
        "tts_provider",
        "status",
    ]
    values = [chapter.get(f) for f in fields]
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO chapters(id, novel_id, idx, title, source_url,
                                 text_path, audio_path, duration_seconds,
                                 tts_provider, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
    except sqlite3.IntegrityError:
        set_clause = ", ".join([f"{field} = ?" for field in fields[1:]])
        cur.execute(
            f"UPDATE chapters SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            values[1:] + [chapter["id"]],
        )
    conn.commit()
    conn.close()


def insert_job(job: Dict[str, Any]) -> None:
    """Insert or update a job row in the jobs table.

    Jobs are uniquely identified by ``id``. The ``payload`` field is
    serialized to JSON when stored.
    """
    fields = ["id", "type", "novel_id", "payload", "status", "progress", "error"]
    payload_json = json.dumps(job.get("payload")) if job.get("payload") is not None else None
    values = [job.get("id"), job.get("type"), job.get("novel_id"), payload_json,
              job.get("status"), job.get("progress", 0), job.get("error")]
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO jobs(id, type, novel_id, payload, status, progress, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )
    except sqlite3.IntegrityError:
        cur.execute(
            """
            UPDATE jobs SET type = ?, novel_id = ?, payload = ?, status = ?, progress = ?, error = ?,
                           updated_at = CURRENT_TIMESTAMP WHERE id = ?
            """,
            values[1:] + [job["id"]],
        )
    conn.commit()
    conn.close()


def update_job(job_id: str, *, status: Optional[str] = None, progress: Optional[int] = None,
               error: Optional[str] = None) -> None:
    """Update status/progress/error fields on a job.

    Only provided arguments will be updated. The ``updated_at`` field
    automatically records when the row was modified.
    """
    conn = get_connection()
    cur = conn.cursor()
    parts: List[str] = []
    params: List[Any] = []
    if status is not None:
        parts.append("status = ?")
        params.append(status)
    if progress is not None:
        parts.append("progress = ?")
        params.append(progress)
    if error is not None:
        parts.append("error = ?")
        params.append(error)
    if not parts:
        return
    params.append(job_id)
    set_clause = ", ".join(parts)
    cur.execute(
        f"UPDATE jobs SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        params,
    )
    conn.commit()
    conn.close()


def update_chapter_status(chapter_id: str, status: str) -> None:
    """Update the status of a chapter to one of new|ready|error.|"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE chapters SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, chapter_id),
    )
    conn.commit()
    conn.close()


def update_novel_status(novel_id: str, status: str) -> None:
    """Update the status of a novel.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE novels SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (status, novel_id),
    )
    conn.commit()
    conn.close()


def update_chapter_audio_info(chapter_id: str, audio_path: str, duration: int,
                              provider: str) -> None:
    """Update the audio information for a chapter.

    Sets the ``audio_path``, ``duration_seconds`` and ``tts_provider``
    fields, and marks the chapter status as ``ready``. Existing fields
    are overwritten.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE chapters SET audio_path = ?, duration_seconds = ?, tts_provider = ?,
                            status = 'ready', updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
        """,
        (audio_path, duration, provider, chapter_id),
    )
    conn.commit()
    conn.close()


def get_novel(novel_id: str) -> Optional[Dict[str, Any]]:
    """Return the novel row for ``novel_id`` as a dict, or None if absent."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM novels WHERE id = ?", (novel_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_chapter(chapter_id: str) -> Optional[Dict[str, Any]]:
    """Return a chapter row as a dict."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM chapters WHERE id = ?", (chapter_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_chapters(novel_id: str) -> List[Dict[str, Any]]:
    """Return all chapters for a novel, ordered by index."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM chapters WHERE novel_id = ? ORDER BY idx ASC",
        (novel_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return a job record as a dict."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        data = dict(row)
        if data.get("payload"):
            try:
                data["payload"] = json.loads(data["payload"])
            except Exception:
                pass
        return data
    return None


def list_novels() -> List[Dict[str, Any]]:
    """Return all novels in the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM novels ORDER BY updated_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]
