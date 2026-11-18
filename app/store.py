"""SQLite storage for LLM configurations, parsed runs, and score reports.

Issue #10: configuration CRUD; Issue #36: runs and scores persistence.
All public functions accept an optional ``db_path`` for test isolation.
"""

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.schemas import ConfigRecord, config_fingerprint


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "platform.db"


def _database_path(db_path: str | Path | None = None) -> Path:
    """Find the requested database, an environment override, or the default."""
    if db_path is not None:
        return Path(db_path)
    return Path(os.environ.get("LLM_CONFIG_DB_PATH", str(DEFAULT_DB_PATH)))


def _utc_now() -> str:
    """Timestamp for sortable SQLite text columns."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_sha256(source_path: str | Path) -> str:
    """Hash original file bytes without loading a possibly large log into memory."""
    sha256 = hashlib.sha256()
    with open(source_path, "rb") as log_file:
        for chunk in iter(lambda: log_file.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def init_db(db_path: str | Path | None = None) -> None:
    """Create or migrate tables without deleting existing M1 configuration rows."""
    path = _database_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS configs (
                    config_id TEXT PRIMARY KEY,
                    config_hash TEXT UNIQUE NOT NULL,
                    model_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    run_status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    notes TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    config_id TEXT REFERENCES configs(config_id),
                    source_path TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    family TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (config_id, source_sha256)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scores (
                    run_id TEXT PRIMARY KEY REFERENCES runs(run_id),
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )


def find_by_hash(
    config_hash: str, db_path: str | Path | None = None
) -> str | None:
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        row = conn.execute(
            "SELECT config_id FROM configs WHERE config_hash = ?",
            (config_hash,),
        ).fetchone()
    return row[0] if row is not None else None


def insert_config(cfg: ConfigRecord, db_path: str | Path | None = None) -> str:
    """Save cfg and return its ID; reject identical functional settings.

    The fingerprint comes from config_fingerprint in app/schemas.py.
    The source object is not modified. SQLite's UNIQUE constraint also protects
    against duplicate saves that happen at nearly the same time.
    """
    init_db(db_path)
    fingerprint = config_fingerprint(cfg)
    saved_cfg = cfg.model_copy(update={"config_hash": fingerprint}, deep=True)

    try:
        with closing(sqlite3.connect(_database_path(db_path))) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO configs (
                        config_id, config_hash, model_id, task_type,
                        run_status, created_at, created_by, notes, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        saved_cfg.config_id,
                        fingerprint,
                        saved_cfg.model_id,
                        saved_cfg.task_type,
                        saved_cfg.run_status,
                        saved_cfg.created_at,
                        saved_cfg.created_by,
                        saved_cfg.notes,
                        saved_cfg.model_dump_json(),
                    ),
                )
    except sqlite3.IntegrityError as exc:
        existing_id = find_by_hash(fingerprint, db_path)
        if existing_id is not None:
            raise ValueError(
                f"An identical experiment config already exists "
                f"(ID: {existing_id})"
            ) from exc
        raise

    return saved_cfg.config_id


def list_configs(db_path: str | Path | None = None) -> list[dict]:
    """Return saved record metadata, sorted by created_at descending."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT config_id, config_hash, model_id, task_type,
                   run_status, created_at, created_by, notes
            FROM configs
            ORDER BY created_at DESC, config_id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_config(
    config_id: str, db_path: str | Path | None = None
) -> ConfigRecord | None:
    """Return the full saved ConfigRecord for config_id if present."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        row = conn.execute(
            "SELECT payload_json FROM configs WHERE config_id = ?",
            (config_id,),
        ).fetchone()
    if row is None:
        return None
    return ConfigRecord.model_validate_json(row[0])


def delete_config(config_id: str, db_path: str | Path | None = None) -> None:
    """Permanently remove a saved configuration by its ID."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn, conn:
        conn.execute("DELETE FROM configs WHERE config_id = ?", (config_id,))


def duplicate_config(
    config_id: str, db_path: str | Path | None = None
) -> ConfigRecord:
    """Build an unsaved copy with new identity and a fresh UTC timestamp.

    The settings remain identical, so the fingerprint is cleared and the copy
    must be edited before it can be saved without triggering deduplication.
    """
    original = get_config(config_id, db_path)
    if original is None:
        raise ValueError(f"Configuration not found: {config_id}")

    return original.model_copy(
        deep=True,
        update={
            "config_id": str(uuid4()),
            "config_hash": "",
            "created_at": _utc_now(),
            "run_status": "pending",
        },
    )


# Issue #36: parsed runs and evaluation score reports.

def _decode_run(row: sqlite3.Row) -> dict:
    """Expose row metadata along with a decoded ParseResult as 'payload'."""
    record = dict(row)
    record["payload"] = json.loads(record["payload_json"])
    return record


def insert_run(
    result: dict, config_id: str | None, db_path: str | Path | None = None
) -> str:
    """Store one ParseResult and return its ID; reuse the same log/config ID.

    A byte-level SHA-256 hash identifies the log even if its path changes.
    The log must still exist when saving so its original bytes can be hashed.
    """
    init_db(db_path)
    path = str(result["source_path"])
    digest = _source_sha256(path)
    run_id = str(uuid4())

    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            if config_id is None:
                # SQLite UNIQUE allows repeated NULL values: handle unlinked logs
                # explicitly so repeated uploads of the same log reuse one run.
                existing = conn.execute(
                    "SELECT run_id FROM runs WHERE config_id IS NULL "
                    "AND source_sha256 = ? LIMIT 1",
                    (digest,),
                ).fetchone()
                if existing is not None:
                    return existing[0]
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, config_id, source_path, source_sha256,
                    family, status, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (config_id, source_sha256) DO NOTHING
                """,
                (
                    run_id,
                    config_id,
                    path,
                    digest,
                    result["family"],
                    result["status"],
                    _utc_now(),
                    json.dumps(result),
                ),
            )
            row = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE config_id IS ? AND source_sha256 = ?
                ORDER BY created_at DESC, run_id DESC LIMIT 1
                """,
                (config_id, digest),
            ).fetchone()
    return row[0] if row is not None else run_id


def get_run(run_id: str, db_path: str | Path | None = None) -> dict | None:
    """Retrieve one run with its stored ParseResult under the 'payload' key."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return _decode_run(row) if row is not None else None


def list_runs(
    config_id: str | None = None, db_path: str | Path | None = None
) -> list[dict]:
    """List runs newest first, optionally restricted to a saved config."""
    init_db(db_path)
    sql = "SELECT * FROM runs"
    args: tuple = ()
    if config_id is not None:
        sql += " WHERE config_id = ?"
        args = (config_id,)
    sql += " ORDER BY created_at DESC, run_id DESC"
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, args).fetchall()
    return [_decode_run(row) for row in rows]


def upsert_score(
    run_id: str, report: dict, db_path: str | Path | None = None
) -> None:
    """Insert or replace a run's single ScoreReport, keeping one score row."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with conn:
            conn.execute(
                """
                INSERT INTO scores (run_id, created_at, payload_json)
                VALUES (?, ?, ?)
                ON CONFLICT (run_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    payload_json = excluded.payload_json
                """,
                (run_id, _utc_now(), json.dumps(report)),
            )


def get_score(
    run_id: str, db_path: str | Path | None = None
) -> dict | None:
    """Return the stored ScoreReport or None when a run is not yet scored."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        row = conn.execute(
            "SELECT payload_json FROM scores WHERE run_id = ?", (run_id,)
        ).fetchone()
    return json.loads(row[0]) if row is not None else None


def list_scores_for_config(
    config_id: str, db_path: str | Path | None = None
) -> list[dict]:
    """Return score history for a config, newest score first."""
    init_db(db_path)
    with closing(sqlite3.connect(_database_path(db_path))) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT s.run_id, r.config_id, r.source_path, r.family, r.status,
                   s.created_at, s.payload_json
            FROM scores AS s
            JOIN runs AS r ON r.run_id = s.run_id
            WHERE r.config_id = ?
            ORDER BY s.created_at DESC, s.run_id DESC
            """,
            (config_id,),
        ).fetchall()
    return [dict(row, report=json.loads(row["payload_json"])) for row in rows]
