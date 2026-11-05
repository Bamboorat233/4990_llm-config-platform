
"""SQLite storage for saved LLM configuration records.

Public API (database requirements in ``specs/parameters.md`` sections 1 and 13;
interface contract in ``specs/interfaces.md`` section 2; GitHub issue #10):

- :func:`init_db` — create the ``configs`` table
- :func:`find_by_hash` — look up an existing configuration by fingerprint
- :func:`insert_config` — save a configuration and reject duplicates
- :func:`list_configs` — list saved configurations, newest first
- :func:`get_config` — retrieve a complete saved configuration
- :func:`duplicate_config` — build an unsaved copy with a new ID and timestamp
"""

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


def init_db(db_path: str | Path | None = None) -> None:

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
    """Save "cfg" and return its ID; reject identical functional settings.

    The fingerprint comes from "config_fingerprint" in "app/schemas.py".
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
    """Return saved record metadata, sorted by "created_at" descending."""
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
    """Return the full saved "ConfigRecord" for "config_id" if present."""
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
            "created_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "run_status": "pending",
        },
    )
