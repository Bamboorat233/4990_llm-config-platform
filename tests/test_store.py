"""Unit tests for GitHub issues #10 and #36: SQLite persistence."""

import json
import sqlite3

import pytest

from app.registry import MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord, config_fingerprint
from app.store import (
    delete_config,
    duplicate_config,
    find_by_hash,
    get_config,
    get_run,
    get_score,
    init_db,
    insert_config,
    insert_run,
    list_configs,
    list_runs,
    list_scores_for_config,
    upsert_score,
)


@pytest.fixture
def db_path(tmp_path):
    """Each test receives a separate temporary database file."""
    return tmp_path / "test-platform.db"


def example_config():
    """Build a real Pydantic configuration without downloading Mistral."""
    return ConfigRecord(**defaults_for(MISTRAL_ID, "inference"))


def test_init_db_creates_required_columns(db_path):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("PRAGMA table_info(configs)").fetchall()
    columns = {row[1] for row in rows}
    assert columns == {
        "config_id", "config_hash", "model_id", "task_type",
        "run_status", "created_at", "created_by", "notes", "payload_json",
    }


def test_duplicate_insert_is_rejected(db_path):
    config = example_config()
    first_id = insert_config(config, db_path)
    assert first_id == config.config_id
    assert find_by_hash(config_fingerprint(config), db_path) == first_id

    identical_settings = example_config()  # Different ID, same settings
    with pytest.raises(ValueError, match="identical experiment config"):
        insert_config(identical_settings, db_path)
    assert len(list_configs(db_path)) == 1


def test_get_config_survives_reopening_database(db_path):
    original = example_config()
    saved_id = insert_config(original, db_path)

    # All store.py connections are already closed by this point.
    with sqlite3.connect(db_path) as reopened:
        assert reopened.execute("SELECT COUNT(*) FROM configs").fetchone()[0] == 1

    restored = get_config(saved_id, db_path)
    assert isinstance(restored, ConfigRecord)
    assert restored.config_id == saved_id
    assert restored.f.temperature == 0.15
    assert restored.config_hash == config_fingerprint(original)
    assert get_config("does-not-exist", db_path) is None


def test_duplicate_creates_new_id_and_time_without_saving(db_path):
    original = example_config()
    insert_config(original, db_path)

    copy = duplicate_config(original.config_id, db_path)
    assert copy.config_id != original.config_id
    assert copy.created_at != original.created_at
    assert copy.config_hash == ""
    assert copy.f.temperature == original.f.temperature
    assert len(list_configs(db_path)) == 1  # Not yet stored

    with pytest.raises(ValueError, match="identical experiment config"):
        insert_config(copy, db_path)

    copy.f.temperature = 0.9
    insert_config(copy, db_path)
    assert len(list_configs(db_path)) == 2


def test_list_configs_newest_first(db_path):
    old_config = example_config()
    old_config.created_at = "2026-01-01T00:00:00Z"
    insert_config(old_config, db_path)

    new_config = example_config()
    new_config.f.temperature = 0.9
    new_config.created_at = "2026-10-31T00:00:00Z"
    insert_config(new_config, db_path)

    rows = list_configs(db_path)
    assert [row["config_id"] for row in rows] == [
        new_config.config_id, old_config.config_id,
    ]


def test_delete_config_removes_record(db_path):
    """Verify that delete_config permanently deletes the record from SQLite."""
    config = example_config()
    insert_config(config, db_path)
    assert len(list_configs(db_path)) == 1

    delete_config(config.config_id, db_path)
    assert len(list_configs(db_path)) == 0
    assert get_config(config.config_id, db_path) is None


# Issue #36 tests.


def example_run(tmp_path, filename="sample.log", data=b"LLM output\r\n"):
    log = tmp_path / filename
    log.write_bytes(data)
    return {
        "source_path": str(log),
        "script": "inferenceMistral.py",
        "model_id": None,
        "family": "inference",
        "status": "success",
        "scenario": "inference_success",
        "phases": [],
        "failed_phase": None,
        "output_format": "banner",
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "progress": {"records_done": 1, "records_expected": 1},
        "errors": [],
        "warnings": [],
        "evidence": [],
        "notes": [],
    }


def example_score(accuracy=0.75):
    return {
        "marker": "### Response: Verdict: ",
        "labels": ["True", "False", "Conflicting"],
        "source": "parsed",
        "records": 4,
        "complete": True,
        "exact": 3,
        "recovered": 0,
        "empty": 1,
        "out_of_set": 0,
        "unparsed": 0,
        "unfinished": 0,
        "empty_rate": 0.25,
        "invalid_rate": 0.25,
        "accuracy": accuracy,
        "macro_f1": 0.7,
        "micro_f1": accuracy,
        "weighted_f1": 0.7,
        "precision": {}, "recall": {}, "f1": {}, "confusion": {},
        "exact_metrics": {}, "inclusive_metrics": {},
    }


def test_init_db_adds_runs_and_scores_tables(db_path):
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        names = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert {"configs", "runs", "scores"}.issubset(names)
        runs = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
        scores = {row[1] for row in conn.execute("PRAGMA table_info(scores)")}
    assert runs == {
        "run_id", "config_id", "source_path", "source_sha256",
        "family", "status", "created_at", "payload_json",
    }
    assert scores == {"run_id", "created_at", "payload_json"}


def test_same_log_and_config_deduplicates_even_with_different_path(db_path, tmp_path):
    cfg = example_config()
    insert_config(cfg, db_path)
    first = example_run(tmp_path, "one.log")
    second = example_run(tmp_path, "two.log", b"LLM output\r\n")

    first_id = insert_run(first, cfg.config_id, db_path)
    second_id = insert_run(second, cfg.config_id, db_path)

    assert first_id == second_id
    assert len(list_runs(cfg.config_id, db_path)) == 1
    assert get_run(first_id, db_path)["payload"] == first
    assert len(get_run(first_id, db_path)["source_sha256"]) == 64


def test_same_log_different_config_creates_two_runs(db_path, tmp_path):
    a = example_config()
    b = example_config()
    b.f.temperature = 0.75
    insert_config(a, db_path)
    insert_config(b, db_path)
    result = example_run(tmp_path)

    first = insert_run(result, a.config_id, db_path)
    second = insert_run(result, b.config_id, db_path)

    assert first != second
    assert len(list_runs(db_path=db_path)) == 2
    assert len(list_runs(a.config_id, db_path)) == 1
    assert len(list_runs(b.config_id, db_path)) == 1
    assert get_run("does-not-exist", db_path) is None


def test_rescoring_keeps_one_score_per_run(db_path, tmp_path):
    cfg = example_config()
    insert_config(cfg, db_path)
    run_id = insert_run(example_run(tmp_path), cfg.config_id, db_path)

    assert get_score(run_id, db_path) is None
    upsert_score(run_id, example_score(), db_path)
    upsert_score(run_id, example_score(0.5), db_path)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scores").fetchone()[0] == 1
    assert get_score(run_id, db_path)["accuracy"] == 0.5
    rows = list_scores_for_config(cfg.config_id, db_path)
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["report"]["accuracy"] == 0.5


def test_foreign_key_rejects_missing_config_or_run(db_path, tmp_path):
    with pytest.raises(sqlite3.IntegrityError):
        insert_run(example_run(tmp_path), "missing-config", db_path)
    with pytest.raises(sqlite3.IntegrityError):
        upsert_score("missing-run", example_score(), db_path)


def test_existing_m1_database_can_be_upgraded(db_path, tmp_path):
    # Simulate a legacy database containing only the original configs table.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE configs (
                config_id TEXT PRIMARY KEY,
                config_hash TEXT UNIQUE NOT NULL,
                model_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                run_status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                created_by TEXT,
                notes TEXT,
                payload_json TEXT NOT NULL
            )"""
        )
    cfg = example_config()
    fingerprint = config_fingerprint(cfg)
    stored = cfg.model_copy(update={"config_hash": fingerprint}, deep=True)
    # Write a legacy config directly, before init_db has created the new tables.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """INSERT INTO configs (
                config_id, config_hash, model_id, task_type, run_status,
                created_at, created_by, notes, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (stored.config_id, fingerprint, stored.model_id, stored.task_type,
             stored.run_status, stored.created_at, stored.created_by,
             stored.notes, stored.model_dump_json()),
        )
    init_db(db_path)
    assert get_config(cfg.config_id, db_path) is not None
    assert len(list_configs(db_path)) == 1

    run_id = insert_run(example_run(tmp_path), cfg.config_id, db_path)
    upsert_score(run_id, example_score(), db_path)
    assert get_score(run_id, db_path)["records"] == 4


def test_run_missing_source_file_raises(db_path, tmp_path):
    cfg = example_config()
    insert_config(cfg, db_path)
    result = {"source_path": str(tmp_path / "absent.log"),
              "family": "inference", "status": "success"}
    with pytest.raises(FileNotFoundError):
        insert_run(result, cfg.config_id, db_path)


def test_unlinked_same_log_deduplicates(db_path, tmp_path):
    result = example_run(tmp_path)
    first = insert_run(result, None, db_path)
    second = insert_run(result, None, db_path)
    assert first == second
    assert len(list_runs(db_path=db_path)) == 1
