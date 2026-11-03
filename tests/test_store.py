"""Unit tests for GitHub issue #10: SQLite persistence of ConfigRecord."""

import sqlite3

import pytest

from app.registry import MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord, config_fingerprint
from app.store import (
    duplicate_config,
    find_by_hash,
    get_config,
    init_db,
    insert_config,
    list_configs,
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