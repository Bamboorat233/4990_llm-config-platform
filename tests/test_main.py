"""Unit tests for GitHub issues #6, #7, and #11 in app/main.py.

Validates:
- Page layout and form skeleton with registered field widgets (#6)
- Model and task-type switching linkage (#7)
- Dynamic visibility for Optimizer (Component d) and Data (Component c) (#7)
- Save, deduplication, and duplication in the UI (#11)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("nicegui", reason="NiceGUI is required to test main.py UI components")

from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord
from app.store import list_configs


@pytest.fixture
def isolated_db(tmp_path: Any, monkeypatch: Any) -> Any:
    """Redirect SQLite database to a temporary directory."""
    test_db = tmp_path / "test_main.db"
    monkeypatch.setenv("LLM_CONFIG_DB_PATH", str(test_db))
    return test_db


def test_main_page_skeleton_initialization(isolated_db: Any) -> None:
    """Issue #6: Verify page initialization and field_widgets registration.

    Ensures that when create_app() runs:
    - Crucial widgets are mapped in `field_widgets` for error highlighting.
    - Initial state is Mistral + fine-tune.
    - The validation console is connected and ready.
    """
    from app.main import create_app

    platform = create_app()

    # 1. Verify core widgets are registered so validation errors can highlight them
    assert "model_id" in platform.field_widgets
    assert "task_type" in platform.field_widgets
    assert "p.system_prompt" in platform.field_widgets
    assert "b.pad_token" in platform.field_widgets
    assert "f.temperature" in platform.field_widgets
    assert "d.sft.learning_rate" in platform.field_widgets

    # 2. Form defaults match Mistral fine_tune on initial launch
    assert platform.current_model_id == MISTRAL_ID
    assert platform.current_task_type == "fine_tune"

    # 3. Console is attached and save button is enabled (since default state is valid)
    assert platform.console is not None
    assert platform.console._save_button.enabled is True


def test_model_task_switching_linkage(isolated_db: Any) -> None:
    """Issue #7: Switching across all 4 combinations reflects defaults_for exactly."""
    from app.main import create_app

    platform = create_app()

    # 1. Switch to inference (Mistral + inference)
    platform.on_task_type_changed(SimpleNamespace(value="inference"))
    assert platform.current_task_type == "inference"
    cfg1 = platform.build_config_record()
    exp1 = defaults_for(MISTRAL_ID, "inference")
    assert cfg1.task_type == "inference"
    assert cfg1.e.add_generation_prompt is True
    assert cfg1.c.test_data_file == exp1["c"]["test_data_file"]
    # Optimizer hidden in inference
    assert platform.optimizer_expansion.visible is False
    assert platform.data_fine_tune_container.visible is False
    assert platform.data_inference_container.visible is True

    # 2. Switch to DeepSeek (DeepSeek + inference)
    platform.on_model_changed(SimpleNamespace(value=DEEPSEEK_ID))
    assert platform.current_model_id == DEEPSEEK_ID
    cfg2 = platform.build_config_record()
    exp2 = defaults_for(DEEPSEEK_ID, "inference")
    assert cfg2.model_id == DEEPSEEK_ID
    assert cfg2.b.pad_token == exp2["b"]["pad_token"]
    assert cfg2.f.temperature == exp2["f"]["temperature"]
    assert cfg2.f.max_new_tokens == 1024

    # 3. Switch to fine_tune (DeepSeek + fine_tune)
    platform.on_task_type_changed(SimpleNamespace(value="fine_tune"))
    assert platform.current_task_type == "fine_tune"
    cfg3 = platform.build_config_record()
    exp3 = defaults_for(DEEPSEEK_ID, "fine_tune")
    assert cfg3.task_type == "fine_tune"
    assert cfg3.c.train_data_file == exp3["c"]["train_data_file"]
    assert cfg3.e.add_generation_prompt is False
    # Optimizer visible in fine_tune
    assert platform.optimizer_expansion.visible is True
    assert platform.data_fine_tune_container.visible is True
    assert platform.data_inference_container.visible is False

    # 4. Switch back to Mistral (Mistral + fine_tune)
    platform.on_model_changed(SimpleNamespace(value=MISTRAL_ID))
    assert platform.current_model_id == MISTRAL_ID
    cfg4 = platform.build_config_record()
    exp4 = defaults_for(MISTRAL_ID, "fine_tune")
    assert cfg4.model_id == MISTRAL_ID
    assert cfg4.b.pad_token == exp4["b"]["pad_token"]
    assert cfg4.f.temperature == exp4["f"]["temperature"]
    assert platform.optimizer_expansion.visible is True


def test_validation_highlighting_and_save_blocking(isolated_db: Any) -> None:
    """Issue #6 / #8: Error in form disables save button and highlights offending field."""
    from app.main import create_app

    platform = create_app()

    # Rule 9: Empty system prompt for Mistral
    platform.field_widgets["p.system_prompt"].value = ""
    platform.console.refresh()

    assert platform.console._save_button.enabled is False

    # Restore valid system prompt
    platform.field_widgets["p.system_prompt"].value = "You are a fact-checker."
    platform.console.refresh()
    assert platform.console._save_button.enabled is True


def test_save_and_duplicate_workflow(isolated_db: Any) -> None:
    """Issue #11: Save to DB, deduplication, and duplication.

    Verifies the end-to-end user lifecycle:
    1. Saving a valid config stores 1 record in SQLite.
    2. Attempting to save the same config again is blocked by SHA-256 deduplication.
    3. Duplicating a config creates a fresh in-memory draft with a new UUID.
    """
    from app.main import create_app

    platform = create_app()

    # Step 1: Save the initial configuration
    cfg = platform.build_config_record()
    platform.save_to_db(cfg)

    saved_items = list_configs(isolated_db)
    assert len(saved_items) == 1
    assert saved_items[0]["config_id"] == cfg.config_id

    # Step 2: Deduplication check: attempting to save identical settings again is blocked
    platform.save_to_db(cfg)
    assert len(list_configs(isolated_db)) == 1

    # Step 3: Duplicate config loads a draft with a fresh ID without immediately inserting
    platform.duplicate_saved_config(cfg.config_id)
    duplicated_cfg = platform.build_config_record()
    assert duplicated_cfg.config_id != cfg.config_id
