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

from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord
from app.store import list_configs


@pytest.fixture
def isolated_db(tmp_path: Any, monkeypatch: Any) -> Any:
    """Redirect SQLite database to a temporary directory."""
    pytest.importorskip("nicegui", reason="NiceGUI is required to test main.py UI components")
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

    # Trigger an error in Component f (temperature > 0 while do_sample is False)
    # and verify that clicking Validate navigates pipeline_tabs to tab 'f'
    platform.pipeline_tabs.value = "a"
    platform.field_widgets["f.temperature"].value = 0.7
    platform.field_widgets["f.do_sample"].value = False
    platform.console._on_validate()
    assert platform.pipeline_tabs.value == "f"


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


def test_user_prompt_autoformatting(isolated_db: Any) -> None:
    """Typing 'Claim: ...', 'Answer:' auto-formats without curly brackets, Space populates defaults, and a.model_id syncs."""
    from app.main import create_app

    platform = create_app()

    # 1. Prompt fields start visually empty (placeholder) and populate default when Space is pressed
    assert platform.field_widgets["p.system_prompt"].value == ""
    platform.field_widgets["p.system_prompt"].value = " "
    assert "You are a fact-checking assistant" in platform.field_widgets["p.system_prompt"].value

    # 2. Blank Answer: formats to '### Answer: ' without '{answer}' curly brackets
    platform.field_widgets["p.user_prompt_template"].value = (
        "Claim: Can pigs fly?\nQuestion: Can pigs Fly?\nAnswer:"
    )
    platform.format_user_prompt_field()

    cfg = platform.build_config_record()
    assert cfg.p.user_prompt_template == (
        "### Claim: Can pigs fly?\n### Question: Can pigs Fly?\n### Answer: "
    )
    assert "{answer}" not in cfg.p.user_prompt_template
    assert "### Claim: Can pigs fly?" in platform.prompt_preview.content

    # 3. Selecting a different model in a.model_id toggles the top-level platform model
    platform.field_widgets["a.model_id"].value = DEEPSEEK_ID
    assert platform.current_model_id == DEEPSEEK_ID


def test_kwargs_panel_stays_bound_after_model_switch(isolated_db: Any) -> None:
    """Issue #12: Component h keeps writing to the live state after a switch.

    ``load_state`` used to rebind ``form_state`` (and its ``"h"`` sub-dict),
    orphaning every ``KwargsSubPanel`` that captured the old dict. A key typed
    after a model switch was written to the orphan and never reached the JSON
    preview.
    """
    from app.main import create_app

    platform = create_app()
    h_state = platform.form_state["h"]
    panel = platform.kwargs_panels["load_model"]

    panel.add_row()
    panel.row_records[-1]["key_widget"].value = "device_map"
    assert platform.form_state["h"]["load_model"] == {"device_map": ""}

    platform.on_model_changed(SimpleNamespace(value=DEEPSEEK_ID))

    # Both the top-level state and the "h" sub-dict keep their identity.
    assert platform.form_state["h"] is h_state
    assert platform.kwargs_panels["load_model"].h_state is h_state

    panel = platform.kwargs_panels["load_model"]
    panel.add_row()
    panel.row_records[-1]["key_widget"].value = "torch_dtype"
    assert platform.form_state["h"]["load_model"] == {"torch_dtype": ""}
    assert platform.build_config_record().h["load_model"] == {"torch_dtype": ""}


def test_per_model_and_task_prompt_visibility(isolated_db: Any) -> None:
    """Prompt visibility follows app/model_prompts.json across (model, task_type)."""
    from app.main import create_app

    platform = create_app()

    # 1. Mistral + fine_tune: all 3 prompts visible
    assert platform.system_prompt_container.visible is True
    assert platform.user_prompt_container.visible is True
    assert platform.assistant_prompt_container.visible is True

    # 2. Mistral + inference: 2 prompts visible (System + User; Assistant hidden)
    platform.on_task_type_changed(SimpleNamespace(value="inference"))
    assert platform.system_prompt_container.visible is True
    assert platform.user_prompt_container.visible is True
    assert platform.assistant_prompt_container.visible is False

    # 3. DeepSeek + inference: 1 prompt visible (User only; System + Assistant hidden)
    platform.on_model_changed(SimpleNamespace(value=DEEPSEEK_ID))
    assert platform.system_prompt_container.visible is False
    assert platform.user_prompt_container.visible is True
    assert platform.assistant_prompt_container.visible is False

    # 4. DeepSeek + fine_tune: 2 prompts visible (User + Assistant; System hidden)
    platform.on_task_type_changed(SimpleNamespace(value="fine_tune"))
    assert platform.system_prompt_container.visible is False
    assert platform.user_prompt_container.visible is True
    assert platform.assistant_prompt_container.visible is True


