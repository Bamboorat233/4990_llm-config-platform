"""Unit tests for GitHub issue #12: Component h (**kwargs) dynamic panels.

Tests:
- Parsing of typed kwargs (int, float, bool, JSON, str)
- Whitelist validation for each of the 4 stages (load_model, load_tokenizer, encode, generate)
- Unknown argument rejection and field-level issue reporting
- Addition and deletion of dynamic rows
- Reflection of valid kwargs in ConfigRecord serialization
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.registry import MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord
from app.ui.kwargs_panel import (
    KWARGS_WHITELISTS,
    KwargsSubPanel,
    parse_kwarg_value,
    validate_kwargs_section,
)


def test_parse_kwarg_value() -> None:
    """Verify intelligent parsing of string values into typed Python objects."""
    assert parse_kwarg_value("123") == 123
    assert parse_kwarg_value("0.85") == 0.85
    assert parse_kwarg_value("true") is True
    assert parse_kwarg_value("False") is False
    assert parse_kwarg_value("null") is None
    assert parse_kwarg_value("none") is None
    assert parse_kwarg_value("[1, 2, 3]") == [1, 2, 3]
    assert parse_kwarg_value('{"key": "val"}') == {"key": "val"}
    assert parse_kwarg_value("bfloat16") == "bfloat16"


def test_validate_kwargs_whitelists() -> None:
    """Issue #12 Done When: Unknown keys produce field-level errors; valid keys pass."""
    # 1. Valid kwargs across stages should produce 0 validation issues
    valid_data = {
        "load_model": {"torch_dtype": "bfloat16", "low_cpu_mem_usage": True},
        "load_tokenizer": {"use_fast": True},
        "encode": {"max_length": 2048},
        "generate": {"min_new_tokens": 10},
    }
    assert validate_kwargs_section(valid_data) == []

    # 2. Unknown keys not in the preset Hugging Face whitelists must produce error issues
    invalid_data = {
        "generate": {"unknown_generation_arg": 123},
        "load_model": {"fake_model_param": True},
    }
    issues = validate_kwargs_section(invalid_data)
    assert len(issues) == 2

    fields = {iss["field"] for iss in issues}
    assert "h.generate.unknown_generation_arg" in fields
    assert "h.load_model.fake_model_param" in fields
    for iss in issues:
        assert iss["level"] == "error"
        assert "Unknown argument" in iss["msg"]


def test_kwargs_in_config_record() -> None:
    """Valid kwargs correctly serialize into ConfigRecord payload."""
    payload = defaults_for(MISTRAL_ID, "fine_tune")
    payload["h"] = {
        "generate": {"min_new_tokens": 10, "repetition_penalty": 1.2},
    }
    record = ConfigRecord(**payload)
    assert record.h["generate"]["min_new_tokens"] == 10
    dumped = record.model_dump()
    assert dumped["h"]["generate"]["repetition_penalty"] == 1.2


def test_kwargs_subpanel_add_and_delete() -> None:
    """Test dynamic row lifecycle in KwargsSubPanel."""
    pytest.importorskip("nicegui", reason="NiceGUI is required for dynamic row UI elements")
    from nicegui import ui

    h_state: dict[str, dict] = {}
    widgets: dict[str, object] = {}
    refresh_calls: list[bool] = []

    panel = KwargsSubPanel(
        stage="generate",
        h_state=h_state,
        field_widgets=widgets,
        on_change_callback=lambda: refresh_calls.append(True),
    )

    with ui.column():
        panel.build_ui()
        panel.add_row("min_new_tokens", "15")

    assert h_state["generate"]["min_new_tokens"] == 15
    assert "h.generate.min_new_tokens" in widgets

    # Test clearing panel
    panel.clear()
    assert h_state["generate"] == {}
    assert "h.generate.min_new_tokens" not in widgets
