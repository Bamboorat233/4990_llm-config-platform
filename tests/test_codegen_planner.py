"""Unit tests for Issue #56: codegen contracts, unified kwargs validation, and planner."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.codegen import (
    ConfigValidationError,
    GenerationResult,
    ScriptPlan,
    build_script_filename,
    build_script_plan,
    generate_bundle,
)
from app.parsing.errors import family_for_script
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord
from app.ui.kwargs_panel import (
    KWARGS_WHITELISTS as UI_KWARGS_WHITELISTS,
    parse_kwarg_value as ui_parse_kwarg_value,
    validate_kwargs_section as ui_validate_kwargs_section,
)
from app.validation import (
    KWARGS_WHITELISTS,
    parse_kwarg_value,
    validate_kwargs_section,
)


def _make_cfg(model_id: str, task_type: str, **overrides) -> ConfigRecord:
    payload = defaults_for(model_id, task_type)  # type: ignore[arg-type]
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    return ConfigRecord(**payload)


def test_unified_kwargs_reexports_identity() -> None:
    """app.ui.kwargs_panel must re-export the exact pure objects from app.validation."""
    assert UI_KWARGS_WHITELISTS is KWARGS_WHITELISTS
    assert ui_validate_kwargs_section is validate_kwargs_section
    assert ui_parse_kwarg_value is parse_kwarg_value


@pytest.mark.parametrize(
    ("model_id", "task_type", "expected_prefix"),
    [
        (MISTRAL_ID, "fine_tune", "fineTune_Mistral24B_"),
        (MISTRAL_ID, "inference", "inference_Mistral24B_"),
        (DEEPSEEK_ID, "fine_tune", "fineTune_DeepSeek32B_"),
        (DEEPSEEK_ID, "inference", "inference_DeepSeek32B_"),
    ],
)
def test_valid_configs_build_script_plan_and_filenames(
    model_id: str,
    task_type: str,
    expected_prefix: str,
) -> None:
    """All four valid Model x Task combinations produce immutable ScriptPlans."""
    cfg = _make_cfg(model_id, task_type, config_id="a1b2c3d4-e5f6-4789-abcd-ef0123456789")
    plan, warnings = build_script_plan(cfg)

    assert isinstance(plan, ScriptPlan)
    assert warnings == []
    assert plan.script_filename == f"{expected_prefix}a1b2c3d4.py"
    assert build_script_filename(cfg) == plan.script_filename
    assert family_for_script(plan.script_filename) == task_type
    assert plan.tokenizer_id == model_id
    assert plan.uses_distributed is True
    assert plan.fix_mistral_regex is (model_id == MISTRAL_ID)
    assert set(plan.kwargs.keys()) == set(KWARGS_WHITELISTS.keys())
    assert len(plan.phase_markers) > 0

    if task_type == "fine_tune":
        assert plan.save_model_dir == "fineTuningOutput/final_model"
    else:
        assert plan.save_model_dir is None

    with pytest.raises(FrozenInstanceError):
        plan.script_filename = "other.py"  # type: ignore[misc]


def test_unknown_kwargs_raise_config_validation_error() -> None:
    """Unknown h.* kwargs must fail fast with ConfigValidationError in planner."""
    cfg = _make_cfg(
        MISTRAL_ID,
        "inference",
        h={"generate": {"not_a_real_hf_kwarg": 123}},
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        build_script_plan(cfg)

    issues = exc_info.value.issues
    assert len(issues) == 1
    assert issues[0]["field"] == "h.generate.not_a_real_hf_kwarg"
    assert issues[0]["level"] == "error"


def test_cross_field_errors_raise_config_validation_error() -> None:
    """Cross-field errors such as empty Mistral system_prompt or <myPadToken> block planning."""
    cfg = _make_cfg(
        MISTRAL_ID,
        "fine_tune",
        b={"pad_token": "<myPadToken>"},
        p={"system_prompt": "   "},
    )

    with pytest.raises(ConfigValidationError) as exc_info:
        build_script_plan(cfg)

    fields = {iss["field"] for iss in exc_info.value.issues}
    assert "b.pad_token" in fields
    assert "p.system_prompt" in fields
    assert all(iss["level"] == "error" for iss in exc_info.value.issues)


def test_warnings_do_not_block_and_are_returned() -> None:
    """DeepSeek non-empty system_prompt emits a warning without raising."""
    cfg = _make_cfg(
        DEEPSEEK_ID,
        "inference",
        p={"system_prompt": "Verify the claim carefully."},
    )

    plan, warnings = build_script_plan(cfg)

    assert plan.include_system_prompt is True
    assert len(warnings) == 1
    assert "DeepSeek" in warnings[0]


def test_generate_bundle_preflight_and_contract() -> None:
    """generate_bundle enforces pre-flight validation and exposes contract §6 keys."""
    invalid_cfg = _make_cfg(MISTRAL_ID, "inference", b={"pad_token": "<myPadToken>"})
    with pytest.raises(ConfigValidationError):
        generate_bundle(invalid_cfg)

    valid_cfg = _make_cfg(MISTRAL_ID, "inference")
    result = generate_bundle(valid_cfg)
    assert result["syntax_ok"] is True
    assert result["syntax_error"] is None
    assert result["parity_ok"] is True
    assert result["parity_diffs"] == []
    assert result["pitfall_issues"] == []
    assert result["marker_issues"] == []
    assert result["slurm_script"] is None

    expected_keys = {
        "config_id",
        "config_hash",
        "task_type",
        "model_id",
        "script_filename",
        "script_code",
        "requirements_txt",
        "config_json",
        "slurm_script",
        "script_sha256",
        "syntax_ok",
        "syntax_error",
        "parity_ok",
        "parity_diffs",
        "pitfall_issues",
        "marker_issues",
        "warnings",
    }
    assert set(GenerationResult.__annotations__.keys()) == expected_keys


def test_pure_modules_import_without_nicegui() -> None:
    """app.validation and app.codegen must import cleanly with no nicegui import."""
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "import app.validation, app.codegen, app.codegen.planner; "
        "assert 'nicegui' not in sys.modules, 'nicegui was imported'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
