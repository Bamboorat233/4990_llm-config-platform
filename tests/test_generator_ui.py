"""Unit and UI tests for Milestone 2.5 Script Export panel (Issue #60).

Covers:
1. Subprocess lazy-import check: importing ``app.ui.generator`` does NOT import ``nicegui``.
2. Pure ``evaluate_generation`` tests across all 4 Model x Task defaults:
   green badges, valid Python AST, and valid ``requirements.txt``.
3. Pure ``build_verification_badges`` failure states (syntax, parity, pitfalls, markers).
4. Blocking invalid configs (empty Mistral ``system_prompt`` and unknown ``h.generate.bogus``)
   and non-blocking warnings (DeepSeek non-empty ``system_prompt``).
5. NiceGUI ``Client`` mount test for ``build_generator_panel`` and ``app/main.py``
   integration (active form, saved config, tabs, copy/download, invalid config blocking,
   and ``refresh_saved_configs_list`` sync).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.codegen import GenerationResult  # noqa: E402
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for  # noqa: E402
from app.schemas import ConfigRecord  # noqa: E402
from app.store import insert_config  # noqa: E402
from app.ui.generator import (  # noqa: E402
    ACTIVE_FORM_SOURCE,
    build_generator_panel,
    build_verification_badges,
    evaluate_generation,
)


def _collect_ui_texts(element: object) -> list[str]:
    texts: list[str] = []
    txt = getattr(element, "text", None)
    if isinstance(txt, str) and txt:
        texts.append(txt)
    slot = getattr(element, "default_slot", None)
    for child in slot.children if slot is not None else []:
        texts.extend(_collect_ui_texts(child))
    return texts


def test_lazy_nicegui_import() -> None:
    """Importing ``app.ui.generator`` must not import ``nicegui`` at module load time."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import app.ui.generator; assert 'nicegui' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("model_id", [MISTRAL_ID, DEEPSEEK_ID])
@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_evaluate_generation_all_four_defaults(
    model_id: str, task_type: str
) -> None:
    """All 4 Model x Task defaults generate valid scripts with 4 green badges."""
    cfg = ConfigRecord(**defaults_for(model_id, task_type))
    state = evaluate_generation(cfg, source_label="Active Form")

    assert state.blocked is False
    assert state.errors == []
    assert state.error_message is None
    assert state.result is not None

    result = state.result
    assert result["syntax_ok"] is True
    assert result["parity_ok"] is True
    assert result["parity_diffs"] == []
    assert result["pitfall_issues"] == []
    assert result["marker_issues"] == []

    ast.parse(result["script_code"], filename=result["script_filename"])
    assert "torch>=" in result["requirements_txt"]
    assert "transformers>=" in result["requirements_txt"]
    parsed_cfg = json.loads(result["config_json"])
    assert parsed_cfg["model_id"] == model_id
    assert parsed_cfg["task_type"] == task_type

    assert len(state.badges) == 4
    assert [b.label for b in state.badges] == [
        "Syntax OK",
        "Parameter Parity (0 diffs)",
        "6/6 Pitfalls Fixed",
        "Parser-Compatible",
    ]
    assert all(b.passed and b.tone == "success" for b in state.badges)


def test_build_verification_badges_failure_states() -> None:
    """Failed verification checks render red error badges with counts and details."""
    fake_result: GenerationResult = {
        "config_id": "12345678-aaaa-bbbb-cccc-1234567890ab",
        "config_hash": "abc",
        "task_type": "inference",
        "model_id": MISTRAL_ID,
        "script_filename": "inference_Mistral24B_12345678.py",
        "script_code": "def broken(:",
        "requirements_txt": "torch>=2.4.0\n",
        "config_json": "{}",
        "slurm_script": None,
        "script_sha256": "deadbeef",
        "syntax_ok": False,
        "syntax_error": "SyntaxError: invalid syntax (line 1)",
        "parity_ok": False,
        "parity_diffs": [
            {
                "field": "f.temperature",
                "expected": 0.0,
                "actual": 0.1,
                "reason": "Value mismatch",
            },
            {
                "field": "b.pad_token",
                "expected": "<pad>",
                "actual": "<myPadToken>",
                "reason": "Value mismatch",
            },
        ],
        "pitfall_issues": ["Found <myPadToken>"],
        "marker_issues": [
            "Missing VERDICTS_BANNER",
            "Missing indexed table",
        ],
        "warnings": [],
    }
    badges = build_verification_badges(fake_result)
    assert [b.label for b in badges] == [
        "Syntax Error",
        "Parameter Parity (2 diffs)",
        "Pitfalls (1 issue)",
        "Parser-Compatible (2 marker issues)",
    ]
    assert all((not b.passed) and b.tone == "error" for b in badges)
    assert badges[0].detail is not None and "SyntaxError" in badges[0].detail
    assert badges[1].detail is not None and "f.temperature" in badges[1].detail


def test_evaluate_generation_blocks_invalid_configs_and_lists_fields() -> None:
    """Empty Mistral system_prompt and unknown h.generate.bogus block generation."""
    # 1. Rule 9: empty Mistral system_prompt
    cfg_empty_prompt = ConfigRecord(**defaults_for(MISTRAL_ID, "fine_tune"))
    cfg_empty_prompt.p.system_prompt = ""
    state1 = evaluate_generation(cfg_empty_prompt)
    assert state1.blocked is True
    assert state1.result is None
    assert state1.badges == []
    assert any(err["field"] == "p.system_prompt" for err in state1.errors)

    # 2. Unknown Component h kwarg
    cfg_bad_kwarg = ConfigRecord(**defaults_for(DEEPSEEK_ID, "inference"))
    cfg_bad_kwarg.h = {"generate": {"bogus": 123}}
    state2 = evaluate_generation(cfg_bad_kwarg)
    assert state2.blocked is True
    assert state2.result is None
    assert any(err["field"] == "h.generate.bogus" for err in state2.errors)


def test_evaluate_generation_carries_non_blocking_warnings() -> None:
    """Rule 10 warning on DeepSeek does not block generation and appears in warnings."""
    cfg = ConfigRecord(**defaults_for(DEEPSEEK_ID, "inference"))
    cfg.p.system_prompt = "Custom instruction"
    state = evaluate_generation(cfg)
    assert state.blocked is False
    assert state.result is not None
    assert len(state.warnings) == 1
    assert "DeepSeek" in state.warnings[0]


def test_nicegui_generator_panel_mount_and_interactions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GeneratorPanel renders, switches active/saved configs, exports, and blocks errors."""
    nicegui = pytest.importorskip("nicegui")
    from nicegui import Client, context, ui
    from nicegui.page import page

    # Ensure NiceGUI script_mode slot is initialized before entering a test Client
    # so subsequent tests calling create_app() (e.g., test_main.py) keep a valid slot.
    _ = context.slot_stack

    db_path = tmp_path / "gen_ui.db"
    saved_cfg = ConfigRecord(**defaults_for(DEEPSEEK_ID, "inference"))
    insert_config(saved_cfg, db_path)

    active_cfg = ConfigRecord(**defaults_for(MISTRAL_ID, "fine_tune"))

    downloaded: list[tuple[bytes, str | None]] = []
    monkeypatch.setattr(
        nicegui.ui,
        "download",
        lambda src, filename=None, media_type="": downloaded.append((src, filename)),
    )

    with Client(page("/test-generator-ui")) as client:
        with client.content:
            container = ui.column()
            panel = build_generator_panel(
                container,
                active_config_fn=lambda: active_cfg,
                db_path=db_path,
            )

        # Initial mount generates for Active Form (Mistral fine_tune)
        assert panel.selected_source == ACTIVE_FORM_SOURCE
        assert panel.current_state is not None
        assert panel.current_state.blocked is False
        assert panel.current_result is not None
        assert panel.current_result["script_filename"].startswith("fineTune_Mistral24B_")
        assert panel.filename_label.text == panel.current_result["script_filename"]
        assert panel.error_banner.visible is False
        assert panel.result_section.visible is True
        assert [lbl.text for lbl in panel.badge_widgets] == [
            "Syntax OK",
            "Parameter Parity (0 diffs)",
            "6/6 Pitfalls Fixed",
            "Parser-Compatible",
        ]
        assert panel.script_code_view.content == panel.current_result["script_code"]
        assert (
            panel.requirements_code_view.content
            == panel.current_result["requirements_txt"]
        )
        assert panel.config_json_view.content == panel.current_result["config_json"]

        # Tab switching + Copy + Download
        panel.code_tabs.value = "requirements"
        assert panel.get_active_tab_text() == panel.current_result["requirements_txt"]
        assert panel.copy_active_code() == panel.current_result["requirements_txt"]
        panel.code_tabs.value = "config"
        assert panel.get_active_tab_text() == panel.current_result["config_json"]
        panel.code_tabs.value = "script"
        assert panel.download_script() is True
        assert downloaded[-1] == (
            panel.current_result["script_code"].encode("utf-8"),
            panel.current_result["script_filename"],
        )

        # Switch to saved DeepSeek inference config
        state_saved = panel.select_source(saved_cfg.config_id)
        assert state_saved.blocked is False
        assert state_saved.result is not None
        expected_prefix = f"inference_DeepSeek32B_{saved_cfg.config_id[:8]}"
        assert state_saved.result["script_filename"].startswith(expected_prefix)
        assert panel.filename_label.text == state_saved.result["script_filename"]

        # Switch back to Active Form after making it invalid -> error banner blocks generation
        active_cfg.p.system_prompt = ""
        active_cfg.h = {"generate": {"bogus": True}}
        state_invalid = panel.select_source(ACTIVE_FORM_SOURCE)
        assert state_invalid.blocked is True
        assert panel.error_banner.visible is True
        assert panel.result_section.visible is False
        assert panel.copy_button.enabled is False
        assert panel.download_button.enabled is False
        error_texts = _collect_ui_texts(panel.error_list_container)
        assert any("p.system_prompt" in t for t in error_texts)
        assert any("h.generate.bogus" in t for t in error_texts)


def test_script_export_tab_mounted_in_main_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Issue #60: Script Export tab is mounted in app/main.py and syncs on save."""
    pytest.importorskip("nicegui")
    monkeypatch.setenv("LLM_CONFIG_DB_PATH", str(tmp_path / "main_gen.db"))

    from app.main import create_app

    platform = create_app()
    assert platform.generator_panel is not None
    assert platform.generator_panel.current_result is not None
    assert platform.generator_panel.current_result["script_filename"].startswith(
        "fineTune_Mistral24B_"
    )

    # Switching to export tab works
    platform.side_tabs.value = "export"

    # Saving a config updates generator_panel options alongside results_panel
    cfg = platform.build_config_record()
    platform.save_to_db(cfg)
    assert cfg.config_id in platform.generator_panel.source_select.options

    # Making active form invalid and clicking Generate blocks cleanly in mounted panel
    platform.field_widgets["p.system_prompt"].value = ""
    state = platform.generator_panel.generate()
    assert state.blocked is True
    assert platform.generator_panel.error_banner.visible is True
    assert any(err["field"] == "p.system_prompt" for err in state.errors)
