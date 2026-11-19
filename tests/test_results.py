"""Unit and UI tests for Milestone 1.5 Run Results panel (`app/ui/results.py`).

Covers:
1. Pure summary view generation (`build_result_summary`) without NiceGUI loaded.
2. Acceptance criterion 1: Loading each of logs 01-15 (`s01`..`s15` + synthetic)
   shows the `family`, `status`, and `phase` outcomes in spec section 3.1.
3. Acceptance criterion 2: Log 12 (`s12_fine_tune_success_warning.log`) shows a
   yellow warning (`tone == "warning"`), NOT a red error (`tone == "error"`).
4. NiceGUI `ResultsPanel` rendering, config selector refresh, and `parse_async`
   with spinner visibility toggling.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.parsing import parse_log_file
from app.ui.results import (
    DEFAULT_PARSE_RESULT,
    FIXTURE_LOGS_DIR,
    build_result_summary,
    build_results_panel,
    fixture_log_options,
    list_fixture_logs,
)

EXPECTED_JSON_PATH = FIXTURE_LOGS_DIR / "expected.json"


def _load_expected() -> dict[str, dict]:
    with EXPECTED_JSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_lazy_nicegui_import() -> None:
    """Importing `app.ui.results` must not import `nicegui` at module load time."""
    repo_root = str(Path(__file__).resolve().parents[1])
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {repo_root!r}); import app.ui.results; assert 'nicegui' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_list_fixture_logs_includes_s01_to_s15() -> None:
    logs = list_fixture_logs()
    names = [p.name for p in logs]
    for idx in range(1, 16):
        prefix = f"s{idx:02d}_"
        assert any(name.startswith(prefix) for name in names), f"Missing {prefix}*"
    opts = fixture_log_options()
    assert len(opts) >= 15


def test_default_parse_result_summary() -> None:
    summary = build_result_summary(DEFAULT_PARSE_RESULT)
    assert summary.family == "fine_tune"
    assert summary.status == "success"
    assert summary.status_tone == "success"
    assert len(summary.phases) == 4
    assert all(p.outcome == "succeeded" for p in summary.phases)
    assert summary.first_alert is None


@pytest.mark.parametrize("log_name", sorted(_load_expected().keys()))
def test_all_logs_01_to_15_match_expected_family_status_and_phases(
    log_name: str,
) -> None:
    """Acceptance Criterion 1: Loading each of logs 01-15 shows family, status, and phase outcomes."""
    expected = _load_expected()[log_name]
    log_path = FIXTURE_LOGS_DIR / log_name
    result = parse_log_file(log_path)
    summary = build_result_summary(result)

    assert summary.family == expected["family"]
    assert summary.status == expected["status"]
    assert summary.failed_phase == expected["failed_phase"]

    actual_phases = [
        {"name": p.name, "outcome": p.outcome} for p in summary.phases
    ]
    assert actual_phases == expected["phases"]


def test_log_12_shows_yellow_warning_not_red_error() -> None:
    """Acceptance Criterion 2: Log 12 shows a yellow warning, not a red error."""
    log_12_path = FIXTURE_LOGS_DIR / "s12_fine_tune_success_warning.log"
    result = parse_log_file(log_12_path)
    summary = build_result_summary(result)

    assert summary.family == "fine_tune"
    assert summary.status == "success"
    # Overall badge tone must be yellow warning, never red error
    assert summary.status_tone == "warning"
    assert summary.status_tone != "error"

    # The post-core failed phase (model_saving) must also render with warning tone, not error
    phase_by_name = {p.name: p for p in summary.phases}
    assert phase_by_name["setup"].outcome == "succeeded"
    assert phase_by_name["fine_tuning"].outcome == "succeeded"
    assert phase_by_name["checkpoint_creation"].outcome == "succeeded"
    assert phase_by_name["model_saving"].outcome == "failed"
    assert phase_by_name["model_saving"].tone == "warning"

    # Primary diagnostic callout must be a warning, not an error
    assert summary.first_alert is not None
    assert summary.first_alert.is_warning is True
    assert summary.first_alert.is_error is False
    assert summary.first_alert.tone == "warning"
    assert "model_saving" in summary.first_alert.message
    assert summary.first_alert.detail is not None
    assert "FileNotFoundError" in summary.first_alert.detail


def test_error_log_shows_red_error_alert() -> None:
    """True error logs (e.g., Log 01) must render a red error alert."""
    log_01_path = FIXTURE_LOGS_DIR / "s01_inference_error.log"
    result = parse_log_file(log_01_path)
    summary = build_result_summary(result)

    assert summary.status == "error"
    assert summary.status_tone == "error"
    assert summary.first_alert is not None
    assert summary.first_alert.is_error is True
    assert summary.first_alert.is_warning is False


def test_results_panel_ui_renders_and_parses_async() -> None:
    """Mount `ResultsPanel` in NiceGUI and verify `load_file` and `parse_async` on Log 12."""
    from nicegui import Client, ui
    from nicegui.page import page

    with Client(page("/test-results")) as client:
        with client.content:
            container = ui.column()
            panel = build_results_panel(
                container,
                list_configs_fn=lambda: [
                    {
                        "config_id": "12345678-aaaa-bbbb-cccc-1234567890ab",
                        "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
                        "task_type": "fine_tune",
                    }
                ],
            )

        assert panel.current_summary is not None
        assert len(panel.config_select.options) == 1

        # Load Log 12 via synchronous helper and verify warning tone
        log_12 = FIXTURE_LOGS_DIR / "s12_fine_tune_success_warning.log"
        summary_12 = panel.load_file(log_12)
        assert summary_12.status_tone == "warning"
        assert summary_12.first_alert is not None
        assert summary_12.first_alert.is_warning is True

        # Also test async parse (`run.io_bound` path) on Log 02
        log_02 = FIXTURE_LOGS_DIR / "s02_inference_success.log"
        summary_02 = asyncio.run(panel.parse_async(log_02))
        assert summary_02.family == "inference"
        assert summary_02.status == "success"
        assert summary_02.status_tone == "success"
        assert panel.spinner_row.visible is False

        # Test Up / Down preset log cycling and error cycling
        cycled_preset = panel.cycle_preset(1)
        assert cycled_preset is not None
        cycled_err = panel.cycle_error(1)
        assert cycled_err.first_alert is not None

        # Test multi-file upload inside the Upload dropdown and per-file Remove
        from types import SimpleNamespace
        import io

        fake_upload_1 = SimpleNamespace(
            name="sampleOutput15.txt",
            content=io.BytesIO(
                (FIXTURE_LOGS_DIR / "s01_inference_error.log").read_bytes()
            ),
        )
        fake_upload_2 = SimpleNamespace(
            name="sampleOutput10.txt",
            content=io.BytesIO(
                (FIXTURE_LOGS_DIR / "s02_inference_success.log").read_bytes()
            ),
        )
        asyncio.run(panel._on_file_uploaded(fake_upload_1))
        asyncio.run(panel._on_file_uploaded(fake_upload_2))
        assert len(panel.uploaded_files) == 2
        assert [f.name for f in panel.uploaded_files] == [
            "sampleOutput15.txt",
            "sampleOutput10.txt",
        ]
        assert panel.uploaded_files_container.visible is True

        # Remove the first uploaded file; the second remains in the dropdown
        first_temp = panel.uploaded_files[0].temp_path
        second_temp = panel.uploaded_files[1].temp_path
        panel.remove_uploaded_file(first_temp)
        assert len(panel.uploaded_files) == 1
        assert panel.uploaded_files[0].name == "sampleOutput10.txt"
        assert not first_temp.exists()
        assert second_temp.exists()

        # Clear remaining uploaded files
        panel.clear_uploaded_file()
        assert len(panel.uploaded_files) == 0
        assert panel.uploaded_files_container.visible is False
        assert not second_temp.exists()

