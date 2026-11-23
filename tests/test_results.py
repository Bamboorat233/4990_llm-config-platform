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
from app.parsing.verdicts import TABLE_ONLY_SINGLE_TABLE
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord
from app.scoring import INVALID_LABEL, render_report_markdown
from app.store import get_run, get_score, insert_config
from app.ui.results import (
    DEFAULT_PARSE_RESULT,
    FIXTURE_LOGS_DIR,
    build_result_summary,
    build_results_panel,
    build_verdict_review_rows,
    count_out_of_set_values,
    evaluate_inference_log,
    filter_verdict_rows,
    fixture_log_options,
    is_anomalous_verdict_row,
    list_fixture_logs,
    load_gold_labels_file,
    resolve_eval_settings,
    resolve_gold_labels_path,
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


# ---------------------------------------------------------------------------
# Milestone 2.0 Dashboard & Persistence Tests (Issue #37)
# ---------------------------------------------------------------------------


def _write_gold_json(path: Path, labels: list[str]) -> Path:
    path.write_text(
        json.dumps([{"id": idx, "label": label} for idx, label in enumerate(labels)]),
        encoding="utf-8",
    )
    return path


def _write_gold_jsonl(path: Path, labels: list[str]) -> Path:
    lines = [json.dumps({"id": idx, "label": label}) for idx, label in enumerate(labels)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _collect_ui_texts(element: object) -> list[str]:
    texts: list[str] = []
    txt = getattr(element, "text", None)
    if isinstance(txt, str) and txt:
        texts.append(txt)
    for child in getattr(element, "default_slot", None).children if hasattr(element, "default_slot") else []:
        texts.extend(_collect_ui_texts(child))
    return texts


def test_resolve_eval_settings_and_gold_resolution(tmp_path: Path) -> None:
    cfg = ConfigRecord(**defaults_for(DEEPSEEK_ID, "inference"))
    cfg.c.label_values = ["True", "False", "Conflicting"]
    cfg.p.verdict_parse_marker = "### Response: Verdict: "
    settings = resolve_eval_settings(cfg)
    assert settings.model_id == DEEPSEEK_ID
    assert settings.eos_token is not None
    assert settings.marker == "### Response: Verdict: "
    assert settings.labels == ("True", "False", "Conflicting")
    assert settings.test_data_file == "data/testData.json"

    # Missing cluster path emits a clear notice; existing local path resolves cleanly
    path_missing, notice = resolve_gold_labels_path(
        config_test_data_file="nonexistent_cluster_dir/testData.json"
    )
    assert path_missing is None
    assert notice is not None
    assert "nonexistent_cluster_dir/testData.json" in notice

    gold_jsonl = _write_gold_jsonl(tmp_path / "gold.jsonl", ["True", "False", "Conflicting"])
    assert load_gold_labels_file(gold_jsonl) == ["True", "False", "Conflicting"]
    found, err_notice = resolve_gold_labels_path(user_gold_path=str(gold_jsonl))
    assert found == gold_jsonl
    assert err_notice is None


def test_verdict_review_rows_and_anomaly_filter() -> None:
    verdicts = [
        {"index": 0, "raw": "True", "parsed": "True", "kind": "exact"},
        {"index": 1, "raw": "False", "parsed": "False", "kind": "exact"},
        {"index": 2, "raw": "Conflicting</s>", "parsed": "Conflicting", "kind": "recovered"},
        {"index": 3, "raw": "Mostly True", "parsed": "Mostly True", "kind": "out_of_set"},
        {"index": 4, "raw": "No response", "parsed": None, "kind": "empty"},
    ]
    gold = ["True", "True", "Conflicting", "False", "True", "False"]

    rows = build_verdict_review_rows(verdicts, gold)
    assert len(rows) == 6
    assert rows[0]["anomalous"] is False  # exact + matches gold
    assert rows[1]["anomalous"] is True   # exact but mismatches gold (False != True)
    assert rows[2]["anomalous"] is True   # recovered (kind != exact)
    assert rows[3]["anomalous"] is True   # out_of_set
    assert rows[4]["anomalous"] is True   # empty
    assert rows[5]["kind"] == "unfinished"
    assert rows[5]["anomalous"] is True

    anomalies = filter_verdict_rows(rows, anomalies_only=True)
    assert [r["index"] for r in anomalies] == [1, 2, 3, 4, 5]
    assert len(filter_verdict_rows(rows, anomalies_only=False)) == 6
    assert is_anomalous_verdict_row(rows[0]) is False
    assert count_out_of_set_values(verdicts) == [("Mostly True", 1)]


def test_log_03_linked_config_dashboard_and_sqlite_persistence(tmp_path: Path) -> None:
    """Acceptance: Log 03 + linked config + gold labels -> dashboard + SQLite persistence."""
    from nicegui import Client, ui
    from nicegui.page import page

    db_path = tmp_path / "m2_test.db"
    gold_path = _write_gold_json(
        tmp_path / "gold_s03.json",
        # Log 03 extracted verdicts: Conflicting, True, False, True, True, True, False
        ["Conflicting", "True", "False", "True", "True", "Conflicting", "False"],
    )
    cfg = ConfigRecord(**defaults_for(MISTRAL_ID, "inference"))
    cfg.c.test_data_file = str(gold_path)
    insert_config(cfg, db_path)

    log_03 = FIXTURE_LOGS_DIR / "s03_inference_success.log"

    with Client(page("/test-m2-log03")) as client:
        with client.content:
            container = ui.column()
            panel = build_results_panel(container, db_path=db_path)

        panel.select_config(cfg.config_id)
        summary = panel.load_file(log_03)

        assert summary.family == "inference"
        assert summary.status == "success"
        assert panel.eval_section.visible is True
        assert panel.dashboard_card.visible is True
        assert panel.incomplete_banner.visible is False
        assert panel.single_table_banner.visible is False
        assert panel.verdict_table_card.visible is True

        report = panel.current_score_report
        assert report is not None
        assert report["records"] == 7
        assert report["complete"] is True
        assert report["exact"] == 7
        assert report["source"] == "parsed"
        assert report["accuracy"] == round(6 / 7, 4)

        # Dual exact/inclusive visibility & toggle
        assert f"{report['accuracy'] * 100:.2f}%" == panel.accuracy_value_label.text
        assert "Exact only:" in panel.accuracy_sub_label.text
        panel.set_exact_only(True)
        assert "Inclusive:" in panel.accuracy_sub_label.text
        panel.set_exact_only(False)

        # Verdict review table + anomalies toggle
        assert len(panel.verdict_table.rows) == 7
        anomalous_rows = panel.set_anomalies_only(True)
        assert [r["index"] for r in anomalous_rows] == [5]
        all_rows = panel.set_anomalies_only(False)
        assert len(all_rows) == 7

        # Markdown export uses render_report_markdown
        md = panel.export_markdown_report()
        assert md == render_report_markdown(report)

        # Save Run & Score -> persists to SQLite runs + scores
        assert panel.save_run_button.text == "Save Run & Score"
        run_id = panel.save_current_run()
        assert run_id is not None
        stored_run = get_run(run_id, db_path)
        stored_score = get_score(run_id, db_path)
        assert stored_run is not None
        assert stored_run["config_id"] == cfg.config_id
        assert stored_run["family"] == "inference"
        assert stored_score == report

        # Non-inference log (Log 12) hides inference dashboard and persists run-only cleanly
        log_12 = FIXTURE_LOGS_DIR / "s12_fine_tune_success_warning.log"
        panel.load_file(log_12)
        assert panel.eval_section.visible is False
        assert panel.save_run_button.text == "Save Run"
        run_12_id = panel.save_current_run()
        assert run_12_id is not None
        assert get_run(run_12_id, db_path) is not None
        assert get_score(run_12_id, db_path) is None


def test_log_15_shows_incomplete_run_banner_and_async_scoring(tmp_path: Path) -> None:
    """Acceptance: Log 15 shows the Incomplete Run banner with unfinished count."""
    from nicegui import Client, ui
    from nicegui.page import page

    db_path = tmp_path / "m2_log15.db"
    # Cluster path does not exist locally -> tests fallback + local gold path override
    cfg = ConfigRecord(**defaults_for(MISTRAL_ID, "inference"))
    cfg.c.test_data_file = "data/nonexistent_cluster_testData.json"
    insert_config(cfg, db_path)

    # Log 15 excerpt has 7 completed verdicts; use 12 gold records so 5 are unfinished
    gold_labels = [
        "Conflicting", "Conflicting", "False", "Conflicting", "True", "False", "False",
        "True", "False", "Conflicting", "True", "False",
    ]
    local_gold = _write_gold_json(tmp_path / "local_gold_15.json", gold_labels)
    log_15 = FIXTURE_LOGS_DIR / "s15_inference_terminated.log"

    with Client(page("/test-m2-log15")) as client:
        with client.content:
            container = ui.column()
            panel = build_results_panel(container, db_path=db_path)

        panel.select_config(cfg.config_id)
        asyncio.run(panel.parse_async(log_15))

        # Before local gold file is supplied, clear feedback is shown
        assert panel.current_score_report is None
        assert panel.incomplete_banner.visible is False
        assert "not found locally" in panel.gold_status_label.text

        # Supply local gold file -> scored over full 12-record gold denominator
        panel.set_gold_labels_path(local_gold)
        report = panel.current_score_report
        assert report is not None
        assert report["complete"] is False
        assert report["records"] == 12
        assert report["unfinished"] == 5
        assert panel.incomplete_banner.visible is True
        assert "5 of 12 records unfinished" in panel.incomplete_banner_label.text
        assert panel.spinner_row.visible is False

        # Anomalies toggle surfaces the 5 unfinished rows
        anomalies = panel.set_anomalies_only(True)
        assert [r["index"] for r in anomalies] == [7, 8, 9, 10, 11]
        assert all(r["kind"] == "unfinished" for r in anomalies)


def test_confusion_matrix_invalid_column_and_out_of_set_counts(tmp_path: Path) -> None:
    """Verify confusion matrix __INVALID__ column, out-of-set counts, and uploaded gold JSON."""
    from types import SimpleNamespace
    import io
    from nicegui import Client, ui
    from nicegui.page import page

    # Synthetic banner inference log containing exact, recovered, empty, out_of_set, unparsed
    synthetic_log = tmp_path / "synthetic_banner.log"
    blocks = [
        "***** INPUTS *****\ntensor([[1]])\n***** OUTPUT *****\n### Response: Verdict: True\n",
        "***** INPUTS *****\ntensor([[2]])\n***** OUTPUT *****\n### Response: Verdict: False</s>\n",
        "***** INPUTS *****\ntensor([[3]])\n***** OUTPUT *****\nNo marker here\n",
        "***** INPUTS *****\ntensor([[4]])\n***** OUTPUT *****\n### Response: Verdict: Mostly True\n",
        "***** INPUTS *****\ntensor([[5]])\n***** OUTPUT *****\n### Response: Verdict: Mostly True\n",
        "***** INPUTS *****\ntensor([[6]])\n***** OUTPUT *****\n### Response: Verdict: Half true\n",
        "***** INPUTS *****\ntensor([[7]])\n***** OUTPUT *****\n### Response: Verdict: totally invalid gibberish\n",
        "***** VERDICTS *****\n",
    ]
    synthetic_log.write_text("".join(blocks), encoding="utf-8")

    gold_bytes = json.dumps(
        [
            {"label": "True"},
            {"label": "False"},
            {"label": "Conflicting"},
            {"label": "True"},
            {"label": "False"},
            {"label": "Conflicting"},
            {"label": "True"},
        ]
    ).encode("utf-8")

    with Client(page("/test-m2-oos")) as client:
        with client.content:
            container = ui.column()
            panel = build_results_panel(container, db_path=tmp_path / "oos.db")

        panel.load_file(synthetic_log)
        upload_event = SimpleNamespace(
            name="uploaded_gold.json",
            content=io.BytesIO(gold_bytes),
        )
        asyncio.run(panel._on_gold_file_uploaded(upload_event))

        report = panel.current_score_report
        assert report is not None
        assert report["exact"] == 1
        assert report["recovered"] == 1
        assert report["empty"] == 1
        assert report["out_of_set"] == 3
        assert report["unparsed"] == 1
        assert report["unfinished"] == 0

        # Confusion matrix has __INVALID__ column populated for empty/out_of_set/unparsed
        assert report["confusion"]["True"][INVALID_LABEL] == 2
        assert report["confusion"]["False"][INVALID_LABEL] == 1
        assert report["confusion"]["Conflicting"][INVALID_LABEL] == 2

        confusion_texts = _collect_ui_texts(panel.confusion_container)
        assert "Invalid" in confusion_texts

        # Out-of-set value counts display
        assert panel.current_eval is not None
        assert panel.current_eval.out_of_set_counts == [
            ("Mostly True", 2),
            ("Half true", 1),
        ]
        oos_texts = _collect_ui_texts(panel.out_of_set_container)
        assert "Mostly True" in oos_texts
        assert "Half true" in oos_texts

        # Dual exact vs inclusive reporting: inclusive has 2/7 correct, exact has 1/7
        assert report["inclusive_metrics"]["accuracy"] == round(2 / 7, 4)
        assert report["exact_metrics"]["accuracy"] == round(1 / 7, 4)
        assert f"{round(2 / 7, 4) * 100:.2f}%" == panel.accuracy_value_label.text
        panel.set_exact_only(True)
        assert f"{round(1 / 7, 4) * 100:.2f}%" == panel.accuracy_value_label.text

        panel.clear_uploaded_gold_file()
        assert panel.uploaded_gold_path is None


def test_table_only_single_table_and_overlength_error_handling(tmp_path: Path) -> None:
    """Verify TABLE_ONLY_SINGLE_TABLE warning and ValueError handling when verdicts > gold."""
    from nicegui import Client, ui
    from nicegui.page import page

    single_table_log = tmp_path / "single_table.log"
    single_table_log.write_text(
        "\n".join(
            [
                "Done Loading the Model : 2026-02-01 00:00:00.000000",
                "Done Loading the Tokenizer : 2026-02-01 00:00:01.000000",
                "Done Preparing the Input : 2026-02-01 00:00:02.000000",
                "0 | True",
                "1 | False",
            ]
        ),
        encoding="utf-8",
    )

    with Client(page("/test-m2-edge")) as client:
        with client.content:
            container = ui.column()
            panel = build_results_panel(container, db_path=tmp_path / "edge.db")

        panel.load_file(single_table_log)
        assert panel.current_eval is not None
        assert TABLE_ONLY_SINGLE_TABLE in panel.current_eval.warnings
        assert panel.single_table_banner.visible is True
        assert panel.current_score_report is None

        # Now load Log 03 (7 verdicts) with only 2 gold labels -> graceful ValueError message
        too_short_gold = _write_gold_json(tmp_path / "short.json", ["True", "False"])
        panel.set_gold_labels_path(too_short_gold)
        panel.load_file(FIXTURE_LOGS_DIR / "s03_inference_success.log")
        assert panel.current_score_report is None
        assert panel.current_eval is not None
        assert panel.current_eval.error_message is not None
        assert "7" in panel.current_eval.error_message
        assert "2" in panel.current_eval.error_message
        assert panel.current_eval.error_message in panel.gold_status_label.text


