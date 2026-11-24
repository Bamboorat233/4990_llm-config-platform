"""M2.0 acceptance checklist tests (issue #38).

Turns each Milestone 1.5 and Milestone 2.0 acceptance criterion into an
executable assertion across the real pipeline:

    log fixture -> app.parsing.parse_log_file()
        -> app.parsing.verdicts.extract_verdicts_with_source()
        -> app.scoring.score_verdicts() / render_report_markdown()
        -> app.store.insert_run() / upsert_score()
        -> app.ui.results.ResultsPanel (M1.5 summary + M2.0 dashboard)

All tests are self-contained against the trimmed fixtures under
``tests/fixtures/logs/`` and temporary SQLite databases. Full-log checks
remain gated by ``LOGS_DIR`` in the module-level slow tests.
"""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import store  # noqa: E402
from app.parsing import parse_log_file  # noqa: E402
from app.parsing.verdicts import (  # noqa: E402
    TABLE_ONLY_SINGLE_TABLE,
    extract_verdicts_with_source,
)
from app.registry import MISTRAL_ID, defaults_for  # noqa: E402
from app.schemas import ConfigRecord  # noqa: E402
from app.scoring import (  # noqa: E402
    INVALID_LABEL,
    VerdictRecord,
    render_report_markdown,
    score_verdicts,
)
from app.ui.results import (  # noqa: E402
    FIXTURE_LOGS_DIR,
    build_result_summary,
    build_results_panel,
    build_verdict_review_rows,
    count_out_of_set_values,
    filter_verdict_rows,
)

EXPECTED_JSON_PATH = FIXTURE_LOGS_DIR / "expected.json"
DEFAULT_LABELS = ["True", "False", "Conflicting"]
DEFAULT_MARKER = "### Response: Verdict: "


def _load_expected() -> dict[str, dict]:
    with EXPECTED_JSON_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_gold_json(path: Path, labels: list[str]) -> Path:
    rows = [{"id": idx, "label": label} for idx, label in enumerate(labels)]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def _make_inference_config(test_data_file: str) -> ConfigRecord:
    payload = defaults_for(MISTRAL_ID, "inference")
    payload["c"]["test_data_file"] = test_data_file
    return ConfigRecord(**payload)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "m2_acceptance.db"


# ======================================================================
# 1. Sample log catalogue, fixtures and detection (M1.5 -> M2.0 base)
# ======================================================================


def test_all_16_fixtures_exist_keep_crlf_and_stay_under_200kb() -> None:
    """M2.0 acceptance: 15 sample excerpts + 1 synthetic fixture, CRLF, < 200 KB."""
    expected = _load_expected()
    assert len(expected) == 16
    for log_name in sorted(expected):
        path = FIXTURE_LOGS_DIR / log_name
        assert path.is_file(), f"Missing fixture {log_name}"
        raw = path.read_bytes()
        assert len(raw) < 200 * 1024, f"Fixture exceeds 200 KB: {log_name}"
        assert b"\r\n" in raw, f"Fixture lost CRLF line endings: {log_name}"


@pytest.mark.parametrize("log_name", sorted(_load_expected().keys()))
def test_every_fixture_matches_oracle_family_status_phases_and_format(
    log_name: str,
) -> None:
    """M1.5/M2.0 acceptance: all 15 real logs + synthetic match expected.json."""
    expected = _load_expected()[log_name]
    result = parse_log_file(FIXTURE_LOGS_DIR / log_name)

    assert result["family"] == expected["family"]
    assert result["status"] == expected["status"]
    assert result["failed_phase"] == expected["failed_phase"]
    assert result["output_format"] == expected["output_format"]
    assert [
        {"name": p["name"], "outcome": p["outcome"]} for p in result["phases"]
    ] == expected["phases"]


def test_log_12_is_success_with_model_saving_warning_and_training_progress() -> None:
    """M2.0 acceptance: log 12 is success with a yellow model_saving warning."""
    result = parse_log_file(FIXTURE_LOGS_DIR / "s12_fine_tune_success_warning.log")
    summary = build_result_summary(result)

    assert result["family"] == "fine_tune"
    assert result["status"] == "success"
    assert result["failed_phase"] == "model_saving"
    assert [p["outcome"] for p in result["phases"]] == [
        "succeeded",
        "succeeded",
        "succeeded",
        "failed",
    ]
    assert result["progress"]["step"] == 9936
    assert result["progress"]["total_steps"] == 9936
    assert len(result["progress"]["eval_losses"]) == 2

    # UI presentation: yellow warning, never red error.
    assert summary.status_tone == "warning"
    assert summary.first_alert is not None
    assert summary.first_alert.is_warning is True
    assert summary.first_alert.is_error is False


def test_log_15_is_terminated_in_generation_with_partial_records() -> None:
    """M2.0 acceptance: log 15 is terminated in generation below full test size."""
    result = parse_log_file(FIXTURE_LOGS_DIR / "s15_inference_terminated.log")
    assert result["family"] == "inference"
    assert result["status"] == "terminated"
    assert result["failed_phase"] == "generation"
    assert result["progress"]["records_done"] < 2495


# ======================================================================
# 2. Verdict extraction and 5-kind normalisation (app/parsing/verdicts.py)
# ======================================================================


def test_log_02_eos_token_verdicts_are_recovered_and_dual_scored() -> None:
    """M2.0 acceptance: False</s> is recovered; exact=0% vs inclusive>0%."""
    log_path = FIXTURE_LOGS_DIR / "s02_inference_success.log"
    source, verdicts = extract_verdicts_with_source(
        log_path,
        marker=DEFAULT_MARKER,
        labels=DEFAULT_LABELS,
    )
    assert source == "parsed"
    assert len(verdicts) > 0
    assert all(v["kind"] == "recovered" for v in verdicts)
    assert any("</s>" in v["raw"] for v in verdicts)

    gold = [v["parsed"] or "False" for v in verdicts]
    report = score_verdicts(
        verdicts,
        gold,
        labels=DEFAULT_LABELS,
        marker=DEFAULT_MARKER,
        source=source,
    )
    assert report["exact"] == 0
    assert report["recovered"] == len(verdicts)
    assert report["exact_metrics"]["accuracy"] == 0.0
    assert report["inclusive_metrics"]["accuracy"] == 1.0
    assert report["accuracy"] == report["exact_metrics"]["accuracy"]


def test_log_03_clean_banner_outputs_are_exact() -> None:
    """M2.0 acceptance: log 03 banner outputs extract as exact verdicts."""
    log_path = FIXTURE_LOGS_DIR / "s03_inference_success.log"
    source, verdicts = extract_verdicts_with_source(
        log_path,
        marker=DEFAULT_MARKER,
        labels=DEFAULT_LABELS,
    )
    assert source == "parsed"
    assert len(verdicts) > 0
    assert all(v["kind"] == "exact" for v in verdicts)


def test_empty_response_placeholder_out_of_set_and_unparsed_rules(
    tmp_path: Path,
) -> None:
    """M2.0 acceptance: Empty Response/<verdict> -> empty; out_of_set & unparsed preserved."""
    synthetic_think = tmp_path / "think_edge_cases.log"
    synthetic_think.write_text(
        "\r\n".join(
            [
                "Done Loading the Model: 2026-11-01 00:00:01.000000",
                "Done Loading the Tokenizer: 2026-11-01 00:00:02.000000",
                "Done Preparing the Input: 2026-11-01 00:00:03.000000",
                "Tokenizing Inputs: 2026-11-01 00:00:03.500000",
                "Done Decoding the Output: 2026-11-01 00:00:04.000000",
                "Decoded output:",
                "### Response:",
                "Verdict: <verdict>",
                "Tokenizing Inputs: 2026-11-01 00:00:04.500000",
                "Done Decoding the Output: 2026-11-01 00:00:05.000000",
                "Decoded output:",
                "### Response:",
                "<think>reasoning here</think>",
                "Verdict: Mostly True",
                "Tokenizing Inputs: 2026-11-01 00:00:05.500000",
                "Done Decoding the Output: 2026-11-01 00:00:06.000000",
                "Decoded output:",
                "### Response:",
                "<think>reasoning here</think>",
                "Verdict: gibberish answer that cannot be mapped",
                "Tokenizing Inputs: 2026-11-01 00:00:06.500000",
                "Done Decoding the Output: 2026-11-01 00:00:07.000000",
                "Decoded output:",
                "### Response:",
                "<think>reasoning here</think>",
                "Verdict: Conflicting.",
            ]
        )
        + "\r\n",
        encoding="utf-8",
    )

    source, verdicts = extract_verdicts_with_source(
        synthetic_think,
        marker=DEFAULT_MARKER,
        layout="think",
        labels=DEFAULT_LABELS,
    )
    assert source == "parsed"
    assert [v["kind"] for v in verdicts] == [
        "empty",
        "out_of_set",
        "unparsed",
        "recovered",
    ]
    assert verdicts[0]["parsed"] is None
    assert verdicts[1]["parsed"] == "Mostly True"
    assert verdicts[2]["parsed"] is None
    assert "gibberish" in verdicts[2]["raw"]
    assert verdicts[3]["parsed"] == "Conflicting"

    # Empty, out_of_set, and unparsed all count as false (__INVALID__) predictions.
    gold = ["True", "True", "False", "Conflicting"]
    report = score_verdicts(
        verdicts,
        gold,
        labels=DEFAULT_LABELS,
        marker=DEFAULT_MARKER,
        source=source,
    )
    assert report["empty"] == 1
    assert report["out_of_set"] == 1
    assert report["unparsed"] == 1
    assert report["recovered"] == 1
    assert report["accuracy"] == 0.0
    assert report["exact_metrics"]["accuracy"] == 0.0
    assert report["inclusive_metrics"]["accuracy"] == 0.25
    assert report["empty_rate"] == 0.25
    assert report["invalid_rate"] == 0.75
    assert report["confusion"]["True"][INVALID_LABEL] == 2
    assert report["confusion"]["False"][INVALID_LABEL] == 1
    assert report["confusion"]["Conflicting"][INVALID_LABEL] == 1
    assert report["inclusive_metrics"]["confusion"]["Conflicting"]["Conflicting"] == 1
    assert count_out_of_set_values(verdicts) == [("Mostly True", 1)]


def test_single_table_only_log_emits_machine_readable_warning(
    tmp_path: Path,
) -> None:
    """M2.0 acceptance: lone table_only output warns via TABLE_ONLY_SINGLE_TABLE."""
    single_table = tmp_path / "single_table.log"
    single_table.write_text(
        "0 | True\r\n1 | False\r\n2 | Conflicting\r\n",
        encoding="utf-8",
    )
    warnings: list[str] = []
    source, verdicts = extract_verdicts_with_source(
        single_table,
        marker=DEFAULT_MARKER,
        layout="table_only",
        labels=DEFAULT_LABELS,
        warnings=warnings,
    )
    assert source == "script_table"
    assert verdicts == []
    assert TABLE_ONLY_SINGLE_TABLE in warnings


# ======================================================================
# 3. Scoring guarantees (app/scoring.py)
# ======================================================================


def test_log_15_scores_against_full_2495_gold_denominator() -> None:
    """M2.0 acceptance: partial run (log 15) scores against 2,495 gold records."""
    log_path = FIXTURE_LOGS_DIR / "s15_inference_terminated.log"
    source, verdicts = extract_verdicts_with_source(
        log_path,
        marker=DEFAULT_MARKER,
        labels=DEFAULT_LABELS,
    )
    assert 0 < len(verdicts) < 2495

    gold = [DEFAULT_LABELS[i % 3] for i in range(2495)]
    report = score_verdicts(
        verdicts,
        gold,
        labels=DEFAULT_LABELS,
        marker=DEFAULT_MARKER,
        source=source,
    )
    assert report["records"] == 2495
    assert report["complete"] is False
    assert report["unfinished"] == 2495 - len(verdicts)
    assert report["invalid_rate"] > 0.99
    assert sum(
        sum(row.values()) for row in report["confusion"].values()
    ) == 2495

    md = render_report_markdown(report)
    assert "2495" in md
    assert INVALID_LABEL in md or "Invalid" in md


def test_more_verdicts_than_gold_raises_value_error() -> None:
    """M2.0 acceptance: more verdicts than gold labels must raise ValueError."""
    verdicts: list[VerdictRecord] = [
        {"index": 0, "raw": "True", "parsed": "True", "kind": "exact"},
        {"index": 1, "raw": "False", "parsed": "False", "kind": "exact"},
    ]
    with pytest.raises(ValueError):
        score_verdicts(
            verdicts,
            ["True"],
            labels=DEFAULT_LABELS,
            marker=DEFAULT_MARKER,
            source="parsed",
        )


def test_zero_support_class_never_produces_nan_or_division_error() -> None:
    """M2.0 acceptance: zero-support classes yield 0.0, never NaN."""
    verdicts: list[VerdictRecord] = [
        {"index": 0, "raw": "True", "parsed": "True", "kind": "exact"},
        {"index": 1, "raw": "True", "parsed": "True", "kind": "exact"},
    ]
    report = score_verdicts(
        verdicts,
        ["True", "True"],
        labels=DEFAULT_LABELS,
        marker=DEFAULT_MARKER,
        source="parsed",
    )
    for key in ("accuracy", "macro_f1", "micro_f1", "weighted_f1"):
        assert math.isfinite(report[key])
    assert report["precision"]["Conflicting"] == 0.0
    assert report["recall"]["Conflicting"] == 0.0
    assert report["f1"]["Conflicting"] == 0.0


# ======================================================================
# 4. Persistence: runs and scores tables (app/store.py)
# ======================================================================


def test_store_persists_runs_and_scores_idempotently_and_migrates_m1_db(
    tmp_path: Path,
    db_path: Path,
) -> None:
    """M2.0 acceptance: duplicate run reuses ID; re-score keeps one row; M1 DB opens cleanly."""
    # Simulate an existing M1 database that only has the configs table.
    with closing(sqlite3.connect(db_path)) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE configs (
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

    gold_path = _write_gold_json(tmp_path / "gold.json", ["True"] * 7)
    cfg = _make_inference_config(str(gold_path))
    config_id = store.insert_config(cfg, db_path=db_path)

    log_path = FIXTURE_LOGS_DIR / "s03_inference_success.log"
    parse_result = parse_log_file(log_path)
    source, verdicts = extract_verdicts_with_source(
        log_path,
        marker=cfg.p.verdict_parse_marker,
        labels=cfg.c.label_values,
    )
    gold = [v["parsed"] or "True" for v in verdicts]
    report_v1 = score_verdicts(
        verdicts,
        gold,
        labels=cfg.c.label_values,
        marker=cfg.p.verdict_parse_marker,
        source=source,
    )

    run_id_1 = store.insert_run(parse_result, config_id, db_path=db_path)
    run_id_2 = store.insert_run(parse_result, config_id, db_path=db_path)
    assert run_id_1 == run_id_2
    assert len(store.list_runs(config_id=config_id, db_path=db_path)) == 1

    store.upsert_score(run_id_1, report_v1, db_path=db_path)
    report_v2 = dict(report_v1, accuracy=0.5)
    store.upsert_score(run_id_1, report_v2, db_path=db_path)

    stored_run = store.get_run(run_id_1, db_path=db_path)
    assert stored_run is not None
    assert stored_run["payload"]["family"] == "inference"

    stored_score = store.get_score(run_id_1, db_path=db_path)
    assert stored_score is not None
    assert stored_score["accuracy"] == 0.5
    assert len(store.list_scores_for_config(config_id, db_path=db_path)) == 1


# ======================================================================
# 5. End-to-end Results Panel M2.0 dashboard (app/ui/results.py)
# ======================================================================


def test_results_panel_m2_end_to_end_log03_save_and_log15_incomplete(
    tmp_path: Path,
    db_path: Path,
) -> None:
    """M2.0 acceptance: Log 03 -> dashboard + SQLite; Log 15 -> incomplete-run banner."""
    from nicegui import Client, ui
    from nicegui.page import page

    log03_path = FIXTURE_LOGS_DIR / "s03_inference_success.log"
    _, verdicts_03 = extract_verdicts_with_source(
        log03_path,
        marker=DEFAULT_MARKER,
        labels=DEFAULT_LABELS,
    )
    # Introduce 1 intentional mismatch to verify anomaly toggle filtering.
    gold_03 = [v["parsed"] or "True" for v in verdicts_03]
    gold_03[0] = "False" if gold_03[0] != "False" else "True"
    gold_03_path = _write_gold_json(tmp_path / "gold03.json", gold_03)

    cfg = _make_inference_config(str(gold_03_path))
    config_id = store.insert_config(cfg, db_path=db_path)

    with Client(page("/test-m2-checklist")) as client:
        with client.content:
            panel = build_results_panel(ui.column(), db_path=db_path)

        # Log 03 with linked config: full dashboard + SQLite persistence.
        panel.select_config(config_id)
        panel.load_file(log03_path)
        assert panel.current_score_report is not None
        assert panel.current_score_report["complete"] is True
        assert panel.incomplete_banner.visible is False
        assert panel.dashboard_card.visible is True

        # Anomaly toggle filters down to the 1 mismatched record.
        anomalous = panel.set_anomalies_only(True)
        assert len(anomalous) == 1
        assert anomalous[0]["index"] == 0
        assert len(panel.set_anomalies_only(False)) == len(verdicts_03)

        saved_run_id = panel.save_current_run()
        assert saved_run_id is not None
        assert store.get_run(saved_run_id, db_path=db_path) is not None
        assert store.get_score(saved_run_id, db_path=db_path) is not None

        # Log 15 with larger gold set: Incomplete Run banner visible.
        log15_path = FIXTURE_LOGS_DIR / "s15_inference_terminated.log"
        gold_15_path = _write_gold_json(
            tmp_path / "gold15.json",
            [DEFAULT_LABELS[i % 3] for i in range(20)],
        )
        panel.load_file(log15_path)
        panel.set_gold_labels_path(gold_15_path)
        assert panel.current_score_report is not None
        assert panel.current_score_report["complete"] is False
        assert panel.current_score_report["unfinished"] > 0
        assert panel.incomplete_banner.visible is True


def test_pure_modules_have_no_top_level_nicegui_import() -> None:
    """M2.0 architecture rule: parsing, scoring, store, and ui.results stay lazy."""
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "import app.parsing, app.parsing.verdicts, app.scoring, app.store, app.ui.results; "
        "assert 'nicegui' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_repository_language_policy_has_no_cjk() -> None:
    """Repo CI rule: check_no_cjk.py must pass with zero CJK characters."""
    script = ROOT / ".github" / "scripts" / "check_no_cjk.py"
    assert script.is_file()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
