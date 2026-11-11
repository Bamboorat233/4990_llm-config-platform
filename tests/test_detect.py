"""Tests for issue #30: the streaming log detector.

The fixture oracle in ``tests/fixtures/logs/expected.json`` (issue #29) is
the source of truth for family, status, failed phase, phase outcomes,
output layout and the warning count.  The remaining tests pin the
issue-specific cases from the assignment, the evidence contract, streaming
behaviour on pathological input, and the ``parse_log_file`` delegation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.parsing import parse_log_file
from app.parsing.detect import detect

FIX = Path(__file__).parent / "fixtures" / "logs"
EXPECTED = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
NAMES = sorted(EXPECTED)

CONTRACT_KEYS = {
    "source_path",
    "script",
    "model_id",
    "family",
    "status",
    "scenario",
    "phases",
    "failed_phase",
    "output_format",
    "started_at",
    "finished_at",
    "duration_seconds",
    "progress",
    "errors",
    "warnings",
    "evidence",
    "notes",
}


def outcomes(result: dict) -> list[str]:
    """Return the phase outcomes in order."""

    return [phase["outcome"] for phase in result["phases"]]


def phase_pairs(result: dict) -> list[tuple[str, str]]:
    """Return ``(name, outcome)`` pairs in order."""

    return [(phase["name"], phase["outcome"]) for phase in result["phases"]]


@pytest.mark.parametrize("name", NAMES)
def test_matches_fixture_oracle(name):
    """Every fixture matches expected.json exactly on the oracle keys."""

    result = detect(FIX / name)
    expected = EXPECTED[name]
    assert result["family"] == expected["family"]
    assert result["status"] == expected["status"]
    assert result["failed_phase"] == expected["failed_phase"]
    assert phase_pairs(result) == [
        (phase["name"], phase["outcome"]) for phase in expected["phases"]
    ]
    assert result["output_format"] == expected["output_format"]
    assert len(result["warnings"]) == expected["warnings"]


@pytest.mark.parametrize("name", NAMES)
def test_result_has_every_contract_key(name):
    """The result has exactly the seventeen ParseResult keys."""

    result = detect(FIX / name)
    assert set(result) == CONTRACT_KEYS
    assert result["scenario"] == f"{result['family']}_{result['status']}"
    for item in result["phases"]:
        assert set(item) == {"name", "outcome", "evidence_line"}
    for item in result["errors"] + result["warnings"]:
        assert set(item) == {"kind", "message", "line", "rank"}
    for item in result["evidence"]:
        assert set(item) == {"rule", "line", "text"}


def test_s12_is_success_with_one_warning():
    """Log 12: three successes, model saving failed, success with one warning."""

    result = detect(FIX / "s12_fine_tune_success_warning.log")
    assert outcomes(result) == ["succeeded", "succeeded", "succeeded", "failed"]
    assert result["status"] == "success"
    assert result["failed_phase"] == "model_saving"
    assert len(result["warnings"]) == 1


@pytest.mark.parametrize(
    "name",
    [
        "s08_fine_tune_error.log",
        "s09_fine_tune_error.log",
        "s11_fine_tune_error.log",
    ],
)
def test_library_failures_stop_in_setup(name):
    """Logs 08, 09 and 11 fail in setup."""

    result = detect(FIX / name)
    assert result["failed_phase"] == "setup"
    assert outcomes(result)[0] == "failed"
    assert result["status"] == "error"


def test_s15_is_terminated_in_generation():
    """Log 15 is terminated in generation."""

    result = detect(FIX / "s15_inference_terminated.log")
    assert result["failed_phase"] == "generation"
    assert outcomes(result)[2] == "terminated"
    assert result["status"] == "terminated"


@pytest.mark.parametrize("name", NAMES)
def test_every_resolved_phase_has_evidence(name):
    """Each outcome other than not_reached carries non-empty evidence."""

    result = detect(FIX / name)
    for phase in result["phases"]:
        if phase["outcome"] == "not_reached":
            continue
        prefix = f"phase_{phase['name']}_"
        matches = [
            entry for entry in result["evidence"] if entry["rule"].startswith(prefix)
        ]
        assert matches, phase["name"]
        assert any(entry["text"].strip() for entry in matches), phase["name"]


@pytest.mark.parametrize("name", NAMES)
def test_errors_and_warnings_carry_a_location(name):
    """Every error and warning has a line, or None plus a note explaining why."""

    result = detect(FIX / name)
    for item in result["errors"] + result["warnings"]:
        assert "line" in item and "rank" in item
        if item["line"] is None:
            assert result["notes"], item


def test_empty_file_is_unknown(tmp_path):
    """An empty file is unclassifiable and never raises."""

    path = tmp_path / "empty.log"
    path.write_bytes(b"")
    result = detect(path)
    assert result["family"] == "unknown"
    assert result["status"] == "unknown"
    assert result["scenario"] == "unknown_unknown"
    assert result["failed_phase"] is None
    assert result["phases"] == []
    assert result["output_format"] is None


def test_large_junk_file_streams_quickly(tmp_path):
    """100k junk lines are processed without loading the file or raising."""

    path = tmp_path / "junk.log"
    path.write_bytes(b"junk line\n" * 100_000)
    start = time.perf_counter()
    result = detect(path)
    elapsed = time.perf_counter() - start
    assert result["family"] == "unknown"
    assert result["status"] == "unknown"
    assert result["phases"] == []
    assert elapsed < 2.0, f"detect() took {elapsed:.2f}s on 100k junk lines"


def test_truncated_run_terminates_in_current_phase(tmp_path):
    """A file cut off before its current phase completes is terminated."""

    path = tmp_path / "truncated.log"
    path.write_text(
        "Done Loading the Train Data :  2026-01-01 00:00:02.000000\n"
        "Done Loading the Eval Data :  2026-01-01 00:00:03.000000\n"
        " 50%|#####     | 50/100 [00:01<00:01,  1.00it/s]\n",
        encoding="utf-8",
    )
    result = detect(path)
    assert result["family"] == "fine_tune"
    assert result["status"] == "terminated"
    assert result["failed_phase"] == "fine_tuning"
    assert result["phases"][1]["outcome"] == "terminated"


def test_parse_log_file_delegates_to_detect():
    """parse_log_file() returns exactly what detect() returns for one fixture."""

    path = FIX / "s12_fine_tune_success_warning.log"
    assert parse_log_file(path) == detect(path)
