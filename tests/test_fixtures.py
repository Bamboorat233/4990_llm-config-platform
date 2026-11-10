"""Self-check for the log fixtures (issue #29).

Checks the frozen excerpts under tests/fixtures/logs and expected.json.
The expectations in expected.json come from the spec's log catalogue, not
from parser output. The slow test compares against the full logs and runs
only when LOGS_DIR points at a folder holding sampleOutput01.txt ...
"""
import json
import os
import re
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures" / "logs"
EXPECTED = json.loads((FIX / "expected.json").read_text(encoding="utf-8"))
LOGS = sorted(p.name for p in FIX.glob("*.log"))
PHASES = {
    "fine_tune": ["setup", "fine_tuning", "checkpoint_creation", "model_saving"],
    "inference": ["setup", "data_preparation", "generation", "verdict_output"],
    "checkpoint_export": ["setup", "model_saving"],
}
CJK_RANGES = ((0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
              (0xF900, 0xFAFF), (0xAC00, 0xD7AF))  # same ranges as .github/scripts/check_no_cjk.py


def has_cjk(text):
    return any(lo <= ord(c) <= hi for c in text for lo, hi in CJK_RANGES)


def test_sixteen_fixtures_and_expected_cover_them():
    assert len(LOGS) == 16
    assert sorted(EXPECTED) == LOGS


@pytest.mark.parametrize("name", LOGS)
def test_file_rules(name):
    raw = (FIX / name).read_bytes()
    assert len(raw) < 200_000, "fixture must stay under 200 KB"
    assert raw.count(b"\r\n") > 0, "CRLF must be preserved"
    assert raw.count(b"\n") == raw.count(b"\r\n"), "mixed line endings"
    text = raw.decode("utf-8")
    assert not has_cjk(text), "no CJK characters (the full-width bar is allowed)"
    assert "gaganm" not in text and not re.search(r"/project/(?!0000000)\d+", text)


@pytest.mark.parametrize("name", LOGS)
def test_expected_entry_is_consistent(name):
    e = EXPECTED[name]
    assert e["family"] in PHASES
    assert [p["name"] for p in e["phases"]] == PHASES[e["family"]]
    outcomes = [p["outcome"] for p in e["phases"]]
    assert set(outcomes) <= {"succeeded", "failed", "terminated", "not_reached"}
    if e["status"] == "error":
        assert outcomes.count("failed") == 1 and e["failed_phase"]
    if e["status"] == "terminated":
        assert outcomes.count("terminated") == 1 and e["failed_phase"]
    if e["status"] == "success" and e["failed_phase"] is None:
        assert set(outcomes) == {"succeeded"}
    if e["failed_phase"]:
        assert dict((p["name"], p["outcome"]) for p in e["phases"])[e["failed_phase"]] != "succeeded"
    # nothing runs after the first non-successful phase
    seen = False
    for o in outcomes:
        assert not (seen and o != "not_reached")
        seen = seen or o != "succeeded"


def test_names_match_status_and_family():
    for name, e in EXPECTED.items():
        assert e["family"].replace("_", "") in name.replace("_", "")
        assert e["status"] in name or (e["status"] == "success" and "success" in name)


def test_marker_lines_are_ascii_only():
    for name in LOGS:
        for line in (FIX / name).read_bytes().split(b"\r\n"):
            if line.startswith(b"[... "):
                assert re.fullmatch(rb"\[\.\.\. \d+ lines omitted from the original log \.\.\.\]", line)


def test_evidence_key_lines_present():
    def has(name, pat):
        return re.search(pat, (FIX / name).read_text(encoding="utf-8"))
    assert has("s06_inference_error.log", r"OutOfMemoryError|out of memory")
    assert has("s10_fine_tune_error.log", r"OutOfMemoryError|out of memory")
    assert has("s12_fine_tune_success_warning.log", r"FileNotFoundError")
    assert has("s15_inference_terminated.log", r"DUE TO TIME LIMIT")
    assert has("synthetic_fine_tune_terminated.log", r"DUE TO TIME LIMIT")
    assert has("synthetic_fine_tune_terminated.log", r"SYNTHETIC FIXTURE")


@pytest.mark.skipif(not os.environ.get("LOGS_DIR"), reason="set LOGS_DIR to the folder with the full sample logs")
def test_every_excerpt_line_exists_in_full_log():
    names = {1: "s01", 2: "s02", 3: "s03", 4: "s04", 5: "s05", 6: "s06", 7: "s07", 8: "s08", 9: "s09",
             10: "s10", 11: "s11", 12: "s12", 13: "s13", 14: "s14", 15: "s15"}
    scrub = lambda b: (b.replace(b"gaganm", b"user"))
    for n, prefix in names.items():
        full = Path(os.environ["LOGS_DIR"]) / f"sampleOutput{n:02d}.txt"
        full_lines = {re.sub(rb"/project/\d+", b"/project/0000000",
                             re.sub(rb"\bfc\d{5}\b", b"fc00000", scrub(l)))
                      for l in full.read_bytes().split(b"\n")}
        fx = next(FIX.glob(prefix + "_*.log"))
        for l in fx.read_bytes().split(b"\n"):
            if l and not l.startswith(b"[... "):
                assert l in full_lines, f"{fx.name}: line not found in full log: {l[:80]!r}"
