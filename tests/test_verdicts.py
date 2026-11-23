"""Regression tests for issue #34 (synthetic tests require no private logs)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.parsing.verdicts import extract_verdicts, normalize_verdict

LABELS = ("True", "False", "Conflicting")
MARKER = "### Response: Verdict: "
INPUTS = "*" * 15 + "INPUTS" + "*" * 15
OUTPUT = "*" * 15 + "OUTPUT" + "*" * 15
TS = " :  2026-02-20 10:45:18.187166"


@pytest.mark.parametrize("raw,parsed,kind", [
    ("True", "True", "exact"),
    ("False", "False", "exact"),
    ("Conflicting", "Conflicting", "exact"),
    ("False</s>", "False", "recovered"),
    ("Conflicting.", "Conflicting", "recovered"),
    ("Verdict: true", "True", "recovered"),
    ("True<｜end▁of▁sentence｜>", "True", "recovered"),
    ("Mostly True", None, "out_of_set"),
    ("Half true", None, "out_of_set"),
    ("Not enough information", None, "out_of_set"),
    ("The claim looks true", None, "unparsed"),
    ("Verdict: <verdict>", None, "empty"),
    ("<verdict>", None, "empty"),
    ("Empty Response", None, "empty"),
    ("No response", None, "empty"),
    ("", None, "empty"),
])
def test_normalize(raw, parsed, kind):
    assert normalize_verdict(raw, LABELS) == (parsed, kind)


def test_custom_labels_case_insensitive_recovery():
    assert normalize_verdict("supported.", ("Supported", "Refuted")) == ("Supported", "recovered")
    assert normalize_verdict("True", ("Supported", "Refuted")) == (None, "unparsed")


def test_banner_rfind_with_prompt_mention_and_multiple_records(tmp_path: Path):
    log = tmp_path / "banner.log"
    log.write_text("\n".join([
        INPUTS, OUTPUT,
        "Give the response in the format - Verdict: <verdict>",
        "### Response: Verdict: True</s>",
        INPUTS, OUTPUT,
        "### Response: Verdict: False</s>",
    ]))
    records = extract_verdicts(log, output_format="banner")
    assert [(r["index"], r["parsed"], r["kind"]) for r in records] == [
        (0, "True", "recovered"), (1, "False", "recovered")
    ]


def test_banner_empty_closed_by_next_inputs_but_cutoff_last_ignored(tmp_path: Path):
    log = tmp_path / "banner.log"
    log.write_text("\n".join([
        INPUTS, OUTPUT, "unusable output with no marker",
        INPUTS, OUTPUT, "incomplete last record",
    ]))
    result = extract_verdicts(log, output_format="banner")
    assert len(result) == 1
    assert result[0]["kind"] == "empty"


def test_banner_last_record_summary_not_extra_records(tmp_path: Path):
    log = tmp_path / "banner.log"
    log.write_text("\n".join([
        INPUTS, OUTPUT, "### Response: Verdict: False",
        "### Response: Verdict: True", "### Response: Verdict: True",
        "### Response: Verdict: True", "### Response: Verdict: True",
    ]))
    result = extract_verdicts(log, output_format="banner")
    assert len(result) == 1
    assert result[0]["parsed"] == "False"


def test_think_reasoning_and_fallback(tmp_path: Path):
    log = tmp_path / "think.log"
    def block(text):
        return "\n".join([
            "Tokenizing Inputs" + TS, "Done Decoding" + TS,
            "Decoded output:",
            "Give the response in the format - Verdict: <verdict>",
            "### Response: " + text,
        ])
    log.write_text("\n".join([
        block("<think>Reason about the evidence</think>\nVerdict: True"),
        block("Verdict: <verdict>"),
        block("<think>Unfinished reasoning"),
        "0 | Verdict: True", "1 | Verdict: <verdict>", "2 | Verdict: <verdict>",
        "0 | True", "1 | False", "2 | Empty Response",
    ]))
    result = extract_verdicts(log, output_format="think")
    assert [(r["parsed"], r["kind"]) for r in result] == [
        ("True", "recovered"), ("False", "recovered"), (None, "empty")
    ]
    assert result[1]["raw"] == "Verdict: <verdict>"


def test_table_only_uses_final_table_and_marks_fallback(tmp_path: Path):
    log = tmp_path / "tables.log"
    log.write_text("\n".join([
        "0 | Verdict: True", "1 | Verdict: <verdict>",
        "2 | Verdict: Half true", "3 | Verdict: <verdict>",
        "0 | True", "1 | Conflicting", "2 | Half true", "3 | Empty Response",
    ]))
    result = extract_verdicts(log, output_format="table_only")
    assert [(r["index"], r["parsed"], r["kind"]) for r in result] == [
        (0, "True", "exact"),
        (1, "Conflicting", "recovered"),
        (2, None, "out_of_set"),
        (3, None, "empty"),
    ]


def test_sparse_tables_preserve_original_indices(tmp_path: Path):
    log = tmp_path / "sparse.log"
    log.write_text("\n".join([
        "0 | Verdict: True", "12 | Verdict: <verdict>", "2494 | Verdict: False",
        "0 | True", "12 | False", "2494 | False",
    ]))
    result = extract_verdicts(log, output_format="table_only")
    assert [r["index"] for r in result] == [0, 12, 2494]
    assert result[1]["parsed"] == "False" and result[1]["kind"] == "recovered"


def test_script_fallback_recovery_three_indices(tmp_path: Path):
    log = tmp_path / "think.log"
    first = [f"{i} | Verdict: {'<verdict>' if i in (128, 925, 1064) else 'True'}" for i in range(2495)]
    final = [f"{i} | {'False' if i in (128, 925, 1064) else 'True'}" for i in range(2495)]
    log.write_text("\n".join(first + final))
    result = extract_verdicts(log, output_format="think")
    assert len(result) == 2495
    assert {r["index"] for r in result if r["kind"] == "recovered"} == {128, 925, 1064}


def test_all_real_layouts_have_a_consistent_verdict_shape(tmp_path: Path):
    log = tmp_path / "no_formats.log"
    log.write_text("not a run")
    assert extract_verdicts(log, output_format="banner") == []
    with pytest.raises(ValueError):
        extract_verdicts(log, output_format="unknown")

# Run these against the repository's small, committed fixture excerpts when
# working inside the real checkout. The standalone patch can be tested too.
FIXTURES = Path(__file__).parent / "fixtures" / "logs"


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="repository fixtures not available")
def test_real_banner_excerpts_and_summary():
    for name in ("s02_inference_success.log", "s03_inference_success.log"):
        records = extract_verdicts(FIXTURES / name, output_format="banner")
        assert len(records) == 7, name
        assert [r["index"] for r in records] == list(range(7)), name
        assert all(r["parsed"] in LABELS or r["parsed"] is None for r in records)


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="repository fixtures not available")
def test_real_think_excerpt_recovers_three_known_fallbacks():
    records = extract_verdicts(FIXTURES / "s04_inference_success.log", output_format="think")
    by_index = {record["index"]: record for record in records}
    assert by_index[0]["parsed"] == "Conflicting"
    assert by_index[1]["parsed"] == "True"
    assert by_index[2]["parsed"] == "False"
    assert by_index[12]["kind"] == "empty"
    for index, expected in ((128, "False"), (925, "False"), (1064, "Conflicting")):
        assert by_index[index]["parsed"] == expected
        assert by_index[index]["kind"] == "recovered"


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="repository fixtures not available")
def test_real_table_only_excerpt_recovery():
    records = extract_verdicts(FIXTURES / "s07_inference_success.log", output_format="table_only")
    by_index = {record["index"]: record for record in records}
    assert by_index[0]["parsed"] == "True"
    assert by_index[170]["kind"] == "recovered"
    assert by_index[170]["parsed"] == "Conflicting"
    assert by_index[208]["kind"] == "empty"
    assert by_index[247]["kind"] == "recovered"


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="repository fixtures not available")
def test_real_partial_banner_excerpt_ignores_incomplete_last_record():
    records = extract_verdicts(FIXTURES / "s15_inference_terminated.log", output_format="banner")
    assert len(records) == 7


def test_scoring_integration_if_available(tmp_path: Path):
    scoring = pytest.importorskip("app.scoring")
    log = tmp_path / "banner.log"
    log.write_text("\n".join([
        INPUTS, OUTPUT, "### Response: Verdict: True",
        INPUTS, OUTPUT, "### Response: Verdict: False</s>",
        INPUTS, OUTPUT, "### Response: Verdict: <verdict>",
    ]))
    records = extract_verdicts(log, output_format="banner")
    report = scoring.score_verdicts(records, ["True", "False", "True"], marker=MARKER, labels=LABELS)
    assert report["exact"] == 1
    assert report["recovered"] == 1
    assert report["empty"] == 1
    assert report["accuracy"] == pytest.approx(2 / 3, abs=0.0001)


def test_full_log04_if_supplied():
    """Optional full-length acceptance check using non-committed research logs."""
    import os

    folder = os.environ.get("LOGS_DIR")
    if not folder:
        pytest.skip("set LOGS_DIR to the folder with sampleOutput04.txt")
    log = Path(folder) / "sampleOutput04.txt"
    if not log.is_file():
        pytest.skip("sampleOutput04.txt not found in LOGS_DIR")
    records = extract_verdicts(log, output_format="think")
    assert len(records) == 2495
    assert [record["index"] for record in records] == list(range(2495))
    by_index = {record["index"]: record for record in records}
    assert by_index[128]["parsed"] == "False" and by_index[128]["kind"] == "recovered"
    assert by_index[925]["parsed"] == "False" and by_index[925]["kind"] == "recovered"
    assert by_index[1064]["parsed"] == "Conflicting" and by_index[1064]["kind"] == "recovered"


def test_banner_uses_last_marker_within_one_decoded_output(tmp_path: Path):
    log = tmp_path / "double-marker.log"
    log.write_text("\n".join([
        INPUTS, OUTPUT,
        "### Response: Verdict: True",
        "explanation text",  # not a consecutive bare-summary list
        "### Response: Verdict: False",
    ]))
    assert extract_verdicts(log, output_format="banner")[0]["parsed"] == "False"


def test_banner_respects_custom_marker_and_labels(tmp_path: Path):
    log = tmp_path / "custom-marker.log"
    log.write_text("\n".join([INPUTS, OUTPUT, "Answer: SUPPORTED</s>"]))
    records = extract_verdicts(
        log, output_format="banner", marker="Answer: ", labels=("Supported", "Refuted")
    )
    assert records[0]["parsed"] == "Supported"
    assert records[0]["kind"] == "recovered"


def test_incomplete_timestamp_record_does_not_create_verdict(tmp_path: Path):
    log = tmp_path / "incomplete.log"
    log.write_text("\n".join([
        "Tokenizing Inputs" + TS,
        "### Response: <think>not done</think>\nVerdict: False",
    ]))
    assert extract_verdicts(log, output_format="think") == []
