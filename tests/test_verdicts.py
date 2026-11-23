"""Tests for issue #34: verdict extraction and normalisation.

The fast tests use the committed trimmed fixtures under
``tests/fixtures/logs`` and small synthetic logs. The ``LOGS_DIR``-gated slow
tests run on the full sample logs, which stay outside the repository; they are
the acceptance evidence for the 2,495-index requirement.

The comment on each table-driven assertion names the independent oracle: the
final ``N | ...`` table printed by the script itself (read here with a small
regex that shares no code with the parser).
"""

from __future__ import annotations

import os
import re
import tracemalloc
from pathlib import Path

import pytest

from app.parsing.verdicts import extract_verdicts, verdict_source

FIX = Path(__file__).parent / "fixtures" / "logs"
LOGS_DIR = os.environ.get("LOGS_DIR")

MARKER = "### Response: Verdict: "
LABELS = ("True", "False", "Conflicting")
TIMESTAMP = " :  2026-02-20 10:45:18.187166"
ROW = re.compile(r"^(\d+) \| (.*)$")

slow = pytest.mark.skipif(
    not LOGS_DIR, reason="set LOGS_DIR to the folder with the full sample logs"
)


def tables(path: Path) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """Return the two ``N | ...`` tables of a log, in file order.

    A new table starts when the row index stops increasing, which is how both
    real layouts print the first pass and then the final table.
    """

    rows: list[tuple[int, str]] = []
    with open(path, encoding="utf-8", errors="replace", newline=None) as handle:
        for raw_line in handle:
            match = ROW.match(raw_line.rstrip("\r\n"))
            if match is not None:
                rows.append((int(match.group(1)), match.group(2).strip()))
    split = next(
        (position for position in range(1, len(rows)) if rows[position][0] == 0),
        len(rows),
    )
    return rows[:split], rows[split:]


def by_index(records):
    return {record["index"]: record for record in records}


def expected_value(final_value: str) -> str | None:
    """The final table's value as a parsed verdict (``Empty Response`` is None)."""

    return None if final_value == "Empty Response" else final_value


def think_block(generated: str) -> str:
    """A minimal ``think`` decoded record around one generated text."""

    return "\n".join(
        [
            "Tokenizing Inputs" + TIMESTAMP,
            "Done Decoding" + TIMESTAMP,
            "Decoded output:  ",
            "Give the response in the format - Verdict: <verdict>",
            "### Response: " + generated,
        ]
    )


# ---------------------------------------------------------------------------
# Fast fixture tests
# ---------------------------------------------------------------------------


def test_banner_eos_token_is_recovered():
    """Log 02's ``### Response: Verdict: False</s>`` is recovered, not exact."""

    records = extract_verdicts(FIX / "s02_inference_success.log", marker=MARKER)
    assert len(records) == 7
    assert [record["index"] for record in records] == list(range(7))
    # Every response in this excerpt carries the model EOS token.
    assert all(record["kind"] == "recovered" for record in records)
    false = [record for record in records if record["parsed"] == "False"]
    assert false
    assert false[0]["raw"] == "False</s>"
    assert false[0]["kind"] == "recovered"
    assert all(
        set(record) == {"index", "raw", "parsed", "kind"}
        for record in records
    )


def test_records_are_accepted_by_the_scoring_engine():
    """The shape and the ``exact``/``recovered`` values satisfy #35."""

    from app.scoring import score_verdicts

    records = extract_verdicts(FIX / "s02_inference_success.log", marker=MARKER)
    # The engine validates the shape and the exact/recovered values; the gold
    # list here only has to be well formed.
    gold = [record["parsed"] for record in records]
    report = score_verdicts(records, gold, marker=MARKER, labels=LABELS)
    assert report["records"] == 7
    assert report["complete"] is True
    assert report["recovered"] == 7
    assert report["accuracy"] == 1.0


def test_banner_trailing_bare_list_is_not_extra_records():
    """s03 ends with a bare list of every response; only the blocks count."""

    records = extract_verdicts(FIX / "s03_inference_success.log", marker=MARKER)
    assert len(records) == 7
    assert [record["index"] for record in records] == list(range(7))
    # These responses have no EOS token, so they are exact.
    assert all(record["kind"] == "exact" for record in records)


def test_instruction_placeholder_output_is_empty(tmp_path: Path):
    """A generated text that starts with ``Verdict: <verdict>`` is empty.

    No committed fixture contains this case (every log 04 generation starts
    with ``<think>``), so it is constructed synthetically, as the issue asks.
    The rule is about the literal instruction placeholder, not about any text
    beginning with ``Verdict:``: a real ``Verdict: False`` after the think
    block is still a verdict (covered by the log 04 tests below).
    """

    log = tmp_path / "placeholder.log"
    log.write_text(
        "\n".join(
            [
                think_block("Verdict: <verdict>"),
                think_block("<think>Reasoning only, no verdict line</think>"),
                "0 | Verdict: <verdict>",
                "1 | Verdict: <verdict>",
                "0 | Empty Response",
                "1 | Empty Response",
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        (None, "empty"),
        (None, "empty"),
    ]


def test_think_fixture_matches_the_visible_final_table():
    """s04: one record per visible index; kinds agree with the final table."""

    path = FIX / "s04_inference_success.log"
    records = extract_verdicts(path, marker=MARKER)
    _, final = tables(path)
    assert len(records) == 162
    assert [record["index"] for record in records] == [
        index for index, _ in final
    ]

    records_by_index = by_index(records)
    for index, value in final:
        record = records_by_index[index]
        if value == "Empty Response":
            assert record["kind"] == "empty", index
            assert record["parsed"] is None, index
        else:
            assert record["parsed"] == value, index
            assert record["kind"] in ("exact", "recovered"), index

    # The three first-table placeholders the script recovered, plus the two
    # trailing-period recoveries visible in this excerpt.
    assert records_by_index[128]["parsed"] == "False"
    assert records_by_index[128]["kind"] == "recovered"
    assert records_by_index[925]["parsed"] == "False"
    assert records_by_index[925]["kind"] == "recovered"
    assert records_by_index[1064]["parsed"] == "Conflicting"
    assert records_by_index[1064]["kind"] == "recovered"
    assert records_by_index[270]["parsed"] == "Conflicting"
    assert records_by_index[270]["kind"] == "recovered"
    assert records_by_index[12]["kind"] == "empty"
    assert records_by_index[0]["kind"] == "exact"


def test_think_mangled_final_table_line_is_unparsed(tmp_path: Path):
    """A final-table cell that is not a verdict is ``unparsed``, not guessed."""

    path = FIX / "s04_inference_success.log"
    text = path.read_text(encoding="utf-8")
    needle = "128 | False\n"
    assert text.count(needle) == 1  # only in the second table
    mangled = tmp_path / "s04_mangled.log"
    mangled.write_text(
        text.replace(needle, "128 | this is not a verdict\n", 1),
        encoding="utf-8",
    )

    records = by_index(extract_verdicts(mangled, marker=MARKER))
    assert records[128]["kind"] == "unparsed"
    assert records[128]["parsed"] == "this is not a verdict"


def test_table_only_fixture_uses_the_second_table():
    """s07: no prompt echo; the final table is the source (``script_table``)."""

    path = FIX / "s07_inference_success.log"
    records = extract_verdicts(path, marker=MARKER, source="script_table")
    _, final = tables(path)
    assert len(records) == 163
    assert [record["index"] for record in records] == [
        index for index, _ in final
    ]

    records_by_index = by_index(records)
    for index, value in final:
        assert records_by_index[index]["parsed"] == expected_value(value), index

    # The six rows the script's fallback recovered, and one it could not.
    for index in (170, 247, 947, 1211, 2277, 2283):
        assert records_by_index[index]["parsed"] == "Conflicting", index
        assert records_by_index[index]["kind"] == "recovered", index
    assert records_by_index[208]["kind"] == "empty"
    assert records_by_index[208]["parsed"] is None

    assert verdict_source("table_only") == "script_table"
    with pytest.raises(ValueError):
        extract_verdicts(path, marker=MARKER, source="parsed")


@pytest.mark.parametrize(
    "value",
    [
        "Mostly True",
        "Half true",
        "Not enough information",
        "Supports",
        "Refutes",
    ],
)
def test_out_of_set_values_are_kept_for_review(tmp_path: Path, value):
    """A plausible verdict outside ``labels`` is reported, never coerced."""

    log = tmp_path / "out_of_set.log"
    log.write_text(
        "\n".join(
            [
                f"0 | Verdict: {value}",
                f"0 | {value}",
            ]
        ),
        encoding="utf-8",
    )
    records = extract_verdicts(log, marker=MARKER, layout="table_only")
    assert len(records) == 1
    assert records[0]["kind"] == "out_of_set"
    assert records[0]["parsed"] == value


def test_verdict_source_mapping():
    assert verdict_source("banner") == "parsed"
    assert verdict_source("think") == "parsed"
    assert verdict_source("table_only") == "script_table"
    with pytest.raises(ValueError):
        verdict_source("unknown")


def test_missing_output_format_raises_unless_layout_is_given():
    """A log that never reached the output stage has no layout to detect."""

    path = FIX / "s01_inference_error.log"
    with pytest.raises(ValueError):
        extract_verdicts(path, marker=MARKER)
    # An explicit layout skips detect(); this log has no OUTPUT block.
    assert extract_verdicts(path, marker=MARKER, layout="banner") == []


def test_source_must_match_the_layout():
    path = FIX / "s02_inference_success.log"
    with pytest.raises(ValueError):
        extract_verdicts(path, marker=MARKER, source="script_table")
    assert len(extract_verdicts(path, marker=MARKER, source="parsed")) == 7


# ---------------------------------------------------------------------------
# Slow full-log acceptance tests (the logs stay outside the repository)
# ---------------------------------------------------------------------------


@slow
def test_full_log04_reproduces_the_final_table():
    """2,495 records; every value equals the script's final table."""

    path = Path(LOGS_DIR) / "sampleOutput04.txt"
    records = extract_verdicts(path, marker=MARKER)
    first, final = tables(path)

    assert len(records) == 2495
    assert [record["index"] for record in records] == list(range(2495))

    records_by_index = by_index(records)
    for index, value in final:
        assert records_by_index[index]["parsed"] == expected_value(value), index

    # The 3 rows where the first pass printed the placeholder and the final
    # table holds a real verdict: the script's fallback recovered them.
    fallback = [
        index
        for (index, first_value), (_, final_value) in zip(first, final)
        if first_value == "Verdict: <verdict>" and final_value in LABELS
    ]
    print("log 04 script-fallback indices:", fallback)
    assert fallback == [128, 925, 1064]
    for index in fallback:
        assert records_by_index[index]["kind"] == "recovered", index


@slow
def test_full_log07_reproduces_the_final_table():
    """2,495 records from the second table, including the 6 fallback rows."""

    path = Path(LOGS_DIR) / "sampleOutput07.txt"
    records = extract_verdicts(path, marker=MARKER)
    first, final = tables(path)

    assert len(records) == 2495
    assert [record["index"] for record in records] == list(range(2495))

    records_by_index = by_index(records)
    for index, value in final:
        assert records_by_index[index]["parsed"] == expected_value(value), index

    fallback = [
        index
        for (index, first_value), (_, final_value) in zip(first, final)
        if first_value == "Verdict: <verdict>" and final_value in LABELS
    ]
    print("log 07 script-fallback indices:", fallback)
    assert fallback == [170, 247, 947, 1211, 2277, 2283]
    for index in fallback:
        assert records_by_index[index]["kind"] == "recovered", index


@slow
def test_full_log02_extracts_2495_and_recovers_eos():
    """Banner layout: 2,495 records, every ``</s>`` trimmed to a label."""

    path = Path(LOGS_DIR) / "sampleOutput02.txt"
    records = extract_verdicts(path, marker=MARKER)
    assert len(records) == 2495
    assert [record["index"] for record in records] == list(range(2495))
    assert sum(record["kind"] == "recovered" for record in records) >= 1
    assert all(
        record["raw"].endswith("</s>")
        for record in records
        if record["kind"] == "recovered"
    )
    assert any(record["raw"] == "False</s>" for record in records)
    # The blocks whose generation stopped before a response line.
    assert sum(record["kind"] == "empty" for record in records) == 7


@slow
def test_full_log04_memory_stays_flat():
    """The 8 MB log is streamed: peak traced memory is far below its size."""

    path = Path(LOGS_DIR) / "sampleOutput04.txt"
    size = path.stat().st_size
    tracemalloc.start()
    try:
        records = extract_verdicts(path, marker=MARKER)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(records) == 2495
    assert peak < 4 * 1024 * 1024, peak
    assert peak < size, (peak, size)
