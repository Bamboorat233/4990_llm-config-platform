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

from app.parsing.verdicts import (
    extract_verdicts,
    extract_verdicts_with_source,
    verdict_source,
)

FIX = Path(__file__).parent / "fixtures" / "logs"
LOGS_DIR = os.environ.get("LOGS_DIR")

MARKER = "### Response: Verdict: "
LABELS = ("True", "False", "Conflicting")
TIMESTAMP = " :  2026-02-20 10:45:18.187166"
ROW = re.compile(r"^(\d+) \| (.*)$")
BANNER = re.compile(r"^[*#]{5,}\s*(INPUTS|OUTPUT|VERDICTS|LENGTH)\s*[*#]{5,}$")

#: The rule the ``think`` script prints between two records, copied from
#: ``tests/fixtures/logs/s04_inference_success.log``.
SEPARATOR = "*" * 78

#: The terminal line of a run killed at the SLURM time limit, copied from
#: ``tests/fixtures/logs/s15_inference_terminated.log``.
TIME_LIMIT = (
    "slurmstepd: error: *** JOB 29242690 ON fc00000 "
    "CANCELLED AT 2026-03-27T18:48:59 DUE TO TIME LIMIT ***"
)

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


def first_pass_value(cell: str) -> str:
    """The first table's cell as a value (the script prints ``Verdict: X``)."""

    return re.sub(r"^Verdict:\s*", "", cell, count=1).strip()


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


def banner_block(generated: str) -> list[str]:
    """The lines of one banner record, up to and including its response."""

    return [
        "*************************************INPUTS****************************************",
        "### Claim: a claim",
        "******************************OUTPUT*****************************",
        "    ### Instruction:",
        "    Give the response in the format - Verdict: <verdict>",
        "### Response: Verdict: " + generated,
    ]


def marker_less_block(claim: str) -> list[str]:
    """A banner record whose prompt was echoed but whose response never came."""

    return [
        "*************************************INPUTS****************************************",
        f"### Claim: {claim}",
        "******************************OUTPUT*****************************",
        "    ### Instruction:",
        "    Give the response in the format - Verdict: <verdict>",
    ]


def bare_list(*values: str | None) -> list[str]:
    """The trailing list some runs print: one line per record, in order.

    A record without a verdict is the literal ``No response``, exactly as the
    list prints it.
    """

    return [
        "No response" if value is None else "### Response: Verdict: " + value
        for value in values
    ]


def complete_output_blocks(path: Path) -> int:
    """Count the ``OUTPUT`` blocks that a later banner closes.

    This is the oracle for "complete record" in the terminated-run tests: a
    block is complete when the next ``INPUTS``, ``VERDICTS`` or ``OUTPUT``
    banner proves the script moved on. A block still open at end of file is
    not complete. It shares no code with the parser.
    """

    complete = 0
    inside = False
    with open(path, encoding="utf-8", errors="replace", newline=None) as handle:
        for raw_line in handle:
            match = BANNER.match(raw_line.rstrip("\r\n"))
            if match is None:
                continue
            section = match.group(1)
            if section == "OUTPUT":
                if inside:
                    complete += 1
                inside = True
            elif inside and section in ("INPUTS", "VERDICTS"):
                complete += 1
                inside = False
    return complete


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


def test_eos_followed_by_a_period_needs_the_trim_loop(tmp_path: Path):
    """``False</s>.``: the period hides the EOS token, so ``_trim`` repeats.

    The trailing period is removed first, which exposes ``</s>``; only a
    second pass of the loop removes the token. The value is recovered, not
    exact, and ``raw`` still shows what the model printed.
    """

    from app.parsing.verdicts import _trim

    assert _trim("False</s>.") == "False"

    log = tmp_path / "eos_then_period.log"
    log.write_text("\n".join(banner_block("False</s>.")), encoding="utf-8")
    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("False", "recovered")
    ]
    assert records[0]["raw"] == "False</s>."


def test_records_are_accepted_by_the_scoring_engine():
    """The shape and the ``exact``/``recovered`` values satisfy #35."""

    from app.scoring import score_verdicts

    # ``source`` is resolved by the extractor and passed on explicitly. The
    # scoring branch makes it a required argument, so a call that relied on
    # this branch's default would raise ``TypeError`` once both branches are
    # merged; passing it by keyword works with either signature.
    source, records = extract_verdicts_with_source(
        FIX / "s02_inference_success.log", marker=MARKER
    )
    # The engine validates the shape and the exact/recovered values; the gold
    # list here only has to be well formed.
    gold = [record["parsed"] for record in records]
    report = score_verdicts(
        records, gold, marker=MARKER, labels=LABELS, source=source
    )
    assert report["records"] == 7
    assert report["complete"] is True
    assert report["recovered"] == 7
    assert report["accuracy"] == 1.0


def test_banner_trailing_bare_list_is_not_extra_records():
    """s03 ends with a bare list of every response; only the blocks count."""

    path = FIX / "s03_inference_success.log"
    records = extract_verdicts(path, marker=MARKER)
    assert len(records) == 7
    assert [record["index"] for record in records] == list(range(7))
    # These responses have no EOS token, so they are exact.
    assert all(record["kind"] == "exact" for record in records)

    # The file prints the seven records' own responses first and then the bare
    # list, so the first seven response lines are the seven records. The last
    # record's value must be its own answer, not a later line of the list
    # (which belongs to another record in the full log).
    listed = [
        line[len("### Response: Verdict: ") :]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("### Response: Verdict: ")
    ]
    assert [record["parsed"] for record in records] == listed[:7]
    assert records[-1]["parsed"] == "False"
    assert records[-1]["raw"] == "False"


def test_bare_list_does_not_verdict_a_marker_less_last_record(tmp_path: Path):
    """The last record has no marker; the list must not supply record 0's.

    The trailing list holds one line per record, so its first line is record
    0's answer. Cutting after the first pair of consecutive marker lines
    lands inside the list and classifies the final record ``exact`` with
    record 0's verdict; the list must instead be removed by its length.
    """

    log = tmp_path / "banner_markerless_last_with_list.log"
    log.write_text(
        "\n".join(
            banner_block("True")
            + banner_block("False")
            + marker_less_block("the last claim")
            + bare_list("True", "False", None)
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert len(records) == 3
    assert records[0]["parsed"] == "True"
    assert records[1]["parsed"] == "False"
    assert records[2]["parsed"] is None
    assert records[2]["kind"] == "empty"
    assert records[2]["raw"] == "No response"
    assert records[2]["parsed"] != records[0]["parsed"]


def test_bare_list_keeps_a_final_verdict_when_record_zero_has_none(
    tmp_path: Path,
):
    """Record 0 has no marker, so the list starts with ``No response``.

    The first pair of consecutive marker lines is then the second and third
    list lines, which lands after record 1's answer: the final record would
    inherit record 1's verdict. The trailing run is the record's own response
    plus the whole list, so dropping one line per record leaves that own
    response and the record keeps its verdict.
    """

    log = tmp_path / "banner_markerless_first_with_list.log"
    log.write_text(
        "\n".join(
            marker_less_block("the first claim")
            + banner_block("False")
            + banner_block("True")
            + bare_list(None, "False", "True")
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        (None, "empty"),
        ("False", "exact"),
        ("True", "exact"),
    ]


def test_terminated_fixture_yields_only_its_complete_records():
    """s15: the number of records equals the number of complete blocks.

    The fixture was killed while the eighth record was being tokenized, so
    the kill sits *between* records: seven ``OUTPUT`` blocks were each closed
    by a later banner, and nothing is left open at end of file. The expected
    count is therefore read from the fixture itself, not from the parser, and
    no record may appear for the record that never started.
    """

    path = FIX / "s15_inference_terminated.log"
    assert complete_output_blocks(path) == 7
    records = extract_verdicts(path, marker=MARKER)

    assert len(records) == 7
    assert [record["index"] for record in records] == list(range(7))
    # Every complete block in this excerpt carries a real verdict. A trailing
    # ``empty`` record would be the invented "No response" of a cut block.
    assert all(record["kind"] != "empty" for record in records)
    assert records[-1]["parsed"] == "False"


def test_terminated_fixture_with_a_cut_block_keeps_it(tmp_path: Path):
    """A cut block that holds a response line is a record: 7 becomes 8.

    The fixture is the real terminated log with one more ``OUTPUT`` block
    spliced in before the terminal line, cut in the middle of its verdict.
    The block holds ``### Response: Verdict: Fal``, so the stream itself
    proves the record was written; the value is ``unparsed``. The run's
    termination status no longer drops it.
    """

    source = (FIX / "s15_inference_terminated.log").read_text(encoding="utf-8")
    lines = source.splitlines()
    assert lines[-1] == TIME_LIMIT
    cut = [
        "******************************OUTPUT*****************************",
        "    ### Instruction:",
        "    Give the response in the format - Verdict: <verdict>",
        "### Response: Verdict: Fal",
    ]
    log = tmp_path / "s15_cut_inside_record.log"
    log.write_text("\n".join(lines[:-1] + cut + [TIME_LIMIT]), encoding="utf-8")

    records = extract_verdicts(log, marker=MARKER)
    assert len(records) == 8
    assert [record["index"] for record in records] == list(range(8))
    assert all(record["kind"] == "exact" for record in records[:7])
    assert records[7]["raw"] == "Fal"
    assert records[7]["parsed"] is None
    assert records[7]["kind"] == "unparsed"


def test_banner_cut_at_eof_is_kept_when_it_has_a_marker(tmp_path: Path):
    """A half-written ``Verdict: Fal`` before the time limit is a record: 1 to 2.

    The final open block contains a ``### Response:`` line, so the stream
    proves the record was written. The partial value is ``unparsed``; only
    the termination status is ignored now.
    """

    log = tmp_path / "banner_terminated.log"
    log.write_text(
        "\n".join(banner_block("True") + banner_block("Fal") + [TIME_LIMIT]),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert len(records) == 2
    assert records[0]["parsed"] == "True"
    assert records[0]["kind"] == "exact"
    assert records[1]["raw"] == "Fal"
    assert records[1]["parsed"] is None
    assert records[1]["kind"] == "unparsed"


def test_banner_cut_at_eof_is_kept_without_a_terminal_line(tmp_path: Path):
    """The same cut with no terminal event keeps its block, as log 02 does."""

    log = tmp_path / "banner_clean_eof.log"
    log.write_text(
        "\n".join(banner_block("True") + banner_block("Fal")),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert len(records) == 2
    assert records[0]["parsed"] == "True"
    assert records[1]["raw"] == "Fal"
    assert records[1]["kind"] == "unparsed"


def test_banner_final_block_without_a_response_is_not_a_record(tmp_path: Path):
    """The final open block needs a ``### Response:`` line; earlier blocks do not.

    A block closed by a later banner is always a record, even with no marker
    (it is ``empty``). The block still open at end of file is a record only
    when it holds a response line, so the third block here is absent rather
    than an invented ``empty``.
    """

    closed_without_marker = [
        "*************************************INPUTS****************************************",
        "### Claim: a claim",
        "******************************OUTPUT*****************************",
        "    ### Instruction:",
        "    Give the response in the format - Verdict: <verdict>",
    ]
    open_without_marker = [
        "*************************************INPUTS****************************************",
        "### Claim: another claim",
        "******************************OUTPUT*****************************",
        "    ### Instruction:",
        "    Give the response in the format - Verdict: <verdict>",
    ]
    log = tmp_path / "banner_missing_response.log"
    log.write_text(
        "\n".join(
            banner_block("True") + closed_without_marker + open_without_marker
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert len(records) == 2
    assert [record["index"] for record in records] == [0, 1]
    assert records[0]["parsed"] == "True"
    assert records[0]["kind"] == "exact"
    # Closed by the next INPUTS: a record, with the literal "No response".
    assert records[1]["raw"] == "No response"
    assert records[1]["parsed"] is None
    assert records[1]["kind"] == "empty"


def test_terminated_think_log_keeps_a_complete_open_block(tmp_path: Path):
    """The think layout applies the same stream test: 2 becomes 3.

    The third block holds both ``Done Decoding`` and a response line, so it
    is a record even though the terminal line follows it. The partial value
    ``Verdict: Tru`` is ``unparsed``.
    """

    log = tmp_path / "think_terminated.log"
    log.write_text(
        "\n".join(
            [
                think_block("Verdict: True"),
                think_block("Verdict: False"),
                think_block("Verdict: Tru"),
                TIME_LIMIT,
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact"),
        ("False", "exact"),
        (None, "unparsed"),
    ]


def test_think_block_needs_both_done_decoding_and_a_response(tmp_path: Path):
    """A record needs ``Done Decoding`` *and* a response line; one is not enough.

    Three decoded blocks, only the middle one complete. The first has ``Done
    Decoding`` but no response (the ~0.14 ms window before the decoded text)
    and the third has a response but no ``Done Decoding`` (a block the run
    never completed). Requiring only one of the two would invent a record:
    either an ``empty`` one or an unstarted one. Exactly one record survives.
    """

    log = tmp_path / "think_incomplete_blocks.log"
    log.write_text(
        "\n".join(
            [
                "Tokenizing Inputs" + TIMESTAMP,
                "Done Decoding" + TIMESTAMP,
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                "Tokenizing Inputs" + TIMESTAMP,
                "Done Decoding" + TIMESTAMP,
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                "### Response: Verdict: True",
                "Tokenizing Inputs" + TIMESTAMP,
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                "### Response: Verdict: False",
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact")
    ]


def test_stray_done_decoding_does_not_carry_into_the_next_block(
    tmp_path: Path,
):
    """The pending phase flag belongs to one record and is cleared at the phase.

    ``Done Decoding`` is printed before ``Decoded output:`` in the real logs,
    so the flag is carried into the block the header opens. A stray one seen
    while no block is open must not survive the next ``Tokenizing Inputs``:
    otherwise a block with no phase evidence of its own becomes a record.
    """

    log = tmp_path / "think_stray_done_decoding.log"
    log.write_text(
        "\n".join(
            [
                "Done Decoding" + TIMESTAMP,
                "Tokenizing Inputs" + TIMESTAMP,
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                "### Response: Verdict: True",
            ]
        ),
        encoding="utf-8",
    )

    assert extract_verdicts(log, marker=MARKER, layout="think") == []


def test_separator_inside_a_prompt_echo_does_not_drop_the_record(
    tmp_path: Path,
):
    """A 20-asterisk rule before the response is prompt echo, not a closer.

    The block holds ``Done Decoding`` and then a separator line inside the
    echoed prompt. Closing there drops the record before its response line
    arrives; the separator closes only a block that already has a response.
    """

    log = tmp_path / "think_separator_in_prompt.log"
    log.write_text(
        "\n".join(
            [
                "Tokenizing Inputs" + TIMESTAMP,
                "Done Decoding" + TIMESTAMP,
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                SEPARATOR,
                "### Response: Verdict: True",
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact")
    ]


def test_think_log_cut_without_a_terminal_line_keeps_the_record(tmp_path: Path):
    """Without a terminal event the open decoded record is completed."""

    log = tmp_path / "think_clean_eof.log"
    log.write_text(
        "\n".join(
            [
                think_block("Verdict: True"),
                think_block("Verdict: False"),
                think_block("Verdict: Tru"),
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert len(records) == 3
    assert records[-1]["raw"] == "Tru"
    assert records[-1]["kind"] == "unparsed"


def test_terminated_think_run_keeps_the_last_complete_record(tmp_path: Path):
    """A kill after the next record started must not lose record N: one record.

    The real record shape includes the phase lines (``Done Decoding`` and the
    response); a block without them is not a record, so the old shape with
    only ``Decoded output:`` and a response yielded zero records. Here the
    block holds both, is closed by the ``***`` separator, and yields record N
    even though the kill lands before the next ``Decoded output:``.
    """

    log = tmp_path / "think_terminated_between_records.log"
    log.write_text(
        "\n".join(
            [
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                "Done Decoding" + TIMESTAMP,
                "### Response: Verdict: True",
                SEPARATOR,
                "Tokenizing Inputs" + TIMESTAMP,
                TIME_LIMIT,
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact")
    ]


def test_terminated_think_run_drops_the_started_record(tmp_path: Path):
    """The record that only started tokenizing stays missing, not empty.

    Same real record shape as the test above, but the kill lands after the
    next record's phase lines and before any ``Decoded output:`` header or
    verdict. Only record N has ``Done Decoding`` and a response, so only it
    is a record.
    """

    log = tmp_path / "think_terminated_started_record.log"
    log.write_text(
        "\n".join(
            [
                "Decoded output:  ",
                "Give the response in the format - Verdict: <verdict>",
                "Done Decoding" + TIMESTAMP,
                "### Response: Verdict: True",
                SEPARATOR,
                "Tokenizing Inputs" + TIMESTAMP,
                "Done Preparing the Input" + TIMESTAMP,
                "Preparing Outputs" + TIMESTAMP,
                TIME_LIMIT,
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact")
    ]


def test_clean_think_log_keeps_its_last_block(tmp_path: Path):
    """A log that ends cleanly, with no terminal event, keeps its last block.

    Guard against over-dropping after closing blocks at the next record's
    start: the last record is complete and has no successor to prove it.
    """

    log = tmp_path / "think_clean_two_records.log"
    log.write_text(
        "\n".join(
            [
                think_block("Verdict: True"),
                think_block("Verdict: False"),
            ]
        ),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact"),
        ("False", "exact"),
    ]


def test_banner_block_with_a_complete_response_before_the_kill_is_kept(
    tmp_path: Path,
):
    """A complete response line is a record even when the kill follows it.

    The mirror of ``test_banner_cut_at_eof_is_kept_when_it_has_a_marker``: the
    block already holds ``### Response: Verdict: True</s>``, so the terminal
    line did not cut anything.
    """

    log = tmp_path / "banner_terminated_complete.log"
    log.write_text(
        "\n".join(banner_block("True</s>") + [TIME_LIMIT]),
        encoding="utf-8",
    )

    records = extract_verdicts(log, marker=MARKER, layout="banner")
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "recovered")
    ]


def test_terminated_and_clean_fixtures_keep_seven_records():
    """Guard: s02 (clean) and s15 (terminated) both still yield seven."""

    for name in (
        "s02_inference_success.log",
        "s15_inference_terminated.log",
    ):
        records = extract_verdicts(FIX / name, marker=MARKER)
        assert len(records) == 7, name
        assert [record["index"] for record in records] == list(range(7)), name


def test_banner_fixture_record_counts_match_records_done():
    """s02 and s15: the record count equals the run's ``records_done`` (7).

    ``records_done`` is the progress collector's independent count for the
    banner layout, so it is the oracle for the completeness rule.
    """

    from app.parsing import parse_log_file

    for name in (
        "s02_inference_success.log",
        "s15_inference_terminated.log",
    ):
        path = FIX / name
        records = extract_verdicts(path, marker=MARKER)
        assert len(records) == parse_log_file(path)["progress"]["records_done"]
        assert len(records) == 7, name


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


def test_synthetic_think_log_covers_the_decoded_record_path(tmp_path: Path):
    """The think main path: decoded record first, final table as fallback.

    ``s04`` is a trimmed excerpt, so it exercises the table path only. This
    log has one decoded record per index, which makes ``aligned`` true and
    runs the primary path: four generations covering a clean label, a label
    with a trailing period, a generation with no verdict line (the script's
    fallback) and an empty generation.
    """

    log = tmp_path / "think_decoded.log"
    generations = [
        "Verdict: True",
        "Verdict: Conflicting.",
        "<think>No verdict line here.</think>",
        "Verdict: <verdict>",
    ]
    first = [
        "0 | Verdict: True",
        "1 | Verdict: Conflicting.",
        "2 | Verdict: <verdict>",
        "3 | Verdict: <verdict>",
    ]
    final = [
        "0 | True",
        "1 | Conflicting",
        "2 | False",
        "3 | Empty Response",
    ]
    log.write_text(
        "\n".join(
            [think_block(generation) for generation in generations]
            + first
            + final
        ),
        encoding="utf-8",
    )

    source, records = extract_verdicts_with_source(
        log, marker=MARKER, layout="think"
    )
    assert source == "parsed"
    assert [(record["parsed"], record["kind"]) for record in records] == [
        ("True", "exact"),
        ("Conflicting", "recovered"),
        ("False", "recovered"),
        (None, "empty"),
    ]
    # These raws are the decoded record, not the first table's cell, which is
    # what proves the decoded-record path ran rather than the table path.
    # A generation with no verdict line keeps its first line only, with the
    # layout space that follows ``### Response:`` stripped.
    assert records[1]["raw"] == "Conflicting."
    assert records[2]["raw"] == "<think>No verdict line here.</think>"
    assert records[3]["raw"] == "<verdict>"


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
    # ``unparsed`` is not a parse: ``parsed`` stays empty, as for ``empty``.
    assert records[128]["parsed"] is None


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


def test_lone_script_table_is_not_treated_as_final(tmp_path: Path):
    """One table is not the final table; the decoded records are the source.

    A lone table may be the script's first pass. Treating it as final turns
    its ``Verdict: True`` cells into ``recovered`` and skips the decoded
    records for ``think``; for ``table_only`` it asserts a final output that
    was never printed. This file has one decoded record and a two-row table,
    so the old rule would return two ``recovered`` records instead of one
    exact decoded record.
    """

    think = tmp_path / "think_lone_table.log"
    think.write_text(
        "\n".join(
            [
                think_block("Verdict: True"),
                "0 | Verdict: True",
                "1 | Verdict: False",
            ]
        ),
        encoding="utf-8",
    )
    think_records = extract_verdicts(think, marker=MARKER, layout="think")
    assert [(record["parsed"], record["kind"]) for record in think_records] == [
        ("True", "exact")
    ]

    table = tmp_path / "table_only_lone_table.log"
    table.write_text(
        "\n".join(["0 | Verdict: True", "1 | Verdict: False"]),
        encoding="utf-8",
    )
    assert extract_verdicts(table, marker=MARKER, layout="table_only") == []


def test_lone_table_only_run_reports_a_machine_readable_reason(
    tmp_path: Path,
):
    """A single table yields no records but tells the UI why.

    The empty list alone cannot distinguish "the run was killed before the
    final table" from a parse failure, so the optional ``warnings`` list
    receives :data:`TABLE_ONLY_SINGLE_TABLE`. A complete log with both tables
    reports nothing, and the records are unchanged.
    """

    from app.parsing.verdicts import TABLE_ONLY_SINGLE_TABLE

    table = tmp_path / "table_only_lone_table.log"
    table.write_text(
        "\n".join(["0 | Verdict: True", "1 | Verdict: False"]),
        encoding="utf-8",
    )
    warnings: list[str] = []
    source, records = extract_verdicts_with_source(
        table, marker=MARKER, layout="table_only", warnings=warnings
    )
    assert source == "script_table"
    assert records == []
    assert warnings == [TABLE_ONLY_SINGLE_TABLE]

    warnings.clear()
    source, records = extract_verdicts_with_source(
        FIX / "s07_inference_success.log", marker=MARKER, warnings=warnings
    )
    assert source == "script_table"
    assert len(records) == 163
    assert warnings == []


@pytest.mark.parametrize(
    "value",
    [
        "Mostly True",
        "Half true",
        "Not enough information",
        "Supports",
        "Refutes",
        "Incorrect",
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


def test_extract_with_source_reports_the_resolved_layout():
    """The caller must be able to score ``table_only`` as ``script_table``.

    Each fixture is extracted through its own resolved layout, so the records
    carry that layout's properties: the banner records trim the model EOS
    token, while the table records come from the 0-based final table. A
    wrapper that resolved the wrong layout could not produce both.
    """

    banner = FIX / "s02_inference_success.log"
    source, records = extract_verdicts_with_source(banner, marker=MARKER)
    assert source == "parsed"
    assert [record["index"] for record in records] == list(range(7))
    assert all(record["kind"] == "recovered" for record in records)

    table = FIX / "s07_inference_success.log"
    source, records = extract_verdicts_with_source(table, marker=MARKER)
    assert source == "script_table"
    assert len(records) == 163
    _, final = tables(table)
    assert [record["index"] for record in records] == [
        index for index, _ in final
    ]

    think = FIX / "s04_inference_success.log"
    source, records = extract_verdicts_with_source(think, marker=MARKER)
    assert source == "parsed"
    assert len(records) == 162
    _, final = tables(think)
    assert [record["index"] for record in records] == [
        index for index, _ in final
    ]


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

    # The first table prints the script's own parse of the same generation,
    # so it decides exact versus recovered for *every* row: a clean label is
    # exact, a placeholder or a trailing period is a recovery.
    for (index, first_value), (_, final_value) in zip(first, final):
        if final_value == "Empty Response":
            expected_kind = "empty"
        elif first_pass_value(first_value) == final_value:
            expected_kind = "exact"
        else:
            expected_kind = "recovered"
        assert records_by_index[index]["kind"] == expected_kind, index

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
def test_full_log15_terminated_keeps_records_done():
    """The terminated full log still yields the 2,480 records it reports.

    ``records_done`` is the script's own count, so it is the oracle: dropping
    the block open at end of file must not remove a record the script
    finished, and the cut record must not be invented as ``empty``.
    """

    from app.parsing.detect import detect

    path = Path(LOGS_DIR) / "sampleOutput15.txt"
    records = extract_verdicts(path, marker=MARKER)

    assert detect(path)["progress"]["records_done"] == 2480
    assert len(records) == 2480
    assert [record["index"] for record in records] == list(range(2480))


@slow
def test_full_logs_yield_exactly_records_done():
    """Logs 02, 04 and 15: record count equals ``records_done`` (2495/2495/2480).

    ``records_done`` is computed by the progress collector from the stream
    (``app.parsing.inference_log``) and shares no code with this extractor,
    so it is the independent oracle for the completeness rules.
    """

    from app.parsing import parse_log_file

    expected = {
        "sampleOutput02.txt": 2495,
        "sampleOutput04.txt": 2495,
        "sampleOutput15.txt": 2480,
    }
    for name, count in expected.items():
        path = Path(LOGS_DIR) / name
        records = extract_verdicts(path, marker=MARKER)
        assert len(records) == parse_log_file(path)["progress"]["records_done"]
        assert len(records) == count, name


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
