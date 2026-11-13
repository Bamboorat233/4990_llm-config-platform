"""Tests for the inference progress collector (issue #31).

Numbers for the full logs come from the survey of the real logs (2495 records
in logs 02, 03, 04 and 07; 2480 in the time-limited log 15) and are checked
against tools/inference_oracle.py, which shares no code with the parser. The
excerpt numbers count the records visible in each trimmed fixture. The slow
test runs the parser and the oracle on the full logs when LOGS_DIR is set.
"""
import importlib.util
import os
from pathlib import Path

import pytest

from app.parsing import parse_log_file
from app.parsing.inference_log import InferenceProgress

FIX = Path(__file__).parent / "fixtures" / "logs"
ROOT = Path(__file__).parent.parent

# fixture -> records_done visible in the trimmed excerpt
EXCERPT = {
    "s01_inference_error.log": None,      # failed while loading the tokenizer
    "s02_inference_success.log": 7,       # banner: first 5 + last 2 kept records
    "s03_inference_success.log": 7,
    "s04_inference_success.log": 14,      # think: timestamped records
    "s05_inference_error.log": 1,         # one record finished, then IndexError
    "s06_inference_error.log": 0,         # record started, out of memory in generation
    "s07_inference_success.log": 19,      # table_only
    "s14_inference_error.log": None,      # failed while preparing the prompt
    "s15_inference_terminated.log": 7,
}
# sample number -> records_done in the full log
FULL = {1: None, 2: 2495, 3: 2495, 4: 2495, 5: 1, 6: 0, 7: 2495, 14: None, 15: 2480}


@pytest.mark.parametrize("name", sorted(EXCERPT))
def test_fixture_progress(name):
    progress = parse_log_file(FIX / name)["progress"]
    assert progress == {"records_done": EXCERPT[name], "records_expected": None}


def test_other_families_keep_their_own_keys():
    assert parse_log_file(FIX / "s13_checkpoint_export_success.log")["progress"] == {}
    ft = parse_log_file(FIX / "s12_fine_tune_success_warning.log")["progress"]
    assert "records_done" not in ft and ft["total_steps"] == 9936


def feed(*lines):
    p = InferenceProgress()
    for n, line in enumerate(lines, 1):
        p.feed(line, n)
    return p.result()


INPUTS = "*************************************INPUTS****************************************"
LENGTH = "#####################################LENGTH#########################################"
OUTPUT = "******************************OUTPUT*****************************"
RESP = "### Response: Verdict: True</s>"


def test_nothing_seen_means_none_not_zero():
    assert feed("Done Loading the Model : 2026-02-14 09:31:00.000001") == {
        "records_done": None, "records_expected": None}


def test_banner_block_closed_by_next_inputs_counts():
    r = feed(INPUTS, "tensor([[1]])", LENGTH, "1", OUTPUT, "text", RESP, INPUTS, "tensor([[2]])")
    assert r["records_done"] == 1


def test_empty_reply_block_still_counts_when_the_script_moved_on():
    r = feed(INPUTS, OUTPUT, "### Question: x", "### Answer", INPUTS, OUTPUT, "p", RESP)
    assert r["records_done"] == 2


def test_last_block_without_response_is_not_counted():
    assert feed(INPUTS, OUTPUT, "### Question: x", "### Answer: y")["records_done"] == 0


def test_record_that_only_reached_inputs_is_not_counted():
    assert feed(INPUTS, OUTPUT, RESP, INPUTS, "tensor([[2]])")["records_done"] == 1


def test_a_block_counts_once_even_with_many_response_lines():
    r = feed(INPUTS, OUTPUT, "### Response: a", "### Response: b", "### Response: c")
    assert r["records_done"] == 1


def test_trailing_bare_verdict_list_after_the_last_block_is_not_extra_records():
    r = feed(INPUTS, OUTPUT, RESP, INPUTS, OUTPUT, RESP, *["### Response: Verdict: False"] * 50)
    assert r["records_done"] == 2


def test_response_line_outside_an_output_block_is_ignored():
    assert feed(INPUTS, "### Response: Verdict: True")["records_done"] == 0


def test_timestamp_layout_counts_done_decoding():
    ts = " :  2026-02-20 11:01:47.724950"
    lines = []
    for _ in range(3):
        lines += ["Tokenizing Inputs" + ts, "Done Preparing the Input" + ts, "Done Decoding" + ts]
    assert feed(*lines)["records_done"] == 3


def test_started_but_never_decoded_is_zero():
    ts = " :  2026-02-20 11:01:47.724950"
    assert feed("Tokenizing Inputs" + ts, "Done Preparing the Input" + ts)["records_done"] == 0


def test_done_decoding_text_without_a_timestamp_is_not_counted():
    ts = " :  2026-02-20 11:01:47.724950"
    assert feed("Tokenizing Inputs" + ts, "Done Decoding the claim is false")["records_done"] == 0


def test_banner_wins_when_both_markers_exist():
    ts = " :  2026-02-20 11:01:47.724950"
    assert feed("Tokenizing Inputs" + ts, INPUTS, OUTPUT, RESP)["records_done"] == 1


def test_records_expected_is_never_guessed():
    assert feed(INPUTS, OUTPUT, RESP)["records_expected"] is None


def test_state_stays_bounded_over_many_records():
    p = InferenceProgress()
    for i in range(50_000):
        for line in (INPUTS, OUTPUT, RESP):
            p.feed(line, i)
    assert p.result()["records_done"] == 50_000 and len(vars(p)) == 6


@pytest.mark.skipif(not os.environ.get("LOGS_DIR"), reason="set LOGS_DIR to the folder with the full sample logs")
def test_parser_matches_numbers_and_oracle_on_full_logs():
    spec = importlib.util.spec_from_file_location("inference_oracle", ROOT / "tools" / "inference_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for n, expected in FULL.items():
        full = Path(os.environ["LOGS_DIR"]) / f"sampleOutput{n:02d}.txt"
        got = parse_log_file(full)["progress"]["records_done"]
        assert got == expected == mod.oracle(str(full))["records_done"], full.name
