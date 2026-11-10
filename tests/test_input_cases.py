"""pytest runner for the input checker (issue #13).

Every case in tests/cases/input_cases.json becomes one test. The case is
run through the real pipeline (defaults, edits, ConfigRecord, validator,
console logic, optional save) and its outcome is compared with the
expectation written in the file.

A few extra tests check the checker itself, so a PASS can be trusted: it
must be able to FAIL.
"""
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
for path in (str(TESTS_DIR), str(TESTS_DIR.parent)):
    if path not in sys.path:
        sys.path.insert(0, path)

import input_checker as ic  # noqa: E402

import input_sweeps as sw  # noqa: E402

CASES = ic.load_cases()
SWEEPS = sw.all_sweeps()


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_case(case, tmp_path):
    if case.get("skip"):
        pytest.skip(case["skip"])
    assert case.get("expect"), "every saved case needs an expectation"
    result = ic.run_case(case, tmp_path / "cases.db")
    verdict, problems = ic.judge(result, case["expect"])
    report = ic.format_report(0, case["name"], case, result, verdict, problems)
    assert verdict == "PASS", "\n" + report


@pytest.mark.parametrize("case", SWEEPS, ids=[c["name"] for c in SWEEPS])
def test_generated_input(case, tmp_path):
    result = ic.run_case(case, tmp_path / "s.db")
    verdict, problems = ic.judge(result, case["expect"])
    assert verdict == "PASS", "\n" + ic.format_report(0, case["name"], case, result, verdict, problems)


def test_case_names_are_unique():
    names = [c["name"] for c in CASES + SWEEPS]
    assert len(names) == len(set(names))


# ----------------------------------------------- user_input dictionary

def test_user_input_dictionary_with_professor_style_keys():
    user_input = {"nameModel": "deepseek", "runMode": "inference", "pad_token": "<pad>"}
    model, mode, pairs = ic.expand_user_input(user_input)
    assert model == ic.DEEPSEEK_ID and mode == "inference"
    assert pairs == [("b.pad_token", "<pad>")]
    result = ic.run_case({"user_input": user_input})
    assert result.status == "BLOCKED" and result.red == ["b.pad_token"]


def test_friendly_names_and_full_paths_are_equivalent():
    a = ic.run_case({"user_input": {"model": "mistral", "run_mode": "inference", "temperature": 0}})
    b = ic.run_case({"user_input": {"model": "mistral", "run_mode": "inference", "f.temperature": 0}})
    assert (a.status, a.red) == (b.status, b.red) == ("BLOCKED", ["f.do_sample"])


def test_unknown_choice_name_is_rejected_with_help():
    with pytest.raises(ValueError, match="unknown choice 'tempature'"):
        ic.expand_user_input({"model": "mistral", "tempature": 0})


def test_unchosen_values_fall_back_to_defaults():
    result = ic.run_case({"user_input": {"model": "mistral", "run_mode": "inference"}})
    assert result.status == "OK"


def test_old_case_form_still_works():
    ui = ic.case_to_user_input({"model": "mistral", "task": "inference", "edits": ["temperature = 0"]})
    assert ui == {"model": "mistral", "run_mode": "inference", "temperature": 0}


def test_sweeps_cover_both_outcomes():
    statuses = {c["expect"]["status"] for c in SWEEPS}
    assert statuses == {"OK", "BLOCKED"} and len(SWEEPS) > 50


# ----------------------------------------------- the checker can fail

def test_a_wrong_expectation_gives_fail():
    case = {"user_input": {"model": "mistral", "run_mode": "inference",
                           "pad_token": "<myPadToken>"}}
    result = ic.run_case(case)
    assert ic.judge(result, {"status": "OK"})[0] == "FAIL"
    assert ic.judge(result, {"status": "BLOCKED", "red": ["f.do_sample"]})[0] == "FAIL"
    assert ic.judge(result, {"status": "BLOCKED", "red": ["b.pad_token"]})[0] == "PASS"


def test_no_expectation_gives_no_verdict():
    result = ic.run_case({"model": "mistral", "task": "inference"})
    assert ic.judge(result, None) == ("NONE", [])


def test_run_all_reports_failures_and_exit_code(tmp_path):
    cases = [
        {"name": "good", "model": "mistral", "task": "inference",
         "expect": {"status": "OK"}},
        {"name": "wrong", "model": "mistral", "task": "inference",
         "expect": {"status": "BLOCKED"}},
        {"name": "skipped", "skip": "open question", "expect": {"status": "OK"}},
    ]
    lines = []
    code = ic.run_all(cases, tmp_path / "x.db", out=lines.append)
    text = "\n".join(lines)
    assert code == 1
    assert "SUMMARY: 1 passed, 1 failed, 1 skipped" in text
    assert "SKIPPED (open question)" in text


# ----------------------------------------------- input parsing

@pytest.mark.parametrize("text,expected", [
    ("0", 0), ("0.9", 0.9), ("true", True), ("null", None), ('""', ""),
    ('"quoted text"', "quoted text"), ("<myPadToken>", "<myPadToken>"),
    ("auto", "auto"), ("</s>", "</s>"),
])
def test_parse_value(text, expected):
    assert ic.parse_value(text) == expected


def test_parse_edit_line_and_errors():
    assert ic.parse_edit_line("b.pad_token = <pad>") == ("b.pad_token", "<pad>")
    assert ic.parse_edit_line("f.temperature=0") == ("f.temperature", 0)
    with pytest.raises(ValueError):
        ic.parse_edit_line("no equals sign here")


def test_parse_expect_text():
    assert ic.parse_expect_text("BLOCKED red=b.pad_token,f.do_sample warn=p.system_prompt") == {
        "status": "BLOCKED", "red": ["b.pad_token", "f.do_sample"],
        "warnings": ["p.system_prompt"]}
    assert ic.parse_expect_text("ok") == {"status": "OK"}
    with pytest.raises(ValueError):
        ic.parse_expect_text("MAYBE")


def test_full_json_config_is_merged_over_the_defaults():
    case = {"user_input": {"model": "mistral", "run_mode": "inference"},
            "config": {"f": {"temperature": 0}}}
    result = ic.run_case(case)
    assert result.status == "BLOCKED" and result.red == ["f.do_sample"]


# ----------------------------------------------- the interactive prompt

def _scripted(answers):
    """Fake input(): returns the answers in order, then behaves like Ctrl+D."""
    answers = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    return fake_input


def test_interactive_edits_session_runs_an_input_and_keeps_the_case(tmp_path):
    cases_file = tmp_path / "kept.json"
    answers = [
        "mistral", "inference", "edits",
        "pad_token = <myPadToken>", "",
        "BLOCKED red=b.pad_token",
        "n", "n",
        "y", "bad-pad-token",
        "n",
    ]
    lines = []
    ic.interactive(tmp_path / "t.db", cases_file, _scripted(answers), lines.append)
    assert "verdict:    PASS" in "\n".join(lines)
    kept = ic.load_cases(cases_file)
    assert [c["name"] for c in kept] == ["bad-pad-token"]
    assert kept[0]["user_input"]["pad_token"] == "<myPadToken>"
    assert kept[0]["expect"]["red"] == ["b.pad_token"]


def test_interactive_guided_session_asks_one_choice_at_a_time(tmp_path):
    # mistral/inference guided order: pad_token, padding_side, fix_mistral_regex, ...
    answers = [
        "mistral", "inference", "guided",
        "<myPadToken>", ".",          # pad_token typed, then stop
        "BLOCKED red=b.pad_token",
        "n", "n", "n", "n",
    ]
    lines = []
    ic.interactive(tmp_path / "t.db", tmp_path / "k.json", _scripted(answers), lines.append)
    text = "\n".join(lines)
    assert "verdict:    PASS" in text
    assert '"pad_token": "<myPadToken>"' in text


def test_interactive_guided_enter_keeps_defaults(tmp_path):
    answers = ["deepseek", "fine_tune", "guided", ".", "OK", "n", "n", "n", "n"]
    lines = []
    ic.interactive(tmp_path / "t.db", tmp_path / "k.json", _scripted(answers), lines.append)
    assert "verdict:    PASS" in "\n".join(lines)


def test_interactive_wrong_expectation_shows_fail(tmp_path):
    answers = ["mistral", "inference", "edits", "temperature = 0", "", "OK", "n", "n", "n", "n"]
    lines = []
    ic.interactive(tmp_path / "t.db", tmp_path / "k.json", _scripted(answers), lines.append)
    assert "verdict:    FAIL" in "\n".join(lines)


def test_interactive_session_ends_cleanly_on_end_of_input(tmp_path):
    lines = []
    ic.interactive(tmp_path / "t.db", tmp_path / "k.json", _scripted([]), lines.append)
    assert "bye" in "\n".join(lines)
