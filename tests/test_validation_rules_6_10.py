"""Unit tests for validation rules 6-10 (issue #5).

Written against specs/interfaces.md:
  - contract 1: field paths ("p.system_prompt")
  - contract 2: defaults_for() returns a plain dict
  - contract 3: validate_cross_fields() returns a list of
    {"field", "level", "msg"} and never raises.

Same approach as tests/test_validation_rules_1_5.py: build a config from
defaults_for(), change one or two values, and look only at the fields that
belong to the rule under test.

Spots marked OPEN are assumptions to confirm with A or against
specs/parameters.md before the tests are treated as final.
"""
import copy

import pytest

from app.registry import defaults_for
from app.schemas import ConfigRecord, validate_cross_fields

MS = "mistralai/Mistral-Small-24B-Instruct-2501"
DS = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

ALL_COMBOS = [
    (MS, "fine_tune"),
    (MS, "inference"),
    (DS, "fine_tune"),
    (DS, "inference"),
]


# ---------------------------------------------------------------- helpers

def set_path(cfg: dict, path: str, value) -> None:
    *parents, leaf = path.split(".")
    node = cfg
    for key in parents:
        node = node[key]
    node[leaf] = value


def get_path(cfg: dict, path: str):
    node = cfg
    for key in path.split("."):
        node = node[key]
    return node


def build(model_id: str, task_type: str, **changes) -> dict:
    """Defaults for a model/task with changes given as {"a.tp_plan": None}."""
    cfg = copy.deepcopy(defaults_for(model_id, task_type))
    for path, value in changes.items():
        set_path(cfg, path, value)
    return cfg


def validate(cfg: dict) -> list:
    return validate_cross_fields(ConfigRecord(**cfg))


def on_fields(issues: list, fields: set, level: str = None) -> list:
    return [
        i for i in issues
        if i["field"] in fields and (level is None or i["level"] == level)
    ]


# --- paths used by these rules
LOAD_BEST = "d.sft.load_best_model_at_end"   # in contract 1
EVAL_STRATEGY = "d.sft.eval_strategy"        # OPEN: assumed spelling, check parameters.md
SAVE_STRATEGY = "d.sft.save_strategy"        # OPEN: assumed spelling, check parameters.md

# --- fields each rule may report on (see OPEN notes)
RULE6_STRATEGY_FIELDS = {EVAL_STRATEGY, SAVE_STRATEGY, LOAD_BEST}  # OPEN: which one?
RULE6_EVAL_FILE_FIELDS = {"c.eval_data_file"}
RULE7_FIELDS = {"p.assistant_prompt", "p.verdict_parse_marker"}    # OPEN: which one?
RULE8_FIELDS = {"a.tp_plan", "a.device_map"}                       # OPEN: which one?
RULE9_FIELDS = {"p.system_prompt"}
RULE10_FIELDS = {"p.system_prompt"}


# ------------------------------------------------------ sanity: paths exist

@pytest.mark.parametrize("model_id", [MS, DS])
def test_assumed_paths_exist_in_fine_tune_defaults(model_id):
    """If this fails, a path in this file is misspelled; fix the constants above."""
    cfg = build(model_id, "fine_tune")
    for path in (LOAD_BEST, EVAL_STRATEGY, SAVE_STRATEGY,
                 "c.eval_data_file", "a.tp_plan", "a.device_map",
                 "p.system_prompt", "p.assistant_prompt", "p.verdict_parse_marker"):
        get_path(cfg, path)


@pytest.mark.parametrize("model_id,task_type", ALL_COMBOS)
def test_defaults_raise_no_errors_on_rules_6_to_10(model_id, task_type):
    """Baseline: untouched defaults pass every rule in this file.

    Rule 10 may legitimately give a warning on DeepSeek if its default system
    prompt is non-empty, so only errors are checked here.
    """
    issues = validate(build(model_id, task_type))
    all_fields = (RULE6_STRATEGY_FIELDS | RULE6_EVAL_FILE_FIELDS | RULE7_FIELDS
                  | RULE8_FIELDS | RULE9_FIELDS | RULE10_FIELDS)
    assert on_fields(issues, all_fields, "error") == []


# ------------- rule 6: load_best_model_at_end needs matching strategies + eval file
# Applies to fine_tune only.

def test_rule6_fail_strategies_differ():
    cfg = build(MS, "fine_tune", **{
        LOAD_BEST: True, EVAL_STRATEGY: "steps", SAVE_STRATEGY: "epoch"})
    assert on_fields(validate(cfg), RULE6_STRATEGY_FIELDS, "error")


def test_rule6_pass_strategies_match():
    cfg = build(MS, "fine_tune", **{
        LOAD_BEST: True, EVAL_STRATEGY: "epoch", SAVE_STRATEGY: "epoch"})
    assert on_fields(validate(cfg), RULE6_STRATEGY_FIELDS, "error") == []


def test_rule6_pass_strategies_differ_but_flag_off():
    """With the flag off, mismatched strategies are fine."""
    cfg = build(MS, "fine_tune", **{
        LOAD_BEST: False, EVAL_STRATEGY: "steps", SAVE_STRATEGY: "epoch"})
    assert on_fields(validate(cfg), RULE6_STRATEGY_FIELDS, "error") == []


@pytest.mark.parametrize("empty", [None, ""])
def test_rule6_fail_missing_eval_file(empty):
    cfg = build(MS, "fine_tune", **{LOAD_BEST: True, "c.eval_data_file": empty})
    assert on_fields(validate(cfg), RULE6_EVAL_FILE_FIELDS, "error")


def test_rule6_pass_eval_file_present():
    cfg = build(MS, "fine_tune", **{
        LOAD_BEST: True, "c.eval_data_file": "data/evalData.json"})
    assert on_fields(validate(cfg), RULE6_EVAL_FILE_FIELDS, "error") == []


def test_rule6_pass_no_eval_file_when_flag_off():
    cfg = build(MS, "fine_tune", **{LOAD_BEST: False, "c.eval_data_file": None})
    assert on_fields(validate(cfg), RULE6_EVAL_FILE_FIELDS, "error") == []


def test_rule6_fail_applies_to_deepseek_too():
    cfg = build(DS, "fine_tune", **{
        LOAD_BEST: True, EVAL_STRATEGY: "steps", SAVE_STRATEGY: "epoch"})
    assert on_fields(validate(cfg), RULE6_STRATEGY_FIELDS, "error")


def test_rule6_not_applied_to_inference():
    """Component d is hidden under inference, so this rule must not fire."""
    cfg = build(MS, "inference", **{
        LOAD_BEST: True, EVAL_STRATEGY: "steps", SAVE_STRATEGY: "epoch",
        "c.eval_data_file": None})
    issues = validate(cfg)
    assert on_fields(issues, RULE6_STRATEGY_FIELDS | RULE6_EVAL_FILE_FIELDS, "error") == []


# ----------------- rule 7: assistant prompt and verdict marker stay consistent
# OPEN: the spec says "between fine-tuning and downstream inference", which
# needs two configs. A single ConfigRecord holds both fields, so these tests
# use one reading: verdict_parse_marker must appear inside assistant_prompt.
# That matches the original scripts: training used "### Response: Verdict: "
# but the eval split used "### Response: ", and inference searched for
# "### Response: Verdict: ". Skipped until A confirms how rule 7 works.

RULE7_SKIP = "OPEN: rule 7 semantics not confirmed with A; enable once settled."


@pytest.mark.skip(reason=RULE7_SKIP)
def test_rule7_pass_marker_inside_assistant_prompt():
    cfg = build(MS, "inference", **{
        "p.assistant_prompt": "\n### Response: Verdict: ",
        "p.verdict_parse_marker": "### Response: Verdict: "})
    assert on_fields(validate(cfg), RULE7_FIELDS, "error") == []


@pytest.mark.skip(reason=RULE7_SKIP)
def test_rule7_fail_marker_missing_from_assistant_prompt():
    """The bug in the original scripts: no 'Verdict:' in the prompt."""
    cfg = build(MS, "inference", **{
        "p.assistant_prompt": "\n### Response: ",
        "p.verdict_parse_marker": "### Response: Verdict: "})
    assert on_fields(validate(cfg), RULE7_FIELDS, "error")


# ------------------------------- rule 8: tp_plan and device_map are exclusive

def test_rule8_fail_both_set():
    cfg = build(MS, "inference", **{"a.tp_plan": "auto", "a.device_map": "auto"})
    assert on_fields(validate(cfg), RULE8_FIELDS, "error")


def test_rule8_pass_only_tp_plan():
    cfg = build(MS, "inference", **{"a.tp_plan": "auto", "a.device_map": None})
    assert on_fields(validate(cfg), RULE8_FIELDS, "error") == []


def test_rule8_pass_only_device_map():
    cfg = build(MS, "inference", **{"a.tp_plan": None, "a.device_map": "auto"})
    assert on_fields(validate(cfg), RULE8_FIELDS, "error") == []


def test_rule8_pass_neither_set():
    cfg = build(MS, "inference", **{"a.tp_plan": None, "a.device_map": None})
    assert on_fields(validate(cfg), RULE8_FIELDS, "error") == []


def test_rule8_empty_string_counts_as_empty():
    """The spec says 'both non-empty', so an empty string is not set."""
    cfg = build(MS, "inference", **{"a.tp_plan": "auto", "a.device_map": ""})
    assert on_fields(validate(cfg), RULE8_FIELDS, "error") == []


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule8_fail_in_both_task_types(task_type):
    cfg = build(DS, task_type, **{"a.tp_plan": "auto", "a.device_map": "balanced"})
    assert on_fields(validate(cfg), RULE8_FIELDS, "error")


# -------------------------- rule 9: Mistral with empty system prompt -> error

@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule9_fail_ms_empty_system_prompt(task_type):
    cfg = build(MS, task_type, **{"p.system_prompt": ""})
    issues = on_fields(validate(cfg), RULE9_FIELDS)
    assert issues
    # issue #5 acceptance: rule 9 must be an error, not a warning
    assert all(i["level"] == "error" for i in issues)


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule9_pass_ms_non_empty_system_prompt(task_type):
    cfg = build(MS, task_type, **{"p.system_prompt": "You are a fact-checker."})
    assert on_fields(validate(cfg), RULE9_FIELDS, "error") == []


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule9_pass_ds_empty_system_prompt(task_type):
    """Rule 9 is Mistral-only; an empty DeepSeek prompt is the normal case."""
    cfg = build(DS, task_type, **{"p.system_prompt": ""})
    assert on_fields(validate(cfg), RULE9_FIELDS, "error") == []


# ---------------------- rule 10: DeepSeek with non-empty system prompt -> warning

@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule10_fail_ds_non_empty_system_prompt(task_type):
    cfg = build(DS, task_type, **{"p.system_prompt": "You are a fact-checker."})
    issues = on_fields(validate(cfg), RULE10_FIELDS)
    assert issues
    # issue #5 acceptance: rule 10 is a warning, and a warning must not block saving
    assert all(i["level"] == "warning" for i in issues)


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule10_pass_ds_empty_system_prompt(task_type):
    cfg = build(DS, task_type, **{"p.system_prompt": ""})
    assert on_fields(validate(cfg), RULE10_FIELDS) == []


def test_rule10_not_applied_to_mistral():
    """A non-empty Mistral prompt is the normal case and gets no warning."""
    cfg = build(MS, "inference", **{"p.system_prompt": "You are a fact-checker."})
    assert on_fields(validate(cfg), RULE10_FIELDS, "warning") == []


def test_rule10_warning_alone_does_not_block_saving():
    """Contract 3 rule 6: only errors block saving."""
    cfg = build(DS, "inference", **{"p.system_prompt": "You are a fact-checker."})
    issues = validate(cfg)
    assert on_fields(issues, RULE10_FIELDS, "warning")
    assert [i for i in issues if i["level"] == "error"] == []


# ----------------------------------------- shape and field paths (all rules)

def test_issue_shape_and_paths_follow_contract():
    """Issue #5: field paths follow the #1 contract and resolve in the payload."""
    cfg = build(MS, "fine_tune", **{
        "p.system_prompt": "",
        "a.tp_plan": "auto", "a.device_map": "auto",
        LOAD_BEST: True, EVAL_STRATEGY: "steps", SAVE_STRATEGY: "epoch"})
    issues = validate(cfg)
    assert issues
    for issue in issues:
        assert set(issue) == {"field", "level", "msg"}
        assert issue["level"] in {"error", "warning"}
        assert isinstance(issue["msg"], str) and issue["msg"]
        get_path(cfg, issue["field"])
