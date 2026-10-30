"""Unit tests for validation rules 1-5 (issue #4).

Written against specs/interfaces.md:
  - contract 1: field paths ("b.pad_token")
  - contract 2: defaults_for() returns a plain dict
  - contract 3: validate_cross_fields() returns a list of
    {"field", "level", "msg"} and never raises.

Each test builds a config from defaults_for(), changes one or two values,
and checks only the fields that belong to the rule under test. That keeps
a failure in one rule from breaking the tests for another.

Where the contract does not say which field a rule reports, the test
accepts either of the two fields involved. These spots are marked OPEN.
"""
import copy
import sys
from pathlib import Path

import pytest

# Let plain `pytest` find the app package, not only `python -m pytest`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.registry import defaults_for  # noqa: E402
from app.schemas import ConfigRecord, validate_cross_fields  # noqa: E402

MS = "mistralai/Mistral-Small-24B-Instruct-2501"
DS = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DS_PAD = "<｜end▁of▁sentence｜>"

ALL_COMBOS = [
    (MS, "fine_tune"),
    (MS, "inference"),
    (DS, "fine_tune"),
    (DS, "inference"),
]


# ---------------------------------------------------------------- helpers

def set_path(cfg: dict, path: str, value) -> None:
    """Set a value by field path, e.g. set_path(cfg, "b.pad_token", "<pad>")."""
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
    """Defaults for a model/task with changes given as {"f.temperature": 0}."""
    cfg = copy.deepcopy(defaults_for(model_id, task_type))
    for path, value in changes.items():
        set_path(cfg, path, value)
    return cfg


def validate(cfg: dict) -> list:
    return validate_cross_fields(ConfigRecord(**cfg))


def on_fields(issues: list, fields: set, level: str = None) -> list:
    """Issues reported on any of the given field paths (optionally one level)."""
    return [
        i for i in issues
        if i["field"] in fields and (level is None or i["level"] == level)
    ]


# --- fields each rule may report on (see OPEN notes in the module docstring)
RULE1_FIELDS = {"e.max_length", "f.max_new_tokens"}      # OPEN: which one?
RULE2_FIELDS = {"b.padding_side", "f.batch_size"}        # OPEN: which one?
RULE3_FIELDS = {"b.pad_token"}
RULE4_FIELDS = {"b.fix_mistral_regex"}
RULE5_FIELDS = {"f.do_sample"}  # contract rule 8 uses this exact example


# ------------------------------------------------- contract (all rules)

@pytest.mark.parametrize("model_id,task_type", ALL_COMBOS)
def test_issue_shape_follows_contract(model_id, task_type):
    """Contract 3: level is error/warning, msg is text, field resolves."""
    cfg = build(
        model_id, task_type,
        **{"f.temperature": 0, "f.do_sample": True, "b.pad_token": "<myPadToken>"},
    )
    issues = validate(cfg)
    assert isinstance(issues, list) and issues
    for issue in issues:
        assert set(issue) == {"field", "level", "msg"}
        assert issue["level"] in {"error", "warning"}
        assert isinstance(issue["msg"], str) and issue["msg"]
        # contract 1 rule 8: a reported path must resolve in the payload
        get_path(cfg, issue["field"])


@pytest.mark.parametrize("model_id,task_type", ALL_COMBOS)
def test_defaults_raise_no_errors_on_rules_1_to_5(model_id, task_type):
    """Baseline: untouched defaults pass every rule in this file."""
    issues = validate(build(model_id, task_type))
    all_fields = RULE1_FIELDS | RULE2_FIELDS | RULE3_FIELDS | RULE4_FIELDS | RULE5_FIELDS
    assert on_fields(issues, all_fields, "error") == []


# ------------------------------------------ rule 1: total length limit
# (e.max_length or 0) + f.max_new_tokens <= effective_max_seq_len
# Both models use an effective limit of 32768 (A's registry:
# effective_max_seq_len). DeepSeek's tokenizer model_max_length is 16384, but
# that is a separate field (b.model_max_length). Confirm with A that rule 1
# uses 32768 for DeepSeek.

def test_rule1_fail_ms_30000_plus_5000():
    """Issue #4 acceptance case: 30000 + 5000 = 35000 > 32768."""
    cfg = build(MS, "inference", **{"e.max_length": 30000, "f.max_new_tokens": 5000})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error")


def test_rule1_pass_ms_defaults_size():
    cfg = build(MS, "inference", **{"e.max_length": 4096, "f.max_new_tokens": 512})
    assert on_fields(validate(cfg), RULE1_FIELDS) == []


def test_rule1_pass_exactly_at_limit():
    """The rule is <=, so a total of exactly 32768 passes."""
    cfg = build(MS, "inference", **{"e.max_length": 32256, "f.max_new_tokens": 512})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error") == []


def test_rule1_fail_one_over_limit():
    cfg = build(MS, "inference", **{"e.max_length": 32257, "f.max_new_tokens": 512})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error")


def test_rule1_none_max_length_counts_as_zero():
    """max_length=None is valid input and is treated as 0."""
    cfg = build(MS, "inference", **{"e.max_length": None, "f.max_new_tokens": 512})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error") == []


def test_rule1_none_max_length_still_checks_new_tokens():
    cfg = build(MS, "inference", **{"e.max_length": None, "f.max_new_tokens": 40000})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error")


def test_rule1_fail_ds_far_over_any_limit():
    """40000 + 1024 is far over DeepSeek's limit."""
    cfg = build(DS, "inference", **{"e.max_length": 40000, "f.max_new_tokens": 1024})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error")


def test_rule1_pass_ds_defaults_size():
    cfg = build(DS, "inference", **{"e.max_length": 4096, "f.max_new_tokens": 1024})
    assert on_fields(validate(cfg), RULE1_FIELDS) == []


def test_rule1_pass_ds_between_16384_and_32768():
    """DeepSeek's effective limit is 32768, not its tokenizer's 16384."""
    cfg = build(DS, "inference", **{"e.max_length": 16000, "f.max_new_tokens": 1024})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error") == []


def test_rule1_fail_ds_just_over_32768():
    """31745 + 1024 = 32769, one over the limit."""
    cfg = build(DS, "inference", **{"e.max_length": 31745, "f.max_new_tokens": 1024})
    assert on_fields(validate(cfg), RULE1_FIELDS, "error")


# ---------------------------------- rule 2: batch inference needs left padding

def test_rule2_fail_inference_batch_with_right_padding():
    cfg = build(MS, "inference", **{"f.batch_size": 4, "b.padding_side": "right"})
    assert on_fields(validate(cfg), RULE2_FIELDS, "error")


def test_rule2_pass_inference_batch_with_left_padding():
    cfg = build(MS, "inference", **{"f.batch_size": 4, "b.padding_side": "left"})
    assert on_fields(validate(cfg), RULE2_FIELDS, "error") == []


def test_rule2_pass_batch_size_one_with_right_padding():
    cfg = build(MS, "inference", **{"f.batch_size": 1, "b.padding_side": "right"})
    assert on_fields(validate(cfg), RULE2_FIELDS, "error") == []


def test_rule2_fail_deepseek_too():
    cfg = build(DS, "inference", **{"f.batch_size": 2, "b.padding_side": "right"})
    assert on_fields(validate(cfg), RULE2_FIELDS, "error")


def test_rule2_not_applied_to_fine_tune():
    """The rule only covers inference; batch_size is unused in fine-tuning."""
    cfg = build(MS, "fine_tune", **{"f.batch_size": 4, "b.padding_side": "right"})
    assert on_fields(validate(cfg), RULE2_FIELDS, "error") == []


# --------------------------------------------- rule 3: pad_token whitelist
# MS allows "<pad>" or "</s>". DS allows only "<｜end▁of▁sentence｜>".

def test_rule3_fail_ms_my_pad_token():
    """Issue #4 acceptance case: <myPadToken> is rejected."""
    cfg = build(MS, "inference", **{"b.pad_token": "<myPadToken>"})
    assert on_fields(validate(cfg), RULE3_FIELDS, "error")


@pytest.mark.parametrize("token", ["<pad>", "</s>"])
def test_rule3_pass_ms_allowed_tokens(token):
    cfg = build(MS, "inference", **{"b.pad_token": token})
    assert on_fields(validate(cfg), RULE3_FIELDS, "error") == []


def test_rule3_fail_ms_with_deepseek_token():
    cfg = build(MS, "inference", **{"b.pad_token": DS_PAD})
    assert on_fields(validate(cfg), RULE3_FIELDS, "error")


def test_rule3_pass_ds_allowed_token():
    cfg = build(DS, "inference", **{"b.pad_token": DS_PAD})
    assert on_fields(validate(cfg), RULE3_FIELDS, "error") == []


@pytest.mark.parametrize("token", ["<pad>", "</s>", "<myPadToken>"])
def test_rule3_fail_ds_rejects_everything_else(token):
    cfg = build(DS, "inference", **{"b.pad_token": token})
    assert on_fields(validate(cfg), RULE3_FIELDS, "error")


def test_rule3_fail_empty_pad_token():
    cfg = build(MS, "inference", **{"b.pad_token": ""})
    assert on_fields(validate(cfg), RULE3_FIELDS, "error")


# ----------------------------------- rule 4: Mistral needs fix_mistral_regex

@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule4_fail_ms_regex_fix_off(task_type):
    cfg = build(MS, task_type, **{"b.fix_mistral_regex": False})
    assert on_fields(validate(cfg), RULE4_FIELDS, "error")


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule4_pass_ms_regex_fix_on(task_type):
    cfg = build(MS, task_type, **{"b.fix_mistral_regex": True})
    assert on_fields(validate(cfg), RULE4_FIELDS, "error") == []


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_rule4_pass_ds_regex_fix_off(task_type):
    """DeepSeek defaults to False and must not be flagged."""
    cfg = build(DS, task_type, **{"b.fix_mistral_regex": False})
    assert on_fields(validate(cfg), RULE4_FIELDS, "error") == []


# ------------------------------ rule 5: temperature and do_sample agree
# temperature == 0 needs do_sample False; temperature > 0 needs do_sample True.
# Contract 3 rule 8: the issue is reported on f.do_sample.

def test_rule5_fail_temperature_zero_with_sampling_on():
    """Issue #4 acceptance case."""
    cfg = build(MS, "inference", **{"f.temperature": 0, "f.do_sample": True})
    assert on_fields(validate(cfg), RULE5_FIELDS, "error")


def test_rule5_fail_positive_temperature_with_sampling_off():
    cfg = build(MS, "inference", **{"f.temperature": 0.6, "f.do_sample": False})
    assert on_fields(validate(cfg), RULE5_FIELDS, "error")


def test_rule5_pass_temperature_zero_with_sampling_off():
    cfg = build(MS, "inference", **{"f.temperature": 0, "f.do_sample": False})
    assert on_fields(validate(cfg), RULE5_FIELDS, "error") == []


def test_rule5_pass_positive_temperature_with_sampling_on():
    cfg = build(MS, "inference", **{"f.temperature": 0.9, "f.do_sample": True})
    assert on_fields(validate(cfg), RULE5_FIELDS, "error") == []


def test_rule5_float_zero_counts_as_zero():
    cfg = build(DS, "inference", **{"f.temperature": 0.0, "f.do_sample": True})
    assert on_fields(validate(cfg), RULE5_FIELDS, "error")


# ----------------------------------------- validation never raises

def test_validate_returns_issues_instead_of_raising():
    """Contract 3 rule 4: a bad config yields a list, never an exception."""
    cfg = build(
        MS, "inference",
        **{"b.pad_token": "<myPadToken>", "f.temperature": 0, "f.do_sample": True},
    )
    assert isinstance(validate(cfg), list)
