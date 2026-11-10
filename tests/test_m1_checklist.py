"""M1.0 acceptance tests, driven by UI input (issue #13).

These tests do not call the validator on a hand-built config. They go
through the same path the form takes:

    user edits a widget  ->  form state  ->  build_config()  (ConfigRecord)
        ->  validate_cross_fields()  ->  console.evaluate()
        ->  console.apply_issue_props()  (red field + disabled Save)
        ->  Save to DB  (app/store.py)

``FakeForm`` stands in for the NiceGUI form that app/main.py (#6-#8) will
build. It keeps one widget per field path (contract 1), sets values the way
widgets emit them (numbers, strings, "" for a cleared text box, booleans),
and re-runs the console logic after every edit, like the live form does.
The real console functions and the real validator, registry and store are
used; only NiceGUI is replaced.

Each test carries a comment naming the M1.0 acceptance item it covers
(planning/TaskList.md, section M1.0).
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import store  # noqa: E402
from app.registry import (  # noqa: E402
    DEEPSEEK_ID,
    MISTRAL_ID,
    MODEL_IDS,
    defaults_for,
)
from app.schemas import ConfigRecord, validate_cross_fields  # noqa: E402
from app.ui import console  # noqa: E402

DS_PAD = "<｜end▁of▁sentence｜>"


# ------------------------------------------------------------ UI stand-ins

class FakeWidget:
    """Mirrors NiceGUI's Element.props: tracks whether the field is red."""

    def __init__(self) -> None:
        self.error_message = None

    def props(self, add: str = None, remove: str = None) -> None:
        if remove and "error" in remove:
            self.error_message = None
        if add and add.startswith("error"):
            start = add.index('"') + 1
            self.error_message = add[start:add.rindex('"')]

    @property
    def is_red(self) -> bool:
        return self.error_message is not None


def leaf_paths(node, prefix=""):
    """All editable field paths of a payload (contract 1: lists are leaves)."""
    paths = []
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths += leaf_paths(value, path + ".")
        else:
            paths.append(path)
    return paths


class FakeForm:
    """The form: widgets by field path, state edited by the 'user'."""

    def __init__(self, model_id=MISTRAL_ID, task_type="fine_tune"):
        self.payload = defaults_for(model_id, task_type)
        self.widgets = {
            path: FakeWidget()
            for path in leaf_paths(defaults_for(MISTRAL_ID, "fine_tune"))
        }
        self.state = None
        self.refresh()

    # --- what the user does
    def pick_model(self, model_id):
        """Page 1: selecting a model reloads the defaults."""
        self.payload = defaults_for(model_id, self.payload["task_type"])
        return self.refresh()

    def pick_task(self, task_type):
        """Page 2: switching fine_tune / inference reloads the defaults."""
        self.payload = defaults_for(self.payload["model_id"], task_type)
        return self.refresh()

    def set(self, path, value):
        """Type or select a value in the widget for ``path``, then re-check."""
        *parents, leaf = path.split(".")
        node = self.payload
        for key in parents:
            node = node[key]
        node[leaf] = value
        return self.refresh()

    def reset_to_defaults(self):
        self.payload = defaults_for(
            self.payload["model_id"], self.payload["task_type"])
        return self.refresh()

    # --- what the console does (same calls as ValidationConsole.refresh)
    def build_config(self) -> ConfigRecord:
        return ConfigRecord(**copy.deepcopy(self.payload))

    def refresh(self):
        cfg = self.build_config()
        issues = list(validate_cross_fields(cfg))
        self.state = console.evaluate(cfg, issues)
        self.issues = issues
        self.highlighted = console.apply_issue_props(self.widgets, issues)
        return self.state

    def click_save(self, db_path):
        """Save to DB: the button is disabled while errors exist."""
        if not self.state.save_enabled:
            return None
        return store.insert_config(self.build_config(), db_path)

    # --- what the user sees
    def red_fields(self):
        return sorted(p for p, w in self.widgets.items() if w.is_red)

    def warnings_for(self, path):
        return [i for i in self.issues
                if i["field"] == path and i["level"] == "warning"]


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "platform.db"


# ======================================================================
# Page 1: select an LLM from a list
# ======================================================================

# M1.0 acceptance: "Page 1: select an LLM from a list (Mistral first,
# leave room for DeepSeek)"
def test_both_models_can_be_selected_and_start_valid():
    assert set(MODEL_IDS) == {MISTRAL_ID, DEEPSEEK_ID}
    for model_id in MODEL_IDS:
        form = FakeForm(model_id, "fine_tune")
        assert form.state.errors == ()
        assert form.state.save_enabled is True
        assert form.red_fields() == []


# ======================================================================
# Page 2: fine-tune / inference switches the rest of the form
# ======================================================================

# M1.0 acceptance: "Page 2: choose task type; the remaining form switches"
def test_switching_task_type_reloads_the_task_specific_fields():
    form = FakeForm(MISTRAL_ID, "fine_tune")
    assert form.payload["e"]["add_generation_prompt"] is False
    assert form.payload["c"]["train_data_file"] is not None
    assert form.payload["c"]["test_data_file"] is None

    form.pick_task("inference")
    assert form.payload["e"]["add_generation_prompt"] is True
    assert form.payload["c"]["train_data_file"] is None
    assert form.payload["c"]["test_data_file"] is not None
    assert form.state.save_enabled is True


# M1.0 acceptance: "d. Optimizer: fixed parameters, displayed read-only
# (r=64, lora_alpha=16, dropout=0.1, lr=2e-4)"
@pytest.mark.parametrize("model_id", [MISTRAL_ID, DEEPSEEK_ID])
@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_optimizer_block_is_fixed_for_every_model_and_task(model_id, task_type):
    d = FakeForm(model_id, task_type).payload["d"]
    assert d["lora"]["r"] == 64
    assert d["lora"]["lora_alpha"] == 16
    assert d["lora"]["lora_dropout"] == 0.1
    assert d["sft"]["learning_rate"] == 0.0002


# M1.0 acceptance: demo step "switch models" must not leave stale errors
def test_switching_model_clears_errors_from_the_previous_model():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("b.pad_token", "<myPadToken>")
    assert form.red_fields() == ["b.pad_token"]

    form.pick_model(DEEPSEEK_ID)
    assert form.payload["b"]["pad_token"] == DS_PAD
    assert form.red_fields() == []
    assert form.state.errors == ()
    assert form.state.save_enabled is True


# ======================================================================
# Page 3: the prompts
# ======================================================================

# M1.0 acceptance: "for Mistral the system prompt must be required"
# (this is the demo step "clear Mistral's system_prompt")
@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_clearing_the_mistral_system_prompt_blocks_save_and_marks_the_field(
        task_type, db_path):
    form = FakeForm(MISTRAL_ID, task_type)
    form.set("p.system_prompt", "")

    assert form.red_fields() == ["p.system_prompt"]
    assert form.widgets["p.system_prompt"].error_message
    assert form.state.save_enabled is False
    assert form.click_save(db_path) is None
    assert store.list_configs(db_path) == []


# M1.0 acceptance: same requirement; whitespace only is also "empty"
def test_whitespace_only_mistral_system_prompt_is_rejected():
    form = FakeForm(MISTRAL_ID, "fine_tune")
    form.set("p.system_prompt", "   ")
    assert form.red_fields() == ["p.system_prompt"]


# M1.0 acceptance: errors are live; fixing the field clears the red box
def test_refilling_the_system_prompt_clears_the_error_and_reenables_save():
    form = FakeForm(MISTRAL_ID, "fine_tune")
    form.set("p.system_prompt", "")
    assert form.state.save_enabled is False

    form.set("p.system_prompt", "You are a fact-checking assistant.")
    assert form.red_fields() == []
    assert form.state.save_enabled is True


# M1.0 acceptance: DeepSeek system prompt is a warning, not a blocker
def test_deepseek_system_prompt_warns_but_does_not_block_or_turn_red(db_path):
    form = FakeForm(DEEPSEEK_ID, "inference")
    form.set("p.system_prompt", "Give a verdict.")

    assert form.warnings_for("p.system_prompt")
    assert form.red_fields() == []
    assert form.state.errors == ()
    assert form.state.save_enabled is True
    assert form.click_save(db_path) is not None


# ======================================================================
# Page 4: per-component parameters (the issue #4 / #5 cases, via the UI)
# ======================================================================

# M1.0 acceptance: "pad_token validity ... <myPadToken> is rejected"
def test_typing_my_pad_token_turns_the_pad_token_field_red():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("b.pad_token", "<myPadToken>")

    assert form.red_fields() == ["b.pad_token"]
    assert "<myPadToken>" in form.widgets["b.pad_token"].error_message
    assert form.state.save_enabled is False


# M1.0 acceptance: "length constraint ... 30000 + 5000 is rejected"
def test_length_overflow_is_marked_on_a_field_the_user_can_edit():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("e.max_length", 30000)
    form.set("f.max_new_tokens", 5000)

    assert form.state.save_enabled is False
    assert form.red_fields() == ["f.max_new_tokens"]

    form.set("f.max_new_tokens", 512)
    assert form.red_fields() == []
    assert form.state.save_enabled is True


# M1.0 acceptance: "temperature=0 + do_sample=True is rejected"
def test_temperature_zero_with_sampling_marks_do_sample_and_fix_clears_it():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("f.temperature", 0)
    assert form.red_fields() == ["f.do_sample"]
    assert form.state.save_enabled is False

    form.set("f.do_sample", False)
    assert form.red_fields() == []
    assert form.state.save_enabled is True


# M1.0 acceptance: "temperature > 0 requires do_sample"
def test_switching_sampling_off_with_positive_temperature_is_rejected():
    form = FakeForm(DEEPSEEK_ID, "inference")
    form.set("f.do_sample", False)
    assert form.red_fields() == ["f.do_sample"]


# M1.0 acceptance: "padding_side must be left for batched inference"
def test_batch_inference_with_right_padding_is_rejected_only_for_inference():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("f.batch_size", 4)
    form.set("b.padding_side", "right")
    assert form.red_fields() == ["b.padding_side"]

    form.pick_task("fine_tune")  # defaults reload: left padding, batch 1
    form.set("f.batch_size", 4)
    form.set("b.padding_side", "right")
    assert form.red_fields() == []


# M1.0 acceptance: "fix_mistral_regex" is required for Mistral
def test_turning_off_the_mistral_regex_fix_is_rejected_but_not_for_deepseek():
    form = FakeForm(MISTRAL_ID, "fine_tune")
    form.set("b.fix_mistral_regex", False)
    assert form.red_fields() == ["b.fix_mistral_regex"]

    ds = FakeForm(DEEPSEEK_ID, "fine_tune")
    ds.set("b.fix_mistral_regex", False)
    assert ds.red_fields() == []


# M1.0 acceptance: "eval/save strategy consistency" (+ eval data file)
def test_best_model_needs_matching_strategies_and_an_eval_file():
    form = FakeForm(MISTRAL_ID, "fine_tune")
    form.set("d.sft.eval_strategy", "steps")
    assert form.red_fields() == ["d.sft.eval_strategy"]
    form.set("d.sft.eval_strategy", "epoch")
    assert form.red_fields() == []

    # a cleared text box emits "", a cleared optional field emits None
    for cleared in ("", None):
        form.set("c.eval_data_file", cleared)
        assert form.red_fields() == ["c.eval_data_file"], repr(cleared)
        form.set("c.eval_data_file", "data/evalData.json")
        assert form.red_fields() == []


# M1.0 acceptance: tp_plan / device_map are exclusive; a cleared text box
# ("") must count as not set, otherwise every default would be rejected
def test_tp_plan_and_device_map_exclusive_and_empty_string_means_unset():
    form = FakeForm(MISTRAL_ID, "inference")
    assert form.payload["a"]["tp_plan"] == "auto"

    form.set("a.device_map", "")
    assert form.red_fields() == []
    form.set("a.device_map", "auto")
    assert form.red_fields() == ["a.device_map"]
    form.set("a.tp_plan", None)
    assert form.red_fields() == []


# M1.0 acceptance: "optional numeric fields can be left empty"
def test_cleared_optional_numbers_do_not_break_validation():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("e.max_length", None)
    form.set("f.top_k", None)
    assert form.state.errors == ()
    assert form.state.save_enabled is True


# ======================================================================
# Component h: optional **kwargs
# ======================================================================

# M1.0 acceptance: "h. Optional arguments per component"; the path form is
# h.<stage>.<argument> (contract 1)
def test_optional_kwargs_reach_the_json_preview():
    form = FakeForm(MISTRAL_ID, "inference")
    form.payload["h"] = {"generate": {"repetition_penalty": 1.1}}
    form.refresh()

    preview = json.loads(form.state.json_text)
    assert preview["h"] == {"generate": {"repetition_penalty": 1.1}}
    assert form.state.save_enabled is True


# ======================================================================
# Full preview, errors located to the field
# ======================================================================

# M1.0 acceptance: "Full preview before submission, with validation errors
# located to the offending field"
def test_every_reported_error_points_at_a_real_widget():
    """Contract 1, rule 8: a reported path must exist in the form."""
    bad_edits = [
        (MISTRAL_ID, "inference", {"b.pad_token": "<myPadToken>"}),
        (MISTRAL_ID, "inference", {"e.max_length": 30000, "f.max_new_tokens": 5000}),
        (MISTRAL_ID, "inference", {"f.temperature": 0}),
        (MISTRAL_ID, "inference", {"f.batch_size": 4, "b.padding_side": "right"}),
        (MISTRAL_ID, "fine_tune", {"b.fix_mistral_regex": False}),
        (MISTRAL_ID, "fine_tune", {"d.sft.eval_strategy": "steps"}),
        (MISTRAL_ID, "fine_tune", {"c.eval_data_file": ""}),
        (MISTRAL_ID, "inference", {"a.device_map": "auto"}),
        (MISTRAL_ID, "fine_tune", {"p.system_prompt": ""}),
    ]
    for model_id, task_type, edits in bad_edits:
        form = FakeForm(model_id, task_type)
        for path, value in edits.items():
            form.set(path, value)
        errors = [i for i in form.issues if i["level"] == "error"]
        assert errors, edits
        for issue in errors:
            assert issue["field"] in form.widgets, (edits, issue)
            assert form.widgets[issue["field"]].is_red, (edits, issue)


def test_json_preview_always_matches_the_current_form_state():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("f.temperature", 0.9)
    form.set("p.system_prompt", "Custom instruction.")

    preview = json.loads(form.state.json_text)
    assert preview["f"]["temperature"] == 0.9
    assert preview["p"]["system_prompt"] == "Custom instruction."
    assert preview == form.build_config().model_dump(mode="json") | {
        "config_id": preview["config_id"],
        "created_at": preview["created_at"],
    }


def test_reset_to_defaults_clears_every_error():
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("b.pad_token", "<myPadToken>")
    form.set("f.temperature", 0)
    assert form.red_fields()

    form.reset_to_defaults()
    assert form.red_fields() == []
    assert form.state.save_enabled is True


# ======================================================================
# Database: save, dedupe, load, duplicate, persistence after restart
# ======================================================================

# M1.0 acceptance: "Persist everything collected by the UI in JSON format"
def test_saved_record_holds_every_field_the_form_collected(db_path):
    form = FakeForm(MISTRAL_ID, "fine_tune")
    form.set("f.temperature", 0.3)
    form.set("p.system_prompt", "Edited in the form.")
    saved_id = form.click_save(db_path)

    stored = store.get_config(saved_id, db_path).model_dump(mode="json")
    expected = form.build_config().model_dump(mode="json")
    for key in ("a", "b", "c", "d", "e", "f", "g", "h", "p",
                "model_id", "task_type"):
        assert stored[key] == expected[key], key
    assert stored["f"]["temperature"] == 0.3
    assert stored["p"]["system_prompt"] == "Edited in the form."


# M1.0 acceptance: "config fingerprint (dedup)"
def test_saving_the_same_form_twice_is_rejected_with_the_existing_id(db_path):
    form = FakeForm(MISTRAL_ID, "inference")
    first = form.click_save(db_path)
    assert first

    with pytest.raises(ValueError, match="identical experiment config") as info:
        form.click_save(db_path)
    assert first in str(info.value)
    assert len(store.list_configs(db_path)) == 1


# M1.0 acceptance: "Config read / list / duplicate (duplicate is the main
# entry point for 'change one parameter and run again')"
def test_duplicate_change_one_parameter_and_save_as_new_experiment(db_path):
    form = FakeForm(MISTRAL_ID, "inference")
    original_id = form.click_save(db_path)

    # the form loads the copy; identity fields are not form fields
    payload = store.duplicate_config(original_id, db_path).model_dump()
    for bookkeeping in ("config_id", "config_hash", "created_at",
                        "created_by", "notes"):
        payload.pop(bookkeeping)
    form.payload = payload
    form.set("f.temperature", 0.9)

    second_id = form.click_save(db_path)
    assert second_id and second_id != original_id
    assert {c["config_id"] for c in store.list_configs(db_path)} == {
        original_id, second_id}
    assert store.get_config(second_id, db_path).f.temperature == 0.9
    assert store.get_config(original_id, db_path).f.temperature == 0.15


# M1.0 acceptance: "Include persistence after a database restart"
# A fresh Python process stands in for the restart: nothing from this
# process (connections, caches, imports) can leak into the check.
def test_saved_config_survives_a_restart(db_path, tmp_path):
    form = FakeForm(DEEPSEEK_ID, "inference")
    form.set("f.temperature", 0.7)
    saved_id = form.click_save(db_path)

    code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]);"
        "from app.store import get_config, list_configs;"
        "cfg = get_config(sys.argv[2]);"
        "print(json.dumps({'temp': cfg.f.temperature, 'model': cfg.model_id,"
        " 'count': len(list_configs())}))"
    )
    env = {**os.environ, "LLM_CONFIG_DB_PATH": str(db_path)}
    out = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), saved_id],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    result = json.loads(out)
    assert result == {"temp": 0.7, "model": DEEPSEEK_ID, "count": 1}


# M1.0 acceptance: "Save is only possible when the config is valid"
def test_save_is_refused_for_every_blocking_error(db_path):
    form = FakeForm(MISTRAL_ID, "inference")
    form.set("b.pad_token", "<myPadToken>")
    assert form.click_save(db_path) is None
    assert store.list_configs(db_path) == []

    form.set("b.pad_token", "<pad>")
    assert form.click_save(db_path) is not None
    assert len(store.list_configs(db_path)) == 1