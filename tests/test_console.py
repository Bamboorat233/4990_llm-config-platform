"""Checks for the validation console logic (issue #8).

Only the NiceGUI half of ``app/ui/console.py`` needs ``nicegui``
installed; the logic tested here does not, which is why importing this
module at collection time also proves the lazy-import requirement.

Runs with either:

    python tests/test_console.py
    pytest tests/test_console.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for  # noqa: E402
from app.schemas import ConfigRecord, ValidationIssue  # noqa: E402
from app.ui import console  # noqa: E402
from app.ui.console import (  # noqa: E402
    apply_issue_props,
    clear_error_props,
    evaluate,
    panel_lines,
)

RULE_9_MESSAGE = (
    "Mistral requires a system prompt; the template would inject a "
    "date-stamped default."
)

RULE_9_ISSUE: ValidationIssue = {
    "field": "p.system_prompt",
    "level": "error",
    "msg": RULE_9_MESSAGE,
}

CLEAR_PROP = "error error-message"


class FakeWidget:
    """Records NiceGUI ``props`` calls; mirrors ``Element.props``."""

    def __init__(self) -> None:
        self.added: list[str] = []
        self.removed: list[str] = []

    def props(self, add: str | None = None, remove: str | None = None) -> None:
        if add:
            self.added.append(add)
        if remove:
            self.removed.append(remove)


def _record(
    model_id: str = MISTRAL_ID,
    task_type: str = "fine_tune",
    *,
    system_prompt: str | None = None,
) -> ConfigRecord:
    payload = defaults_for(model_id, task_type)
    if system_prompt is not None:
        payload["p"]["system_prompt"] = system_prompt
    return ConfigRecord(**payload)


def _error_prop(message: str) -> str:
    return f'error error-message="{message}"'


def _stub_validator(cfg: ConfigRecord) -> list[ValidationIssue]:
    """Stub for ``validate_cross_fields`` (#4/#5); only rule 9 matters here."""
    if cfg.model_id == MISTRAL_ID and cfg.p.system_prompt == "":
        return [RULE_9_ISSUE]
    return []


def test_rule_9_acceptance_case() -> None:
    cfg = _record(system_prompt="")
    issues = _stub_validator(cfg)
    assert issues == [RULE_9_ISSUE]

    widget = FakeWidget()
    highlighted = apply_issue_props({"p.system_prompt": widget}, issues)
    state = evaluate(cfg, issues)

    assert widget.added == [_error_prop(RULE_9_MESSAGE)]
    assert highlighted == ["p.system_prompt"]
    assert panel_lines(issues) == [
        ("error", f"p.system_prompt: {RULE_9_MESSAGE}")
    ]
    assert state.errors == (RULE_9_ISSUE,)
    assert state.warnings == ()
    assert state.save_enabled is False
    assert state.payload["p"]["system_prompt"] == ""


def test_clearing_errors_removes_props_and_enables_save() -> None:
    cfg = _record(system_prompt="")
    widget = FakeWidget()
    widgets = {"p.system_prompt": widget}

    apply_issue_props(widgets, [RULE_9_ISSUE])
    assert widget.added == [_error_prop(RULE_9_MESSAGE)]
    assert evaluate(cfg, [RULE_9_ISSUE]).save_enabled is False

    highlighted = apply_issue_props(widgets, [])
    state = evaluate(cfg, [])

    assert highlighted == []
    assert CLEAR_PROP in widget.removed
    assert len(widget.added) == 1
    assert state.errors == ()
    assert state.save_enabled is True


def test_warning_does_not_block_or_highlight() -> None:
    cfg = _record(DEEPSEEK_ID, "inference", system_prompt="Give a verdict.")
    warning: ValidationIssue = {
        "field": "p.system_prompt",
        "level": "warning",
        "msg": (
            "DeepSeek concatenates the system prompt right after BOS with "
            "no separator; prefer putting the instruction in the user prompt."
        ),
    }
    widget = FakeWidget()
    highlighted = apply_issue_props({"p.system_prompt": widget}, [warning])
    state = evaluate(cfg, [warning])

    assert highlighted == []
    assert widget.added == []
    assert widget.removed == [CLEAR_PROP]
    assert state.errors == ()
    assert state.warnings == (warning,)
    assert state.save_enabled is True
    assert panel_lines([warning]) == [
        ("warning", f"p.system_prompt: {warning['msg']}")
    ]


def test_json_preview_round_trips_to_the_payload() -> None:
    cfg = _record()
    state = evaluate(cfg, [])
    expected = cfg.model_dump(mode="json")

    assert state.payload == expected
    assert json.loads(state.json_text) == expected
    assert state.json_text == json.dumps(
        expected, indent=2, ensure_ascii=False
    )
    assert '"config_id"' in state.json_text
    assert json.loads(state.json_text)["config_id"] == cfg.config_id

    deepseek = evaluate(_record(DEEPSEEK_ID, "inference"), [])
    assert "\\u" not in deepseek.json_text


def test_unknown_field_paths_are_skipped() -> None:
    cfg = _record(system_prompt="")
    pad_widget = FakeWidget()
    issues = [
        {"field": "does.not.exist", "level": "error", "msg": "Unknown path."},
        {
            "field": "b.pad_token",
            "level": "error",
            "msg": "Mistral has no such token.",
        },
    ]

    highlighted = apply_issue_props({"b.pad_token": pad_widget}, issues)
    state = evaluate(cfg, issues)

    assert highlighted == ["b.pad_token"]
    assert pad_widget.added == [_error_prop("Mistral has no such token.")]
    assert len(state.errors) == 2
    assert state.save_enabled is False


def test_empty_field_widgets_are_tolerated() -> None:
    clear_error_props({})
    assert apply_issue_props({}, [RULE_9_ISSUE]) == []
    assert apply_issue_props({}, []) == []


def test_double_quotes_are_replaced_inside_the_prop() -> None:
    message = (
        'Mistral does not have the token "<myPadToken>"; use "<pad>" or '
        '"</s>".'
    )
    widget = FakeWidget()
    apply_issue_props(
        {"b.pad_token": widget},
        [{"field": "b.pad_token", "level": "error", "msg": message}],
    )

    prop = widget.added[-1]
    escaped = (
        "Mistral does not have the token '<myPadToken>'; use '<pad>' or "
        "'</s>'."
    )
    assert prop == _error_prop(escaped)
    assert prop.count('"') == 2
    assert '"' not in prop[prop.index('"') + 1 : prop.rindex('"')]


def test_reapplying_issues_keeps_only_the_latest_message() -> None:
    widget = FakeWidget()
    widgets = {"b.pad_token": widget}
    first = {"field": "b.pad_token", "level": "error", "msg": "First problem."}
    second = {
        "field": "b.pad_token",
        "level": "error",
        "msg": "Second problem.",
    }

    apply_issue_props(widgets, [first])
    apply_issue_props(widgets, [first])
    assert widget.added == [
        _error_prop("First problem."),
        _error_prop("First problem."),
    ]

    apply_issue_props(widgets, [second])
    assert widget.added[-1] == _error_prop("Second problem.")
    assert widget.removed == [CLEAR_PROP, CLEAR_PROP, CLEAR_PROP]


def test_console_state_is_frozen() -> None:
    state = evaluate(_record(), [])
    try:
        state.save_enabled = True  # type: ignore[misc]
    except FrozenInstanceError:
        return
    raise AssertionError("ConsoleState must be frozen")


def test_console_imports_without_nicegui() -> None:
    assert "nicegui" not in sys.modules
    assert console.ERROR_PROP == "error"
    assert console.SAVE_LABEL == "Save to DB"
    assert console.VALIDATE_LABEL == "Validate"
    assert console.RESET_LABEL == "Reset to Defaults"
    for name in (
        "ConsoleState",
        "ValidationConsole",
        "apply_issue_props",
        "build_console",
        "clear_error_props",
        "evaluate",
        "panel_lines",
    ):
        assert hasattr(console, name), name
    assert callable(console.build_console)


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:  # noqa: BLE001 - report every failure
                failures += 1
                print(f"FAIL {name}: {error}")
            else:
                print(f"ok   {name}")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
