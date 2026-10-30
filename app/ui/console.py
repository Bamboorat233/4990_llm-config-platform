"""Validation console and live JSON preview for the NiceGUI form (issue #8).

The console is the sticky right column described in
``design/Spec_and_roadmap.md`` section 3.4: a validation status panel, a
live JSON preview rendered with ``ui.code(language="json")`` and the
``Validate`` / ``Save to DB`` / ``Reset to Defaults`` button group.

The module keeps its logic separate from NiceGUI so that it can be
imported and tested without NiceGUI installed. Only
:class:`ValidationConsole` and :func:`build_console` touch the
``nicegui`` package, and they import it lazily inside the function body.

Public API:

- :data:`ERROR_PROP` -- the Quasar prop that marks an invalid widget
- :data:`SAVE_LABEL`, :data:`VALIDATE_LABEL`, :data:`RESET_LABEL` -- the
  three button texts
- :class:`ConsoleState` -- the snapshot returned by :func:`evaluate`
- :func:`evaluate` -- split issues, decide whether saving is allowed and
  render the JSON payload
- :func:`panel_lines` -- the status-panel rows as ``(level, text)`` pairs
- :func:`apply_issue_props` -- highlight the widgets whose field has an
  error and return the highlighted paths
- :func:`clear_error_props` -- remove every error prop
- :class:`ValidationConsole` / :func:`build_console` -- the right-column UI

The form hooks and the validator are injected by ``app/main.py`` (#6): the
console owns neither the rules (#4/#5) nor the form state. A refresh runs
``build_config()``, then ``validate(cfg)`` on the resulting record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from app.schemas import ConfigRecord, ValidationIssue

ERROR_PROP = "error"
SAVE_LABEL = "Save to DB"
VALIDATE_LABEL = "Validate"
RESET_LABEL = "Reset to Defaults"

_ACTIVE_PROPS = "error error-message"
_VALID_MESSAGE = "Configuration is valid."
_ERROR_CLASSES = "text-sm text-red-600"
_WARNING_CLASSES = "text-sm text-orange-600"
_VALID_CLASSES = "text-sm text-green-600"


@dataclass(frozen=True)
class ConsoleState:
    """One console refresh: the split issues and the rendered payload."""

    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    save_enabled: bool
    payload: dict[str, Any]
    json_text: str


def evaluate(
    cfg: ConfigRecord, issues: Sequence[ValidationIssue]
) -> ConsoleState:
    """Build the console snapshot for ``cfg`` and its ``issues``.

    Errors and warnings are separated by ``level``; any error disables
    saving. The payload is the JSON-mode dump of the record, rendered with
    ``indent=2`` and ``ensure_ascii=False`` so the preview keeps the
    original non-ASCII tokens readable.
    """
    payload = cfg.model_dump(mode="json")
    errors = tuple(issue for issue in issues if issue["level"] == "error")
    warnings = tuple(issue for issue in issues if issue["level"] == "warning")
    return ConsoleState(
        errors=errors,
        warnings=warnings,
        save_enabled=not errors,
        payload=payload,
        json_text=json.dumps(payload, indent=2, ensure_ascii=False),
    )


def panel_lines(issues: Sequence[ValidationIssue]) -> list[tuple[str, str]]:
    """Return ``(level, "path: message")`` rows for the status panel."""
    return [
        (issue["level"], f"{issue['field']}: {issue['msg']}")
        for issue in issues
    ]


def apply_issue_props(
    field_widgets: Mapping[str, Any], issues: Sequence[ValidationIssue]
) -> list[str]:
    """Highlight the widget of every field that has an error.

    Previous highlights are cleared first, so calling this twice never
    accumulates stale messages. Only ``error`` issues touch widgets;
    warnings stay in the panel. A field path the form does not render is
    skipped silently, because the validator may report paths that have no
    dedicated widget. Returns the highlighted paths in issue order.
    """
    clear_error_props(field_widgets)
    highlighted: list[str] = []
    for issue in issues:
        if issue["level"] != "error":
            continue
        widget = field_widgets.get(issue["field"])
        if widget is None:
            continue
        message = str(issue["msg"]).replace('"', "'")
        widget.props(f'{ERROR_PROP} error-message="{message}"')
        highlighted.append(issue["field"])
    return highlighted


def clear_error_props(field_widgets: Mapping[str, Any]) -> None:
    """Remove the error state from every registered widget."""
    for widget in field_widgets.values():
        if widget is None:
            continue
        widget.props(remove=_ACTIVE_PROPS)


class ValidationConsole:
    """The sticky right column: status panel, JSON preview and buttons.

    ``build_config`` returns the :class:`ConfigRecord` produced by the
    current form state, ``validate`` returns its issues and the optional
    ``save`` / ``reset`` hooks perform the button side effects. Every
    refresh rebuilds the record and re-runs the validator, so the console
    always reflects the live form.
    """

    def __init__(
        self,
        container: Any,
        field_widgets: Mapping[str, Any],
        *,
        build_config: Callable[[], ConfigRecord],
        validate: Callable[[ConfigRecord], Sequence[ValidationIssue]],
        save: Callable[[ConfigRecord], Any] | None = None,
        reset: Callable[[], Any] | None = None,
    ) -> None:
        from nicegui import ui

        self._field_widgets = field_widgets
        self._build_config = build_config
        self._validate = validate
        self._save = save
        self._reset = reset

        with container:
            with ui.column().classes("sticky top-0"):
                with ui.card().classes("w-full"):
                    ui.label("Validation").classes("text-subtitle2")
                    self._status = ui.column().classes("w-full gap-1")
                self._code = ui.code("", language="json").classes("w-full")
                with ui.row().classes("w-full gap-2"):
                    self._validate_button = ui.button(
                        VALIDATE_LABEL, on_click=self._on_validate
                    )
                    self._save_button = ui.button(
                        SAVE_LABEL, on_click=self._on_save
                    )
                    self._reset_button = ui.button(
                        RESET_LABEL, on_click=self._on_reset
                    )

        self.refresh()

    def refresh(self) -> ConsoleState:
        """Re-run validation and update the console from the live form."""
        cfg = self._build_config()
        issues = list(self._validate(cfg))
        state = evaluate(cfg, issues)
        apply_issue_props(self._field_widgets, issues)
        self._render_status(issues)
        self._code.content = state.json_text
        self._save_button.set_enabled(state.save_enabled)
        return state

    def _render_status(self, issues: Sequence[ValidationIssue]) -> None:
        from nicegui import ui

        self._status.clear()
        with self._status:
            if not issues:
                ui.label(_VALID_MESSAGE).classes(_VALID_CLASSES)
            for level, text in panel_lines(issues):
                classes = (
                    _ERROR_CLASSES if level == "error" else _WARNING_CLASSES
                )
                ui.label(text).classes(classes)

    def _on_validate(self) -> None:
        self.refresh()

    def _on_save(self) -> None:
        if self._save is None:
            return
        self.refresh()
        self._save(self._build_config())

    def _on_reset(self) -> None:
        if self._reset is not None:
            self._reset()
        self.refresh()


def build_console(
    container: Any,
    field_widgets: Mapping[str, Any],
    *,
    build_config: Callable[[], ConfigRecord],
    validate: Callable[[ConfigRecord], Sequence[ValidationIssue]],
    save: Callable[[ConfigRecord], Any] | None = None,
    reset: Callable[[], Any] | None = None,
) -> ValidationConsole:
    """Build the validation console inside ``container`` and return it."""
    return ValidationConsole(
        container,
        field_widgets,
        build_config=build_config,
        validate=validate,
        save=save,
        reset=reset,
    )


__all__ = [
    "ConsoleState",
    "ERROR_PROP",
    "RESET_LABEL",
    "SAVE_LABEL",
    "VALIDATE_LABEL",
    "ValidationConsole",
    "apply_issue_props",
    "build_console",
    "clear_error_props",
    "evaluate",
    "panel_lines",
]
