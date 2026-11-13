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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.schemas import ConfigRecord, ValidationIssue

ERROR_PROP = "error"
SAVE_LABEL = "Save to DB"
VALIDATE_LABEL = "Validate"
RESET_LABEL = "Reset to Defaults"

_ACTIVE_PROPS = "error error-message"
_VALID_MESSAGE = "Configuration is valid."
_ERROR_CLASSES = "text-xs text-red-600 dark:text-red-400 font-mono"
_WARNING_CLASSES = "text-xs text-amber-600 dark:text-amber-400 font-mono"
_VALID_CLASSES = "text-xs text-emerald-600 dark:text-emerald-400 font-medium"


@dataclass(frozen=True)
class ConsoleState:
    """One console refresh: the split issues and the rendered payload."""

    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    save_enabled: bool
    payload: dict[str, Any]
    json_text: str


def evaluate(cfg: ConfigRecord, issues: Sequence[ValidationIssue]) -> ConsoleState:
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
    return [(issue["level"], f"{issue['field']}: {issue['msg']}") for issue in issues]


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
    ``save`` / ``reset`` / ``on_navigate`` hooks perform the button and
    issue-navigation side effects. Every refresh rebuilds the record and
    re-runs the validator, so the console always reflects the live form.
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
        on_navigate: Callable[[str], Any] | None = None,
    ) -> None:
        from nicegui import ui

        self._field_widgets = field_widgets
        self._build_config = build_config
        self._validate = validate
        self._save = save
        self._reset = reset
        self._on_navigate = on_navigate

        with container:
            with ui.column().classes("w-full gap-3"):
                # Action toolbar
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 no-wrap"
                ):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        self._validate_button = (
                            ui.button(
                                VALIDATE_LABEL,
                                icon="fact_check",
                                on_click=self._on_validate,
                            )
                            .props("flat dense no-caps color=primary")
                            .classes("px-2.5 text-xs font-medium")
                        )
                        self._reset_button = (
                            ui.button(
                                RESET_LABEL,
                                icon="restart_alt",
                                on_click=self._on_reset,
                            )
                            .props("flat dense no-caps color=secondary")
                            .classes("px-2.5 text-xs font-medium")
                        )
                    self._save_button = (
                        ui.button(SAVE_LABEL, icon="save", on_click=self._on_save)
                        .props("unelevated dense no-caps color=primary")
                        .classes("px-3.5 text-xs font-medium")
                    )

                # Validation status box
                with ui.element("div").classes(
                    "w-full px-3 py-2.5 rounded-md surface-subtle"
                ):
                    self._status = ui.column().classes("w-full gap-1.5")

                # Scrollable JSON Output Viewer (no redundant heading)
                self._code = ui.code("", language="json").classes(
                    "json-viewer w-full text-xs rounded-md border-token max-h-[560px] overflow-y-auto"
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
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.element("span").classes(
                        "w-2 h-2 rounded-full bg-emerald-500 shrink-0"
                    )
                    ui.label(_VALID_MESSAGE).classes(_VALID_CLASSES)
            for issue in issues:
                level = issue["level"]
                field_path = issue["field"]
                text = f"{field_path}: {issue['msg']}"
                dot_cls = (
                    "w-2 h-2 rounded-full bg-red-500 shrink-0 mt-1"
                    if level == "error"
                    else "w-2 h-2 rounded-full bg-amber-500 shrink-0 mt-1"
                )
                text_cls = _ERROR_CLASSES if level == "error" else _WARNING_CLASSES
                row = ui.row().classes(
                    "w-full items-start gap-2 no-wrap cursor-pointer hover:opacity-80"
                )
                if self._on_navigate is not None:
                    row.on("click", lambda _, fp=field_path: self._on_navigate(fp))
                with row:
                    ui.element("span").classes(dot_cls)
                    ui.label(text).classes(f"{text_cls} leading-relaxed")
                    if self._on_navigate is not None:
                        ui.icon("open_in_new").classes(
                            "text-xs opacity-60 mt-0.5 shrink-0"
                        )

    def _on_validate(self) -> None:
        state = self.refresh()
        # Jump directly to the first offending field in the GUI when Validate is clicked
        if self._on_navigate is not None:
            if state.errors:
                self._on_navigate(state.errors[0]["field"])
            elif state.warnings:
                self._on_navigate(state.warnings[0]["field"])

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
    on_navigate: Callable[[str], Any] | None = None,
) -> ValidationConsole:
    """Build the validation console inside ``container`` and return it."""
    return ValidationConsole(
        container,
        field_widgets,
        build_config=build_config,
        validate=validate,
        save=save,
        reset=reset,
        on_navigate=on_navigate,
    )


__all__ = [
    "ERROR_PROP",
    "RESET_LABEL",
    "SAVE_LABEL",
    "VALIDATE_LABEL",
    "ConsoleState",
    "ValidationConsole",
    "apply_issue_props",
    "build_console",
    "clear_error_props",
    "evaluate",
    "panel_lines",
]
