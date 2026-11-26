"""Component h (**kwargs) dynamic key-value panels and whitelist validator.

GitHub issue: #12 (M1.0 milestone)
Owner: C (Daniel Laurin)

Scope:
- Four sub-panels: load_model, load_tokenizer, encode, generate.
- Dynamic row addition and deletion.
- Key names verified against preset library whitelists.
- Field-level error highlighting on unknown keys.
- Real-time serialization into the ConfigRecord JSON preview.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from app.validation import (
    KWARGS_WHITELISTS,
    parse_kwarg_value,
    validate_kwargs_section,
)


class KwargsSubPanel:
    """Manages dynamic key-value rows for one stage (e.g. 'generate')."""

    def __init__(
        self,
        stage: str,
        h_state: dict[str, dict[str, Any]],
        field_widgets: dict[str, Any],
        on_change_callback: Callable[[], None],
    ) -> None:
        self.stage = stage
        self.h_state = h_state
        self.field_widgets = field_widgets
        self.on_change = on_change_callback
        self.allowed = KWARGS_WHITELISTS.get(stage, frozenset())
        self.rows_container: Any = None
        self.row_records: list[dict[str, Any]] = []

    def build_ui(self) -> None:
        """Construct the NiceGUI sub-panel with proper padding and readable font size."""
        from nicegui import ui

        with (
            ui.card()
            .props("flat")
            .classes("w-full p-4 gap-3 rounded-lg surface-subtle")
        ):
            with ui.row().classes("w-full items-center justify-between no-wrap"):
                with ui.row().classes("items-center gap-2.5 no-wrap"):
                    ui.icon("data_object", color="primary").classes("text-lg")
                    ui.label(self.stage).classes("font-semibold text-sm")
                    ui.label(f"({len(self.allowed)} allowed)").classes(
                        "text-sm text-secondary-token"
                    )
                ui.button(
                    "Add Argument",
                    icon="add",
                    on_click=lambda: self.add_row(),
                ).props("flat dense no-caps color=primary").classes(
                    "px-2.5 text-sm font-medium"
                )

            self.rows_container = ui.column().classes("w-full gap-2.5")

            existing = self.h_state.get(self.stage, {})
            if existing:
                for k, v in existing.items():
                    val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    self.add_row(initial_key=k, initial_val=val_str)

    def add_row(self, initial_key: str = "", initial_val: str = "") -> None:
        """Add a dynamic key-value row with top-aligned inputs and delete button."""
        from nicegui import ui

        if self.stage not in self.h_state:
            self.h_state[self.stage] = {}

        record: dict[str, Any] = {"current_key": initial_key}

        with self.rows_container:
            row_el = ui.row().classes("w-full items-start gap-2.5 no-wrap pt-1")
            record["element"] = row_el

            with row_el:
                key_input = (
                    ui.input(
                        placeholder="Argument name (e.g. revision)",
                        value=initial_key,
                    )
                    .props("outlined dense hide-bottom-space")
                    .classes("flex-1 min-w-0")
                )
                val_input = (
                    ui.input(
                        placeholder="Value",
                        value=initial_val,
                    )
                    .props("outlined dense hide-bottom-space")
                    .classes("flex-1 min-w-0")
                )

                record["key_widget"] = key_input
                record["val_widget"] = val_input

                def handle_key_change(e: Any) -> None:
                    raw_key = (e.value or "").strip()
                    old_key = record.get("current_key", "")

                    if old_key and old_key in self.h_state[self.stage]:
                        del self.h_state[self.stage][old_key]
                    if old_key:
                        self.field_widgets.pop(f"h.{self.stage}.{old_key}", None)

                    record["current_key"] = raw_key
                    if raw_key:
                        path = f"h.{self.stage}.{raw_key}"
                        self.field_widgets[path] = key_input

                        if raw_key not in self.allowed:
                            key_input.props('error error-message="Unknown argument"')
                        else:
                            key_input.props(remove="error error-message")

                        parsed_val = parse_kwarg_value(val_input.value or "")
                        self.h_state[self.stage][raw_key] = parsed_val

                    self.on_change()

                def handle_val_change(e: Any) -> None:
                    cur_key = record.get("current_key", "").strip()
                    if cur_key:
                        self.h_state[self.stage][cur_key] = parse_kwarg_value(
                            e.value or ""
                        )
                    self.on_change()

                def handle_delete() -> None:
                    cur_key = record.get("current_key", "").strip()
                    if cur_key:
                        self.h_state[self.stage].pop(cur_key, None)
                        self.field_widgets.pop(f"h.{self.stage}.{cur_key}", None)
                    row_el.delete()
                    if record in self.row_records:
                        self.row_records.remove(record)
                    self.on_change()

                key_input.on_value_change(handle_key_change)
                val_input.on_value_change(handle_val_change)

                ui.button(
                    icon="delete_outline",
                    on_click=handle_delete,
                ).props("flat round dense size=sm color=negative").classes(
                    "mt-0.5 shrink-0"
                )

        self.row_records.append(record)

        if initial_key:
            path = f"h.{self.stage}.{initial_key}"
            self.field_widgets[path] = key_input
            self.h_state[self.stage][initial_key] = parse_kwarg_value(initial_val)

        self.on_change()

    def clear(self) -> None:
        """Clear all rows from this sub-panel."""
        for rec in list(self.row_records):
            cur_key = rec.get("current_key", "")
            if cur_key:
                self.field_widgets.pop(f"h.{self.stage}.{cur_key}", None)
        self.row_records.clear()
        if self.stage in self.h_state:
            self.h_state[self.stage] = {}
        if self.rows_container is not None:
            self.rows_container.clear()


__all__ = [
    "KWARGS_WHITELISTS",
    "KwargsSubPanel",
    "parse_kwarg_value",
    "validate_kwargs_section",
]
