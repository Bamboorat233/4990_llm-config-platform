"""Milestone 2.5 Script Export UI panel (Issue #60).

Provides pure frozen verification/generation snapshot helpers and the lazy
NiceGUI :class:`GeneratorPanel` mounted as the **Script Export** tab in
``app/main.py``.

NiceGUI is imported lazily inside :class:`GeneratorPanel` so this module can be
imported and unit-tested without starting or loading NiceGUI.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.codegen import ConfigValidationError, GenerationResult, generate_bundle
from app.schemas import ConfigRecord, ValidationIssue
from app.store import get_config, list_configs

ACTIVE_FORM_SOURCE = "__active__"


@dataclass(frozen=True)
class VerificationBadgeView:
    """Presentation model for one verification pill badge."""

    key: str
    label: str
    passed: bool
    tone: str  # "success" | "error"
    detail: str | None = None


@dataclass(frozen=True)
class GeneratorStateView:
    """Pure presentation snapshot for a single script-generation attempt."""

    blocked: bool
    result: GenerationResult | None = None
    badges: list[VerificationBadgeView] = field(default_factory=list)
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    source_label: str = ""
    config_id: str | None = None
    error_message: str | None = None


def build_verification_badges(
    result: GenerationResult,
) -> list[VerificationBadgeView]:
    """Build the four verification badges from a :class:`GenerationResult`."""
    syntax_ok = bool(result.get("syntax_ok"))
    syntax_error = result.get("syntax_error")
    syntax_badge = VerificationBadgeView(
        key="syntax",
        label="Syntax OK" if syntax_ok else "Syntax Error",
        passed=syntax_ok,
        tone="success" if syntax_ok else "error",
        detail=None if syntax_ok else str(syntax_error or "Syntax check failed."),
    )

    parity_diffs = list(result.get("parity_diffs") or [])
    parity_ok = bool(result.get("parity_ok")) and len(parity_diffs) == 0
    diff_count = len(parity_diffs)
    diff_noun = "diff" if diff_count == 1 else "diffs"
    parity_detail = (
        None
        if parity_ok
        else "; ".join(
            f"{d.get('field', '?')}: {d.get('reason', 'mismatch')}"
            for d in parity_diffs
        )
    )
    parity_badge = VerificationBadgeView(
        key="parity",
        label=(
            "Parameter Parity (0 diffs)"
            if parity_ok
            else f"Parameter Parity ({diff_count} {diff_noun})"
        ),
        passed=parity_ok,
        tone="success" if parity_ok else "error",
        detail=parity_detail,
    )

    pitfall_issues = list(result.get("pitfall_issues") or [])
    pitfalls_ok = len(pitfall_issues) == 0
    pitfall_count = len(pitfall_issues)
    pitfall_noun = "issue" if pitfall_count == 1 else "issues"
    pitfall_badge = VerificationBadgeView(
        key="pitfalls",
        label=(
            "6/6 Pitfalls Fixed"
            if pitfalls_ok
            else f"Pitfalls ({pitfall_count} {pitfall_noun})"
        ),
        passed=pitfalls_ok,
        tone="success" if pitfalls_ok else "error",
        detail=None if pitfalls_ok else "; ".join(str(item) for item in pitfall_issues),
    )

    marker_issues = list(result.get("marker_issues") or [])
    markers_ok = len(marker_issues) == 0
    marker_count = len(marker_issues)
    marker_noun = "marker issue" if marker_count == 1 else "marker issues"
    marker_badge = VerificationBadgeView(
        key="markers",
        label=(
            "Parser-Compatible"
            if markers_ok
            else f"Parser-Compatible ({marker_count} {marker_noun})"
        ),
        passed=markers_ok,
        tone="success" if markers_ok else "error",
        detail=None if markers_ok else "; ".join(str(item) for item in marker_issues),
    )

    return [syntax_badge, parity_badge, pitfall_badge, marker_badge]


def evaluate_generation(
    cfg: ConfigRecord,
    *,
    parent_cfg: ConfigRecord | None = None,
    source_label: str = "",
) -> GeneratorStateView:
    """Run ``generate_bundle(cfg)`` and return an immutable :class:`GeneratorStateView`.

    Pre-flight :class:`ConfigValidationError` issues are captured into ``errors``
    with ``blocked=True`` so the UI can display an explicit error banner.
    Non-blocking warnings are preserved in ``warnings``.
    """
    try:
        result = generate_bundle(cfg, parent_cfg=parent_cfg)
    except ConfigValidationError as exc:
        return GeneratorStateView(
            blocked=True,
            result=None,
            badges=[],
            errors=list(exc.issues),
            warnings=[],
            source_label=source_label,
            config_id=cfg.config_id,
            error_message=str(exc),
        )

    return GeneratorStateView(
        blocked=False,
        result=result,
        badges=build_verification_badges(result),
        errors=[],
        warnings=[str(w) for w in (result.get("warnings") or [])],
        source_label=source_label,
        config_id=result.get("config_id"),
        error_message=None,
    )


class GeneratorPanel:
    """NiceGUI Script Export tab controller: pre-flight gate, badges, code preview, and export."""

    def __init__(
        self,
        container: Any,
        *,
        active_config_fn: Callable[[], ConfigRecord] | None = None,
        list_configs_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        get_config_fn: Callable[[str], ConfigRecord | None] | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        from nicegui import ui

        self._ui = ui
        self.container = container
        self._db_path = db_path
        self._active_config_fn = active_config_fn

        if list_configs_fn is not None:
            self._list_configs_fn = list_configs_fn
        else:
            self._list_configs_fn = lambda: list_configs(db_path)

        if get_config_fn is not None:
            self._get_config_fn = get_config_fn
        else:
            self._get_config_fn = lambda cid: get_config(cid, db_path)

        self.selected_source: str = (
            ACTIVE_FORM_SOURCE if self._active_config_fn is not None else ""
        )
        self._suppress_source_change: bool = False
        self.current_state: GeneratorStateView | None = None
        self.badge_widgets: list[Any] = []

        options = self._build_source_options()
        if not self.selected_source and options:
            self.selected_source = next(iter(options.keys()))

        with self.container:
            with ui.column().classes("w-full gap-3"):
                # 1. Config Source Selector & Action Toolbar
                with ui.column().classes(
                    "w-full p-4 gap-3 rounded-md border-token surface-card"
                ):
                    with ui.row().classes(
                        "w-full items-center justify-between gap-2 flex-wrap"
                    ):
                        ui.label("Script Export").classes(
                            "text-base font-bold tracking-tight"
                        )
                        self.filename_label = ui.label("").classes(
                            "mono text-xs font-medium text-secondary-token"
                        )

                    with ui.row().classes("w-full items-end gap-2 flex-wrap"):
                        with ui.column().classes("flex-grow min-w-[200px] gap-1"):
                            ui.label("Configuration Source").classes(
                                "text-xs font-medium text-secondary-token select-none"
                            )
                            self.source_select = (
                                ui.select(
                                    options=options,
                                    value=self.selected_source or None,
                                    on_change=self._on_source_changed,
                                )
                                .props("outlined dense")
                                .classes("w-full")
                            )

                        with ui.row().classes("items-center gap-2 no-wrap shrink-0"):
                            self.generate_button = (
                                ui.button(
                                    "Generate",
                                    icon="play_arrow",
                                    on_click=self.generate,
                                )
                                .props("unelevated dense no-caps size=sm color=primary")
                                .classes("px-3 text-xs font-medium")
                            )
                            self.copy_button = (
                                ui.button(
                                    "Copy",
                                    icon="content_copy",
                                    on_click=self.copy_active_code,
                                )
                                .props("outlined dense no-caps size=sm color=primary")
                                .classes("px-2.5 text-xs font-medium")
                            )
                            self.download_button = (
                                ui.button(
                                    "Download .py",
                                    icon="download",
                                    on_click=self.download_script,
                                )
                                .props("outlined dense no-caps size=sm color=primary")
                                .classes("px-2.5 text-xs font-medium")
                            )

                # 2. Pre-Flight Status Banners (errors block generation; warnings do not)
                self.error_banner = ui.column().classes(
                    "w-full p-3.5 gap-1.5 diagnostic-box diagnostic-box-error"
                )
                with self.error_banner:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("error_outline", color="negative").classes(
                            "text-base shrink-0"
                        )
                        ui.label("Generation Blocked — Pre-Flight Validation Failed").classes(
                            "text-xs font-bold tracking-tight text-red-600 dark:text-red-400"
                        )
                    self.error_list_container = ui.column().classes("w-full gap-1")
                self.error_banner.set_visibility(False)

                self.warning_banner = ui.column().classes(
                    "w-full p-3.5 gap-1.5 diagnostic-box diagnostic-box-warning"
                )
                with self.warning_banner:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("warning_amber", color="warning").classes(
                            "text-base shrink-0"
                        )
                        ui.label("Pre-Flight Warnings").classes(
                            "text-xs font-bold tracking-tight text-amber-600 dark:text-amber-400"
                        )
                    self.warning_list_container = ui.column().classes("w-full gap-1")
                self.warning_banner.set_visibility(False)

                # 3. Verification Badges & Code Viewer Card
                self.result_section = ui.column().classes("w-full gap-3")
                with self.result_section:
                    # 3a. Verification Badge Strip
                    with ui.column().classes(
                        "w-full p-3.5 gap-2 rounded-md border-token surface-card"
                    ):
                        with ui.row().classes(
                            "w-full items-center justify-between gap-2 flex-wrap"
                        ):
                            ui.label("Verification Checks").classes(
                                "text-xs font-semibold uppercase tracking-wider text-secondary-token"
                            )
                            self.sha_label = ui.label("").classes(
                                "mono text-[11px] text-secondary-token"
                            )
                        self.badges_container = ui.row().classes(
                            "w-full items-center gap-2 flex-wrap"
                        )
                        self.badge_details_container = ui.column().classes(
                            "w-full gap-1"
                        )
                        self.badge_details_container.set_visibility(False)

                    # 3b. Code Viewer Tabs
                    with ui.column().classes(
                        "w-full gap-0 rounded-md border-token overflow-hidden surface-card"
                    ):
                        with (
                            ui.tabs(value="script")
                            .props(
                                "dense no-caps align=left active-color=primary indicator-color=primary"
                            )
                            .classes(
                                "w-full px-2 pt-1 surface-subtle border-b border-token"
                            ) as self.code_tabs
                        ):
                            ui.tab(
                                "script", label="Python Script (.py)", icon="code"
                            ).classes("text-xs font-medium")
                            ui.tab(
                                "requirements",
                                label="requirements.txt",
                                icon="list_alt",
                            ).classes("text-xs font-medium")
                            ui.tab(
                                "config", label="config.json", icon="data_object"
                            ).classes("text-xs font-medium")

                        with (
                            ui.tab_panels(self.code_tabs, value="script")
                            .props("animated")
                            .classes("w-full bg-transparent")
                        ):
                            with ui.tab_panel("script").classes("p-3"):
                                self.script_code_view = ui.code(
                                    "", language="python"
                                ).classes(
                                    "w-full text-xs rounded-md border-token max-h-[520px] overflow-y-auto overflow-x-auto"
                                )
                            with ui.tab_panel("requirements").classes("p-3"):
                                self.requirements_code_view = ui.code(
                                    "", language="plaintext"
                                ).classes(
                                    "w-full text-xs rounded-md border-token max-h-[520px] overflow-y-auto overflow-x-auto"
                                )
                            with ui.tab_panel("config").classes("p-3"):
                                self.config_json_view = ui.code(
                                    "", language="json"
                                ).classes(
                                    "json-viewer w-full text-xs rounded-md border-token max-h-[520px] overflow-y-auto overflow-x-auto"
                                )

                self.result_section.set_visibility(False)

        if self.selected_source:
            self.generate()
        else:
            self._set_action_buttons_enabled(False)

    @property
    def current_result(self) -> GenerationResult | None:
        """Return the latest :class:`GenerationResult`, or ``None`` if blocked."""
        return self.current_state.result if self.current_state is not None else None

    def _safe_notify(self, message: str, *, type: str = "info") -> None:
        try:
            self._ui.notify(message, type=type, position="top")
        except RuntimeError:
            pass

    def _build_source_options(self) -> dict[str, str]:
        options: dict[str, str] = {}
        if self._active_config_fn is not None:
            options[ACTIVE_FORM_SOURCE] = "Active Form"
        try:
            rows = self._list_configs_fn()
        except Exception:
            rows = []
        for row in rows:
            cid = str(row.get("config_id") or "")
            if not cid:
                continue
            model_short = str(row.get("model_id") or "").split("/")[-1]
            task = str(row.get("task_type") or "")
            options[cid] = f"{cid[:8]} · {model_short} ({task})"
        return options

    def refresh_config_options(self) -> None:
        """Refresh the config source selector options from SQLite."""
        options = self._build_source_options()
        self.source_select.options = options
        if self.selected_source not in options:
            if ACTIVE_FORM_SOURCE in options:
                self.selected_source = ACTIVE_FORM_SOURCE
            elif options:
                self.selected_source = next(iter(options.keys()))
            else:
                self.selected_source = ""
            self._suppress_source_change = True
            try:
                self.source_select.value = self.selected_source or None
            finally:
                self._suppress_source_change = False
        self.source_select.update()

    def _on_source_changed(self, e: Any) -> None:
        if self._suppress_source_change:
            return
        val = e.value if hasattr(e, "value") else self.source_select.value
        self.selected_source = str(val or "")
        self.generate()

    def select_source(self, source_id: str) -> GeneratorStateView:
        """Select a config source (``ACTIVE_FORM_SOURCE`` or saved ``config_id``) and generate."""
        self.selected_source = source_id
        self._suppress_source_change = True
        try:
            self.source_select.value = source_id or None
        finally:
            self._suppress_source_change = False
        return self.generate()

    def _set_action_buttons_enabled(self, enabled: bool) -> None:
        for btn in (self.copy_button, self.download_button):
            if enabled:
                btn.enable()
            else:
                btn.disable()

    def _resolve_selected_config(self) -> tuple[ConfigRecord, str]:
        if self.selected_source == ACTIVE_FORM_SOURCE:
            if self._active_config_fn is None:
                raise ValueError("Active form config provider is not configured.")
            return self._active_config_fn(), "Active Form"

        if not self.selected_source:
            raise ValueError("No configuration source selected.")

        cfg = self._get_config_fn(self.selected_source)
        if cfg is None:
            raise ValueError(
                f"Saved configuration '{self.selected_source}' was not found."
            )
        return cfg, f"Saved Config ({self.selected_source[:8]})"

    def generate(self) -> GeneratorStateView:
        """Evaluate the selected config and update banners, badges, and code viewers."""
        try:
            cfg, source_label = self._resolve_selected_config()
            state = evaluate_generation(cfg, source_label=source_label)
        except Exception as exc:
            state = GeneratorStateView(
                blocked=True,
                result=None,
                badges=[],
                errors=[{"field": "config", "level": "error", "msg": str(exc)}],
                warnings=[],
                source_label=self.selected_source or "",
                config_id=None,
                error_message=str(exc),
            )

        self.current_state = state
        self._render_state(state)
        return state

    def _render_state(self, state: GeneratorStateView) -> None:
        ui = self._ui

        # 1. Error Banner
        self.error_list_container.clear()
        if state.errors:
            with self.error_list_container:
                for issue in state.errors:
                    field_name = issue.get("field", "config")
                    msg = issue.get("msg", "Invalid configuration.")
                    ui.label(f"{field_name}: {msg}").classes(
                        "text-xs text-red-600 dark:text-red-400 font-mono break-words"
                    )
            self.error_banner.set_visibility(True)
        else:
            self.error_banner.set_visibility(False)

        # 2. Warning Banner
        self.warning_list_container.clear()
        if state.warnings:
            with self.warning_list_container:
                for warn in state.warnings:
                    ui.label(warn).classes(
                        "text-xs text-amber-600 dark:text-amber-400 font-medium break-words"
                    )
            self.warning_banner.set_visibility(True)
        else:
            self.warning_banner.set_visibility(False)

        # 3. Result Section (Badges + Code Tabs)
        self.badges_container.clear()
        self.badge_details_container.clear()
        self.badge_widgets = []

        if state.blocked or state.result is None:
            self.filename_label.text = "Generation blocked"
            self.sha_label.text = ""
            self.script_code_view.content = ""
            self.requirements_code_view.content = ""
            self.config_json_view.content = ""
            self.result_section.set_visibility(False)
            self._set_action_buttons_enabled(False)
            return

        result = state.result
        self.filename_label.text = result["script_filename"]
        self.sha_label.text = f"sha256:{result['script_sha256'][:12]}"

        failed_details: list[str] = []
        with self.badges_container:
            for badge in state.badges:
                dot_cls = (
                    "status-dot dot-success"
                    if badge.passed
                    else "status-dot dot-error"
                )
                tag_cls = (
                    "info-tag tag-success" if badge.passed else "info-tag tag-error"
                )
                with ui.row().classes(
                    "items-center gap-1.5 px-2.5 py-1 rounded-full surface-subtle border-token no-wrap"
                ) as pill:
                    ui.element("span").classes(dot_cls)
                    lbl = ui.label(badge.label).classes(tag_cls)
                    if badge.detail:
                        pill.tooltip(badge.detail)
                        failed_details.append(f"{badge.label}: {badge.detail}")
                    self.badge_widgets.append(lbl)

        if failed_details:
            with self.badge_details_container:
                for detail in failed_details:
                    ui.label(detail).classes(
                        "text-xs text-red-600 dark:text-red-400 font-mono break-words"
                    )
            self.badge_details_container.set_visibility(True)
        else:
            self.badge_details_container.set_visibility(False)

        self.script_code_view.content = result["script_code"]
        self.requirements_code_view.content = result["requirements_txt"]
        self.config_json_view.content = result["config_json"]
        self.result_section.set_visibility(True)
        self._set_action_buttons_enabled(True)

    def get_active_tab_text(self) -> str:
        """Return the text displayed in the currently active code viewer tab."""
        result = self.current_result
        if result is None:
            return ""
        active = str(getattr(self.code_tabs, "value", "script") or "script")
        if active == "requirements":
            return result["requirements_txt"]
        if active == "config":
            return result["config_json"]
        return result["script_code"]

    def copy_active_code(self) -> str:
        """Copy the active code tab's text to the browser clipboard and return it."""
        text = self.get_active_tab_text()
        if not text:
            self._safe_notify("Nothing to copy.", type="warning")
            return ""
        ui = self._ui
        if getattr(ui.context.client, "has_socket_connection", False):
            ui.run_javascript(f"navigator.clipboard.writeText({json.dumps(text)})")
        active = str(getattr(self.code_tabs, "value", "script") or "script")
        self._safe_notify(f"Copied {active} content to clipboard.", type="positive")
        return text

    def download_script(self) -> bool:
        """Trigger browser download of the generated Python script."""
        result = self.current_result
        if result is None:
            self._safe_notify("Generate a valid script first.", type="warning")
            return False
        self._ui.download(
            result["script_code"].encode("utf-8"),
            result["script_filename"],
        )
        return True


def build_generator_panel(
    container: Any,
    *,
    active_config_fn: Callable[[], ConfigRecord] | None = None,
    list_configs_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    get_config_fn: Callable[[str], ConfigRecord | None] | None = None,
    db_path: str | Path | None = None,
) -> GeneratorPanel:
    """Mount the Script Export panel inside ``container``."""
    return GeneratorPanel(
        container=container,
        active_config_fn=active_config_fn,
        list_configs_fn=list_configs_fn,
        get_config_fn=get_config_fn,
        db_path=db_path,
    )


__all__ = [
    "ACTIVE_FORM_SOURCE",
    "GeneratorPanel",
    "GeneratorStateView",
    "VerificationBadgeView",
    "build_generator_panel",
    "build_verification_badges",
    "evaluate_generation",
]
