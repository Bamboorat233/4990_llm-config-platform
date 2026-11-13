"""Milestone 1.5 Run Results UI panel: log file upload/preset selector, async parse, and summary card.

Implements GitHub Issue (Owner: C / @daniellaurin):
- Scope:
  1. ``app/ui/results.py`` mounted as a tab in ``app/main.py``.
  2. Upload or preset selector (with multi-file upload history inside the upload dropdown,
     per-file Remove button, and Up/Down error & log cycling), config selector,
     ``run.io_bound`` parse with a spinner.
  3. Summary card with one row per phase and the first error/warning.
- Acceptance criteria:
  - Loading each of logs 01-15 shows the ``family``, ``status``, and ``phase`` outcomes
    in spec section 3.1 without opening the file.
  - Log 12 (``s12_fine_tune_success_warning.log``) shows a yellow warning, not a red error.

NiceGUI is imported lazily inside :class:`ResultsPanel` so the pure summary
formatting and severity rules can be unit-tested without starting a web server.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.parsing import ParseResult, parse_log_file

FIXTURE_LOGS_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "logs"

FAMILY_LABELS: dict[str, str] = {
    "fine_tune": "Fine-Tune",
    "inference": "Inference",
    "checkpoint_export": "Checkpoint Export",
    "unknown": "Unknown",
}

PHASE_LABELS: dict[str, str] = {
    "setup": "Setup & Model Load",
    "fine_tuning": "Fine-Tuning Loop",
    "checkpoint_creation": "Checkpoint Creation",
    "model_saving": "Final Model Export",
    "data_preparation": "Data Preparation",
    "generation": "Token Generation",
    "verdict_output": "Verdict Output",
}

OUTCOME_LABELS: dict[str, str] = {
    "succeeded": "Succeeded",
    "failed": "Failed",
    "terminated": "Terminated",
    "not_reached": "Not Reached",
}

# Initial fallback ParseResult before a log is loaded (or when initialized without a path)
DEFAULT_PARSE_RESULT: ParseResult = {
    "family": "fine_tune",
    "status": "success",
    "scenario": "fine_tune_success",
    "script": "fineTuning_1_11.py",
    "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
    "output_format": None,
    "failed_phase": None,
    "phases": [
        {"name": "setup", "outcome": "succeeded", "evidence_line": 124},
        {"name": "fine_tuning", "outcome": "succeeded", "evidence_line": 225},
        {"name": "checkpoint_creation", "outcome": "succeeded", "evidence_line": 227},
        {"name": "model_saving", "outcome": "succeeded", "evidence_line": 236},
    ],
    "progress": {
        "step": 9936,
        "total_steps": 9936,
        "epoch": 2.0,
        "last_loss": 0.4227,
        "eval_losses": [0.6401, 0.6355],
    },
    "errors": [],
    "warnings": [],
    "evidence": [
        {
            "phase": "setup",
            "rule": "trainer_banner",
            "line": 124,
            "text": "***** Running training *****",
        },
        {
            "phase": "fine_tuning",
            "rule": "training_completed",
            "line": 225,
            "text": "Training completed.",
        },
        {
            "phase": "checkpoint_creation",
            "rule": "saving_checkpoint",
            "line": 227,
            "text": "Saving model checkpoint to fineTuningOutput/checkpoint-9936",
        },
        {
            "phase": "model_saving",
            "rule": "saving_final_model",
            "line": 236,
            "text": "Saving final model to fineTuningOutput/final_model",
        },
    ],
    "started_at": None,
    "ended_at": None,
    "duration_seconds": None,
    "notes": [],
}


@dataclass(frozen=True)
class UploadedLogEntry:
    """Metadata for a user-uploaded log file stored inside the Upload dropdown."""

    name: str
    temp_path: Path
    size_bytes: int


@dataclass(frozen=True)
class PhaseRowView:
    """Presentation model for one phase row in the Summary card."""

    name: str
    label: str
    outcome: str
    outcome_label: str
    tone: str  # "success" | "warning" | "error" | "neutral"
    evidence_line: int | None = None
    evidence_text: str | None = None


@dataclass(frozen=True)
class DiagnosticAlertView:
    """Presentation model for an error or warning callout."""

    tone: str  # "warning" | "error"
    title: str
    kind: str
    message: str
    detail: str | None = None
    line: int | None = None
    rank: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_warning(self) -> bool:
        return self.tone == "warning"

    @property
    def is_error(self) -> bool:
        return self.tone == "error"


@dataclass(frozen=True)
class ResultSummaryView:
    """Pure presentation view derived from a ``ParseResult``."""

    family: str
    family_label: str
    status: str
    status_label: str
    status_tone: str  # "success" | "warning" | "error" | "terminated" | "neutral"
    scenario: str | None
    script: str | None
    model_id: str | None
    output_format: str | None
    failed_phase: str | None
    duration_text: str | None
    progress_metrics: list[tuple[str, str]]
    phases: list[PhaseRowView]
    first_alert: DiagnosticAlertView | None
    alerts: list[DiagnosticAlertView]
    notes: list[str]
    raw: ParseResult


def list_fixture_logs(directory: Path | None = None) -> list[Path]:
    """Return sorted ``s01``–``s15`` preset ``.log`` fixture files."""
    base = directory if directory is not None else FIXTURE_LOGS_DIR
    if not base.exists():
        return []
    return sorted(
        p
        for p in base.glob("s[0-9][0-9]_*.log")
        if p.is_file()
    )


def list_error_or_warning_fixture_logs(directory: Path | None = None) -> list[Path]:
    """Return ``s01``–``s15`` preset fixture logs that contain an error, warning, or termination."""
    flagged: list[Path] = []
    for p in list_fixture_logs(directory):
        if "success" not in p.name or "warning" in p.name:
            flagged.append(p)
    return flagged


def fixture_log_options(directory: Path | None = None) -> dict[str, str]:
    """Return ``{str(path): display_label}`` mapping for the Preset Log selector (s01–s15 only)."""
    options: dict[str, str] = {}
    for p in list_fixture_logs(directory):
        options[str(p)] = p.name
    return options


def _format_duration(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_file_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _format_progress(
    family: str, progress: Mapping[str, Any] | None
) -> list[tuple[str, str]]:
    if not progress:
        return []
    metrics: list[tuple[str, str]] = []
    if family == "fine_tune":
        step = progress.get("step")
        total_steps = progress.get("total_steps")
        if step is not None and total_steps is not None:
            metrics.append(("Steps", f"{step:,} / {total_steps:,}"))
        elif step is not None:
            metrics.append(("Step", f"{step:,}"))
        epoch = progress.get("epoch")
        if epoch is not None:
            metrics.append(("Epoch", f"{float(epoch):.2f}"))
        last_loss = progress.get("last_loss")
        if last_loss is not None:
            metrics.append(("Last Train Loss", f"{float(last_loss):.4f}"))
        eval_losses = progress.get("eval_losses")
        if isinstance(eval_losses, Sequence) and eval_losses:
            formatted_evals = ", ".join(f"{float(v):.4f}" for v in eval_losses)
            metrics.append(("Eval Losses", formatted_evals))
    elif family == "inference":
        records_done = progress.get("records_done")
        if records_done is not None:
            metrics.append(("Records Processed", f"{int(records_done):,}"))
    return metrics


def build_result_summary(result: ParseResult) -> ResultSummaryView:
    """Transform a :class:`ParseResult` into a UI-ready :class:`ResultSummaryView`.

    Special handling for Log 12 (``s12_fine_tune_success_warning.log``):
    When ``status == "success"`` and a post-core phase fails (generating an entry in
    ``warnings`` and ``failed_phase == "model_saving"``), the overall status badge,
    the failed post-core phase row, and the diagnostic callout all use the yellow
    ``"warning"`` tone rather than a red ``"error"`` tone.
    """
    family = str(result.get("family") or "unknown")
    status = str(result.get("status") or "unknown")
    warnings = list(result.get("warnings") or [])
    errors = list(result.get("errors") or [])
    notes = [str(n) for n in (result.get("notes") or [])]
    failed_phase = result.get("failed_phase")

    has_post_core_warning = status == "success" and (
        bool(warnings) or failed_phase is not None
    )

    if has_post_core_warning:
        status_tone = "warning"
        status_label = "Success · Warning"
    elif status == "success":
        status_tone = "success"
        status_label = "Success"
    elif status == "error":
        status_tone = "error"
        status_label = "Error"
    elif status == "terminated":
        status_tone = "terminated"
        status_label = "Terminated"
    else:
        status_tone = "neutral"
        status_label = "Unknown"

    evidence_by_phase: dict[str, str] = {}
    for ev in result.get("evidence") or []:
        ev_phase = ev.get("phase")
        ev_text = ev.get("text")
        if ev_phase and ev_text:
            evidence_by_phase[str(ev_phase)] = str(ev_text)

    phase_rows: list[PhaseRowView] = []
    for ph in result.get("phases") or []:
        p_name = str(ph.get("name") or "")
        p_outcome = str(ph.get("outcome") or "not_reached")
        p_line = ph.get("evidence_line")
        if p_outcome == "succeeded":
            p_tone = "success"
        elif p_outcome == "failed":
            # In Log 12, status is 'success' and model_saving failed after core fine_tuning succeeded:
            # show yellow warning on that row instead of red error.
            p_tone = "warning" if status == "success" else "error"
        elif p_outcome == "terminated":
            p_tone = "warning"
        else:
            p_tone = "neutral"

        phase_rows.append(
            PhaseRowView(
                name=p_name,
                label=PHASE_LABELS.get(p_name, p_name.replace("_", " ").title()),
                outcome=p_outcome,
                outcome_label=OUTCOME_LABELS.get(p_outcome, p_outcome.title()),
                tone=p_tone,
                evidence_line=int(p_line) if p_line is not None else None,
                evidence_text=evidence_by_phase.get(p_name),
            )
        )

    alerts: list[DiagnosticAlertView] = []
    if has_post_core_warning:
        first_warn = warnings[0] if warnings else {}
        first_err = errors[0] if errors else {}
        warn_msg = str(
            first_warn.get("message")
            or f"{failed_phase} failed after core phase succeeded"
        )
        err_detail = str(first_err.get("message") or "") or None
        line_no = first_warn.get("line") or first_err.get("line")
        rank_no = (
            first_warn.get("rank")
            if first_warn.get("rank") is not None
            else first_err.get("rank")
        )
        alerts.append(
            DiagnosticAlertView(
                tone="warning",
                title="Post-Core Phase Warning",
                kind=str(first_warn.get("kind") or first_err.get("kind") or "warning"),
                message=warn_msg,
                detail=err_detail,
                line=int(line_no) if line_no is not None else None,
                rank=int(rank_no) if rank_no is not None else None,
                notes=notes,
            )
        )
    else:
        for idx, err in enumerate(errors):
            alert_tone = "warning" if status == "terminated" else "error"
            alert_title = (
                "Process Terminated"
                if status == "terminated"
                else ("First Error Detected" if idx == 0 else f"Error #{idx + 1}")
            )
            line_no = err.get("line")
            rank_no = err.get("rank")
            alerts.append(
                DiagnosticAlertView(
                    tone=alert_tone,
                    title=alert_title,
                    kind=str(err.get("kind") or "error"),
                    message=str(err.get("message") or "Unknown error"),
                    detail=None,
                    line=int(line_no) if line_no is not None else None,
                    rank=int(rank_no) if rank_no is not None else None,
                    notes=notes if idx == 0 else [],
                )
            )
        for warn in warnings:
            line_no = warn.get("line")
            rank_no = warn.get("rank")
            alerts.append(
                DiagnosticAlertView(
                    tone="warning",
                    title="Warning",
                    kind=str(warn.get("kind") or "warning"),
                    message=str(warn.get("message") or ""),
                    detail=None,
                    line=int(line_no) if line_no is not None else None,
                    rank=int(rank_no) if rank_no is not None else None,
                    notes=notes if not alerts else [],
                )
            )

    first_alert = alerts[0] if alerts else None

    return ResultSummaryView(
        family=family,
        family_label=FAMILY_LABELS.get(family, family),
        status=status,
        status_label=status_label,
        status_tone=status_tone,
        scenario=result.get("scenario"),
        script=result.get("script"),
        model_id=result.get("model_id"),
        output_format=result.get("output_format"),
        failed_phase=failed_phase,
        duration_text=_format_duration(result.get("duration_seconds")),
        progress_metrics=_format_progress(family, result.get("progress")),
        phases=phase_rows,
        first_alert=first_alert,
        alerts=alerts,
        notes=notes,
        raw=result,
    )


_TONE_PILL_CLASSES: dict[str, str] = {
    "success": "status-pill pill-success",
    "warning": "status-pill pill-warning",
    "error": "status-pill pill-error",
    "terminated": "status-pill pill-terminated",
    "neutral": "status-pill pill-neutral",
}

_TONE_DOT_CLASSES: dict[str, str] = {
    "success": "status-dot dot-success",
    "warning": "status-dot dot-warning",
    "error": "status-dot dot-error",
    "terminated": "status-dot dot-terminated",
    "neutral": "status-dot dot-neutral",
}


class ResultsPanel:
    """NiceGUI Run Results tab controller: config selector, preset/upload dropdown with multi-file list, error cycler, and Summary card."""

    def __init__(
        self,
        container: Any,
        *,
        list_configs_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        parse_fn: Callable[[str | Path], ParseResult] = parse_log_file,
        initial_result: ParseResult | None = None,
    ) -> None:
        from nicegui import ui

        self._ui = ui
        self.container = container
        self._list_configs_fn = list_configs_fn
        self._parse_fn = parse_fn

        self.selected_config_id: str | None = None
        self.current_log_path: str = ""
        self.current_source_name: str | None = None
        self.current_summary: ResultSummaryView | None = None
        self.active_alert_index: int = 0

        # Multi-file upload list stored strictly inside the Upload dropdown
        self.uploaded_files: list[UploadedLogEntry] = []
        self.uploaded_temp_path: Path | None = None
        self.uploaded_file_name: str | None = None
        self.uploaded_file_size: int = 0
        self._suppress_fixture_change: bool = False

        fixtures = list_fixture_logs()
        if fixtures:
            self.current_log_path = str(fixtures[0])

        with self.container:
            with ui.column().classes("w-full gap-3"):
                # 1. Top Selectors: Linked Configuration + Preset Log (Logs 01-15) with Up/Down cycle arrows
                with ui.grid(columns=2).classes("w-full gap-3 items-end"):
                    with ui.column().classes("w-full min-w-0 gap-1"):
                        ui.label("Linked Configuration").classes(
                            "text-xs font-medium text-secondary-token select-none"
                        )
                        self.config_select = (
                            ui.select(
                                options=self._build_config_options(),
                                value=None,
                                on_change=lambda e: setattr(
                                    self, "selected_config_id", e.value
                                ),
                            )
                            .props("outlined dense clearable")
                            .classes("w-full")
                        )

                    with ui.column().classes("w-full min-w-0 gap-1"):
                        ui.label("Preset Log (Logs 01–15)").classes(
                            "text-xs font-medium text-secondary-token select-none"
                        )
                        with ui.row().classes("w-full items-center gap-1 no-wrap"):
                            self.fixture_select = (
                                ui.select(
                                    options=fixture_log_options(),
                                    value=self.current_log_path or None,
                                    on_change=self._on_fixture_selected,
                                )
                                .props("outlined dense")
                                .classes("flex-grow min-w-0")
                            )
                            self.prev_log_button = (
                                ui.button(
                                    icon="keyboard_arrow_up",
                                    on_click=lambda: self._on_cycle_preset(-1),
                                )
                                .props("flat dense round size=sm color=primary")
                                .tooltip("Previous preset log (01–15)")
                            )
                            self.next_log_button = (
                                ui.button(
                                    icon="keyboard_arrow_down",
                                    on_click=lambda: self._on_cycle_preset(1),
                                )
                                .props("flat dense round size=sm color=primary")
                                .tooltip("Next preset log (01–15)")
                            )

                # 2. Upload Log File Dropdown: properly padded & centered inside its border
                self.upload_expansion = ui.expansion(
                    "Upload Log File (.log / .out / .txt)", icon="upload_file"
                ).classes(
                    "upload-expansion w-full rounded-md surface-subtle text-xs font-medium"
                )
                with self.upload_expansion:
                    with ui.column().classes(
                        "w-full gap-0 rounded-md border-token overflow-hidden surface-card"
                    ):
                        self.uploader = (
                            ui.upload(
                                label="Drop or select Slurm / Python log files",
                                auto_upload=True,
                                multiple=True,
                                on_upload=self._on_file_uploaded,
                            )
                            .props("flat dense accept=.log,.out,.txt")
                            .classes("w-full compact-uploader")
                        )
                        # Uploaded files list rendered directly inside the dropdown box
                        self.uploaded_files_container = ui.column().classes(
                            "w-full gap-0"
                        )
                        # Alias kept for test compatibility
                        self.uploaded_file_row = self.uploaded_files_container

                self._render_uploaded_files_list()

                # 3. Spinner row shown while run.io_bound(parse_log_file, ...) is running
                self.spinner_row = ui.row().classes(
                    "w-full items-center gap-2.5 px-3 py-2 rounded-md surface-subtle"
                )
                with self.spinner_row:
                    self.spinner = ui.spinner("dots", size="sm", color="primary")
                    self.spinner_label = ui.label("Parsing log file...").classes(
                        "text-xs font-medium text-secondary-token"
                    )
                self.spinner_row.set_visibility(False)

                # 4. Summary Card Container
                self.summary_container = ui.column().classes("w-full gap-3")

        if initial_result is not None:
            self.render_result(initial_result)
        elif fixtures:
            self.load_file(fixtures[0])
        else:
            self.render_result(DEFAULT_PARSE_RESULT)

    def _safe_notify(self, message: str, *, type: str = "info") -> None:
        try:
            self._ui.notify(message, type=type, position="top")
        except RuntimeError:
            pass

    def _build_config_options(self) -> dict[str, str]:
        if self._list_configs_fn is None:
            return {}
        try:
            rows = self._list_configs_fn()
        except Exception:
            return {}
        options: dict[str, str] = {}
        for row in rows:
            cid = str(row.get("config_id") or "")
            if not cid:
                continue
            model_short = str(row.get("model_id") or "").split("/")[-1]
            task = str(row.get("task_type") or "")
            options[cid] = f"{cid[:8]} · {model_short} ({task})"
        return options

    def refresh_config_options(self) -> None:
        """Refresh the Linked Configuration dropdown from SQLite."""
        self.config_select.options = self._build_config_options()
        self.config_select.update()

    async def _on_fixture_selected(self, e: Any) -> None:
        if self._suppress_fixture_change:
            return
        val = e.value if hasattr(e, "value") else self.fixture_select.value
        if not val:
            return
        path_str = str(val)
        await self.parse_async(path_str)
        self._render_uploaded_files_list()

    def cycle_preset(self, delta: int) -> ResultSummaryView:
        """Synchronous helper to cycle strictly through preset logs 01–15."""
        preset_paths = [str(p) for p in list_fixture_logs()]
        if not preset_paths:
            return self.current_summary or self.render_result(DEFAULT_PARSE_RESULT)
        curr = self.fixture_select.value or self.current_log_path
        idx = preset_paths.index(curr) if curr in preset_paths else 0
        next_path = preset_paths[(idx + delta) % len(preset_paths)]
        self._suppress_fixture_change = True
        try:
            self.fixture_select.value = next_path
        finally:
            self._suppress_fixture_change = False
        summary = self.load_file(next_path)
        self._render_uploaded_files_list()
        return summary

    async def _on_cycle_preset(self, delta: int) -> None:
        preset_paths = [str(p) for p in list_fixture_logs()]
        if not preset_paths:
            return
        curr = self.fixture_select.value or self.current_log_path
        idx = preset_paths.index(curr) if curr in preset_paths else 0
        next_path = preset_paths[(idx + delta) % len(preset_paths)]
        self._suppress_fixture_change = True
        try:
            self.fixture_select.value = next_path
        finally:
            self._suppress_fixture_change = False
        await self.parse_async(next_path)
        self._render_uploaded_files_list()

    def cycle_error(self, delta: int) -> ResultSummaryView:
        """Cycle through alerts within the current log, or step to the previous/next s01–s15 error log."""
        if (
            self.current_summary is not None
            and len(self.current_summary.alerts) > 1
            and 0 <= self.active_alert_index + delta < len(self.current_summary.alerts)
        ):
            self.active_alert_index += delta
            return self.render_result(
                self.current_summary.raw,
                source_name=self.current_source_name,
                alert_index=self.active_alert_index,
            )

        error_logs = [str(p) for p in list_error_or_warning_fixture_logs()]
        if not error_logs:
            return self.current_summary or self.render_result(DEFAULT_PARSE_RESULT)
        curr = self.current_log_path
        if curr in error_logs:
            idx = error_logs.index(curr)
            next_path = error_logs[(idx + delta) % len(error_logs)]
        else:
            next_path = error_logs[0 if delta >= 0 else -1]
        self._suppress_fixture_change = True
        try:
            self.fixture_select.value = next_path
        finally:
            self._suppress_fixture_change = False
        return self.load_file(next_path)

    async def _on_cycle_error(self, delta: int) -> None:
        if (
            self.current_summary is not None
            and len(self.current_summary.alerts) > 1
            and 0 <= self.active_alert_index + delta < len(self.current_summary.alerts)
        ):
            self.active_alert_index += delta
            self.render_result(
                self.current_summary.raw,
                source_name=self.current_source_name,
                alert_index=self.active_alert_index,
            )
            return

        error_logs = [str(p) for p in list_error_or_warning_fixture_logs()]
        if not error_logs:
            return
        curr = self.current_log_path
        if curr in error_logs:
            idx = error_logs.index(curr)
            next_path = error_logs[(idx + delta) % len(error_logs)]
        else:
            next_path = error_logs[0 if delta >= 0 else -1]
        self._suppress_fixture_change = True
        try:
            self.fixture_select.value = next_path
        finally:
            self._suppress_fixture_change = False
        await self.parse_async(next_path)

    def _render_uploaded_files_list(self) -> None:
        """Render all uploaded log files inside the Upload dropdown box."""
        ui = self._ui
        self.uploaded_files_container.clear()
        if not self.uploaded_files:
            self.uploaded_files_container.set_visibility(False)
            return

        self.uploaded_files_container.set_visibility(True)
        with self.uploaded_files_container:
            for idx, entry in enumerate(self.uploaded_files):
                is_active = self.current_log_path == str(entry.temp_path)
                border_cls = "border-t border-token" if idx >= 0 else ""
                row_bg = "bg-blue-500/5" if is_active else ""
                with ui.row().classes(
                    f"w-full items-center justify-between px-3.5 py-2.5 no-wrap phase-row {border_cls} {row_bg}"
                ):
                    with (
                        ui.row()
                        .classes(
                            "items-center gap-2 no-wrap min-w-0 flex-grow cursor-pointer"
                        )
                        .on(
                            "click",
                            lambda _, p=entry.temp_path, n=entry.name: self._on_select_uploaded_entry(
                                p, n
                            ),
                        )
                    ):
                        ui.icon(
                            "description",
                            color="primary" if is_active else "grey",
                        ).classes("text-base shrink-0")
                        ui.label(
                            f"{entry.name} ({_format_file_size(entry.size_bytes)})"
                        ).classes(
                            "text-xs font-semibold truncate"
                            if is_active
                            else "text-xs font-medium truncate"
                        )
                        if is_active:
                            with ui.element("span").classes(
                                "status-pill pill-success"
                            ):
                                ui.label("Active")
                    ui.button(
                        "Remove",
                        icon="delete_outline",
                        on_click=lambda _, p=entry.temp_path: self.remove_uploaded_file(
                            p
                        ),
                    ).props("flat dense no-caps size=sm color=negative").tooltip(
                        f"Remove {entry.name}"
                    )

    async def _on_select_uploaded_entry(self, temp_path: Path, name: str) -> None:
        self.uploaded_temp_path = temp_path
        self.uploaded_file_name = name
        await self.parse_async(temp_path, display_name=name)
        self._render_uploaded_files_list()

    async def _on_file_uploaded(self, e: Any) -> None:
        file_obj = getattr(e, "file", None)
        orig_name = str(
            getattr(file_obj, "name", None) or getattr(e, "name", "uploaded.log")
        )
        suffix = Path(orig_name).suffix or ".log"

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=suffix, delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        if file_obj is not None and hasattr(file_obj, "save"):
            await file_obj.save(tmp_path)
        elif hasattr(e, "content") and e.content is not None:
            data = e.content.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            tmp_path.write_bytes(data)

        size_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        new_entry = UploadedLogEntry(
            name=orig_name,
            temp_path=tmp_path,
            size_bytes=size_bytes,
        )

        # If a file with the same name was already uploaded, replace its temp file; otherwise append
        existing_idx = next(
            (i for i, item in enumerate(self.uploaded_files) if item.name == orig_name),
            None,
        )
        if existing_idx is not None:
            old_entry = self.uploaded_files[existing_idx]
            try:
                old_entry.temp_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.uploaded_files[existing_idx] = new_entry
        else:
            self.uploaded_files.append(new_entry)

        self.uploaded_temp_path = tmp_path
        self.uploaded_file_name = orig_name
        self.uploaded_file_size = size_bytes

        # Reset Quasar's internal uploader queue so its empty box stays hidden and more files can be added
        try:
            self.uploader.reset()
        except Exception:
            pass

        self._render_uploaded_files_list()
        await self.parse_async(tmp_path, display_name=orig_name)
        self._render_uploaded_files_list()

    def remove_uploaded_file(self, temp_path: Path | str) -> ResultSummaryView:
        """Remove one specific uploaded file from the Upload dropdown list."""
        target_str = str(temp_path)
        removed: UploadedLogEntry | None = None
        remaining: list[UploadedLogEntry] = []
        for entry in self.uploaded_files:
            if str(entry.temp_path) == target_str:
                removed = entry
            else:
                remaining.append(entry)
        self.uploaded_files = remaining

        if removed is not None:
            try:
                removed.temp_path.unlink(missing_ok=True)
            except OSError:
                pass

        if self.uploaded_files:
            latest = self.uploaded_files[-1]
            self.uploaded_temp_path = latest.temp_path
            self.uploaded_file_name = latest.name
            self.uploaded_file_size = latest.size_bytes
        else:
            self.uploaded_temp_path = None
            self.uploaded_file_name = None
            self.uploaded_file_size = 0

        # If the removed file was currently displayed, switch to the latest remaining uploaded file or preset
        if self.current_log_path == target_str:
            if self.uploaded_files:
                nxt = self.uploaded_files[-1]
                summary = self.load_file(nxt.temp_path, display_name=nxt.name)
                self._render_uploaded_files_list()
                return summary

            fixtures = list_fixture_logs()
            if fixtures:
                self._suppress_fixture_change = True
                try:
                    self.fixture_select.value = str(fixtures[0])
                finally:
                    self._suppress_fixture_change = False
                summary = self.load_file(fixtures[0])
                self._render_uploaded_files_list()
                return summary

        self._render_uploaded_files_list()
        return self.current_summary or self.render_result(DEFAULT_PARSE_RESULT)

    def clear_uploaded_file(self) -> ResultSummaryView:
        """Clear all uploaded sample outputs, reset the uploader, and restore the preset log."""
        for entry in self.uploaded_files:
            try:
                entry.temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.uploaded_files.clear()
        self.uploaded_temp_path = None
        self.uploaded_file_name = None
        self.uploaded_file_size = 0
        try:
            self.uploader.reset()
        except Exception:
            pass

        self._render_uploaded_files_list()

        fixtures = list_fixture_logs()
        if fixtures:
            self._suppress_fixture_change = True
            try:
                self.fixture_select.value = str(fixtures[0])
            finally:
                self._suppress_fixture_change = False
            return self.load_file(fixtures[0])
        return self.render_result(DEFAULT_PARSE_RESULT)

    async def parse_async(
        self, path: str | Path, *, display_name: str | None = None
    ) -> ResultSummaryView:
        """Parse ``path`` off the event loop with ``run.io_bound`` while showing a spinner."""
        from nicegui import run

        target = Path(path)
        self.current_log_path = str(target)
        shown_name = display_name or target.name
        self.spinner_label.text = f"Parsing {shown_name}..."
        self.spinner_row.set_visibility(True)
        try:
            if not target.exists():
                self._safe_notify(
                    f"Log file not found: {target}",
                    type="negative",
                )
                return self.current_summary or self.render_result(DEFAULT_PARSE_RESULT)
            result = await run.io_bound(self._parse_fn, target)
            return self.render_result(result, source_name=shown_name)
        finally:
            self.spinner_row.set_visibility(False)

    def load_file(
        self, path: str | Path, *, display_name: str | None = None
    ) -> ResultSummaryView:
        """Synchronous helper to parse ``path`` and update the Summary card."""
        target = Path(path)
        self.current_log_path = str(target)
        shown_name = display_name or target.name
        result = self._parse_fn(target)
        return self.render_result(result, source_name=shown_name)

    def render_result(
        self,
        result: ParseResult,
        *,
        source_name: str | None = None,
        alert_index: int = 0,
    ) -> ResultSummaryView:
        """Render a :class:`ParseResult` into the Summary card using standard project typography."""
        ui = self._ui
        summary = build_result_summary(result)
        self.current_summary = summary
        self.current_source_name = source_name
        if summary.alerts:
            self.active_alert_index = max(
                0, min(alert_index, len(summary.alerts) - 1)
            )
        else:
            self.active_alert_index = 0

        self.summary_container.clear()
        with self.summary_container:
            with ui.card().props("flat").classes(
                "w-full p-4 gap-3.5 rounded-lg surface-card"
            ):
                # 1. Top Run Classification Header (no redundant filename tag next to Run Summary)
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 flex-wrap"
                ):
                    ui.label("Run Summary").classes(
                        "text-sm font-semibold tracking-tight"
                    )
                    with ui.row().classes("items-center gap-1.5 no-wrap"):
                        with ui.element("span").classes("status-pill pill-neutral"):
                            ui.label(summary.family_label)
                        pill_cls = _TONE_PILL_CLASSES.get(
                            summary.status_tone, "status-pill pill-neutral"
                        )
                        dot_cls = _TONE_DOT_CLASSES.get(
                            summary.status_tone, "status-dot dot-neutral"
                        )
                        with ui.element("span").classes(pill_cls):
                            ui.element("span").classes(dot_cls)
                            ui.label(summary.status_label)

                # 2. Metadata Strip (Script, Model, Output Format, Duration)
                meta_items: list[tuple[str, str]] = [
                    ("Family", summary.family_label),
                    ("Status", summary.status.title()),
                ]
                if summary.script:
                    meta_items.append(("Script", summary.script))
                if summary.model_id:
                    meta_items.append(("Model", summary.model_id.split("/")[-1]))
                if summary.output_format:
                    meta_items.append(("Format", summary.output_format.upper()))
                if summary.duration_text:
                    meta_items.append(("Duration", summary.duration_text))

                with ui.row().classes(
                    "w-full items-center gap-x-5 gap-y-1.5 px-3 py-2 rounded-md surface-subtle flex-wrap"
                ):
                    for k, v in meta_items:
                        with ui.row().classes("items-baseline gap-1.5 no-wrap"):
                            ui.label(f"{k}:").classes(
                                "text-xs font-medium text-secondary-token"
                            )
                            ui.label(v).classes("text-xs font-semibold")

                # 3. Progress Metrics Strip (single-line values, equal height across all tiles)
                if summary.progress_metrics:
                    with ui.row().classes("w-full gap-2.5 items-stretch flex-wrap"):
                        for label, val in summary.progress_metrics:
                            with ui.column().classes(
                                "flex-1 min-w-[110px] p-2.5 gap-0.5 justify-center rounded-md surface-subtle"
                            ):
                                ui.label(label).classes(
                                    "text-[10px] font-semibold uppercase tracking-wider text-secondary-token whitespace-nowrap"
                                )
                                ui.label(val).classes(
                                    "text-sm font-semibold whitespace-nowrap"
                                )

                # 4. Phase Outcomes Table (one crisp, vertically centered horizontal row per phase)
                with ui.column().classes("w-full gap-0"):
                    ui.label("Pipeline Phases").classes(
                        "text-xs font-medium text-secondary-token mb-1.5 select-none"
                    )
                    with ui.column().classes(
                        "w-full rounded-md border-token overflow-hidden gap-0"
                    ):
                        for idx, phase in enumerate(summary.phases):
                            row_border = (
                                "border-b border-token"
                                if idx < len(summary.phases) - 1
                                else ""
                            )
                            with ui.row().classes(
                                f"w-full items-center justify-between px-3.5 py-2.5 no-wrap gap-2 phase-row {row_border}"
                            ):
                                with ui.row().classes(
                                    "items-center gap-2.5 no-wrap min-w-0"
                                ):
                                    p_dot = _TONE_DOT_CLASSES.get(
                                        phase.tone, "status-dot dot-neutral"
                                    )
                                    ui.element("span").classes(p_dot)
                                    ui.label(phase.label).classes(
                                        "text-xs font-medium truncate"
                                    )

                                with ui.row().classes(
                                    "items-center gap-2.5 no-wrap shrink-0"
                                ):
                                    if phase.evidence_line is not None:
                                        ui.label(
                                            f"Line {phase.evidence_line}"
                                        ).classes(
                                            "text-[11px] font-medium text-secondary-token"
                                        )
                                    p_pill = _TONE_PILL_CLASSES.get(
                                        phase.tone, "status-pill pill-neutral"
                                    )
                                    with ui.element("span").classes(p_pill):
                                        ui.label(phase.outcome_label)

                # 5. Error / Warning Callout (with Up / Down cycle arrows when a log has multiple issues)
                if summary.alerts:
                    alert = summary.alerts[self.active_alert_index]
                    if alert.is_warning:
                        box_cls = "w-full p-3 gap-1.5 rounded-md bg-amber-500/10 border border-amber-500/30"
                        icon_name = "warning_amber"
                        icon_color = "warning"
                        title_cls = "text-xs font-semibold text-amber-600 dark:text-amber-400"
                        btn_color = "warning"
                    else:
                        box_cls = "w-full p-3 gap-1.5 rounded-md bg-red-500/10 border border-red-500/30"
                        icon_name = "error_outline"
                        icon_color = "negative"
                        title_cls = "text-xs font-semibold text-red-600 dark:text-red-400"
                        btn_color = "negative"

                    with ui.column().classes(box_cls):
                        with ui.row().classes(
                            "w-full items-center justify-between no-wrap gap-2"
                        ):
                            with ui.row().classes("items-center gap-1.5 no-wrap min-w-0"):
                                ui.icon(icon_name, color=icon_color).classes(
                                    "text-base shrink-0"
                                )
                                ui.label(f"{alert.title} · {alert.kind}").classes(
                                    f"{title_cls} truncate"
                                )

                            with ui.row().classes(
                                "items-center gap-1.5 no-wrap shrink-0"
                            ):
                                meta_badges: list[str] = []
                                if alert.line is not None:
                                    meta_badges.append(f"Line {alert.line}")
                                if alert.rank is not None:
                                    meta_badges.append(f"Rank {alert.rank}")
                                if len(summary.alerts) > 1:
                                    meta_badges.append(
                                        f"{self.active_alert_index + 1} / {len(summary.alerts)}"
                                    )
                                if meta_badges:
                                    ui.label(" · ".join(meta_badges)).classes(
                                        "text-[11px] font-medium text-secondary-token"
                                    )
                                if len(summary.alerts) > 1:
                                    ui.button(
                                        icon="keyboard_arrow_up",
                                        on_click=lambda: self._on_cycle_error(-1),
                                    ).props(
                                        f"flat dense round size=xs color={btn_color}"
                                    ).tooltip("Previous issue in this log")
                                    ui.button(
                                        icon="keyboard_arrow_down",
                                        on_click=lambda: self._on_cycle_error(1),
                                    ).props(
                                        f"flat dense round size=xs color={btn_color}"
                                    ).tooltip("Next issue in this log")

                        ui.label(alert.message).classes(
                            "text-xs font-medium break-words leading-relaxed"
                        )
                        if alert.detail:
                            ui.label(alert.detail).classes(
                                "text-xs text-secondary-token break-words leading-relaxed"
                            )
                        for note in alert.notes:
                            ui.label(note).classes(
                                "text-[11px] text-secondary-token pt-0.5"
                            )

        return summary


def build_results_panel(
    container: Any,
    *,
    list_configs_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    parse_fn: Callable[[str | Path], ParseResult] = parse_log_file,
    initial_result: ParseResult | None = None,
) -> ResultsPanel:
    """Mount the Milestone 1.5 Run Results panel inside ``container``."""
    return ResultsPanel(
        container=container,
        list_configs_fn=list_configs_fn,
        parse_fn=parse_fn,
        initial_result=initial_result,
    )
