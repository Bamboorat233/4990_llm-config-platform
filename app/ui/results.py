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

import json
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from app.parsing import ParseResult, parse_log_file
from app.parsing.verdicts import (
    DEFAULT_LABELS,
    TABLE_ONLY_SINGLE_TABLE,
    extract_verdicts_with_source,
)
from app.registry import DEFAULT_VERDICT_MARKER, MODELS
from app.scoring import (
    INVALID_LABEL,
    ScoreReport,
    VerdictRecord,
    load_gold_labels,
    render_report_markdown,
    score_verdicts,
)
from app.store import get_config, get_score, insert_run, list_configs, upsert_score

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_LOGS_DIR = REPO_ROOT / "tests" / "fixtures" / "logs"

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
    "generation": "Model Inference",
    "verdict_output": "Final Verdict Summary",
}

PHASE_DESCRIPTIONS: dict[str, str] = {
    "setup": "Model weights, tokenizer & distributed environment init",
    "fine_tuning": "LoRA / SFT training iterations & loss evaluation",
    "checkpoint_creation": "Intermediate trainer state & adapter checkpointing",
    "model_saving": "Final model weights & tokenizer serialization",
    "data_preparation": "Dataset split loading & prompt template formatting",
    "generation": "Model inference & response generation across test records",
    "verdict_output": "Final verdict summary & marker extraction output",
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
            metrics.append(("Train loss", f"{float(last_loss):.4f}"))
        eval_losses = progress.get("eval_losses")
        if isinstance(eval_losses, Sequence) and eval_losses:
            formatted_evals = ", ".join(f"{float(v):.4f}" for v in eval_losses)
            metrics.append(("Eval loss", formatted_evals))
    elif family == "inference":
        records_done = progress.get("records_done")
        if records_done is not None:
            metrics.append(("Records generated by the model", f"{int(records_done):,}"))
            metrics.append(("Records used for scoring", "-"))
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


@dataclass(frozen=True)
class ConfigEvalSettings:
    """Extraction and scoring settings resolved from a linked config or defaults."""

    config_id: str | None = None
    model_id: str | None = None
    eos_token: str | None = None
    marker: str = DEFAULT_VERDICT_MARKER
    labels: tuple[str, ...] = DEFAULT_LABELS
    test_data_file: str | None = None


@dataclass(frozen=True)
class InferenceEvaluationView:
    """Pure presentation snapshot for inference verdict extraction and scoring."""

    available: bool
    output_format: str | None = None
    settings: ConfigEvalSettings = field(default_factory=ConfigEvalSettings)
    source: str | None = None
    verdicts: list[VerdictRecord] = field(default_factory=list)
    review_rows: list[dict[str, Any]] = field(default_factory=list)
    out_of_set_counts: list[tuple[str, int]] = field(default_factory=list)
    gold_path: str | None = None
    gold_count: int | None = None
    gold_notice: str | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    score_report: ScoreReport | None = None


def resolve_eval_settings(
    cfg: Any | None, *, config_id: str | None = None
) -> ConfigEvalSettings:
    """Extract marker, label values, test data path, and EOS token from a config."""
    if cfg is None:
        return ConfigEvalSettings(config_id=config_id)

    if isinstance(cfg, Mapping):
        resolved_id = str(cfg.get("config_id") or config_id or "") or None
        model_id = str(cfg.get("model_id") or "") or None
        p_sec = cfg.get("p") or {}
        c_sec = cfg.get("c") or {}
        raw_marker = p_sec.get("verdict_parse_marker")
        raw_labels = c_sec.get("label_values")
        raw_test_file = c_sec.get("test_data_file")
    else:
        resolved_id = getattr(cfg, "config_id", None) or config_id
        model_id = getattr(cfg, "model_id", None)
        p_sec = getattr(cfg, "p", None)
        c_sec = getattr(cfg, "c", None)
        raw_marker = getattr(p_sec, "verdict_parse_marker", None)
        raw_labels = getattr(c_sec, "label_values", None)
        raw_test_file = getattr(c_sec, "test_data_file", None)

    marker = (
        str(raw_marker)
        if isinstance(raw_marker, str) and raw_marker
        else DEFAULT_VERDICT_MARKER
    )
    if isinstance(raw_labels, Sequence) and not isinstance(raw_labels, (str, bytes)):
        labels = tuple(str(x) for x in raw_labels if str(x)) or DEFAULT_LABELS
    else:
        labels = DEFAULT_LABELS

    test_data_file = (
        str(raw_test_file).strip()
        if isinstance(raw_test_file, str) and raw_test_file.strip()
        else None
    )
    meta = MODELS.get(model_id) if model_id else None
    eos_token = meta.eos_token if meta is not None else None

    return ConfigEvalSettings(
        config_id=resolved_id,
        model_id=model_id,
        eos_token=eos_token,
        marker=marker,
        labels=labels,
        test_data_file=test_data_file,
    )


def _find_existing_file(raw_path: str | Path | None) -> Path | None:
    """Return an existing file Path for ``raw_path`` (checking CWD then REPO_ROOT)."""
    if raw_path is None:
        return None
    text = str(raw_path).strip()
    if not text:
        return None
    candidate = Path(text)
    if candidate.is_file():
        return candidate
    if not candidate.is_absolute():
        rooted = REPO_ROOT / candidate
        if rooted.is_file():
            return rooted
    return None


def load_gold_labels_file(path: str | Path) -> list[str]:
    """Load gold ``label`` strings from a JSON list or JSONL file."""
    target = Path(path)
    if target.suffix.lower() != ".jsonl":
        try:
            return load_gold_labels(target)
        except json.JSONDecodeError:
            pass

    labels: list[str] = []
    with target.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"gold data in {target!s} is not valid JSON or JSONL (line {line_no}): {exc.msg}"
                ) from exc
            if not isinstance(record, dict) or "label" not in record:
                raise ValueError(
                    f"gold record on line {line_no} has no 'label' field"
                )
            label = record["label"]
            if not isinstance(label, str):
                raise ValueError(
                    f"gold label on line {line_no} is not a string: {label!r}"
                )
            labels.append(label)

    if not labels:
        raise ValueError(f"gold file {target!s} contains no records")
    return labels


def resolve_gold_labels_path(
    *,
    uploaded_gold_path: Path | None = None,
    user_gold_path: str | None = None,
    config_test_data_file: str | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve the active gold-labels file and return ``(path, notice_if_missing)``."""
    if uploaded_gold_path is not None and uploaded_gold_path.is_file():
        return uploaded_gold_path, None

    if user_gold_path is not None and user_gold_path.strip():
        found = _find_existing_file(user_gold_path)
        if found is not None:
            return found, None
        return None, f"Gold labels file not found locally: {user_gold_path.strip()}"

    if config_test_data_file is not None and config_test_data_file.strip():
        found = _find_existing_file(config_test_data_file)
        if found is not None:
            return found, None
        return None, (
            f"Configured test_data_file '{config_test_data_file}' was not found locally. "
            "Enter a local path or upload a gold-labels JSON/JSONL file to compute scores."
        )

    return None, (
        "No gold-labels file loaded. Link an inference config with a local test_data_file, "
        "or enter/upload a gold-labels JSON/JSONL file to compute metrics."
    )


def count_out_of_set_values(
    verdicts: Sequence[VerdictRecord],
) -> list[tuple[str, int]]:
    """Count distinct ``parsed`` values among ``out_of_set`` verdicts."""
    counts = Counter(
        str(v["parsed"])
        for v in verdicts
        if v.get("kind") == "out_of_set" and v.get("parsed") is not None
    )
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def is_anomalous_verdict_row(row: Mapping[str, Any]) -> bool:
    """Return True when a verdict review row is non-exact or mismatches gold."""
    if row.get("kind") != "exact":
        return True
    gold = row.get("gold")
    parsed = row.get("parsed")
    if gold is not None and gold != "—" and parsed != gold:
        return True
    return False


def build_verdict_review_rows(
    verdicts: Sequence[VerdictRecord],
    gold_labels: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build index-sorted rows for the Verdict Review table."""
    by_index: dict[int, VerdictRecord] = {}
    for item in verdicts:
        idx = item.get("index")
        if isinstance(idx, int):
            by_index[idx] = item

    rows: list[dict[str, Any]] = []
    gold_len = len(gold_labels) if gold_labels is not None else 0

    for idx, item in by_index.items():
        gold_val = (
            gold_labels[idx]
            if gold_labels is not None and 0 <= idx < gold_len
            else "—"
        )
        parsed_val = item.get("parsed")
        parsed_text = str(parsed_val) if parsed_val is not None else "—"
        kind = str(item.get("kind") or "unparsed")
        row = {
            "index": idx,
            "gold": gold_val,
            "raw": str(item.get("raw") or ""),
            "parsed": parsed_text,
            "kind": kind,
        }
        row["anomalous"] = is_anomalous_verdict_row(row)
        rows.append(row)

    if gold_labels is not None:
        for idx, gold_val in enumerate(gold_labels):
            if idx not in by_index:
                rows.append(
                    {
                        "index": idx,
                        "gold": gold_val,
                        "raw": "—",
                        "parsed": "—",
                        "kind": "unfinished",
                        "anomalous": True,
                    }
                )

    rows.sort(key=lambda r: int(r["index"]))
    return rows


def filter_verdict_rows(
    rows: Sequence[dict[str, Any]], *, anomalies_only: bool
) -> list[dict[str, Any]]:
    """Filter verdict review rows according to the anomalies-only toggle."""
    if not anomalies_only:
        return list(rows)
    return [row for row in rows if is_anomalous_verdict_row(row)]


def evaluate_inference_log(
    result: ParseResult,
    log_path: str | Path,
    *,
    settings: ConfigEvalSettings | None = None,
    gold_path: str | Path | None = None,
    gold_labels: Sequence[str] | None = None,
    gold_notice: str | None = None,
) -> InferenceEvaluationView:
    """Extract verdicts and optionally score an inference log."""
    active_settings = settings or ConfigEvalSettings()
    if result.get("family") != "inference":
        return InferenceEvaluationView(available=False, settings=active_settings)

    output_format = result.get("output_format")
    if output_format is None:
        return InferenceEvaluationView(
            available=True,
            output_format=None,
            settings=active_settings,
            gold_notice=(
                "No verdict output available: this inference run stopped before generation."
            ),
        )

    target = Path(log_path)
    if not target.is_file():
        return InferenceEvaluationView(
            available=True,
            output_format=output_format,
            settings=active_settings,
            error_message=f"Log file not found on disk: {target}",
        )

    warnings_out: list[str] = []
    try:
        source, verdicts = extract_verdicts_with_source(
            target,
            marker=active_settings.marker,
            labels=active_settings.labels,
            layout=output_format,
            warnings=warnings_out,
        )
    except (ValueError, OSError) as exc:
        return InferenceEvaluationView(
            available=True,
            output_format=output_format,
            settings=active_settings,
            error_message=str(exc),
        )

    out_of_set_counts = count_out_of_set_values(verdicts)

    loaded_gold: list[str] | None = (
        list(gold_labels) if gold_labels is not None else None
    )
    resolved_gold_path = str(gold_path) if gold_path is not None else None
    error_message: str | None = None

    if loaded_gold is None and gold_path is not None:
        try:
            loaded_gold = load_gold_labels_file(gold_path)
        except (ValueError, OSError) as exc:
            error_message = f"Failed to load gold labels: {exc}"

    review_rows = build_verdict_review_rows(verdicts, loaded_gold)

    score_report: ScoreReport | None = None
    if (
        loaded_gold is not None
        and error_message is None
        and TABLE_ONLY_SINGLE_TABLE not in warnings_out
    ):
        try:
            score_report = score_verdicts(
                verdicts,
                loaded_gold,
                marker=active_settings.marker,
                labels=active_settings.labels,
                source=source,
            )
        except ValueError as exc:
            error_message = str(exc)

    return InferenceEvaluationView(
        available=True,
        output_format=output_format,
        settings=active_settings,
        source=source,
        verdicts=verdicts,
        review_rows=review_rows,
        out_of_set_counts=out_of_set_counts,
        gold_path=resolved_gold_path,
        gold_count=len(loaded_gold) if loaded_gold is not None else None,
        gold_notice=gold_notice,
        warnings=warnings_out,
        error_message=error_message,
        score_report=score_report,
    )


_TONE_TAG_CLASSES: dict[str, str] = {
    "success": "info-tag tag-success",
    "info": "info-tag tag-info",
    "warning": "info-tag tag-warning",
    "error": "info-tag tag-error",
    "terminated": "info-tag tag-terminated",
    "neutral": "info-tag tag-neutral",
}

_FAMILY_TAG_CLASSES: dict[str, str] = {
    "fine_tune": "info-tag tag-success",
    "inference": "info-tag tag-info",
    "checkpoint_export": "info-tag tag-neutral",
    "unknown": "info-tag tag-neutral",
}

_TONE_DOT_CLASSES: dict[str, str] = {
    "success": "status-dot dot-success",
    "info": "status-dot dot-info",
    "warning": "status-dot dot-warning",
    "error": "status-dot dot-error",
    "terminated": "status-dot dot-terminated",
    "neutral": "status-dot dot-neutral",
}


class ResultsPanel:
    """NiceGUI Run Results tab controller: log parsing, verdict review, scoring dashboard, and SQLite persistence."""

    def __init__(
        self,
        container: Any,
        *,
        list_configs_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
        get_config_fn: Callable[[str], Any] | None = None,
        insert_run_fn: Callable[[dict[str, Any], str | None], str] | None = None,
        upsert_score_fn: Callable[[str, dict[str, Any]], None] | None = None,
        get_score_fn: Callable[[str], dict[str, Any] | None] | None = None,
        parse_fn: Callable[[str | Path], ParseResult] = parse_log_file,
        initial_result: ParseResult | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        from nicegui import ui

        self._ui = ui
        self.container = container
        self._db_path = db_path
        if list_configs_fn is not None:
            self._list_configs_fn = list_configs_fn
        elif db_path is not None:
            self._list_configs_fn = lambda: list_configs(db_path)
        else:
            self._list_configs_fn = None

        self._get_config_fn = get_config_fn or (lambda cid: get_config(cid, db_path))
        self._insert_run_fn = insert_run_fn or (
            lambda res, cid: insert_run(res, cid, db_path)
        )
        self._upsert_score_fn = upsert_score_fn or (
            lambda rid, rep: upsert_score(rid, rep, db_path)
        )
        self._get_score_fn = get_score_fn or (lambda rid: get_score(rid, db_path))
        self._parse_fn = parse_fn

        self.selected_config_id: str | None = None
        self.current_log_path: str = ""
        self.current_source_name: str | None = None
        self.current_summary: ResultSummaryView | None = None
        self.current_eval: InferenceEvaluationView | None = None
        self.active_alert_index: int = 0
        self.anomalies_only: bool = False
        self.include_recovered: bool = False

        self.uploaded_gold_path: Path | None = None
        self.uploaded_gold_name: str | None = None
        self._gold_path_user_edited: bool = False
        self._suppress_gold_input_change: bool = False
        self.last_saved_run_id: str | None = None
        self.last_markdown_report: str | None = None

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
                                on_change=self._on_config_selected,
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
                        # Quasar emits "finish" once every queued file is done; only then is
                        # it safe to reset (resetting earlier aborts in-flight uploads).
                        self.uploader.on("finish", self._on_uploads_finished)
                        # Uploaded files list rendered directly inside the dropdown box
                        self.uploaded_files_container = ui.column().classes(
                            "w-full gap-0"
                        )
                        # Alias kept for test compatibility
                        self.uploaded_file_row = self.uploaded_files_container

                self._render_uploaded_files_list()

                # 3. Summary Card Container
                self.summary_container = ui.column().classes("w-full gap-0")
                with self.summary_container:
                    # Persistent summary card and sub-containers updated in-place to eliminate DOM jumps
                    self.summary_card = ui.column().classes(
                        "w-full p-5 gap-3.5 rounded-md border-token surface-card"
                    )
                    with self.summary_card:
                        with ui.row().classes(
                            "w-full h-6 items-center justify-between gap-3 no-wrap"
                        ):
                            with ui.row().classes("items-center gap-2.5 no-wrap"):
                                ui.label("Run Summary").classes(
                                    "text-base font-bold tracking-tight leading-none"
                                )
                                # Inline spinner next to Run Summary so the right-side status never shifts
                                self.spinner_row = ui.row().classes(
                                    "items-center gap-1.5 no-wrap pointer-events-none"
                                )
                                with self.spinner_row:
                                    self.spinner = ui.spinner(
                                        "dots", size="xs", color="primary"
                                    )
                                    self.spinner_label = ui.label(
                                        "Parsing log file..."
                                    ).classes(
                                        "text-[11px] font-medium text-secondary-token hidden"
                                    )
                                self.spinner_row.set_visibility(False)

                            # Persistent header elements: never cleared/recreated so text never jumps
                            self.header_status_container = ui.row().classes(
                                "items-center gap-3 no-wrap shrink-0"
                            )
                            with self.header_status_container:
                                self.header_family_label = ui.label("").classes(
                                    "info-tag tag-neutral"
                                )
                                self.header_status_wrap = ui.row().classes(
                                    "items-center gap-1.5 no-wrap info-tag tag-neutral"
                                )
                                with self.header_status_wrap:
                                    self.header_status_dot = ui.element("span").classes(
                                        "status-dot dot-neutral"
                                    )
                                    self.header_status_label = ui.label("")

                        self.metrics_strip_row = ui.row().classes(
                            "w-full items-center h-9 py-2 divider-t divider-b no-wrap overflow-x-auto"
                        )
                        self._metric_slots: list[tuple[Any, Any, Any]] = []
                        with self.metrics_strip_row:
                            for _ in range(6):
                                m_row = ui.row().classes(
                                    "items-baseline gap-1.5 no-wrap shrink-0"
                                )
                                with m_row:
                                    m_lbl = ui.label("").classes(
                                        "text-xs font-medium text-secondary-token whitespace-nowrap"
                                    )
                                    m_val = ui.label("").classes(
                                        "text-xs font-bold tracking-tight whitespace-nowrap"
                                    )
                                m_row.set_visibility(False)
                                self._metric_slots.append((m_row, m_lbl, m_val))

                        with ui.column().classes("w-full gap-0"):
                            with ui.row().classes(
                                "w-full items-center justify-between py-2 divider-b text-xs font-medium text-secondary-token select-none no-wrap"
                            ):
                                ui.label("Pipeline Phase").classes("flex-grow")
                                with ui.row().classes(
                                    "items-center gap-6 no-wrap shrink-0"
                                ):
                                    ui.label("Line").classes(
                                        "w-20 pl-2.5 atera-col-header"
                                    )
                                    ui.label("Status").classes(
                                        "w-28 pl-2.5 atera-col-header"
                                    )
                            self.phases_body_container = ui.column().classes(
                                "w-full gap-0"
                            )
                            self._phase_slots: list[tuple[Any, Any, Any, Any, Any]] = []
                            with self.phases_body_container:
                                for _ in range(4):
                                    p_row = ui.row().classes(
                                        "w-full items-center justify-between py-3.5 no-wrap gap-4 phase-row divider-b"
                                    )
                                    with p_row:
                                        with ui.row().classes(
                                            "items-center gap-2.5 no-wrap min-w-0 flex-grow"
                                        ):
                                            p_dot = ui.element("span").classes(
                                                "status-dot dot-neutral"
                                            )
                                            p_name = ui.label("").classes(
                                                "text-sm font-semibold tracking-tight truncate"
                                            )
                                        with ui.row().classes(
                                            "items-center gap-6 no-wrap shrink-0"
                                        ):
                                            p_line = ui.label("—").classes(
                                                "w-20 pl-2.5 text-xs font-medium text-secondary-token whitespace-nowrap"
                                            )
                                            with ui.row().classes(
                                                "w-28 pl-2.5 items-center no-wrap"
                                            ):
                                                p_status = ui.label("").classes(
                                                    "info-tag tag-neutral"
                                                )
                                    p_row.set_visibility(False)
                                    self._phase_slots.append(
                                        (p_row, p_dot, p_name, p_line, p_status)
                                    )

                        self.alert_container = ui.column().classes("w-full gap-0")
                        with self.alert_container:
                            self.alert_box = ui.column().classes(
                                "w-full p-3.5 gap-1.5 diagnostic-box diagnostic-box-error"
                            )
                            with self.alert_box:
                                with ui.row().classes(
                                    "w-full items-center justify-between no-wrap gap-2"
                                ):
                                    with ui.row().classes(
                                        "items-center gap-2 no-wrap min-w-0"
                                    ):
                                        self.alert_icon = ui.icon(
                                            "error_outline", color="negative"
                                        ).classes("text-base shrink-0")
                                        self.alert_title_label = ui.label("").classes(
                                            "text-xs font-bold tracking-tight text-red-600 dark:text-red-400 truncate"
                                        )
                                    self.alert_nav_row = ui.row().classes(
                                        "items-center gap-1 no-wrap shrink-0"
                                    )
                                    with self.alert_nav_row:
                                        self.alert_count_label = ui.label("").classes(
                                            "text-[11px] font-semibold text-secondary-token"
                                        )
                                        self.alert_prev_btn = (
                                            ui.button(
                                                icon="keyboard_arrow_up",
                                                on_click=lambda: self._on_cycle_error(
                                                    -1
                                                ),
                                            )
                                            .props("flat dense round size=xs color=negative")
                                            .tooltip("Previous issue in this log")
                                        )
                                        self.alert_next_btn = (
                                            ui.button(
                                                icon="keyboard_arrow_down",
                                                on_click=lambda: self._on_cycle_error(
                                                    1
                                                ),
                                            )
                                            .props("flat dense round size=xs color=negative")
                                            .tooltip("Next issue in this log")
                                        )
                                self.alert_message_label = ui.label("").classes(
                                    "text-xs font-semibold break-words leading-relaxed"
                                )
                                self.alert_detail_label = ui.label("").classes(
                                    "text-xs font-medium text-secondary-token break-words leading-relaxed"
                                )
                                self.alert_notes_label = ui.label("").classes(
                                    "text-[11px] font-medium text-secondary-token pt-0.5"
                                )
                        self.alert_container.set_visibility(False)

                        # Save & Export Action Row inside Summary Card footer
                        with ui.row().classes(
                            "w-full items-center justify-between gap-2 pt-2 divider-t flex-wrap"
                        ):
                            with ui.row().classes("items-center gap-2 no-wrap"):
                                self.save_run_button = (
                                    ui.button(
                                        "Save Run",
                                        icon="save",
                                        on_click=self.save_current_run,
                                    )
                                    .props("unelevated dense no-caps size=sm color=primary")
                                    .classes("px-3 text-xs font-medium")
                                )
                                self.export_md_button = (
                                    ui.button(
                                        "Export Markdown",
                                        icon="description",
                                        on_click=self.export_markdown_report,
                                    )
                                    .props("outlined dense no-caps size=sm color=primary")
                                    .classes("px-2.5 text-xs font-medium")
                                )
                                self.export_md_button.set_visibility(False)
                            self.save_status_label = ui.label("").classes(
                                "text-[11px] font-medium text-secondary-token"
                            )
                            self.save_status_label.set_visibility(False)

                # 4. Inference Evaluation & M2.0 Dashboard (shown only for inference logs)
                self.eval_section = ui.column().classes("w-full gap-3")
                with self.eval_section:
                    # 4a. Gold Labels & Extraction Settings Card
                    self.gold_card = ui.column().classes(
                        "w-full p-4 gap-2.5 rounded-md border-token surface-card"
                    )
                    with self.gold_card:
                        with ui.row().classes(
                            "w-full items-center justify-between gap-2 flex-wrap"
                        ):
                            ui.label("Gold Labels & Scoring Context").classes(
                                "text-sm font-bold tracking-tight"
                            )
                            self.eval_context_label = ui.label("").classes(
                                "text-[11px] font-medium text-secondary-token"
                            )

                        with ui.column().classes("w-full min-w-0 gap-1"):
                            ui.label("Local Gold Labels Path (.json / .jsonl)").classes(
                                "text-xs font-medium text-secondary-token select-none"
                            )
                            with ui.row().classes("w-full items-center gap-2 no-wrap"):
                                self.gold_path_input = (
                                    ui.input(
                                        placeholder="e.g. data/testData.json",
                                        on_change=self._on_gold_path_input_changed,
                                    )
                                    .props("outlined dense clearable")
                                    .classes("flex-grow min-w-0")
                                )
                                self.gold_load_button = (
                                    ui.button(
                                        "Score",
                                        icon="play_arrow",
                                        on_click=self._on_gold_load_clicked,
                                    )
                                    .props("unelevated dense no-caps size=sm color=primary")
                                    .classes("px-3 text-xs font-medium shrink-0")
                                )

                        with ui.row().classes(
                            "w-full items-center justify-between gap-2 flex-wrap"
                        ):
                            self.gold_uploader = (
                                ui.upload(
                                    label="Or upload gold labels (.json / .jsonl)",
                                    auto_upload=True,
                                    multiple=False,
                                    on_upload=self._on_gold_file_uploaded,
                                )
                                .props("flat dense accept=.json,.jsonl")
                                .classes("w-full compact-uploader")
                            )
                            self.clear_gold_upload_btn = (
                                ui.button(
                                    "Clear uploaded gold file",
                                    icon="close",
                                    on_click=self.clear_uploaded_gold_file,
                                )
                                .props("flat dense no-caps size=xs color=negative")
                            )
                            self.clear_gold_upload_btn.set_visibility(False)

                        self.gold_status_box = ui.row().classes(
                            "w-full items-start gap-2 px-3 py-2 rounded-md surface-subtle no-wrap"
                        )
                        with self.gold_status_box:
                            self.gold_status_icon = ui.icon(
                                "info_outline", color="primary"
                            ).classes("text-sm shrink-0 mt-0.5")
                            self.gold_status_label = ui.label("").classes(
                                "text-xs text-secondary-token min-w-0 break-all"
                            )

                    # 4b. Special Banners: TABLE_ONLY_SINGLE_TABLE & Incomplete Run
                    self.single_table_banner = ui.column().classes(
                        "w-full p-3.5 gap-1 diagnostic-box diagnostic-box-warning"
                    )
                    with self.single_table_banner:
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.icon("warning_amber", color="warning").classes(
                                "text-base shrink-0"
                            )
                            ui.label("Single Script Table Only").classes(
                                "text-xs font-bold tracking-tight text-amber-600 dark:text-amber-400"
                            )
                        self.single_table_banner_label = ui.label(
                            "Single script table detected: this table_only run was killed before printing its final verdict table, so no final verdicts can be asserted."
                        ).classes("text-xs text-secondary-token leading-relaxed")
                    self.single_table_banner.set_visibility(False)

                    self.incomplete_banner = ui.column().classes(
                        "w-full p-3.5 gap-1 diagnostic-box diagnostic-box-warning"
                    )
                    with self.incomplete_banner:
                        with ui.row().classes("items-center gap-2 no-wrap"):
                            ui.icon("hourglass_bottom", color="warning").classes(
                                "text-base shrink-0"
                            )
                            ui.label("Incomplete Run").classes(
                                "text-xs font-bold tracking-tight text-amber-600 dark:text-amber-400"
                            )
                        self.incomplete_banner_label = ui.label("").classes(
                            "text-xs font-medium leading-relaxed"
                        )
                    self.incomplete_banner.set_visibility(False)

                    # 4c. Metrics Dashboard Card
                    self.dashboard_card = ui.column().classes(
                        "w-full p-5 gap-3.5 rounded-md border-token surface-card"
                    )
                    with self.dashboard_card:
                        with ui.row().classes(
                            "w-full items-center justify-between gap-2 flex-wrap"
                        ):
                            with ui.row().classes("items-center gap-2 no-wrap"):
                                ui.label("Metrics Dashboard").classes(
                                    "text-base font-bold tracking-tight"
                                )
                                self.dashboard_source_tag = ui.label("").classes(
                                    "info-tag tag-info"
                                )
                            self.include_recovered_switch = (
                                ui.switch(
                                    "Include recovered",
                                    value=False,
                                    on_change=lambda e: self.set_include_recovered(
                                        bool(e.value)
                                    ),
                                )
                                .props("dense color=primary")
                                .classes("text-xs font-medium")
                            )

                        # 2x2 Headline Metric Cards (Accuracy, Macro-F1, Empty Rate, Invalid Rate)
                        with ui.grid(columns=2).classes("w-full gap-2.5"):
                            with ui.column().classes(
                                "p-3 gap-0.5 rounded-md surface-subtle"
                            ):
                                ui.label("Accuracy").classes(
                                    "text-[11px] font-semibold uppercase tracking-wider text-secondary-token"
                                )
                                self.accuracy_value_label = ui.label("—").classes(
                                    "text-lg font-bold tracking-tight"
                                )
                                self.accuracy_sub_label = ui.label("").classes(
                                    "text-[11px] text-secondary-token"
                                )

                            with ui.column().classes(
                                "p-3 gap-0.5 rounded-md surface-subtle"
                            ):
                                ui.label("Macro-F1").classes(
                                    "text-[11px] font-semibold uppercase tracking-wider text-secondary-token"
                                )
                                self.macro_f1_value_label = ui.label("—").classes(
                                    "text-lg font-bold tracking-tight"
                                )
                                self.macro_f1_sub_label = ui.label("").classes(
                                    "text-[11px] text-secondary-token"
                                )

                            with ui.column().classes(
                                "p-3 gap-0.5 rounded-md surface-subtle"
                            ):
                                ui.label("Empty Rate").classes(
                                    "text-[11px] font-semibold uppercase tracking-wider text-secondary-token"
                                )
                                self.empty_rate_value_label = ui.label("—").classes(
                                    "text-lg font-bold tracking-tight"
                                )
                                self.empty_rate_sub_label = ui.label("").classes(
                                    "text-[11px] text-secondary-token"
                                )

                            with ui.column().classes(
                                "p-3 gap-0.5 rounded-md surface-subtle"
                            ):
                                ui.label("Invalid Rate").classes(
                                    "text-[11px] font-semibold uppercase tracking-wider text-secondary-token"
                                )
                                self.invalid_rate_value_label = ui.label("—").classes(
                                    "text-lg font-bold tracking-tight"
                                )
                                self.invalid_rate_sub_label = ui.label("").classes(
                                    "text-[11px] text-secondary-token"
                                )

                        # Six Verdict Kind Counts Strip
                        self.kind_counts_row = ui.row().classes(
                            "w-full items-center gap-3 py-2 divider-t divider-b flex-wrap text-xs"
                        )

                        # Per-Class Precision / Recall / F1
                        with ui.column().classes("w-full gap-1.5"):
                            ui.label("Per-Class Precision / Recall / F1").classes(
                                "text-xs font-semibold text-secondary-token"
                            )
                            self.per_class_container = ui.column().classes(
                                "w-full overflow-x-auto"
                            )

                        # 3x4 Confusion Matrix (Gold x Predicted + __INVALID__)
                        with ui.column().classes("w-full gap-1.5"):
                            ui.label("Confusion Matrix (Gold vs Predicted)").classes(
                                "text-xs font-semibold text-secondary-token"
                            )
                            self.confusion_container = ui.column().classes(
                                "w-full overflow-x-auto"
                            )

                        # Out-of-Set Value Counts
                        with ui.column().classes("w-full gap-1.5"):
                            ui.label("Out-of-Set Verdict Counts").classes(
                                "text-xs font-semibold text-secondary-token"
                            )
                            self.out_of_set_container = ui.row().classes(
                                "w-full items-center gap-2 flex-wrap"
                            )
                    self.dashboard_card.set_visibility(False)

                    # 4d. Out-of-set summary when verdicts exist without ScoreReport
                    self.unscored_oos_card = ui.column().classes(
                        "w-full p-4 gap-1.5 rounded-md border-token surface-card"
                    )
                    with self.unscored_oos_card:
                        ui.label("Out-of-Set Verdict Counts").classes(
                            "text-xs font-semibold text-secondary-token"
                        )
                        self.unscored_oos_container = ui.row().classes(
                            "w-full items-center gap-2 flex-wrap"
                        )
                    self.unscored_oos_card.set_visibility(False)

                    # 4e. Verdict Review Table Card (paginated ui.table + Anomalies only switch)
                    self.verdict_table_card = ui.column().classes(
                        "w-full p-5 gap-3 rounded-md border-token surface-card"
                    )
                    with self.verdict_table_card:
                        with ui.row().classes(
                            "w-full items-center justify-between gap-2 flex-wrap"
                        ):
                            with ui.row().classes("items-center gap-2 no-wrap"):
                                ui.label("Verdict Review").classes(
                                    "text-base font-bold tracking-tight"
                                )
                                self.verdict_count_label = ui.label("").classes(
                                    "text-[11px] font-medium text-secondary-token"
                                )
                            self.anomalies_switch = (
                                ui.switch(
                                    "Anomalies only",
                                    value=False,
                                    on_change=lambda e: self.set_anomalies_only(
                                        bool(e.value)
                                    ),
                                )
                                .props("dense color=primary")
                                .classes("text-xs font-medium")
                            )

                        self.verdict_table = (
                            ui.table(
                                columns=[
                                    {
                                        "name": "index",
                                        "label": "Index",
                                        "field": "index",
                                        "sortable": True,
                                        "align": "left",
                                    },
                                    {
                                        "name": "gold",
                                        "label": "Gold",
                                        "field": "gold",
                                        "sortable": True,
                                        "align": "left",
                                    },
                                    {
                                        "name": "raw",
                                        "label": "Raw",
                                        "field": "raw",
                                        "align": "left",
                                    },
                                    {
                                        "name": "parsed",
                                        "label": "Parsed",
                                        "field": "parsed",
                                        "sortable": True,
                                        "align": "left",
                                    },
                                    {
                                        "name": "kind",
                                        "label": "Kind",
                                        "field": "kind",
                                        "sortable": True,
                                        "align": "left",
                                    },
                                ],
                                rows=[],
                                row_key="index",
                                pagination=10,
                            )
                            .props("flat dense")
                            .classes("w-full surface-card border-token")
                        )
                        self.verdict_table.add_slot(
                            "body-cell-raw",
                            '<q-td :props="props" class="max-w-[160px] truncate mono text-xs" :title="props.row.raw">{{ props.row.raw }}</q-td>',
                        )
                        self.verdict_table.add_slot(
                            "body-cell-kind",
                            """
                            <q-td :props="props">
                                <span :class="props.value === 'exact' ? 'info-tag tag-success' : (props.value === 'recovered' ? 'info-tag tag-info' : (props.value === 'out_of_set' || props.value === 'unfinished' ? 'info-tag tag-warning' : 'info-tag tag-error'))">
                                    {{ props.value }}
                                </span>
                            </q-td>
                            """,
                        )
                    self.verdict_table_card.set_visibility(False)

                self.eval_section.set_visibility(False)

                # Markdown Report Preview Dialog
                with (
                    ui.dialog() as self.md_dialog,
                    ui.card()
                    .props("flat")
                    .classes("p-5 gap-3 w-full max-w-2xl rounded-lg surface-card"),
                ):
                    with ui.row().classes(
                        "w-full items-center justify-between no-wrap"
                    ):
                        ui.label("Markdown Score Report").classes(
                            "text-sm font-bold tracking-tight"
                        )
                        ui.button(
                            icon="close", on_click=self.md_dialog.close
                        ).props("flat dense round size=sm")
                    self.md_code = ui.code("", language="markdown").classes(
                        "prompt-preview-box w-full text-xs rounded-md border-token max-h-[420px] overflow-y-auto overflow-x-auto"
                    )
                    with ui.row().classes("w-full justify-end gap-2"):
                        ui.button(
                            "Download .md",
                            icon="download",
                            on_click=self._download_markdown_report,
                        ).props("unelevated dense no-caps size=sm color=primary")
                        ui.button(
                            "Close", on_click=self.md_dialog.close
                        ).props("flat dense no-caps size=sm")

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

    @property
    def current_score_report(self) -> ScoreReport | None:
        """Return the active ``ScoreReport`` when the current inference log is scored."""
        return self.current_eval.score_report if self.current_eval is not None else None

    def refresh_config_options(self) -> None:
        """Refresh the Linked Configuration dropdown from SQLite."""
        options = self._build_config_options()
        self.config_select.options = options
        if self.selected_config_id and self.selected_config_id not in options:
            self.selected_config_id = None
            self.config_select.value = None
        self.config_select.update()

    def _resolve_active_settings(self) -> ConfigEvalSettings:
        cfg = None
        if self.selected_config_id:
            try:
                cfg = self._get_config_fn(self.selected_config_id)
            except Exception:
                cfg = None
        return resolve_eval_settings(cfg, config_id=self.selected_config_id)

    def _sync_gold_input_from_config(self, settings: ConfigEvalSettings) -> None:
        if self._gold_path_user_edited:
            return
        self._suppress_gold_input_change = True
        try:
            self.gold_path_input.value = settings.test_data_file or ""
        finally:
            self._suppress_gold_input_change = False

    def _current_gold_resolution(
        self, settings: ConfigEvalSettings
    ) -> tuple[Path | None, str | None]:
        user_val = (
            str(self.gold_path_input.value or "").strip()
            if self._gold_path_user_edited
            else None
        )
        return resolve_gold_labels_path(
            uploaded_gold_path=self.uploaded_gold_path,
            user_gold_path=user_val,
            config_test_data_file=settings.test_data_file,
        )

    async def _on_config_selected(self, e: Any) -> None:
        self.selected_config_id = e.value if hasattr(e, "value") else self.config_select.value
        settings = self._resolve_active_settings()
        self._sync_gold_input_from_config(settings)
        self.last_saved_run_id = None
        self.save_status_label.set_visibility(False)
        if (
            self.current_summary is not None
            and self.current_summary.family == "inference"
        ):
            await self.refresh_evaluation_async()

    def select_config(self, config_id: str | None) -> InferenceEvaluationView | None:
        """Synchronous helper to link a config and refresh inference evaluation."""
        self.selected_config_id = config_id
        self.config_select.value = config_id
        settings = self._resolve_active_settings()
        self._sync_gold_input_from_config(settings)
        self.last_saved_run_id = None
        self.save_status_label.set_visibility(False)
        return self.refresh_evaluation()

    def _on_gold_path_input_changed(self, e: Any) -> None:
        if self._suppress_gold_input_change:
            return
        val = str(e.value if hasattr(e, "value") else self.gold_path_input.value or "").strip()
        self._gold_path_user_edited = bool(val)

    async def _on_gold_load_clicked(self) -> None:
        val = str(self.gold_path_input.value or "").strip()
        self._gold_path_user_edited = bool(val)
        await self.refresh_evaluation_async()

    async def _on_gold_file_uploaded(self, e: Any) -> None:
        file_obj = getattr(e, "file", None)
        orig_name = str(
            getattr(file_obj, "name", None) or getattr(e, "name", "gold.json")
        )
        suffix = Path(orig_name).suffix or ".json"
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

        if self.uploaded_gold_path is not None:
            try:
                self.uploaded_gold_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.uploaded_gold_path = tmp_path
        self.uploaded_gold_name = orig_name
        self.clear_gold_upload_btn.set_visibility(True)
        try:
            self.gold_uploader.reset()
        except Exception:
            pass
        await self.refresh_evaluation_async()

    def clear_uploaded_gold_file(self) -> InferenceEvaluationView | None:
        """Remove an uploaded gold-labels temp file and re-score with fallback source."""
        if self.uploaded_gold_path is not None:
            try:
                self.uploaded_gold_path.unlink(missing_ok=True)
            except OSError:
                pass
        self.uploaded_gold_path = None
        self.uploaded_gold_name = None
        self.clear_gold_upload_btn.set_visibility(False)
        return self.refresh_evaluation()

    def set_gold_labels_path(
        self, path: str | Path | None
    ) -> InferenceEvaluationView | None:
        """Synchronous helper to set a local gold-labels path and re-score."""
        text = str(path).strip() if path is not None else ""
        self._suppress_gold_input_change = True
        try:
            self.gold_path_input.value = text
        finally:
            self._suppress_gold_input_change = False
        self._gold_path_user_edited = bool(text)
        return self.refresh_evaluation()

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
                eval_view=self.current_eval,
                preserve_save_status=True,
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
                eval_view=self.current_eval,
                preserve_save_status=True,
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
            for entry in self.uploaded_files:
                is_active = self.current_log_path == str(entry.temp_path)
                row_bg = "bg-blue-500/5" if is_active else ""
                with ui.row().classes(
                    f"w-full items-center justify-between px-3.5 py-2.5 no-wrap phase-row divider-t {row_bg}"
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
                            ui.label("Active").classes("info-tag tag-success")
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

        # Do NOT reset the uploader here: with multiple files, other uploads may still be
        # in flight and reset() would abort them. See _on_uploads_finished.
        self._render_uploaded_files_list()
        await self.parse_async(tmp_path, display_name=orig_name)
        self._render_uploaded_files_list()

    def _on_uploads_finished(self, _: Any = None) -> None:
        """Clear Quasar's internal queue after the whole batch finished uploading."""
        try:
            self.uploader.reset()
        except Exception:
            pass

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

    def _parse_and_evaluate_sync(
        self,
        target: Path,
        settings: ConfigEvalSettings,
        gold_path: Path | None,
        gold_notice: str | None,
    ) -> tuple[ParseResult, InferenceEvaluationView | None]:
        result = self._parse_fn(target)
        if result.get("family") != "inference":
            return result, InferenceEvaluationView(
                available=False, settings=settings
            )
        eval_view = evaluate_inference_log(
            result,
            target,
            settings=settings,
            gold_path=gold_path,
            gold_notice=gold_notice,
        )
        return result, eval_view

    async def parse_async(
        self, path: str | Path, *, display_name: str | None = None
    ) -> ResultSummaryView:
        """Parse ``path`` and evaluate inference verdicts off the event loop with ``run.io_bound``."""
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
            settings = self._resolve_active_settings()
            gold_path, gold_notice = self._current_gold_resolution(settings)
            result, eval_view = await run.io_bound(
                self._parse_and_evaluate_sync,
                target,
                settings,
                gold_path,
                gold_notice,
            )
            return self.render_result(
                result, source_name=shown_name, eval_view=eval_view
            )
        finally:
            self.spinner_row.set_visibility(False)

    def load_file(
        self, path: str | Path, *, display_name: str | None = None
    ) -> ResultSummaryView:
        """Synchronous helper to parse ``path``, evaluate inference verdicts, and update the panel."""
        target = Path(path)
        self.current_log_path = str(target)
        shown_name = display_name or target.name
        settings = self._resolve_active_settings()
        gold_path, gold_notice = self._current_gold_resolution(settings)
        result, eval_view = self._parse_and_evaluate_sync(
            target, settings, gold_path, gold_notice
        )
        return self.render_result(
            result, source_name=shown_name, eval_view=eval_view
        )

    def refresh_evaluation(self) -> InferenceEvaluationView | None:
        """Synchronously re-extract verdicts and re-score the current inference log."""
        if self.current_summary is None:
            return None
        result = self.current_summary.raw
        settings = self._resolve_active_settings()
        if result.get("family") != "inference":
            eval_view = InferenceEvaluationView(available=False, settings=settings)
            self._render_evaluation(eval_view)
            return eval_view
        log_path = result.get("source_path") or self.current_log_path
        gold_path, gold_notice = self._current_gold_resolution(settings)
        eval_view = evaluate_inference_log(
            result,
            log_path,
            settings=settings,
            gold_path=gold_path,
            gold_notice=gold_notice,
        )
        self._render_evaluation(eval_view)
        return eval_view

    async def refresh_evaluation_async(self) -> InferenceEvaluationView | None:
        """Re-extract verdicts and re-score the current inference log via ``run.io_bound``."""
        from nicegui import run

        if self.current_summary is None:
            return None
        result = self.current_summary.raw
        settings = self._resolve_active_settings()
        if result.get("family") != "inference":
            eval_view = InferenceEvaluationView(available=False, settings=settings)
            self._render_evaluation(eval_view)
            return eval_view
        log_path = result.get("source_path") or self.current_log_path
        gold_path, gold_notice = self._current_gold_resolution(settings)
        self.spinner_label.text = "Scoring verdicts..."
        self.spinner_row.set_visibility(True)
        try:
            eval_view = await run.io_bound(
                evaluate_inference_log,
                result,
                log_path,
                settings=settings,
                gold_path=gold_path,
                gold_notice=gold_notice,
            )
            self._render_evaluation(eval_view)
            return eval_view
        finally:
            self.spinner_row.set_visibility(False)

    def set_anomalies_only(self, enabled: bool) -> list[dict[str, Any]]:
        """Toggle the Verdict Review table between all records and anomalies only."""
        self.anomalies_only = bool(enabled)
        if self.anomalies_switch.value != self.anomalies_only:
            self.anomalies_switch.value = self.anomalies_only
        return self._refresh_verdict_table()

    @property
    def exact_only(self) -> bool:
        """True when the Metrics Dashboard is showing strict Exact Match metrics."""
        return not self.include_recovered

    def set_include_recovered(self, enabled: bool) -> None:
        """Switch the Metrics Dashboard between Exact Match (default) and Inclusive metrics."""
        self.include_recovered = bool(enabled)
        if self.include_recovered_switch.value != self.include_recovered:
            self.include_recovered_switch.value = self.include_recovered
        self._refresh_summary_metrics_strip()
        if self.current_eval is not None and self.current_eval.score_report is not None:
            self._update_incomplete_banner(self.current_eval.score_report)
            self._render_metrics_dashboard(self.current_eval)

    def set_exact_only(self, enabled: bool) -> None:
        """Backward-compatible wrapper around :meth:`set_include_recovered`."""
        self.set_include_recovered(not bool(enabled))

    def _refresh_verdict_table(self) -> list[dict[str, Any]]:
        if self.current_eval is None:
            self.verdict_table.rows = []
            self.verdict_table.update()
            self.verdict_count_label.text = "0 records"
            return []
        all_rows = self.current_eval.review_rows
        shown_rows = filter_verdict_rows(
            all_rows, anomalies_only=self.anomalies_only
        )
        self.verdict_table.rows = shown_rows
        self.verdict_table.update()
        if self.anomalies_only:
            self.verdict_count_label.text = (
                f"{len(shown_rows):,} anomalies of {len(all_rows):,} records"
            )
        else:
            self.verdict_count_label.text = f"{len(all_rows):,} records"
        return shown_rows

    def _render_out_of_set_badges(
        self, container: Any, counts: Sequence[tuple[str, int]]
    ) -> None:
        ui = self._ui
        container.clear()
        with container:
            if not counts:
                ui.label("None (0 out-of-set verdicts)").classes(
                    "text-xs text-secondary-token"
                )
                return
            for value, count in counts:
                with ui.row().classes(
                    "items-center gap-1.5 px-2.5 py-1 rounded-md surface-subtle border-token no-wrap"
                ):
                    ui.label(value).classes("text-xs font-medium mono")
                    ui.label(f"×{count}").classes("info-tag tag-warning")

    def _render_metrics_dashboard(self, eval_view: InferenceEvaluationView) -> None:
        ui = self._ui
        report = eval_view.score_report
        if report is None:
            self.dashboard_card.set_visibility(False)
            return

        self.dashboard_card.set_visibility(True)
        self.dashboard_source_tag.text = f"Source: {report['source']}"

        active_block = (
            report["inclusive_metrics"]
            if self.include_recovered
            else report["exact_metrics"]
        )
        alt_block = (
            report["exact_metrics"]
            if self.include_recovered
            else report["inclusive_metrics"]
        )
        alt_prefix = "Exact" if self.include_recovered else "Inclusive"

        self.accuracy_value_label.text = f"{active_block['accuracy'] * 100:.2f}%"
        self.accuracy_sub_label.text = (
            f"{alt_prefix}: {alt_block['accuracy'] * 100:.2f}%"
        )

        self.macro_f1_value_label.text = f"{active_block['macro_f1']:.4f}"
        self.macro_f1_sub_label.text = (
            f"{alt_prefix}: {alt_block['macro_f1']:.4f} · Weighted: {active_block['weighted_f1']:.4f}"
        )

        records = report["records"]
        self.empty_rate_value_label.text = f"{report['empty_rate'] * 100:.2f}%"
        self.empty_rate_sub_label.text = f"{report['empty']:,} / {records:,} empty"

        invalid_total = (
            report["empty"]
            + report["out_of_set"]
            + report["unparsed"]
            + report["unfinished"]
        )
        self.invalid_rate_value_label.text = f"{report['invalid_rate'] * 100:.2f}%"
        sub_parts = [f"{invalid_total:,} / {records:,} invalid"]
        if report["unfinished"] > 0:
            sub_parts.append(f"({report['unfinished']:,} unfinished)")
        if not self.include_recovered and report["recovered"] > 0:
            sub_parts.append(f"· {report['recovered']:,} recovered in exact __INVALID__")
        self.invalid_rate_sub_label.text = " ".join(sub_parts)

        # Six kind counts strip
        self.kind_counts_row.clear()
        with self.kind_counts_row:
            kind_items = [
                ("Exact", report["exact"], "tag-success"),
                ("Recovered", report["recovered"], "tag-info"),
                ("Empty", report["empty"], "tag-error" if report["empty"] else "tag-neutral"),
                ("Out-of-Set", report["out_of_set"], "tag-warning" if report["out_of_set"] else "tag-neutral"),
                ("Unparsed", report["unparsed"], "tag-error" if report["unparsed"] else "tag-neutral"),
                ("Unfinished", report["unfinished"], "tag-warning" if report["unfinished"] else "tag-neutral"),
            ]
            for label, count, cls in kind_items:
                with ui.row().classes("items-baseline gap-1 no-wrap"):
                    ui.label(f"{label}:").classes("text-secondary-token font-medium")
                    ui.label(f"{count:,}").classes(f"font-bold {cls}")

        # Per-class P / R / F1 table
        self.per_class_container.clear()
        confusion = active_block["confusion"]
        with self.per_class_container:
            with ui.column().classes("w-full gap-0 rounded-md border-token overflow-hidden"):
                with ui.row().classes(
                    "w-full items-center px-3 py-1.5 surface-subtle divider-b text-[11px] font-semibold text-secondary-token no-wrap"
                ):
                    ui.label("Class").classes("flex-1 min-w-[90px]")
                    ui.label("Precision").classes("w-16 text-right")
                    ui.label("Recall").classes("w-16 text-right")
                    ui.label("F1").classes("w-16 text-right")
                    ui.label("Support").classes("w-16 text-right")
                for label in report["labels"]:
                    support = sum(confusion[label].values())
                    with ui.row().classes(
                        "w-full items-center px-3 py-2 divider-b phase-row text-xs no-wrap"
                    ):
                        ui.label(label).classes("flex-1 min-w-[90px] font-semibold truncate")
                        ui.label(f"{active_block['precision'][label]:.4f}").classes(
                            "w-16 text-right mono"
                        )
                        ui.label(f"{active_block['recall'][label]:.4f}").classes(
                            "w-16 text-right mono"
                        )
                        ui.label(f"{active_block['f1'][label]:.4f}").classes(
                            "w-16 text-right mono font-semibold"
                        )
                        ui.label(f"{support:,}").classes(
                            "w-16 text-right mono text-secondary-token"
                        )

        # 3x4 Confusion Matrix (Gold rows x Predicted columns + __INVALID__)
        self.confusion_container.clear()
        pred_columns = [*report["labels"], INVALID_LABEL]
        with self.confusion_container:
            with ui.column().classes(
                "w-full min-w-[420px] gap-0 rounded-md border-token overflow-hidden"
            ):
                with ui.row().classes(
                    "w-full items-center px-3 py-1.5 surface-subtle divider-b text-[11px] font-semibold text-secondary-token no-wrap"
                ):
                    ui.label("Gold \\ Pred").classes("flex-1 min-w-[90px]")
                    for col in pred_columns:
                        col_label = "Invalid" if col == INVALID_LABEL else col
                        ui.label(col_label).classes(
                            "w-20 text-right truncate"
                        ).tooltip(col)
                for gold_label in report["labels"]:
                    row_counts = confusion[gold_label]
                    with ui.row().classes(
                        "w-full items-center px-3 py-2 divider-b phase-row text-xs no-wrap"
                    ):
                        ui.label(gold_label).classes(
                            "flex-1 min-w-[90px] font-semibold truncate"
                        )
                        for col in pred_columns:
                            val = row_counts.get(col, 0)
                            if col == gold_label and val > 0:
                                val_cls = "w-20 text-right mono font-bold tag-success"
                            elif col == INVALID_LABEL and val > 0:
                                val_cls = "w-20 text-right mono font-semibold tag-error"
                            elif val > 0:
                                val_cls = "w-20 text-right mono font-semibold tag-warning"
                            else:
                                val_cls = "w-20 text-right mono text-secondary-token"
                            ui.label(f"{val:,}").classes(val_cls)

        # Out-of-set value counts
        self._render_out_of_set_badges(
            self.out_of_set_container, eval_view.out_of_set_counts
        )

    def _scored_records_count(self, report: ScoreReport | None) -> int | None:
        if report is None:
            return None
        if self.include_recovered:
            return int(report["exact"]) + int(report["recovered"])
        return int(report["exact"])

    def _refresh_summary_metrics_strip(self) -> None:
        summary = self.current_summary
        if summary is None:
            return
        strip_items: list[tuple[str, str]] = []
        scored_count = self._scored_records_count(self.current_score_report)
        for label, val in summary.progress_metrics:
            if label == "Records used for scoring":
                val = f"{scored_count:,}" if scored_count is not None else "-"
            strip_items.append((label, val))
        if summary.output_format:
            strip_items.append(("Format", summary.output_format.upper()))
        if summary.duration_text:
            strip_items.append(("Duration", summary.duration_text))
        if not strip_items and summary.script:
            strip_items.append(("Script", summary.script))
        if not strip_items:
            strip_items.append(("Mode", summary.family_label))
            strip_items.append(("Outcome", summary.status_label))

        num_metrics = min(len(strip_items), len(self._metric_slots))
        for idx, (m_row, m_lbl, m_val) in enumerate(self._metric_slots):
            if idx < num_metrics:
                label, val = strip_items[idx]
                sep_cls = "divider-r pr-4 mr-4" if idx < num_metrics - 1 else ""
                m_row.classes(
                    replace=f"items-baseline gap-1.5 no-wrap shrink-0 {sep_cls}"
                )
                m_lbl.text = label
                m_val.text = val
                m_row.set_visibility(True)
            else:
                m_row.set_visibility(False)

    def _update_incomplete_banner(self, report: ScoreReport | None) -> None:
        if report is not None and not report["complete"]:
            completed = report["records"] - report["unfinished"]
            if self.include_recovered:
                scored = report["exact"] + report["recovered"]
                mode_desc = f"{scored:,} exact + recovered verdicts"
            else:
                scored = report["exact"]
                mode_desc = f"{scored:,} exact verdicts"
            self.incomplete_banner_label.text = (
                f"Incomplete run: {report['unfinished']:,} of {report['records']:,} records unfinished "
                f"({completed:,} completed). "
                f"Accuracy and F1 metrics are scored over {mode_desc} "
                f"(rates use the full {report['records']:,}-record gold set)."
            )
            self.incomplete_banner.set_visibility(True)
        else:
            self.incomplete_banner.set_visibility(False)

    def _render_evaluation(self, eval_view: InferenceEvaluationView) -> None:
        self.current_eval = eval_view
        self._refresh_summary_metrics_strip()
        if not eval_view.available:
            self.eval_section.set_visibility(False)
            self.export_md_button.set_visibility(False)
            self.save_run_button.text = "Save Run"
            return

        self.eval_section.set_visibility(True)
        settings = eval_view.settings
        eos_desc = repr(settings.eos_token) if settings.eos_token else "auto"
        source_desc = eval_view.source or "—"
        self.eval_context_label.text = (
            f"Marker: {settings.marker!r} · Labels: {', '.join(settings.labels)} · "
            f"EOS: {eos_desc} · Source: {source_desc}"
        )

        # Gold status message
        if eval_view.error_message:
            self.gold_status_icon.name = "error_outline"
            self.gold_status_icon.props("color=negative")
            self.gold_status_label.text = eval_view.error_message
            self.gold_status_label.classes(
                replace="text-xs font-medium text-red-600 dark:text-red-400 min-w-0 break-all"
            )
        elif eval_view.score_report is not None:
            gold_src = self.uploaded_gold_name or eval_view.gold_path or "gold labels"
            self.gold_status_icon.name = "check_circle_outline"
            self.gold_status_icon.props("color=positive")
            self.gold_status_label.text = (
                f"Scored against {eval_view.gold_count:,} gold labels ({gold_src})."
            )
            self.gold_status_label.classes(
                replace="text-xs font-medium text-secondary-token min-w-0 break-all"
            )
        else:
            self.gold_status_icon.name = "info_outline"
            self.gold_status_icon.props("color=primary")
            self.gold_status_label.text = (
                eval_view.gold_notice
                or "Gold labels not loaded yet. Enter a local path or upload a JSON/JSONL file."
            )
            self.gold_status_label.classes(
                replace="text-xs text-secondary-token min-w-0 break-all"
            )

        # Single-table warning banner
        has_single_table = TABLE_ONLY_SINGLE_TABLE in eval_view.warnings
        self.single_table_banner.set_visibility(has_single_table)

        # Incomplete run banner
        report = eval_view.score_report
        self._update_incomplete_banner(report)

        # Metrics dashboard & save/export button states
        if report is not None:
            self._render_metrics_dashboard(eval_view)
            self.unscored_oos_card.set_visibility(False)
            self.save_run_button.text = "Save Run & Score"
            self.export_md_button.set_visibility(True)
        else:
            self.dashboard_card.set_visibility(False)
            self.save_run_button.text = "Save Run"
            self.export_md_button.set_visibility(False)
            if eval_view.verdicts and eval_view.out_of_set_counts:
                self._render_out_of_set_badges(
                    self.unscored_oos_container, eval_view.out_of_set_counts
                )
                self.unscored_oos_card.set_visibility(True)
            else:
                self.unscored_oos_card.set_visibility(False)

        # Verdict review table
        if eval_view.output_format is not None and not has_single_table and eval_view.review_rows:
            self.verdict_table_card.set_visibility(True)
            self._refresh_verdict_table()
        else:
            self.verdict_table_card.set_visibility(False)
            self.verdict_table.rows = []
            self.verdict_table.update()

    def save_current_run(self) -> str | None:
        """Persist the current ParseResult (and ScoreReport when present) via ``app/store.py``."""
        if self.current_summary is None:
            self._safe_notify("No parsed run loaded to save.", type="warning")
            return None

        result = dict(self.current_summary.raw)
        if not result.get("source_path") and self.current_log_path:
            result["source_path"] = self.current_log_path

        source_path = result.get("source_path")
        if not source_path or not Path(str(source_path)).is_file():
            self._safe_notify(
                "Cannot save run: source log file is unavailable on disk.",
                type="negative",
            )
            return None

        try:
            run_id = self._insert_run_fn(result, self.selected_config_id)
            report = self.current_score_report
            if report is not None:
                self._upsert_score_fn(run_id, dict(report))
        except Exception as exc:
            self._safe_notify(f"Failed to save run: {exc}", type="negative")
            return None

        self.last_saved_run_id = run_id
        if report is not None:
            status_msg = f"Saved run {run_id[:8]} + score"
        else:
            status_msg = f"Saved run {run_id[:8]} (run only)"
        self.save_status_label.text = status_msg
        self.save_status_label.classes(
            replace="text-[11px] font-semibold tag-success"
        )
        self.save_status_label.set_visibility(True)
        self._safe_notify(status_msg, type="positive")
        return run_id

    def export_markdown_report(self) -> str | None:
        """Render the current ScoreReport as Markdown and open the preview dialog."""
        report = self.current_score_report
        if report is None:
            self._safe_notify(
                "Score the inference run with gold labels before exporting.",
                type="warning",
            )
            return None
        md = render_report_markdown(report)
        self.last_markdown_report = md
        self.md_code.content = md
        try:
            self.md_dialog.open()
        except Exception:
            pass
        return md

    def _download_markdown_report(self) -> None:
        md = self.last_markdown_report
        if md is None and self.current_score_report is not None:
            md = render_report_markdown(self.current_score_report)
            self.last_markdown_report = md
        if md is None:
            return
        stem = Path(self.current_source_name or self.current_log_path or "score_report").stem
        try:
            self._ui.download(md.encode("utf-8"), filename=f"{stem}_score_report.md")
        except Exception:
            pass

    def render_result(
        self,
        result: ParseResult,
        *,
        source_name: str | None = None,
        alert_index: int = 0,
        eval_view: InferenceEvaluationView | None = None,
        preserve_save_status: bool = False,
    ) -> ResultSummaryView:
        """Render a :class:`ParseResult` into the Summary card and inference dashboard."""
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

        # 1. Top Run Classification Header (updated in-place without clearing DOM nodes so text never jumps)
        family_tag_cls = _FAMILY_TAG_CLASSES.get(
            summary.family, "info-tag tag-neutral"
        )
        self.header_family_label.text = summary.family_label
        self.header_family_label.classes(replace=family_tag_cls)

        tag_cls = _TONE_TAG_CLASSES.get(
            summary.status_tone, "info-tag tag-neutral"
        )
        dot_cls = _TONE_DOT_CLASSES.get(
            summary.status_tone, "status-dot dot-neutral"
        )
        self.header_status_wrap.classes(
            replace=f"items-center gap-1.5 no-wrap {tag_cls}"
        )
        self.header_status_dot.classes(replace=dot_cls)
        self.header_status_label.text = summary.status_label

        # 2. Single-Row Horizontal Metrics Bar (updated in-place without clearing DOM nodes so layout never jumps)
        self._refresh_summary_metrics_strip()

        # 3. Pipeline Phases Table Body (updated in-place without clearing DOM nodes so rows never jump)
        num_phases = min(len(summary.phases), len(self._phase_slots))
        for idx, (p_row, p_dot_el, p_name_lbl, p_line_lbl, p_status_lbl) in enumerate(
            self._phase_slots
        ):
            if idx < num_phases:
                phase = summary.phases[idx]
                line_txt = (
                    f"Line {phase.evidence_line}"
                    if phase.evidence_line is not None
                    else "—"
                )
                p_tag = _TONE_TAG_CLASSES.get(
                    phase.tone, "info-tag tag-neutral"
                )
                p_dot = _TONE_DOT_CLASSES.get(
                    phase.tone, "status-dot dot-neutral"
                )
                p_dot_el.classes(replace=p_dot)
                p_name_lbl.text = phase.label
                p_line_lbl.text = line_txt
                p_status_lbl.text = phase.outcome_label
                p_status_lbl.classes(replace=p_tag)
                p_row.set_visibility(True)
            else:
                p_row.set_visibility(False)

        # 4. Error / Warning Callout (updated in-place without clearing DOM nodes)
        if summary.alerts:
            alert = summary.alerts[self.active_alert_index]
            if alert.is_warning:
                box_cls = "w-full p-3.5 gap-1.5 diagnostic-box diagnostic-box-warning"
                icon_name = "warning_amber"
                icon_color = "warning"
                title_cls = "text-xs font-bold tracking-tight text-amber-600 dark:text-amber-400 truncate"
                btn_color = "warning"
            else:
                box_cls = "w-full p-3.5 gap-1.5 diagnostic-box diagnostic-box-error"
                icon_name = "error_outline"
                icon_color = "negative"
                title_cls = "text-xs font-bold tracking-tight text-red-600 dark:text-red-400 truncate"
                btn_color = "negative"

            heading_text = (
                f"{alert.title} ({alert.kind})"
                if alert.kind and alert.kind.lower() not in alert.title.lower()
                else alert.title
            )

            self.alert_box.classes(replace=box_cls)
            self.alert_icon.name = icon_name
            self.alert_icon.props(f"color={icon_color}")
            self.alert_title_label.text = heading_text
            self.alert_title_label.classes(replace=title_cls)

            if len(summary.alerts) > 1:
                self.alert_count_label.text = (
                    f"{self.active_alert_index + 1} / {len(summary.alerts)}"
                )
                self.alert_prev_btn.props(
                    f"flat dense round size=xs color={btn_color}"
                )
                self.alert_next_btn.props(
                    f"flat dense round size=xs color={btn_color}"
                )
                self.alert_nav_row.set_visibility(True)
            else:
                self.alert_nav_row.set_visibility(False)

            self.alert_message_label.text = alert.message
            if alert.detail:
                self.alert_detail_label.text = alert.detail
                self.alert_detail_label.set_visibility(True)
            else:
                self.alert_detail_label.text = ""
                self.alert_detail_label.set_visibility(False)

            if alert.notes:
                self.alert_notes_label.text = " ".join(alert.notes)
                self.alert_notes_label.set_visibility(True)
            else:
                self.alert_notes_label.text = ""
                self.alert_notes_label.set_visibility(False)

            self.alert_container.set_visibility(True)
        else:
            self.alert_container.set_visibility(False)

        if not preserve_save_status:
            self.last_saved_run_id = None
            self.save_status_label.set_visibility(False)

        if eval_view is None:
            settings = self._resolve_active_settings()
            if summary.family == "inference":
                log_path = result.get("source_path") or self.current_log_path
                gold_path, gold_notice = self._current_gold_resolution(settings)
                eval_view = evaluate_inference_log(
                    result,
                    log_path,
                    settings=settings,
                    gold_path=gold_path,
                    gold_notice=gold_notice,
                )
            else:
                eval_view = InferenceEvaluationView(
                    available=False, settings=settings
                )

        self._render_evaluation(eval_view)

        return summary


def build_results_panel(
    container: Any,
    *,
    list_configs_fn: Callable[[], Sequence[Mapping[str, Any]]] | None = None,
    get_config_fn: Callable[[str], Any] | None = None,
    insert_run_fn: Callable[[dict[str, Any], str | None], str] | None = None,
    upsert_score_fn: Callable[[str, dict[str, Any]], None] | None = None,
    get_score_fn: Callable[[str], dict[str, Any] | None] | None = None,
    parse_fn: Callable[[str | Path], ParseResult] = parse_log_file,
    initial_result: ParseResult | None = None,
    db_path: str | Path | None = None,
) -> ResultsPanel:
    """Mount the Run Results panel inside ``container``."""
    return ResultsPanel(
        container=container,
        list_configs_fn=list_configs_fn,
        get_config_fn=get_config_fn,
        insert_run_fn=insert_run_fn,
        upsert_score_fn=upsert_score_fn,
        get_score_fn=get_score_fn,
        parse_fn=parse_fn,
        initial_result=initial_result,
        db_path=db_path,
    )
