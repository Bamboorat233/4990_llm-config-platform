"""Log parsing: one SLURM stdout log becomes one typed ``ParseResult``.

Pipeline position (``design/M1.5-2_Spec_and_roadmap.md`` section 1)::

    log file
        -> app.parsing.parse_log_file        (this package, issue #30)
        -> ParseResult                       (specs/interfaces.md section 4)
        -> verdict extraction                (app/parsing/verdicts.py)
        -> scoring                           (app/scoring.py)
        -> ScoreReport                       (specs/interfaces.md section 5)
        -> SQLite, then the Results panel

This module owns the public entry point and the result shape. The detection
patterns, script-to-family map, phase order and completion markers live in
:mod:`app.parsing.errors`; the detector and the family parsers arrive in
issue #30. Nothing here imports NiceGUI: parsing must run in a worker thread
through ``run.io_bound`` and stay usable from the command line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

__all__ = ["ParseResult", "parse_log_file"]


class ParseResult(TypedDict):
    """The frozen result shape of ``specs/interfaces.md`` section 4.

    Every key is always present. ``None`` means "not observed in this log";
    a missing key is a bug. ``scenario`` is derived as
    ``f"{family}_{status}"`` and ``phases`` is ordered by family.
    """

    source_path: str
    script: str | None
    model_id: str | None
    family: Literal["fine_tune", "inference", "checkpoint_export", "unknown"]
    status: Literal["success", "error", "terminated", "unknown"]
    scenario: str
    # Ordered; each item is
    # {"name": str, "outcome": PhaseOutcome, "evidence_line": int | None}.
    # PhaseOutcome is "succeeded" | "failed" | "terminated" | "not_reached".
    phases: list[dict]
    failed_phase: str | None
    output_format: Literal["banner", "think", "table_only"] | None
    started_at: str | None
    finished_at: str | None
    duration_seconds: float | None
    # fine_tune: step, total_steps, epoch, last_loss, eval_losses
    # inference: records_done, records_expected
    # checkpoint_export: {}
    progress: dict
    # Each item is {"kind": str, "message": str, "line": int | None,
    # "rank": int | None}.
    errors: list[dict]
    warnings: list[dict]
    # Each item is {"rule": str, "line": int | None, "text": str}.
    evidence: list[dict]
    notes: list[str]


def parse_log_file(path: str | Path) -> ParseResult:
    """Parse one SLURM stdout log into a :class:`ParseResult`.

    Contract (``specs/interfaces.md`` section 4):

    - always returns a :class:`ParseResult`; malformed, truncated or
      unrecognisable content never raises. A log that cannot be classified
      is reported with ``status="unknown"`` and an explanatory ``notes``
      entry; a log cut off by SLURM is reported with
      ``status="terminated"``.
    - makes a single streaming pass over the file, reading with
      ``encoding="utf-8", errors="replace", newline=None``, so a 20 MB log
      is never held in memory in full.
    - raises only for an unreadable path (``OSError``) or a non-file path;
      those are caller errors, not log-content errors.
    - has no side effects: no writes, no NiceGUI import, no global state.

    The implementation lands in issue #30 (detector plus family parsers).
    """
    raise NotImplementedError(
        "parse_log_file is not implemented yet; issue #30 adds the "
        "streaming detector and the family parsers."
    )
