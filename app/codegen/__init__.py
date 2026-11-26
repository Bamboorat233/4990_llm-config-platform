"""Code generation, pre-flight planning, and verification package (issues #56-#60).

Pure Python package; never imports ``nicegui``.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from app.codegen.planner import (
    ConfigValidationError,
    ScriptPlan,
    build_script_filename,
    build_script_plan,
)
from app.schemas import ConfigRecord


class GenerationResult(TypedDict):
    """Contract §6 output bundle returned by :func:`generate_bundle`."""

    config_id: str
    config_hash: str
    task_type: Literal["fine_tune", "inference"]
    model_id: str
    script_filename: str
    script_code: str
    requirements_txt: str
    config_json: str
    slurm_script: str | None
    script_sha256: str
    syntax_ok: bool
    syntax_error: str | None
    parity_ok: bool
    parity_diffs: list[dict]
    pitfall_issues: list[str]
    marker_issues: list[str]
    warnings: list[str]


def generate_bundle(
    cfg: ConfigRecord,
    *,
    include_slurm: bool = False,
    parent_cfg: ConfigRecord | None = None,
) -> GenerationResult:
    """Validate ``cfg`` via the pre-flight gate and generate a script bundle.

    Raises :class:`ConfigValidationError` on pre-flight errors. Script and
    bundle generators are implemented in issues #57-#59.
    """
    _plan, _warnings = build_script_plan(cfg, parent_cfg=parent_cfg)
    raise NotImplementedError(
        "Script generators and verification engine are not implemented yet "
        "(issues #57-#59)."
    )


__all__ = [
    "ConfigValidationError",
    "GenerationResult",
    "ScriptPlan",
    "build_script_filename",
    "build_script_plan",
    "generate_bundle",
]
