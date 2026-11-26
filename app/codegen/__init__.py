"""Code generation, pre-flight planning, and verification package (issues #56-#60).

Pure Python package; never imports ``nicegui``.
"""

from __future__ import annotations

import hashlib
from typing import Literal, TypedDict

from app.codegen.deps_gen import generate_requirements_txt
from app.codegen.fine_tune_gen import generate_fine_tune_script
from app.codegen.inference_gen import generate_inference_script
from app.codegen.planner import (
    ConfigValidationError,
    ScriptPlan,
    build_script_filename,
    build_script_plan,
)
from app.codegen.verify import (
    check_known_pitfalls,
    check_parameter_parity,
    check_parser_compatibility,
    check_reference_parity,
    check_syntax,
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
    """Validate ``cfg`` via the pre-flight gate and generate a verified script bundle.

    Raises :class:`ConfigValidationError` on pre-flight errors. Warnings never
    block generation and are returned in ``result["warnings"]``.
    """
    plan, warnings = build_script_plan(cfg, parent_cfg=parent_cfg)

    if plan.task_type == "fine_tune":
        script_code = generate_fine_tune_script(plan, cfg)
    else:
        script_code = generate_inference_script(plan, cfg)

    requirements_txt = generate_requirements_txt(cfg)
    config_json = cfg.model_dump_json(indent=2)
    # SLURM launcher generation is added in issue #63.
    slurm_script: str | None = None
    _ = include_slurm
    script_sha256 = hashlib.sha256(script_code.encode("utf-8")).hexdigest()

    syntax_ok, syntax_error = check_syntax(script_code, filename=plan.script_filename)
    parity_diffs = check_parameter_parity(cfg, script_code) + check_reference_parity(
        cfg, script_code
    )
    parity_ok = len(parity_diffs) == 0
    pitfall_issues = check_known_pitfalls(cfg, script_code)
    marker_issues = check_parser_compatibility(cfg, plan.script_filename, script_code)

    return {
        "config_id": plan.config_id,
        "config_hash": plan.config_hash,
        "task_type": plan.task_type,
        "model_id": plan.model_id,
        "script_filename": plan.script_filename,
        "script_code": script_code,
        "requirements_txt": requirements_txt,
        "config_json": config_json,
        "slurm_script": slurm_script,
        "script_sha256": script_sha256,
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "parity_ok": parity_ok,
        "parity_diffs": parity_diffs,
        "pitfall_issues": pitfall_issues,
        "marker_issues": marker_issues,
        "warnings": warnings,
    }


__all__ = [
    "ConfigValidationError",
    "GenerationResult",
    "ScriptPlan",
    "build_script_filename",
    "build_script_plan",
    "generate_bundle",
]
