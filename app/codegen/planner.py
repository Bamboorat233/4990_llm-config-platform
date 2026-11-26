"""Pre-flight validation gate and immutable ScriptPlan builder (issue #56).

This module is pure Python and must never import ``nicegui``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Literal

import app.validation as _validation
from app.parsing.errors import (
    DECODING_OUTPUT,
    DONE_LOADING_EVAL_DATA,
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_LOADING_TRAIN_DATA,
    DONE_PREPARING_INPUT,
    SHARD_WRITING_COMPLETE,
    TOKENIZING_INPUTS,
    VERDICTS_BANNER,
    family_for_script,
)
from app.registry import DEEPSEEK_ID, MISTRAL_ID
from app.schemas import ConfigRecord, ValidationIssue, config_fingerprint

MODEL_TAGS: dict[str, str] = {
    MISTRAL_ID: "Mistral24B",
    DEEPSEEK_ID: "DeepSeek32B",
}

TASK_FILENAME_PREFIXES: dict[str, str] = {
    "fine_tune": "fineTune",
    "inference": "inference",
}

FINE_TUNE_PHASE_MARKERS: tuple[str, ...] = (
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_LOADING_TRAIN_DATA,
    DONE_LOADING_EVAL_DATA,
    SHARD_WRITING_COMPLETE,
)

INFERENCE_PHASE_MARKERS: tuple[str, ...] = (
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_PREPARING_INPUT,
    TOKENIZING_INPUTS,
    DECODING_OUTPUT,
    VERDICTS_BANNER,
)


class ConfigValidationError(ValueError):
    """Raised when pre-flight configuration validation produces error issues."""

    def __init__(
        self,
        issues: list[ValidationIssue],
        message: str | None = None,
    ) -> None:
        self.issues: list[ValidationIssue] = list(issues)
        if message is None:
            details = "; ".join(
                f"{issue['field']}: {issue['msg']}" for issue in self.issues
            )
            message = (
                f"Configuration validation failed with {len(self.issues)} error(s): {details}"
                if self.issues
                else "Configuration validation failed."
            )
        super().__init__(message)


@dataclass(frozen=True)
class ScriptPlan:
    """Normalized, immutable execution plan consumed by codegen and verification."""

    config_id: str
    config_hash: str
    task_type: Literal["fine_tune", "inference"]
    model_id: str
    model_tag: str
    script_filename: str
    parent_config_id: str | None
    tokenizer_id: str
    adapter_path: str | None
    save_model_dir: str | None
    uses_distributed: bool
    fix_mistral_regex: bool
    system_prompt: str
    include_system_prompt: bool
    user_prompt_template: str
    assistant_prompt: str
    chat_template_override: str | None
    verdict_parse_marker: str
    evidence_config: str
    evidence_answers_used: int
    missing_answer_placeholder: str
    label_values: tuple[str, ...]
    max_seq_length: int | None
    kwargs: dict[str, dict[str, Any]]
    phase_markers: tuple[str, ...]
    config: ConfigRecord


def _model_tag_for(model_id: str) -> str:
    """Return the canonical filename model tag for ``model_id``."""
    if model_id in MODEL_TAGS:
        return MODEL_TAGS[model_id]
    tail = model_id.rsplit("/", 1)[-1]
    sanitized = re.sub(r"[^0-9A-Za-z]+", "", tail)
    return sanitized or "Model"


def _id8_for(config_id: str) -> str:
    """Return up to the first 8 alphanumeric/hex characters of ``config_id``."""
    cleaned = re.sub(r"[^0-9A-Za-z]", "", config_id)
    return cleaned[:8] or "00000000"


def build_script_filename(cfg: ConfigRecord) -> str:
    """Return ``fineTune_<ModelTag>_<id8>.py`` or ``inference_<ModelTag>_<id8>.py``."""
    prefix = TASK_FILENAME_PREFIXES.get(cfg.task_type)
    if prefix is None:
        raise ValueError(f"Unsupported task_type {cfg.task_type!r}")
    model_tag = _model_tag_for(cfg.a.model_id or cfg.model_id)
    id8 = _id8_for(cfg.config_id)
    filename = f"{prefix}_{model_tag}_{id8}.py"
    if family_for_script(filename) != cfg.task_type:
        raise ValueError(
            f"Generated filename {filename!r} does not map to task_type {cfg.task_type!r}"
        )
    return filename


def build_script_plan(
    cfg: ConfigRecord,
    *,
    parent_cfg: ConfigRecord | None = None,
) -> tuple[ScriptPlan, list[str]]:
    """Validate ``cfg`` (and optional ``parent_cfg``) and build an immutable ``ScriptPlan``.

    Raises :class:`ConfigValidationError` if any validation issue has
    ``level == "error"``. Warning issues are returned as English strings and
    never block plan construction.
    """
    issues: list[ValidationIssue] = []
    issues.extend(_validation.validate_cross_fields(cfg))
    issues.extend(_validation.validate_kwargs_section(cfg.h))

    if parent_cfg is not None:
        pair_validator = getattr(_validation, "validate_pair_consistency", None)
        if callable(pair_validator):
            issues.extend(pair_validator(parent_cfg, cfg))

    errors = [issue for issue in issues if issue["level"] == "error"]
    if errors:
        raise ConfigValidationError(errors)

    warnings = [issue["msg"] for issue in issues if issue["level"] == "warning"]

    base_model_id = cfg.a.model_id or cfg.model_id
    model_tag = _model_tag_for(base_model_id)
    script_filename = build_script_filename(cfg)
    config_hash = cfg.config_hash or config_fingerprint(cfg)
    tokenizer_id = cfg.b.tokenizer_id or base_model_id

    raw_adapter = getattr(cfg.a, "adapter_path", None)
    adapter_path = (
        raw_adapter.strip()
        if isinstance(raw_adapter, str) and raw_adapter.strip()
        else None
    )

    save_model_dir: str | None = None
    if cfg.task_type == "fine_tune":
        output_dir = cfg.d.sft.output_dir.rstrip("/\\") or "fineTuningOutput"
        save_model_dir = f"{output_dir}/final_model"

    include_system_prompt = (
        bool(cfg.p.system_prompt.strip()) if base_model_id == DEEPSEEK_ID else True
    )

    normalized_kwargs: dict[str, dict[str, Any]] = {
        stage: copy.deepcopy(cfg.h.get(stage, {}))
        for stage in _validation.KWARGS_WHITELISTS
    }

    phase_markers = (
        FINE_TUNE_PHASE_MARKERS
        if cfg.task_type == "fine_tune"
        else INFERENCE_PHASE_MARKERS
    )

    plan = ScriptPlan(
        config_id=cfg.config_id,
        config_hash=config_hash,
        task_type=cfg.task_type,
        model_id=base_model_id,
        model_tag=model_tag,
        script_filename=script_filename,
        parent_config_id=parent_cfg.config_id if parent_cfg is not None else None,
        tokenizer_id=tokenizer_id,
        adapter_path=adapter_path,
        save_model_dir=save_model_dir,
        uses_distributed=bool(cfg.a.tp_plan),
        fix_mistral_regex=bool(cfg.b.fix_mistral_regex and base_model_id == MISTRAL_ID),
        system_prompt=cfg.p.system_prompt,
        include_system_prompt=include_system_prompt,
        user_prompt_template=cfg.p.user_prompt_template,
        assistant_prompt=cfg.p.assistant_prompt,
        chat_template_override=cfg.p.chat_template_override,
        verdict_parse_marker=cfg.p.verdict_parse_marker,
        evidence_config=cfg.c.evidence_config,
        evidence_answers_used=cfg.c.evidence_answers_used,
        missing_answer_placeholder=cfg.c.missing_answer_placeholder,
        label_values=tuple(cfg.c.label_values),
        max_seq_length=cfg.c.max_seq_length or cfg.e.max_length,
        kwargs=normalized_kwargs,
        phase_markers=phase_markers,
        config=cfg,
    )
    return plan, warnings


__all__ = [
    "ConfigValidationError",
    "FINE_TUNE_PHASE_MARKERS",
    "INFERENCE_PHASE_MARKERS",
    "MODEL_TAGS",
    "ScriptPlan",
    "TASK_FILENAME_PREFIXES",
    "build_script_filename",
    "build_script_plan",
]
