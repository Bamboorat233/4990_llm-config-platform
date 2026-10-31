"""Pydantic models for the stored configuration record (issue #3).

The shapes mirror the contracts in ``specs/interfaces.md`` and the JSON
keys of ``specs/parameters.md`` sections 1-13 exactly: the execution
components ``a``-``g``, the optional-arguments bucket ``h``, the prompt
section ``p`` and the top-level :class:`ConfigRecord`.

Public API:

- :class:`ModelSection` (a) — loading the model
- :class:`TokenizerSection` (b) — loading the tokenizer
- :class:`DataSection` (c) — loading the data
- :class:`LoraSection` / :class:`SftSection` / :class:`OptimizerSection` (d)
- :class:`EncodingSection` (e) — encoding the tokens
- :class:`GenerationSection` (f) — generating the text
- :class:`DecodingSection` (g) — decoding the tokens
- :class:`PromptSection` (p) — prompt parameters
- :class:`ConfigRecord` — the complete stored record
- :class:`ValidationIssue` — the issue shape of ``specs/interfaces.md`` section 3
- :func:`config_fingerprint` — the content hash used for deduplication

Every model forbids unknown keys so a drifted payload fails loudly.
Cross-field validation rules belong to issues #4/#5 and are not part of
this module; :class:`ValidationIssue` defines only the shape.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

_DEFAULT_LABEL_VALUES: tuple[str, ...] = ("True", "False", "Conflicting")

_FINGERPRINT_EXCLUDED_KEYS = frozenset(
    {
        "config_id",
        "config_hash",
        "run_status",
        "created_at",
        "created_by",
        "notes",
    }
)


def _utc_now_isoformat() -> str:
    """Return the current UTC time as an ISO 8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ModelSection(BaseModel):
    """Component a: how the model is loaded (``specs/parameters.md`` section 3)."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    local_files_only: bool = True
    tp_plan: str | None = "auto"
    device_map: str | None = None
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] | None = None
    quantization: Literal["4bit", "8bit"] | None = None
    trust_remote_code: bool = False


class TokenizerSection(BaseModel):
    """Component b: how the tokenizer is loaded (``specs/parameters.md`` section 4)."""

    model_config = ConfigDict(extra="forbid")

    tokenizer_id: str | None = None
    padding_side: Literal["left", "right"] = "left"
    pad_token: str
    pad_token_id: int | None = None
    fix_mistral_regex: bool = True
    use_fast: bool = True
    model_max_length: int | None = None
    truncation_side: Literal["left", "right"] = "right"


class DataSection(BaseModel):
    """Component c: how the data is loaded (``specs/parameters.md`` section 5)."""

    model_config = ConfigDict(extra="forbid")

    train_data_file: str | None = None
    eval_data_file: str | None = None
    test_data_file: str | None = None
    data_format: Literal["json"] = "json"
    label_values: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_LABEL_VALUES)
    )
    evidence_answers_used: int = 1
    missing_answer_placeholder: str = "No answer could be found !"
    max_seq_length: int | None = None
    max_samples: int | None = None
    evidence_config: Literal[
        "with_evidence",
        "without_evidence",
        "random_evidence",
        "question_only",
        "answer_only",
        "all_questions_last_answer",
        "all_questions_all_but_last_answer",
    ] = "with_evidence"


class LoraSection(BaseModel):
    """LoRA adapter settings (``specs/parameters.md`` section 6.1)."""

    model_config = ConfigDict(extra="forbid")

    r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str]
    use_dora: bool = False
    init_lora_weights: bool = True


class SftSection(BaseModel):
    """SFT trainer arguments (``specs/parameters.md`` section 6.2)."""

    model_config = ConfigDict(extra="forbid")

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 2
    warmup_steps: int = 5
    optim: str = "adamw_8bit"
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_strategy: str = "steps"
    logging_steps: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    push_to_hub: bool = False
    output_dir: str = "fineTuningOutput"
    bf16: bool = True
    gradient_checkpointing: bool = False
    seed: int = 42


class OptimizerSection(BaseModel):
    """Component d: the fixed optimizer block (``specs/parameters.md`` section 6)."""

    model_config = ConfigDict(extra="forbid")

    lora: LoraSection
    sft: SftSection


class EncodingSection(BaseModel):
    """Component e: how the tokens are encoded (``specs/parameters.md`` section 7)."""

    model_config = ConfigDict(extra="forbid")

    apply_chat_template: bool = True
    add_generation_prompt: bool
    tokenize: bool = True
    return_tensors: Literal["pt"] = "pt"
    padding: bool = True
    truncation: bool = True
    max_length: int | None = None
    device: str = "cuda"


class GenerationSection(BaseModel):
    """Component f: how the text is generated (``specs/parameters.md`` section 8)."""

    model_config = ConfigDict(extra="forbid")

    max_new_tokens: int = 512
    temperature: float = 0.15
    top_p: float | None = 0.9
    top_k: int | None = 10
    do_sample: bool = True
    use_cache: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    forced_prefix: str | None = None
    batch_size: int = 1


class DecodingSection(BaseModel):
    """Component g: how the tokens are decoded (``specs/parameters.md`` section 9)."""

    model_config = ConfigDict(extra="forbid")

    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = False
    spaces_between_special_tokens: bool = True


class PromptSection(BaseModel):
    """Component p: prompt parameters (``specs/parameters.md`` section 11)."""

    model_config = ConfigDict(extra="forbid")

    system_prompt: str
    user_prompt_template: str
    assistant_prompt: str = "### Response: Verdict: "
    chat_template_override: str | None = None
    verdict_parse_marker: str = "### Response: Verdict: "


class ConfigRecord(BaseModel):
    """One complete configuration record (``specs/parameters.md`` section 1 + 13).

    Identity and bookkeeping fields default here so that
    ``ConfigRecord(**defaults_for(model_id, task_type))`` is valid; the
    payload itself never carries them (``specs/interfaces.md`` section 2).
    """

    model_config = ConfigDict(extra="forbid")

    config_id: str = Field(default_factory=lambda: str(uuid4()))
    config_hash: str = ""
    model_id: str
    task_type: Literal["fine_tune", "inference"]
    run_status: Literal[
        "pending", "running", "success", "error", "terminated"
    ] = "pending"
    created_at: str = Field(default_factory=_utc_now_isoformat)
    created_by: str | None = None
    notes: str | None = None
    a: ModelSection
    b: TokenizerSection
    c: DataSection
    d: OptimizerSection
    e: EncodingSection
    f: GenerationSection
    g: DecodingSection
    h: dict[str, dict[str, Any]] = Field(default_factory=dict)
    p: PromptSection


class ValidationIssue(TypedDict):
    """One validation finding (shape only; rules live in issues #4/#5).

    ``field`` is a field path from ``specs/interfaces.md`` section 1 and
    ``msg`` is a one-sentence English message shown verbatim in the UI.
    """

    field: str
    level: Literal["error", "warning"]
    msg: str


def config_fingerprint(cfg: ConfigRecord) -> str:
    """Return the ``sha256:``-prefixed content hash of ``cfg``.

    Bookkeeping fields (``config_id``, ``config_hash``, ``run_status``,
    ``created_at``, ``created_by``, ``notes``) are excluded, so the hash
    identifies the functional configuration only. The digest is
    order-independent and does not mutate ``cfg``.
    """
    payload = {
        key: value
        for key, value in cfg.model_dump(mode="json").items()
        if key not in _FINGERPRINT_EXCLUDED_KEYS
    }
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(serialised.encode("utf-8")).hexdigest()


__all__ = [
    "ConfigRecord",
    "DataSection",
    "DecodingSection",
    "EncodingSection",
    "GenerationSection",
    "LoraSection",
    "ModelSection",
    "OptimizerSection",
    "PromptSection",
    "SftSection",
    "TokenizerSection",
    "ValidationIssue",
    "config_fingerprint",
]

from app.validation import validate_cross_fields  # noqa: E402,F401

