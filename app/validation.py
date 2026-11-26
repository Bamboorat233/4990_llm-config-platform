"""Cross-field validation rules 1-10 (issues #4 and #5).

Contract (specs/interfaces.md, contract 3):
  - validate_cross_fields(cfg) returns a list of
    {"field": <field path>, "level": "error" | "warning", "msg": <sentence>}
  - it never raises for a validation failure
  - an empty list means the config passes

The config is read through ``cfg.model_dump()`` and the dotted field paths
from contract 1 (for example "b.pad_token"), so this module does not depend
on how the section models are laid out. It does not import app.schemas, so
schemas.py can import it without a circular import.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.registry import DEEPSEEK_ID, MISTRAL_ID, get_model

if TYPE_CHECKING:
    from app.schemas import ConfigRecord, ValidationIssue

_MISSING = object()


def _get(payload: dict, path: str, default: Any = None) -> Any:
    """Read a value by field path, or return the default if any step is absent."""
    node: Any = payload
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node


def _issue(field: str, level: str, msg: str) -> dict:
    return {"field": field, "level": level, "msg": msg}


def _is_set(value: Any) -> bool:
    """A value counts as set if it is not None and not an empty string/container."""
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple)):
        return len(value) > 0
    return True


# ------------------------------------------------------------ rules 1-5

def _rule1_length(p: dict, meta) -> list:
    """(e.max_length or 0) + f.max_new_tokens <= effective_max_seq_len."""
    if meta is None:
        return []
    max_length = _get(p, "e.max_length") or 0
    max_new = _get(p, "f.max_new_tokens")
    if max_new is None:
        return []
    limit = meta.effective_max_seq_len
    total = max_length + max_new
    if total > limit:
        return [_issue(
            "f.max_new_tokens", "error",
            f"max_length ({max_length}) + max_new_tokens ({max_new}) = {total} "
            f"exceeds the model limit of {limit}; lower one of them.",
        )]
    return []


def _rule2_batch_padding(p: dict) -> list:
    """Inference with batch_size > 1 requires left padding."""
    if p.get("task_type") != "inference":
        return []
    batch = _get(p, "f.batch_size")
    side = _get(p, "b.padding_side")
    if batch is not None and batch > 1 and side != "left":
        return [_issue(
            "b.padding_side", "error",
            f"batch_size is {batch} but padding_side is '{side}'; "
            "use 'left' for batched generation.",
        )]
    return []


def _rule3_pad_token(p: dict, meta) -> list:
    """b.pad_token must be one of the model's allowed pad tokens."""
    if meta is None:
        return []
    allowed = tuple(getattr(meta, "allowed_pad_tokens", ()) or ())
    if not allowed:
        return []
    token = _get(p, "b.pad_token")
    if token not in allowed:
        shown = " or ".join(f"'{t}'" for t in allowed)
        return [_issue(
            "b.pad_token", "error",
            f"The model does not have the pad token '{token}'; use {shown}.",
        )]
    return []


def _rule4_mistral_regex(p: dict, meta) -> list:
    """Mistral requires b.fix_mistral_regex == True."""
    if meta is None or not getattr(meta, "needs_fix_mistral_regex", False):
        return []
    if _get(p, "b.fix_mistral_regex") is not True:
        return [_issue(
            "b.fix_mistral_regex", "error",
            "Mistral needs fix_mistral_regex set to true; turn it on.",
        )]
    return []


def _rule5_temperature(p: dict) -> list:
    """temperature == 0 <-> do_sample False; temperature > 0 <-> do_sample True."""
    temperature = _get(p, "f.temperature")
    do_sample = _get(p, "f.do_sample")
    if temperature is None or do_sample is None:
        return []
    if temperature == 0 and do_sample is True:
        return [_issue(
            "f.do_sample", "error",
            "temperature is 0 but do_sample is true; set do_sample to false "
            "or raise the temperature.",
        )]
    if temperature > 0 and do_sample is False:
        return [_issue(
            "f.do_sample", "error",
            f"temperature is {temperature} but do_sample is false; set "
            "do_sample to true or set the temperature to 0.",
        )]
    return []


# ------------------------------------------------------------ rules 6-10

def _rule6_best_model(p: dict) -> list:
    """Fine-tune with load_best_model_at_end needs matching strategies + eval file."""
    if p.get("task_type") != "fine_tune":
        return []
    if _get(p, "d.sft.load_best_model_at_end") is not True:
        return []
    issues = []
    eval_strategy = _get(p, "d.sft.eval_strategy")
    save_strategy = _get(p, "d.sft.save_strategy")
    if eval_strategy != save_strategy:
        issues.append(_issue(
            "d.sft.eval_strategy", "error",
            f"load_best_model_at_end needs eval_strategy ('{eval_strategy}') "
            f"to equal save_strategy ('{save_strategy}'); make them match.",
        ))
    if not _is_set(_get(p, "c.eval_data_file")):
        issues.append(_issue(
            "c.eval_data_file", "error",
            "load_best_model_at_end needs an evaluation data file; "
            "set eval_data_file.",
        ))
    return issues


def _rule7_prompt_consistency(p: dict) -> list:
    """The verdict marker must appear in the assistant prompt.

    PROVISIONAL: the spec says "consistent between fine-tuning and inference",
    which needs two configs. This checks the single-record version: the text
    inference searches for (verdict_parse_marker) must be what training wrote
    (assistant_prompt). It catches the original script bug, where training
    wrote "### Response: Verdict: " but the eval split wrote "### Response: ".
    Confirm the intended meaning with A.
    """
    marker = _get(p, "p.verdict_parse_marker")
    prompt = _get(p, "p.assistant_prompt")
    if not _is_set(marker) or not _is_set(prompt):
        return []
    if marker.strip() not in prompt:
        return [_issue(
            "p.verdict_parse_marker", "error",
            "verdict_parse_marker does not appear in assistant_prompt, so "
            "inference cannot find the verdict; make them consistent.",
        )]
    return []


def _rule8_tp_vs_device_map(p: dict) -> list:
    """a.tp_plan and a.device_map cannot both be set."""
    if _is_set(_get(p, "a.tp_plan")) and _is_set(_get(p, "a.device_map")):
        return [_issue(
            "a.device_map", "error",
            "tp_plan and device_map cannot both be set; clear device_map "
            "or tp_plan.",
        )]
    return []


def _rule9_mistral_system_prompt(p: dict) -> list:
    """Mistral with an empty system prompt is an error (default template injects the date)."""
    if p.get("model_id") != MISTRAL_ID:
        return []
    prompt = _get(p, "p.system_prompt")
    if prompt is None or not str(prompt).strip():
        return [_issue(
            "p.system_prompt", "error",
            "Mistral needs a non-empty system prompt; an empty one makes the "
            "default template insert the current date.",
        )]
    return []


def _rule10_deepseek_system_prompt(p: dict) -> list:
    """DeepSeek with a non-empty system prompt is a warning."""
    if p.get("model_id") != DEEPSEEK_ID:
        return []
    prompt = _get(p, "p.system_prompt")
    if prompt is not None and str(prompt).strip():
        return [_issue(
            "p.system_prompt", "warning",
            "DeepSeek concatenates the system prompt right after BOS with no "
            "separator; prefer putting the instruction in the user prompt.",
        )]
    return []


# ------------------------------------------------------------ kwargs whitelist

# Whitelists of supported kwargs per stage, aligned with transformers/accelerate signatures
KWARGS_WHITELISTS: dict[str, frozenset[str]] = {
    "load_model": frozenset(
        {
            "torch_dtype",
            "device_map",
            "low_cpu_mem_usage",
            "load_in_8bit",
            "load_in_4bit",
            "trust_remote_code",
            "attn_implementation",
            "revision",
            "subfolder",
            "offload_folder",
            "quantization_config",
        }
    ),
    "load_tokenizer": frozenset(
        {
            "use_fast",
            "trust_remote_code",
            "revision",
            "subfolder",
            "add_prefix_space",
            "padding_side",
            "truncation_side",
            "model_max_length",
            "clean_up_tokenization_spaces",
        }
    ),
    "encode": frozenset(
        {
            "padding",
            "truncation",
            "max_length",
            "return_tensors",
            "add_special_tokens",
            "return_attention_mask",
            "return_token_type_ids",
        }
    ),
    "generate": frozenset(
        {
            "max_new_tokens",
            "min_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "repetition_penalty",
            "length_penalty",
            "no_repeat_ngram_size",
            "bad_words_ids",
            "suppress_tokens",
            "forced_bos_token_id",
            "forced_eos_token_id",
        }
    ),
}


def parse_kwarg_value(val_str: str) -> Any:
    """Parse a string input into a typed Python value (bool, int, float, json, or str)."""
    text = val_str.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() in {"null", "none"}:
        return None

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        try:
            return json.loads(text)
        except Exception:
            pass

    return text


def validate_kwargs_section(
    h_data: dict[str, dict[str, Any]],
) -> list[ValidationIssue]:
    """Validate that all keys in Component h belong to the respective stage whitelist."""
    issues: list[ValidationIssue] = []
    for stage, kwargs_map in h_data.items():
        allowed = KWARGS_WHITELISTS.get(stage, frozenset())
        if not isinstance(kwargs_map, dict):
            continue
        for key in kwargs_map:
            if key not in allowed:
                issues.append(
                    {
                        "field": f"h.{stage}.{key}",
                        "level": "error",
                        "msg": (
                            f"Unknown argument '{key}' for stage '{stage}'; "
                            f"must be one of {sorted(allowed)}"
                        ),
                    }
                )
    return issues


# ------------------------------------------------------------ entry point


def validate_cross_fields(cfg: ConfigRecord) -> list[ValidationIssue]:
    """Run all ten rules and return the list of issues (empty = passes)."""
    payload = cfg.model_dump()

    try:
        meta = get_model(payload.get("model_id"))
    except ValueError:
        meta = None  # unknown model: skip the model-specific limits

    issues: list[ValidationIssue] = []
    issues += _rule1_length(payload, meta)
    issues += _rule2_batch_padding(payload)
    issues += _rule3_pad_token(payload, meta)
    issues += _rule4_mistral_regex(payload, meta)
    issues += _rule5_temperature(payload)
    issues += _rule6_best_model(payload)
    issues += _rule7_prompt_consistency(payload)
    issues += _rule8_tp_vs_device_map(payload)
    issues += _rule9_mistral_system_prompt(payload)
    issues += _rule10_deepseek_system_prompt(payload)
    return issues


__all__ = [
    "KWARGS_WHITELISTS",
    "parse_kwarg_value",
    "validate_cross_fields",
    "validate_kwargs_section",
]
