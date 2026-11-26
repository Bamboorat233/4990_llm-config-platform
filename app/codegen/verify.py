"""AST syntax, parameter parity, reference parity, pitfall, and log-marker checks (issue #59).

This module is pure Python and must never import ``nicegui``.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from app.parsing.errors import (
    DECODING_OUTPUT,
    DONE_LOADING_EVAL_DATA,
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_LOADING_TRAIN_DATA,
    DONE_PREPARING_INPUT,
    TOKENIZING_INPUTS,
    VERDICTS_BANNER,
    family_for_script,
)
from app.registry import MISTRAL_ID
from app.schemas import ConfigRecord, config_fingerprint

_DEFAULT_BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "reference_sections.json"
)

_FINE_TUNE_REQUIRED_MARKERS: tuple[str, ...] = (
    "Start - Loading the model -",
    "End - Loading the model -",
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_LOADING_TRAIN_DATA,
    DONE_LOADING_EVAL_DATA,
    "Done Saving the Model",
)

_INFERENCE_REQUIRED_MARKERS: tuple[str, ...] = (
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_PREPARING_INPUT,
    TOKENIZING_INPUTS,
    DECODING_OUTPUT,
    VERDICTS_BANNER,
    "*****INPUTS*****",
    "*****OUTPUT*****",
    "*****VERDICTS*****",
)


def _diff(field: str, expected: Any, actual: Any, reason: str) -> dict[str, Any]:
    return {
        "field": field,
        "expected": expected,
        "actual": actual,
        "reason": reason,
    }


def _parse_ast(code: str, filename: str = "<generated>") -> tuple[ast.Module | None, str | None]:
    try:
        tree = ast.parse(code, filename=filename)
        return tree, None
    except (SyntaxError, ValueError, TypeError) as exc:
        if isinstance(exc, SyntaxError):
            msg = f"{exc.__class__.__name__}: {exc.msg} (line {exc.lineno}, offset {exc.offset})"
        else:
            msg = f"{exc.__class__.__name__}: {exc}"
        return None, msg


def _normalize_expr(expr_text: str) -> str:
    try:
        return ast.unparse(ast.parse(expr_text, mode="eval").body)
    except SyntaxError:
        return expr_text.strip()


def _extract_config_literal(tree: ast.Module) -> tuple[bool, Any, str | None]:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError) as exc:
                        return False, None, f"CONFIG is not a pure literal: {exc}"
                    if not isinstance(value, dict):
                        return False, value, "CONFIG must be a dict literal."
                    return True, value, None
    return False, None, "Top-level CONFIG dict literal was not found."


def _find_calls(tree: ast.AST, call_site: str) -> list[ast.Call]:
    calls: list[ast.Call] = []
    if "." in call_site:
        owner, attr = call_site.split(".", 1)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == attr
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == owner
            ):
                calls.append(node)
    else:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == call_site
            ):
                calls.append(node)
    return calls


def _call_kwargs(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


def _eval_node(node: ast.expr) -> tuple[bool, Any]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return False, ast.unparse(node)


def _function_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _resolve_cfg_path(cfg: ConfigRecord, path: str) -> Any:
    if path == "effective_max_seq_length":
        return cfg.c.max_seq_length or cfg.e.max_length
    node: Any = cfg.model_dump(mode="json")
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _resolve_adapter_path(cfg: ConfigRecord) -> str | None:
    raw = getattr(cfg.a, "adapter_path", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _check_kw_literal(
    diffs: list[dict[str, Any]],
    kw_map: dict[str, ast.expr],
    kw_name: str,
    expected: Any,
    field_path: str,
    call_site: str,
) -> None:
    if kw_name not in kw_map:
        diffs.append(
            _diff(
                field_path,
                expected,
                None,
                f"Missing keyword '{kw_name}' in {call_site}().",
            )
        )
        return
    is_lit, actual = _eval_node(kw_map[kw_name])
    if not is_lit or actual != expected:
        diffs.append(
            _diff(
                field_path,
                expected,
                actual,
                f"Value mismatch for '{kw_name}' in {call_site}().",
            )
        )


def _check_kw_expr(
    diffs: list[dict[str, Any]],
    kw_map: dict[str, ast.expr],
    kw_name: str,
    expected_expr: str,
    field_path: str,
    call_site: str,
) -> None:
    if kw_name not in kw_map:
        diffs.append(
            _diff(
                field_path,
                expected_expr,
                None,
                f"Missing keyword '{kw_name}' in {call_site}().",
            )
        )
        return
    actual_expr = ast.unparse(kw_map[kw_name])
    if _normalize_expr(actual_expr) != _normalize_expr(expected_expr):
        diffs.append(
            _diff(
                field_path,
                _normalize_expr(expected_expr),
                actual_expr,
                f"Expression mismatch for '{kw_name}' in {call_site}().",
            )
        )


def check_syntax(code: str, filename: str = "<generated>") -> tuple[bool, str | None]:
    """Verify ``code`` parses with ``ast.parse`` and compiles with ``compile``."""
    tree, err = _parse_ast(code, filename=filename)
    if tree is None:
        return False, err
    try:
        compile(tree, filename, "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        return False, f"{exc.__class__.__name__}: {exc}"
    return True, None


def _check_config_dict_parity(
    cfg: ConfigRecord,
    config_dict: dict[str, Any],
) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    base_model_id = cfg.a.model_id or cfg.model_id
    tokenizer_id = cfg.b.tokenizer_id or base_model_id
    expected_hash = cfg.config_hash or config_fingerprint(cfg)

    top_expected: dict[str, Any] = {
        "config_id": cfg.config_id,
        "config_hash": expected_hash,
        "model_id": base_model_id,
        "tokenizer_id": tokenizer_id,
        "max_seq_length": cfg.c.max_seq_length or cfg.e.max_length,
    }
    if cfg.task_type == "fine_tune":
        output_dir = cfg.d.sft.output_dir.rstrip("/\\") or "fineTuningOutput"
        top_expected.update(
            {
                "task_type": cfg.task_type,
                "train_data_file": cfg.c.train_data_file,
                "eval_data_file": cfg.c.eval_data_file,
                "output_dir": output_dir,
                "save_model_dir": f"{output_dir}/final_model",
                "uses_distributed": bool(cfg.a.tp_plan),
                "fix_mistral_regex": bool(
                    cfg.b.fix_mistral_regex and base_model_id == MISTRAL_ID
                ),
            }
        )
        section_names = ("a", "b", "c", "d", "p")
    else:
        output_dir = cfg.d.sft.output_dir.rstrip("/\\") or "inferenceOutput"
        top_expected.update(
            {
                "adapter_path": _resolve_adapter_path(cfg),
                "test_data_file": cfg.c.test_data_file,
                "output_dir": output_dir,
                "seed": cfg.d.sft.seed,
                "max_samples": cfg.c.max_samples,
                "evidence_config": cfg.c.evidence_config,
                "evidence_answers_used": cfg.c.evidence_answers_used,
                "missing_answer_placeholder": cfg.c.missing_answer_placeholder,
                "label_values": list(cfg.c.label_values),
                "system_prompt": cfg.p.system_prompt,
                "user_prompt_template": cfg.p.user_prompt_template,
                "assistant_prompt": cfg.p.assistant_prompt,
                "chat_template_override": cfg.p.chat_template_override,
                "verdict_parse_marker": cfg.p.verdict_parse_marker,
            }
        )
        section_names = ("a", "b", "c", "e", "f", "g", "p")

    for key, exp_val in top_expected.items():
        if key not in config_dict:
            diffs.append(
                _diff(
                    f"CONFIG.{key}",
                    exp_val,
                    None,
                    f"Missing key '{key}' in embedded CONFIG dictionary.",
                )
            )
        elif config_dict[key] != exp_val:
            diffs.append(
                _diff(
                    f"CONFIG.{key}",
                    exp_val,
                    config_dict[key],
                    f"CONFIG['{key}'] does not match ConfigRecord.",
                )
            )

    dumped = cfg.model_dump(mode="json")
    for sec in section_names:
        sec_actual = config_dict.get(sec)
        if not isinstance(sec_actual, dict):
            diffs.append(
                _diff(
                    f"CONFIG.{sec}",
                    dumped[sec],
                    sec_actual,
                    f"CONFIG['{sec}'] must be a dict.",
                )
            )
            continue
        for k, exp_val in dumped[sec].items():
            if cfg.task_type == "inference" and sec == "c" and k in ("train_data_file", "eval_data_file"):
                continue
            if cfg.task_type == "inference" and sec == "b" and k == "fix_mistral_regex":
                exp_val = bool(cfg.b.fix_mistral_regex and base_model_id == MISTRAL_ID)
            if k not in sec_actual:
                diffs.append(
                    _diff(
                        f"CONFIG.{sec}.{k}",
                        exp_val,
                        None,
                        f"Missing '{k}' in CONFIG['{sec}'].",
                    )
                )
            elif sec_actual[k] != exp_val:
                diffs.append(
                    _diff(
                        f"CONFIG.{sec}.{k}",
                        exp_val,
                        sec_actual[k],
                        f"CONFIG['{sec}']['{k}'] does not match cfg.{sec}.{k}.",
                    )
                )

    h_actual = config_dict.get("h")
    if not isinstance(h_actual, dict):
        diffs.append(
            _diff("CONFIG.h", cfg.h, h_actual, "CONFIG['h'] must be a dict.")
        )
    else:
        for stage, stage_kwargs in cfg.h.items():
            if not stage_kwargs:
                continue
            actual_stage = h_actual.get(stage)
            if actual_stage != stage_kwargs:
                diffs.append(
                    _diff(
                        f"CONFIG.h.{stage}",
                        stage_kwargs,
                        actual_stage,
                        f"CONFIG['h']['{stage}'] does not match cfg.h['{stage}'].",
                    )
                )

    return diffs


def check_parameter_parity(cfg: ConfigRecord, code: str) -> list[dict[str, Any]]:
    """Inspect embedded ``CONFIG`` and library call-site kwargs against ``cfg``."""
    tree, err = _parse_ast(code)
    if tree is None:
        return [
            _diff(
                "<syntax>",
                "valid Python AST",
                err,
                "Cannot inspect parameter parity because script has a syntax error.",
            )
        ]

    diffs: list[dict[str, Any]] = []
    found_cfg, config_dict, cfg_err = _extract_config_literal(tree)
    if not found_cfg:
        diffs.append(_diff("CONFIG", "dict literal", config_dict, cfg_err or "Missing CONFIG."))
    else:
        diffs.extend(_check_config_dict_parity(cfg, config_dict))

    base_model_id = cfg.a.model_id or cfg.model_id
    fix_mistral_regex = bool(cfg.b.fix_mistral_regex and base_model_id == MISTRAL_ID)
    h_load_model = dict(cfg.h.get("load_model", {}))
    h_load_tok = dict(cfg.h.get("load_tokenizer", {}))
    h_encode = dict(cfg.h.get("encode", {}))
    h_generate = dict(cfg.h.get("generate", {}))

    # 1. AutoModelForCausalLM.from_pretrained
    model_calls = _find_calls(tree, "AutoModelForCausalLM.from_pretrained")
    if not model_calls:
        diffs.append(
            _diff(
                "AutoModelForCausalLM.from_pretrained",
                "call present",
                None,
                "Missing AutoModelForCausalLM.from_pretrained() call.",
            )
        )
    else:
        mkw = _call_kwargs(model_calls[0])
        expected_dtype = (
            h_load_model["dtype"]
            if cfg.task_type == "inference" and "dtype" in h_load_model
            else cfg.a.dtype
        )
        _check_kw_literal(
            diffs, mkw, "dtype", expected_dtype, "a.dtype", "AutoModelForCausalLM.from_pretrained"
        )
        _check_kw_literal(
            diffs,
            mkw,
            "local_files_only",
            cfg.a.local_files_only,
            "a.local_files_only",
            "AutoModelForCausalLM.from_pretrained",
        )
        if cfg.a.tp_plan:
            _check_kw_literal(
                diffs,
                mkw,
                "tp_plan",
                cfg.a.tp_plan,
                "a.tp_plan",
                "AutoModelForCausalLM.from_pretrained",
            )
        elif "device_map" in h_load_model:
            _check_kw_literal(
                diffs,
                mkw,
                "device_map",
                h_load_model["device_map"],
                "h.load_model.device_map",
                "AutoModelForCausalLM.from_pretrained",
            )
        elif cfg.a.device_map:
            _check_kw_literal(
                diffs,
                mkw,
                "device_map",
                cfg.a.device_map,
                "a.device_map",
                "AutoModelForCausalLM.from_pretrained",
            )

        if "attn_implementation" in h_load_model:
            _check_kw_literal(
                diffs,
                mkw,
                "attn_implementation",
                h_load_model["attn_implementation"],
                "h.load_model.attn_implementation",
                "AutoModelForCausalLM.from_pretrained",
            )
        elif cfg.a.attn_implementation is not None:
            _check_kw_literal(
                diffs,
                mkw,
                "attn_implementation",
                cfg.a.attn_implementation,
                "a.attn_implementation",
                "AutoModelForCausalLM.from_pretrained",
            )

        _check_kw_literal(
            diffs,
            mkw,
            "trust_remote_code",
            h_load_model.get("trust_remote_code", cfg.a.trust_remote_code),
            "a.trust_remote_code",
            "AutoModelForCausalLM.from_pretrained",
        )

        if "quantization_config" in h_load_model:
            _check_kw_literal(
                diffs,
                mkw,
                "quantization_config",
                h_load_model["quantization_config"],
                "h.load_model.quantization_config",
                "AutoModelForCausalLM.from_pretrained",
            )
        elif cfg.a.quantization is not None:
            _check_kw_expr(
                diffs,
                mkw,
                "quantization_config",
                "quantization_config",
                "a.quantization",
                "AutoModelForCausalLM.from_pretrained",
            )
            bnb_calls = _find_calls(tree, "BitsAndBytesConfig")
            if not bnb_calls:
                diffs.append(
                    _diff(
                        "a.quantization",
                        f"BitsAndBytesConfig for {cfg.a.quantization}",
                        None,
                        "Missing BitsAndBytesConfig() call for quantized loading.",
                    )
                )
            else:
                bkw = _call_kwargs(bnb_calls[0])
                flag = "load_in_4bit" if cfg.a.quantization == "4bit" else "load_in_8bit"
                _check_kw_literal(diffs, bkw, flag, True, "a.quantization", "BitsAndBytesConfig")

        for k, v in h_load_model.items():
            if k not in {"device_map", "attn_implementation", "trust_remote_code", "quantization_config"}:
                _check_kw_literal(
                    diffs,
                    mkw,
                    k,
                    v,
                    f"h.load_model.{k}",
                    "AutoModelForCausalLM.from_pretrained",
                )

    # 2. AutoTokenizer.from_pretrained + pad_token assignment
    tok_calls = _find_calls(tree, "AutoTokenizer.from_pretrained")
    if not tok_calls:
        diffs.append(
            _diff(
                "AutoTokenizer.from_pretrained",
                "call present",
                None,
                "Missing AutoTokenizer.from_pretrained() call.",
            )
        )
    else:
        tkw = _call_kwargs(tok_calls[0])
        _check_kw_literal(
            diffs,
            tkw,
            "local_files_only",
            cfg.a.local_files_only,
            "a.local_files_only",
            "AutoTokenizer.from_pretrained",
        )
        _check_kw_literal(
            diffs,
            tkw,
            "padding_side",
            h_load_tok.get("padding_side", cfg.b.padding_side),
            "b.padding_side",
            "AutoTokenizer.from_pretrained",
        )
        _check_kw_literal(
            diffs,
            tkw,
            "use_fast",
            h_load_tok.get("use_fast", cfg.b.use_fast),
            "b.use_fast",
            "AutoTokenizer.from_pretrained",
        )
        _check_kw_literal(
            diffs,
            tkw,
            "truncation_side",
            h_load_tok.get("truncation_side", cfg.b.truncation_side),
            "b.truncation_side",
            "AutoTokenizer.from_pretrained",
        )
        if "model_max_length" in h_load_tok or cfg.b.model_max_length is not None:
            _check_kw_literal(
                diffs,
                tkw,
                "model_max_length",
                h_load_tok.get("model_max_length", cfg.b.model_max_length),
                "b.model_max_length",
                "AutoTokenizer.from_pretrained",
            )
        if fix_mistral_regex:
            _check_kw_literal(
                diffs,
                tkw,
                "fix_mistral_regex",
                True,
                "b.fix_mistral_regex",
                "AutoTokenizer.from_pretrained",
            )
        elif "fix_mistral_regex" in tkw:
            _, actual_fmr = _eval_node(tkw["fix_mistral_regex"])
            diffs.append(
                _diff(
                    "b.fix_mistral_regex",
                    "omitted",
                    actual_fmr,
                    "fix_mistral_regex should not be passed for non-Mistral models or when disabled.",
                )
            )
        for k, v in h_load_tok.items():
            if k not in {"padding_side", "use_fast", "truncation_side", "model_max_length"}:
                _check_kw_literal(
                    diffs,
                    tkw,
                    k,
                    v,
                    f"h.load_tokenizer.{k}",
                    "AutoTokenizer.from_pretrained",
                )

    pad_assigned: list[Any] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "pad_token"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "tokenizer"
                ):
                    _, val = _eval_node(node.value)
                    pad_assigned.append(val)
    if not pad_assigned:
        diffs.append(
            _diff(
                "b.pad_token",
                cfg.b.pad_token,
                None,
                "Missing tokenizer.pad_token assignment.",
            )
        )
    elif pad_assigned[-1] != cfg.b.pad_token:
        diffs.append(
            _diff(
                "b.pad_token",
                cfg.b.pad_token,
                pad_assigned[-1],
                "tokenizer.pad_token assignment does not match cfg.b.pad_token.",
            )
        )

    if cfg.task_type == "fine_tune":
        lora_calls = _find_calls(tree, "LoraConfig")
        if not lora_calls:
            diffs.append(_diff("LoraConfig", "call present", None, "Missing LoraConfig() call."))
        else:
            lkw = _call_kwargs(lora_calls[0])
            for attr in (
                "r",
                "lora_alpha",
                "lora_dropout",
                "bias",
                "target_modules",
                "use_dora",
                "init_lora_weights",
            ):
                _check_kw_literal(
                    diffs,
                    lkw,
                    attr,
                    getattr(cfg.d.lora, attr),
                    f"d.lora.{attr}",
                    "LoraConfig",
                )

        sft_calls = _find_calls(tree, "SFTConfig")
        if not sft_calls:
            diffs.append(_diff("SFTConfig", "call present", None, "Missing SFTConfig() call."))
        else:
            skw = _call_kwargs(sft_calls[0])
            if "output_dir" not in skw:
                diffs.append(
                    _diff(
                        "d.sft.output_dir",
                        cfg.d.sft.output_dir,
                        None,
                        "Missing output_dir in SFTConfig().",
                    )
                )
            for attr in (
                "learning_rate",
                "weight_decay",
                "num_train_epochs",
                "warmup_steps",
                "optim",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "lr_scheduler_type",
                "max_grad_norm",
                "eval_strategy",
                "save_strategy",
                "logging_strategy",
                "logging_steps",
                "load_best_model_at_end",
                "metric_for_best_model",
                "push_to_hub",
                "bf16",
                "gradient_checkpointing",
                "seed",
            ):
                _check_kw_literal(
                    diffs,
                    skw,
                    attr,
                    getattr(cfg.d.sft, attr),
                    f"d.sft.{attr}",
                    "SFTConfig",
                )
            max_seq = cfg.c.max_seq_length or cfg.e.max_length
            if max_seq is not None:
                _check_kw_literal(
                    diffs,
                    skw,
                    "max_length",
                    max_seq,
                    "c.max_seq_length",
                    "SFTConfig",
                )

        trainer_calls = _find_calls(tree, "SFTTrainer")
        if not trainer_calls:
            diffs.append(_diff("SFTTrainer", "call present", None, "Missing SFTTrainer() call."))
        else:
            tr_kw = _call_kwargs(trainer_calls[0])
            for req in ("model", "args", "processing_class", "train_dataset", "eval_dataset"):
                if req not in tr_kw:
                    diffs.append(
                        _diff(
                            f"SFTTrainer.{req}",
                            "present",
                            None,
                            f"Missing '{req}' keyword in SFTTrainer().",
                        )
                    )
    else:
        # Inference call sites
        adapter_path = _resolve_adapter_path(cfg)
        if adapter_path is not None:
            peft_calls = _find_calls(tree, "PeftModel.from_pretrained")
            if not peft_calls:
                diffs.append(
                    _diff(
                        "a.adapter_path",
                        adapter_path,
                        None,
                        "Missing PeftModel.from_pretrained() call when a.adapter_path is set.",
                    )
                )
            else:
                pkw = _call_kwargs(peft_calls[0])
                _check_kw_literal(
                    diffs,
                    pkw,
                    "local_files_only",
                    cfg.a.local_files_only,
                    "a.local_files_only",
                    "PeftModel.from_pretrained",
                )

        chat_calls = _find_calls(tree, "tokenizer.apply_chat_template")
        if not chat_calls:
            diffs.append(
                _diff(
                    "tokenizer.apply_chat_template",
                    "call present",
                    None,
                    "Missing tokenizer.apply_chat_template() call.",
                )
            )
        else:
            ckw = _call_kwargs(chat_calls[0])
            _check_kw_literal(diffs, ckw, "tokenize", False, "e.tokenize", "tokenizer.apply_chat_template")
            _check_kw_literal(
                diffs,
                ckw,
                "add_generation_prompt",
                True,
                "e.add_generation_prompt",
                "tokenizer.apply_chat_template",
            )

        enc_calls = _find_calls(tree, "tokenizer")
        if not enc_calls:
            diffs.append(_diff("tokenizer", "call present", None, "Missing tokenizer() encoding call."))
        else:
            ekw = _call_kwargs(enc_calls[0])
            _check_kw_literal(
                diffs,
                ekw,
                "return_tensors",
                h_encode.get("return_tensors", cfg.e.return_tensors),
                "e.return_tensors",
                "tokenizer",
            )
            _check_kw_literal(
                diffs,
                ekw,
                "padding",
                h_encode.get("padding", cfg.e.padding),
                "e.padding",
                "tokenizer",
            )
            _check_kw_literal(
                diffs,
                ekw,
                "truncation",
                h_encode.get("truncation", cfg.e.truncation),
                "e.truncation",
                "tokenizer",
            )
            if "max_length" in h_encode or cfg.e.max_length is not None:
                _check_kw_literal(
                    diffs,
                    ekw,
                    "max_length",
                    h_encode.get("max_length", cfg.e.max_length),
                    "e.max_length",
                    "tokenizer",
                )
            _check_kw_literal(
                diffs,
                ekw,
                "add_special_tokens",
                h_encode.get("add_special_tokens", False),
                "e.add_special_tokens",
                "tokenizer",
            )
            _check_kw_literal(
                diffs,
                ekw,
                "return_attention_mask",
                h_encode.get("return_attention_mask", True),
                "e.return_attention_mask",
                "tokenizer",
            )
            for k, v in h_encode.items():
                if k not in {
                    "return_tensors",
                    "padding",
                    "truncation",
                    "max_length",
                    "add_special_tokens",
                    "return_attention_mask",
                }:
                    _check_kw_literal(diffs, ekw, k, v, f"h.encode.{k}", "tokenizer")

        gen_calls = _find_calls(tree, "model.generate")
        if not gen_calls:
            diffs.append(_diff("model.generate", "call present", None, "Missing model.generate() call."))
        else:
            gkw = _call_kwargs(gen_calls[0])
            _check_kw_expr(diffs, gkw, "input_ids", "encoded['input_ids']", "e.input_ids", "model.generate")
            _check_kw_expr(
                diffs,
                gkw,
                "attention_mask",
                "encoded['attention_mask']",
                "e.attention_mask",
                "model.generate",
            )
            _check_kw_literal(
                diffs,
                gkw,
                "max_new_tokens",
                h_generate.get("max_new_tokens", cfg.f.max_new_tokens),
                "f.max_new_tokens",
                "model.generate",
            )
            _check_kw_literal(
                diffs,
                gkw,
                "do_sample",
                h_generate.get("do_sample", cfg.f.do_sample),
                "f.do_sample",
                "model.generate",
            )
            if "temperature" in h_generate:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "temperature",
                    h_generate["temperature"],
                    "h.generate.temperature",
                    "model.generate",
                )
            elif cfg.f.do_sample:
                if "temperature" not in gkw:
                    diffs.append(
                        _diff(
                            "f.temperature",
                            cfg.f.temperature,
                            None,
                            "Missing 'temperature' in model.generate() when do_sample=True.",
                        )
                    )
                else:
                    is_lit, val = _eval_node(gkw["temperature"])
                    if is_lit and val != cfg.f.temperature:
                        diffs.append(
                            _diff(
                                "f.temperature",
                                cfg.f.temperature,
                                val,
                                "temperature in model.generate() does not match cfg.f.temperature.",
                            )
                        )
                    elif not is_lit and _normalize_expr(str(val)) != "temperature":
                        diffs.append(
                            _diff(
                                "f.temperature",
                                "temperature",
                                val,
                                "Unexpected expression for temperature in model.generate().",
                            )
                        )
            elif "temperature" in gkw:
                _, actual_temp = _eval_node(gkw["temperature"])
                diffs.append(
                    _diff(
                        "f.temperature",
                        "omitted when do_sample=False",
                        actual_temp,
                        "temperature should not be passed to model.generate() when do_sample=False.",
                    )
                )

            if "top_p" in h_generate or cfg.f.top_p is not None:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "top_p",
                    h_generate.get("top_p", cfg.f.top_p),
                    "f.top_p",
                    "model.generate",
                )
            if "top_k" in h_generate or cfg.f.top_k is not None:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "top_k",
                    h_generate.get("top_k", cfg.f.top_k),
                    "f.top_k",
                    "model.generate",
                )

            for attr in (
                "use_cache",
                "num_return_sequences",
                "repetition_penalty",
                "no_repeat_ngram_size",
            ):
                _check_kw_literal(
                    diffs,
                    gkw,
                    attr,
                    h_generate.get(attr, getattr(cfg.f, attr)),
                    f"f.{attr}",
                    "model.generate",
                )

            if "pad_token_id" in h_generate:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "pad_token_id",
                    h_generate["pad_token_id"],
                    "h.generate.pad_token_id",
                    "model.generate",
                )
            elif cfg.f.pad_token_id is not None:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "pad_token_id",
                    cfg.f.pad_token_id,
                    "f.pad_token_id",
                    "model.generate",
                )
            else:
                _check_kw_expr(
                    diffs,
                    gkw,
                    "pad_token_id",
                    "tokenizer.pad_token_id",
                    "f.pad_token_id",
                    "model.generate",
                )

            if "eos_token_id" in h_generate:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "eos_token_id",
                    h_generate["eos_token_id"],
                    "h.generate.eos_token_id",
                    "model.generate",
                )
            elif cfg.f.eos_token_id is not None:
                _check_kw_literal(
                    diffs,
                    gkw,
                    "eos_token_id",
                    cfg.f.eos_token_id,
                    "f.eos_token_id",
                    "model.generate",
                )
            else:
                _check_kw_expr(
                    diffs,
                    gkw,
                    "eos_token_id",
                    "tokenizer.eos_token_id",
                    "f.eos_token_id",
                    "model.generate",
                )

            for k, v in h_generate.items():
                if k not in {
                    "max_new_tokens",
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "use_cache",
                    "num_return_sequences",
                    "repetition_penalty",
                    "no_repeat_ngram_size",
                    "pad_token_id",
                    "eos_token_id",
                }:
                    _check_kw_literal(diffs, gkw, k, v, f"h.generate.{k}", "model.generate")

        dec_calls = _find_calls(tree, "tokenizer.decode")
        if not dec_calls:
            diffs.append(
                _diff(
                    "tokenizer.decode",
                    "call present",
                    None,
                    "Missing tokenizer.decode() call.",
                )
            )
        else:
            dkw = _call_kwargs(dec_calls[0])
            for attr in (
                "skip_special_tokens",
                "clean_up_tokenization_spaces",
                "spaces_between_special_tokens",
            ):
                _check_kw_literal(
                    diffs,
                    dkw,
                    attr,
                    getattr(cfg.g, attr),
                    f"g.{attr}",
                    "tokenizer.decode",
                )

    return diffs


def _condition_applies(cfg: ConfigRecord, spec: dict[str, Any], stage: str | None) -> bool:
    when = spec.get("when", "always")
    h_stage = dict(cfg.h.get(stage, {})) if stage else {}
    path = spec.get("path")
    base_model_id = cfg.a.model_id or cfg.model_id

    if when == "always":
        return True
    if when == "truthy":
        return bool(_resolve_cfg_path(cfg, path)) if path else False
    if when == "not_none":
        return (_resolve_cfg_path(cfg, path) is not None) if path else False
    if when == "no_tp_plan_and_set":
        return (not cfg.a.tp_plan) and bool(h_stage.get("device_map") or cfg.a.device_map)
    if when == "quantization_enabled":
        return cfg.a.quantization is not None or "quantization_config" in h_stage
    if when == "mistral_fix_regex":
        return bool(cfg.b.fix_mistral_regex and base_model_id == MISTRAL_ID)
    if when == "do_sample_enabled":
        return bool(h_stage.get("do_sample", cfg.f.do_sample))
    return False


def check_reference_parity(
    cfg: ConfigRecord,
    code: str,
    *,
    baseline_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Verify ``code`` matches ``tests/fixtures/reference_sections.json`` per section."""
    path = baseline_path if baseline_path is not None else _DEFAULT_BASELINE_PATH
    try:
        baseline = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [
            _diff(
                "reference_baseline",
                str(path),
                str(exc),
                "Failed to load reference_sections.json baseline.",
            )
        ]

    tree, err = _parse_ast(code)
    if tree is None:
        return [
            _diff(
                "<syntax>",
                "valid Python AST",
                err,
                "Cannot inspect reference parity because script has a syntax error.",
            )
        ]

    task_spec = baseline.get("tasks", {}).get(cfg.task_type)
    if not isinstance(task_spec, dict):
        return [
            _diff(
                "task_type",
                "fine_tune or inference",
                cfg.task_type,
                f"No baseline task entry for {cfg.task_type!r} in reference_sections.json.",
            )
        ]

    diffs: list[dict[str, Any]] = []
    defined_helpers = _function_names(tree)
    for helper in task_spec.get("required_helpers", []):
        if helper not in defined_helpers:
            diffs.append(
                _diff(
                    f"helpers.{helper}",
                    f"def {helper}(...)",
                    None,
                    f"Missing required reference helper '{helper}'.",
                )
            )

    for sec in task_spec.get("sections", []):
        sec_name = sec["section"]
        call_site = sec["call_site"]
        stage = sec.get("stage")
        h_stage = dict(cfg.h.get(stage, {})) if stage else {}
        calls = _find_calls(tree, call_site)
        if not calls:
            diffs.append(
                _diff(
                    f"{sec_name}.call_site",
                    call_site,
                    None,
                    f"Missing reference section '{sec_name}' call site '{call_site}'.",
                )
            )
            continue

        actual_kw = _call_kwargs(calls[0])
        kw_specs: dict[str, dict[str, Any]] = sec.get("keywords", {})
        expected_kw_names: set[str] = set()

        for kw_name, kw_spec in kw_specs.items():
            applies = _condition_applies(cfg, kw_spec, stage) or (kw_name in h_stage)
            if not applies:
                continue
            expected_kw_names.add(kw_name)
            if kw_name not in actual_kw:
                diffs.append(
                    _diff(
                        f"{sec_name}.{kw_name}",
                        kw_spec.get("path") or kw_spec.get("value"),
                        None,
                        f"Missing expected keyword '{kw_name}' in {call_site}().",
                    )
                )
                continue

            source = kw_spec.get("source")
            if kw_name in h_stage:
                expected_val = h_stage[kw_name]
                _check_kw_literal(
                    diffs,
                    actual_kw,
                    kw_name,
                    expected_val,
                    f"{sec_name}.{kw_name}",
                    call_site,
                )
            elif source == "cfg":
                expected_val = _resolve_cfg_path(cfg, kw_spec["path"])
                _check_kw_literal(
                    diffs,
                    actual_kw,
                    kw_name,
                    expected_val,
                    f"{sec_name}.{kw_name}",
                    call_site,
                )
            elif source == "fixed":
                _check_kw_literal(
                    diffs,
                    actual_kw,
                    kw_name,
                    kw_spec["value"],
                    f"{sec_name}.{kw_name}",
                    call_site,
                )
            elif source == "cfg_or_expr":
                cfg_val = _resolve_cfg_path(cfg, kw_spec["path"])
                if cfg_val is not None:
                    _check_kw_literal(
                        diffs,
                        actual_kw,
                        kw_name,
                        cfg_val,
                        f"{sec_name}.{kw_name}",
                        call_site,
                    )
                else:
                    _check_kw_expr(
                        diffs,
                        actual_kw,
                        kw_name,
                        kw_spec["fallback_expr"],
                        f"{sec_name}.{kw_name}",
                        call_site,
                    )
            elif source == "expr":
                _check_kw_expr(
                    diffs,
                    actual_kw,
                    kw_name,
                    kw_spec["value"],
                    f"{sec_name}.{kw_name}",
                    call_site,
                )

        allowed_all = expected_kw_names | set(h_stage.keys())
        for kw_name, expr_node in actual_kw.items():
            if kw_name not in allowed_all:
                _, actual_val = _eval_node(expr_node)
                diffs.append(
                    _diff(
                        f"{sec_name}.{kw_name}",
                        "not present",
                        actual_val,
                        f"Unexpected keyword '{kw_name}' in {call_site}() not in reference baseline or Component h.",
                    )
                )

    return diffs


def check_known_pitfalls(cfg: ConfigRecord, code: str) -> list[str]:
    """Check that none of the 6 reference-script pitfalls or rank=0 bugs exist."""
    issues: list[str] = []
    tree, err = _parse_ast(code)
    if tree is None:
        return [f"Script has invalid syntax so pitfall checks could not run: {err}"]

    defined_helpers = _function_names(tree)
    base_model_id = cfg.a.model_id or cfg.model_id

    # Defect 1: format_evidence_items helper present and no unparenthesized top_k_doc ternary
    if "format_evidence_items" not in defined_helpers:
        issues.append(
            "Missing format_evidence_items() helper (Defect 1: unparenthesized ternary drops questions)."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.IfExp) and "### Question:" in ast.unparse(node.body):
            issues.append(
                "Detected inline ternary formatting '### Question:' (Defect 1 risk)."
            )

    # Defect 2: No <myPadToken> and valid tokenizer.pad_token assignment
    if "<myPadToken>" in code:
        issues.append("Generated script contains invalid pad token literal '<myPadToken>' (Defect 2).")

    # Defect 3: No post-load tokenizer.fix_mistral_regex = ...
    if re.search(r"\btokenizer\.fix_mistral_regex\s*=", code):
        issues.append(
            "Post-load 'tokenizer.fix_mistral_regex =' assignment detected (Defect 3); pass into from_pretrained()."
        )
    if base_model_id == MISTRAL_ID and cfg.b.fix_mistral_regex:
        tok_calls = _find_calls(tree, "AutoTokenizer.from_pretrained")
        if tok_calls:
            tkw = _call_kwargs(tok_calls[0])
            if "fix_mistral_regex" not in tkw:
                issues.append(
                    "Mistral AutoTokenizer.from_pretrained() is missing fix_mistral_regex=True (Defect 3)."
                )

    # Defect 4: Consistent assistant prompt / DRY helpers
    if "NOTE TO SELF" in code:
        issues.append("Generated script contains leftover 'NOTE TO SELF' marker (Defect 4).")
    if "build_messages" not in defined_helpers:
        issues.append("Missing shared build_messages() helper (Defect 4).")
    if cfg.task_type == "fine_tune":
        if "build_dataset" not in defined_helpers:
            issues.append("Missing shared build_dataset() helper (Defect 4).")
        if "Verdict:" in cfg.p.assistant_prompt:
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.strip() == "### Response:":
                        issues.append(
                            "Inconsistent '### Response:' prefix found alongside Verdict assistant prompt (Defect 4)."
                        )

    # Defect 5: Top-level CONFIG, CLI parse_args, non-hardcoded paths, no 0.1 temperature overwrite
    found_cfg, _, _ = _extract_config_literal(tree)
    if not found_cfg:
        issues.append("Missing top-level CONFIG dictionary (Defect 5: non-hardcoded configuration).")
    if "parse_args" not in defined_helpers:
        issues.append("Missing parse_args() CLI override helper (Defect 5).")
    if "model.generation_config.temperature = 0.1" in code:
        issues.append(
            "Found hardcoded 'model.generation_config.temperature = 0.1' overwrite (Defect 5)."
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if ast.unparse(target) == "model.generation_config.temperature":
                    issues.append(
                        "Found direct assignment to model.generation_config.temperature (Defect 5)."
                    )
    model_calls = _find_calls(tree, "AutoModelForCausalLM.from_pretrained")
    if model_calls and model_calls[0].args:
        first_arg = model_calls[0].args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            issues.append(
                "AutoModelForCausalLM.from_pretrained() uses a hardcoded string literal instead of model_id variable (Defect 5)."
            )
    for open_call in _find_calls(tree, "open"):
        if open_call.args:
            arg0 = open_call.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                issues.append(
                    f"open() uses hardcoded path literal {arg0.value!r} instead of variable (Defect 5)."
                )

    # Defect 6: Reproducibility seed helper + SFTConfig(seed=...)
    if "set_reproducibility_seed" not in defined_helpers:
        issues.append("Missing set_reproducibility_seed() helper (Defect 6).")
    else:
        seed_calls = _find_calls(tree, "set_reproducibility_seed")
        if not seed_calls:
            issues.append("set_reproducibility_seed() is defined but never called (Defect 6).")
    if cfg.task_type == "fine_tune":
        sft_calls = _find_calls(tree, "SFTConfig")
        if sft_calls and "seed" not in _call_kwargs(sft_calls[0]):
            issues.append("SFTConfig() is missing seed keyword argument (Defect 6).")

    # Distributed init rank=0 pitfall
    if "rank=0" in code:
        issues.append("Found forbidden 'rank=0' in generated script.")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            is_init_pg = (
                (isinstance(fn, ast.Name) and fn.id == "init_process_group")
                or (isinstance(fn, ast.Attribute) and fn.attr == "init_process_group")
            )
            if is_init_pg and (node.args or any(kw.arg == "rank" for kw in node.keywords)):
                issues.append(
                    "init_process_group() must not pass rank=0 or positional rank arguments."
                )

    return issues


def check_parser_compatibility(
    cfg: ConfigRecord,
    script_filename: str,
    code: str,
) -> list[str]:
    """Verify filename family and required log-parser phase markers are present."""
    issues: list[str] = []

    detected_family = family_for_script(script_filename)
    if detected_family != cfg.task_type:
        issues.append(
            f"Script filename {script_filename!r} maps to family {detected_family!r}, "
            f"expected {cfg.task_type!r}."
        )

    if cfg.task_type == "fine_tune":
        for marker in _FINE_TUNE_REQUIRED_MARKERS:
            if marker not in code:
                issues.append(f"Missing required fine-tune phase marker: {marker!r}.")
    elif cfg.task_type == "inference":
        for marker in _INFERENCE_REQUIRED_MARKERS:
            if marker not in code:
                issues.append(f"Missing required inference phase marker: {marker!r}.")
        if " | " not in code:
            issues.append("Missing indexed verdict table row format ('{idx} | {verdict}').")
    else:
        issues.append(f"Unsupported task_type for parser compatibility: {cfg.task_type!r}.")

    return issues


__all__ = [
    "check_known_pitfalls",
    "check_parameter_parity",
    "check_parser_compatibility",
    "check_reference_parity",
    "check_syntax",
]
