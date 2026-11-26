"""Inference script generator for Mistral and DeepSeek (issue #58).

Generates self-contained, human-reviewable Python inference scripts from a
validated :class:`~app.codegen.planner.ScriptPlan` and
:class:`~app.schemas.ConfigRecord`.

This module is pure Python and must never import ``nicegui``.
"""

from __future__ import annotations

import pprint
from typing import Any

from app.codegen.planner import ScriptPlan
from app.schemas import ConfigRecord


def _format_kwarg_lines(
    base_items: list[tuple[str, str]],
    extra_kwargs: dict[str, Any],
    *,
    indent: int = 8,
) -> str:
    """Merge base call-site kwargs with Component h overrides without duplicates."""
    merged: dict[str, str] = {}
    for key, expr in base_items:
        merged[key] = expr
    for key, val in extra_kwargs.items():
        merged[key] = repr(val)
    pad = " " * indent
    return "\n".join(f"{pad}{key}={expr}," for key, expr in merged.items())


def _resolve_adapter_path(plan: ScriptPlan, cfg: ConfigRecord) -> str | None:
    """Return normalized adapter_path from ``plan`` or ``cfg.a`` if present."""
    if plan.adapter_path:
        return plan.adapter_path
    raw = getattr(cfg.a, "adapter_path", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def _build_config_dict(
    plan: ScriptPlan,
    cfg: ConfigRecord,
    adapter_path: str | None,
) -> dict[str, Any]:
    """Build the deterministic top-level CONFIG dictionary embedded in the script."""
    output_dir = cfg.d.sft.output_dir.rstrip("/\\") or "inferenceOutput"
    return {
        "config_id": plan.config_id,
        "config_hash": plan.config_hash,
        "model_id": plan.model_id,
        "tokenizer_id": plan.tokenizer_id,
        "adapter_path": adapter_path,
        "test_data_file": cfg.c.test_data_file,
        "output_dir": output_dir,
        "seed": cfg.d.sft.seed,
        "max_samples": cfg.c.max_samples,
        "evidence_config": plan.evidence_config,
        "evidence_answers_used": plan.evidence_answers_used,
        "missing_answer_placeholder": plan.missing_answer_placeholder,
        "label_values": list(plan.label_values),
        "max_seq_length": plan.max_seq_length,
        "system_prompt": plan.system_prompt,
        "include_system_prompt": plan.include_system_prompt,
        "user_prompt_template": plan.user_prompt_template,
        "assistant_prompt": plan.assistant_prompt,
        "chat_template_override": plan.chat_template_override,
        "verdict_parse_marker": plan.verdict_parse_marker,
        "a": {
            "model_id": plan.model_id,
            "dtype": cfg.a.dtype,
            "local_files_only": cfg.a.local_files_only,
            "tp_plan": cfg.a.tp_plan,
            "device_map": cfg.a.device_map,
            "attn_implementation": cfg.a.attn_implementation,
            "quantization": cfg.a.quantization,
            "trust_remote_code": cfg.a.trust_remote_code,
            "adapter_path": adapter_path,
        },
        "b": {
            "tokenizer_id": plan.tokenizer_id,
            "padding_side": cfg.b.padding_side,
            "pad_token": cfg.b.pad_token,
            "pad_token_id": cfg.b.pad_token_id,
            "fix_mistral_regex": plan.fix_mistral_regex,
            "use_fast": cfg.b.use_fast,
            "model_max_length": cfg.b.model_max_length,
            "truncation_side": cfg.b.truncation_side,
        },
        "c": {
            "test_data_file": cfg.c.test_data_file,
            "data_format": cfg.c.data_format,
            "label_values": list(cfg.c.label_values),
            "evidence_answers_used": cfg.c.evidence_answers_used,
            "missing_answer_placeholder": cfg.c.missing_answer_placeholder,
            "max_seq_length": cfg.c.max_seq_length,
            "max_samples": cfg.c.max_samples,
            "evidence_config": cfg.c.evidence_config,
        },
        "e": {
            "apply_chat_template": cfg.e.apply_chat_template,
            "add_generation_prompt": cfg.e.add_generation_prompt,
            "tokenize": cfg.e.tokenize,
            "return_tensors": cfg.e.return_tensors,
            "padding": cfg.e.padding,
            "truncation": cfg.e.truncation,
            "max_length": cfg.e.max_length,
            "device": cfg.e.device,
        },
        "f": {
            "max_new_tokens": cfg.f.max_new_tokens,
            "temperature": cfg.f.temperature,
            "top_p": cfg.f.top_p,
            "top_k": cfg.f.top_k,
            "do_sample": cfg.f.do_sample,
            "use_cache": cfg.f.use_cache,
            "num_return_sequences": cfg.f.num_return_sequences,
            "repetition_penalty": cfg.f.repetition_penalty,
            "no_repeat_ngram_size": cfg.f.no_repeat_ngram_size,
            "pad_token_id": cfg.f.pad_token_id,
            "eos_token_id": cfg.f.eos_token_id,
            "forced_prefix": cfg.f.forced_prefix,
            "batch_size": cfg.f.batch_size,
        },
        "g": {
            "skip_special_tokens": cfg.g.skip_special_tokens,
            "clean_up_tokenization_spaces": cfg.g.clean_up_tokenization_spaces,
            "spaces_between_special_tokens": cfg.g.spaces_between_special_tokens,
        },
        "h": {stage: dict(values) for stage, values in plan.kwargs.items()},
        "p": {
            "system_prompt": cfg.p.system_prompt,
            "user_prompt_template": cfg.p.user_prompt_template,
            "assistant_prompt": cfg.p.assistant_prompt,
            "chat_template_override": cfg.p.chat_template_override,
            "verdict_parse_marker": cfg.p.verdict_parse_marker,
        },
    }


def _build_imports_block(
    quantization: str | None,
    adapter_path: str | None,
) -> str:
    """Render minimal standard-library and ML imports."""
    tf_imports = ["AutoModelForCausalLM", "AutoTokenizer"]
    if quantization in ("4bit", "8bit"):
        tf_imports.append("BitsAndBytesConfig")
    tf_imports.append("set_seed")

    lines = [
        "import argparse",
        "import datetime",
        "import json",
        "import os",
        "import random",
        "",
        "import numpy as np",
        "import torch",
        f"from transformers import {', '.join(tf_imports)}",
    ]
    if adapter_path is not None:
        lines.append("from peft import PeftModel")
    return "\n".join(lines)


def _build_quantization_block(quantization: str | None) -> list[tuple[str, str]]:
    """Return setup statements (handled inline) and model-loading kwargs for quantization."""
    if quantization in ("4bit", "8bit"):
        return [("quantization_config", "quantization_config")]
    return []


def _render_quantization_setup(quantization: str | None) -> str:
    """Render BitsAndBytesConfig instantiation when quantization is enabled."""
    if quantization == "4bit":
        return (
            "    quantization_config = BitsAndBytesConfig(\n"
            "        load_in_4bit=True,\n"
            "        bnb_4bit_compute_dtype=torch.bfloat16,\n"
            "    )\n"
        )
    if quantization == "8bit":
        return "    quantization_config = BitsAndBytesConfig(load_in_8bit=True)\n"
    return ""


def _build_model_kwargs_block(plan: ScriptPlan, cfg: ConfigRecord) -> str:
    """Render keyword arguments for ``AutoModelForCausalLM.from_pretrained``."""
    base: list[tuple[str, str]] = [
        ("local_files_only", repr(cfg.a.local_files_only)),
        ("dtype", repr(cfg.a.dtype)),
    ]
    if cfg.a.tp_plan:
        base.append(("tp_plan", repr(cfg.a.tp_plan)))
    elif cfg.a.device_map:
        base.append(("device_map", repr(cfg.a.device_map)))
    if cfg.a.attn_implementation is not None:
        base.append(("attn_implementation", repr(cfg.a.attn_implementation)))
    base.append(("trust_remote_code", repr(cfg.a.trust_remote_code)))
    base.extend(_build_quantization_block(cfg.a.quantization))
    return _format_kwarg_lines(base, plan.kwargs.get("load_model", {}), indent=8)


def _build_tokenizer_kwargs_block(plan: ScriptPlan, cfg: ConfigRecord) -> str:
    """Render keyword arguments for ``AutoTokenizer.from_pretrained``."""
    base: list[tuple[str, str]] = [
        ("local_files_only", repr(cfg.a.local_files_only)),
        ("padding_side", repr(cfg.b.padding_side)),
        ("use_fast", repr(cfg.b.use_fast)),
        ("truncation_side", repr(cfg.b.truncation_side)),
    ]
    if cfg.b.model_max_length is not None:
        base.append(("model_max_length", repr(cfg.b.model_max_length)))
    if plan.fix_mistral_regex:
        base.append(("fix_mistral_regex", "True"))
    return _format_kwarg_lines(base, plan.kwargs.get("load_tokenizer", {}), indent=8)


def _build_encode_kwargs_block(plan: ScriptPlan, cfg: ConfigRecord) -> str:
    """Render keyword arguments for ``tokenizer(...)`` encoding call."""
    base: list[tuple[str, str]] = [
        ("return_tensors", repr(cfg.e.return_tensors)),
        ("padding", repr(cfg.e.padding)),
        ("truncation", repr(cfg.e.truncation)),
    ]
    if cfg.e.max_length is not None:
        base.append(("max_length", repr(cfg.e.max_length)))
    base.append(("add_special_tokens", "False"))
    base.append(("return_attention_mask", "True"))
    return _format_kwarg_lines(base, plan.kwargs.get("encode", {}), indent=12)


def _build_generate_kwargs_block(plan: ScriptPlan, cfg: ConfigRecord) -> str:
    """Render keyword arguments for ``model.generate(...)``."""
    base: list[tuple[str, str]] = [
        ("input_ids", 'encoded["input_ids"]'),
        ("attention_mask", 'encoded["attention_mask"]'),
        ("max_new_tokens", repr(cfg.f.max_new_tokens)),
        ("do_sample", repr(cfg.f.do_sample)),
    ]
    if cfg.f.do_sample:
        base.append(("temperature", "temperature"))
    if cfg.f.top_p is not None:
        base.append(("top_p", repr(cfg.f.top_p)))
    if cfg.f.top_k is not None:
        base.append(("top_k", repr(cfg.f.top_k)))
    base.extend(
        [
            ("use_cache", repr(cfg.f.use_cache)),
            ("num_return_sequences", repr(cfg.f.num_return_sequences)),
            ("repetition_penalty", repr(cfg.f.repetition_penalty)),
            ("no_repeat_ngram_size", repr(cfg.f.no_repeat_ngram_size)),
            (
                "pad_token_id",
                repr(cfg.f.pad_token_id)
                if cfg.f.pad_token_id is not None
                else "tokenizer.pad_token_id",
            ),
            (
                "eos_token_id",
                repr(cfg.f.eos_token_id)
                if cfg.f.eos_token_id is not None
                else "tokenizer.eos_token_id",
            ),
        ]
    )
    return _format_kwarg_lines(base, plan.kwargs.get("generate", {}), indent=16)


def _build_decode_kwargs_block(cfg: ConfigRecord) -> str:
    """Render keyword arguments for ``tokenizer.decode(...)``."""
    base: list[tuple[str, str]] = [
        ("skip_special_tokens", repr(cfg.g.skip_special_tokens)),
        ("clean_up_tokenization_spaces", repr(cfg.g.clean_up_tokenization_spaces)),
        (
            "spaces_between_special_tokens",
            repr(cfg.g.spaces_between_special_tokens),
        ),
    ]
    return _format_kwarg_lines(base, {}, indent=16)


_HELPERS_TEMPLATE = '''
def _timestamp() -> str:
    """Return a full microsecond timestamp compatible with the log parser."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def set_reproducibility_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and Transformers RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides without mutating CONFIG defaults unexpectedly."""
    parser = argparse.ArgumentParser(
        description="Run fact-checking inference and emit parser-compatible logs."
    )
    parser.add_argument("--modelId", type=str, default=CONFIG["model_id"])
    parser.add_argument("--testDataFile", type=str, default=CONFIG["test_data_file"])
    parser.add_argument("--outputDir", type=str, default=CONFIG["output_dir"])
    parser.add_argument(
        "--temperature",
        type=float,
        default=CONFIG["f"]["temperature"],
        help="Sampling temperature (used when do_sample=True).",
    )
    return parser.parse_args()


def _resolve_answer(
    evidence: dict,
    answers_used: int,
    missing_placeholder: str,
) -> str:
    """Return up to ``answers_used`` documents, or ``missing_placeholder`` if empty."""
    docs = evidence.get("top_k_doc") or []
    if docs:
        count = max(1, int(answers_used))
        return "\\n".join(str(doc) for doc in docs[:count])
    return missing_placeholder


def select_evidence_items(
    record: dict,
    mode: str,
    answers_used: int,
    missing_placeholder: str,
    rng: random.Random,
    donor_evidences: list[dict] | None = None,
) -> list[dict[str, str | None]]:
    """Select question/answer evidence pairs according to ``c.evidence_config``."""
    evidences = list(record.get("evidences") or [])

    if mode == "without_evidence":
        return []

    if mode == "with_evidence":
        items: list[dict[str, str | None]] = []
        for ev in evidences:
            question = str(ev.get("questions") or ev.get("question") or "")
            answer = _resolve_answer(ev, answers_used, missing_placeholder)
            items.append({"question": question, "answer": answer})
        return items

    if mode == "random_evidence":
        source = list(donor_evidences) if donor_evidences is not None else list(evidences)
        rng.shuffle(source)
        items = []
        for ev in source:
            question = str(ev.get("questions") or ev.get("question") or "")
            answer = _resolve_answer(ev, answers_used, missing_placeholder)
            items.append({"question": question, "answer": answer})
        return items

    if mode == "question_only":
        return [
            {
                "question": str(ev.get("questions") or ev.get("question") or ""),
                "answer": None,
            }
            for ev in evidences
        ]

    if mode == "answer_only":
        return [
            {
                "question": None,
                "answer": _resolve_answer(ev, answers_used, missing_placeholder),
            }
            for ev in evidences
        ]

    if mode == "all_questions_last_answer":
        last_idx = len(evidences) - 1
        items = []
        for idx, ev in enumerate(evidences):
            question = str(ev.get("questions") or ev.get("question") or "")
            if idx == last_idx:
                answer: str | None = _resolve_answer(ev, answers_used, missing_placeholder)
            else:
                answer = None
            items.append({"question": question, "answer": answer})
        return items

    if mode == "all_questions_all_but_last_answer":
        last_idx = len(evidences) - 1
        items = []
        for idx, ev in enumerate(evidences):
            question = str(ev.get("questions") or ev.get("question") or "")
            if idx < last_idx:
                answer = _resolve_answer(ev, answers_used, missing_placeholder)
            else:
                answer = None
            items.append({"question": question, "answer": answer})
        return items

    raise ValueError(f"Unsupported evidence_config: {mode!r}")


def format_evidence_items(items: list[dict[str, str | None]]) -> str:
    """Format selected evidence items with explicit branching preserving questions."""
    blocks: list[str] = []
    for item in items:
        question = item.get("question")
        answer = item.get("answer")
        if question is not None and answer is not None:
            blocks.append(f"### Question: {question}\\n### Answer: {answer}")
        elif question is not None:
            blocks.append(f"### Question: {question}")
        elif answer is not None:
            blocks.append(f"### Answer: {answer}")
    return "\\n".join(blocks)


def build_user_prompt(
    record: dict,
    mode: str,
    answers_used: int,
    missing_placeholder: str,
    user_prompt_template: str,
    rng: random.Random,
    donor_evidences: list[dict] | None = None,
) -> str:
    """Build the user message for one record respecting ``user_prompt_template``."""
    claim = str(record.get("claim") or "")
    items = select_evidence_items(
        record,
        mode,
        answers_used,
        missing_placeholder,
        rng,
        donor_evidences=donor_evidences,
    )
    evidence_text = format_evidence_items(items)

    if "{evidence}" in user_prompt_template:
        return user_prompt_template.replace("{claim}", claim).replace(
            "{evidence}", evidence_text
        )

    claim_section = user_prompt_template
    for marker in ("### Question:", "{question}", "### Answer:", "{answer}"):
        pos = claim_section.find(marker)
        if pos != -1:
            claim_section = claim_section[:pos]
    claim_header = claim_section.rstrip("\\n").replace("{claim}", claim)
    if not claim_header and claim:
        claim_header = f"### Claim: {claim}"
    if evidence_text:
        return f"{claim_header}\\n{evidence_text}"
    return claim_header


def build_messages(
    user_prompt: str,
    system_prompt: str,
    include_system_prompt: bool,
) -> list[dict[str, str]]:
    """Build chat messages, omitting empty system prompts when configured."""
    messages: list[dict[str, str]] = []
    if include_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def extract_verdict(decoded_output: str, marker: str) -> str:
    """Extract the verdict following the last occurrence of ``marker``."""
    verdict_index = decoded_output.rfind(marker)
    if verdict_index == -1:
        return "No response"
    tail = decoded_output[verdict_index + len(marker) :].strip()
    if not tail:
        return "No response"
    first_line = tail.splitlines()[0].strip()
    return first_line or "No response"
'''


def generate_inference_script(plan: ScriptPlan, cfg: ConfigRecord) -> str:
    """Generate a complete Python inference script for Mistral or DeepSeek.

    Parameters
    ----------
    plan:
        Normalized :class:`~app.codegen.planner.ScriptPlan` produced by
        :func:`~app.codegen.planner.build_script_plan`.
    cfg:
        Validated :class:`~app.schemas.ConfigRecord` for ``task_type="inference"``.
    """
    if plan.task_type != "inference" or cfg.task_type != "inference":
        raise ValueError(
            f"generate_inference_script requires task_type='inference', "
            f"got plan={plan.task_type!r}, cfg={cfg.task_type!r}"
        )

    adapter_path = _resolve_adapter_path(plan, cfg)
    config_dict = _build_config_dict(plan, cfg, adapter_path)
    config_literal = pprint.pformat(
        config_dict, indent=4, width=100, sort_dicts=False
    )

    imports_block = _build_imports_block(cfg.a.quantization, adapter_path)
    quantization_setup = _render_quantization_setup(cfg.a.quantization)
    model_kwargs_block = _build_model_kwargs_block(plan, cfg)
    tokenizer_kwargs_block = _build_tokenizer_kwargs_block(plan, cfg)
    encode_kwargs_block = _build_encode_kwargs_block(plan, cfg)
    generate_kwargs_block = _build_generate_kwargs_block(plan, cfg)
    decode_kwargs_block = _build_decode_kwargs_block(cfg)

    dist_init_block = ""
    dist_cleanup_block = ""
    if plan.uses_distributed:
        dist_init_block = (
            "    if (\n"
            "        torch.distributed.is_available()\n"
            "        and not torch.distributed.is_initialized()\n"
            '        and "RANK" in os.environ\n'
            '        and "WORLD_SIZE" in os.environ\n'
            "    ):\n"
            "        torch.distributed.init_process_group()\n"
        )
        dist_cleanup_block = (
            "    if torch.distributed.is_available() and torch.distributed.is_initialized():\n"
            "        torch.distributed.destroy_process_group()\n"
        )

    adapter_block = ""
    if adapter_path is not None:
        adapter_block = (
            "    model = PeftModel.from_pretrained(\n"
            '        model,\n'
            '        CONFIG["adapter_path"],\n'
            f"        local_files_only={repr(cfg.a.local_files_only)},\n"
            "    )\n"
        )

    chat_template_override_block = ""
    if plan.chat_template_override:
        chat_template_override_block = (
            '    tokenizer.chat_template = CONFIG["chat_template_override"]\n'
        )

    forced_prefix_block = ""
    if cfg.f.forced_prefix:
        forced_prefix_block = (
            '        if CONFIG["f"]["forced_prefix"]:\n'
            '            prompt = prompt + CONFIG["f"]["forced_prefix"]\n'
        )

    target_device_expr = (
        '"cuda" if torch.cuda.is_available() else "cpu"'
        if cfg.e.device == "cuda"
        else repr(cfg.e.device)
    )

    main_block = f'''
def main() -> None:
    """Load model/tokenizer/data, run batched inference, and print verdicts."""
    args = parse_args()
    model_id = args.modelId
    tokenizer_id = (
        model_id
        if CONFIG["tokenizer_id"] == CONFIG["model_id"]
        else CONFIG["tokenizer_id"]
    )
    test_data_file = args.testDataFile
    output_dir = args.outputDir
    temperature = args.temperature

    if not test_data_file:
        raise ValueError("testDataFile must be provided via CONFIG or --testDataFile.")

    set_reproducibility_seed(CONFIG["seed"])
    rng = random.Random(CONFIG["seed"])

{dist_init_block}    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading the Model : {{_timestamp()}}", flush=True)
{quantization_setup}    model = AutoModelForCausalLM.from_pretrained(
        model_id,
{model_kwargs_block}
    )
{adapter_block}    model.eval()
    print(f"Done Loading the Model : {{_timestamp()}}", flush=True)

    print(f"Loading the Tokenizer : {{_timestamp()}}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
{tokenizer_kwargs_block}
    )
    tokenizer.pad_token = {repr(cfg.b.pad_token)}
{chat_template_override_block}    print(f"Done Loading the Tokenizer : {{_timestamp()}}", flush=True)

    print(f"Loading the Data : {{_timestamp()}}", flush=True)
    with open(test_data_file, encoding="utf-8") as handle:
        test_data = json.load(handle)
    if CONFIG["max_samples"] is not None:
        test_data = test_data[: CONFIG["max_samples"]]
    print(f"Done Loading the Data : {{_timestamp()}}", flush=True)

    donor_pool: list[list[dict]] | None = None
    if CONFIG["evidence_config"] == "random_evidence":
        donor_pool = [list(rec.get("evidences") or []) for rec in test_data]
        rng.shuffle(donor_pool)

    test_prompts: list[str] = []
    for idx, record in enumerate(test_data):
        donor_evidences = donor_pool[idx] if donor_pool is not None else None
        user_prompt = build_user_prompt(
            record,
            CONFIG["evidence_config"],
            CONFIG["evidence_answers_used"],
            CONFIG["missing_answer_placeholder"],
            CONFIG["user_prompt_template"],
            rng,
            donor_evidences=donor_evidences,
        )
        messages = build_messages(
            user_prompt,
            CONFIG["system_prompt"],
            CONFIG["include_system_prompt"],
        )
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
{forced_prefix_block}        test_prompts.append(prompt)
    print(f"Done Preparing the Input : {{_timestamp()}}", flush=True)

    target_device = {target_device_expr}
    batch_size = max(1, int(CONFIG["f"]["batch_size"]))
    num_return_sequences = max(1, int(CONFIG["f"]["num_return_sequences"]))
    verdicts: list[str] = []

    for batch_start in range(0, len(test_prompts), batch_size):
        batch_prompts = test_prompts[batch_start : batch_start + batch_size]
        print(f"Tokenizing Inputs : {{_timestamp()}}", flush=True)
        encoded = tokenizer(
            batch_prompts,
{encode_kwargs_block}
        ).to(target_device)

        print("*****INPUTS*****", flush=True)
        print(encoded["input_ids"], flush=True)

        with torch.no_grad():
            outputs = model.generate(
{generate_kwargs_block}
            )

        print(f"Decoding the Output : {{_timestamp()}}", flush=True)
        for offset in range(len(batch_prompts)):
            seq = outputs[offset * num_return_sequences]
            decoded_output = tokenizer.decode(
                seq,
{decode_kwargs_block}
            )
            print("*****OUTPUT*****", flush=True)
            print(decoded_output, flush=True)
            verdict = extract_verdict(decoded_output, CONFIG["verdict_parse_marker"])
            verdicts.append(verdict)

    print("*****VERDICTS*****", flush=True)
    for idx, verdict in enumerate(verdicts):
        print(f"{{idx}} | {{verdict}}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    verdicts_path = os.path.join(output_dir, "verdicts.json")
    with open(verdicts_path, "w", encoding="utf-8") as out_handle:
        json.dump(
            [{{"index": idx, "verdict": verdict}} for idx, verdict in enumerate(verdicts)],
            out_handle,
            indent=2,
        )

{dist_cleanup_block}

if __name__ == "__main__":
    main()
'''

    header = (
        f'"""Generated inference script for {plan.model_id}.\n\n'
        f"Config ID: {plan.config_id}\n"
        f"Config Hash: {plan.config_hash}\n"
        f"Script Filename: {plan.script_filename}\n"
        '"""\n\n'
        "from __future__ import annotations\n\n"
    )

    return (
        header
        + imports_block
        + "\n\nCONFIG = "
        + config_literal
        + "\n"
        + _HELPERS_TEMPLATE
        + main_block
    )


__all__ = ["generate_inference_script"]
