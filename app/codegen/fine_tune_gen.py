"""Mistral & DeepSeek SFT + LoRA fine-tuning script generator (issue #57).

This module is pure Python and must never import ``nicegui``.
"""

from __future__ import annotations

from typing import Any

from app.codegen.planner import ScriptPlan
from app.schemas import ConfigRecord


def _format_py_literal(value: Any, indent: int = 0) -> str:
    """Format a Python value deterministically with readable dict/list indentation."""
    pad = " " * indent
    child_pad = " " * (indent + 4)

    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = ["{"]
        for key, item in value.items():
            rendered = _format_py_literal(item, indent + 4)
            lines.append(f"{child_pad}{key!r}: {rendered},")
        lines.append(f"{pad}}}")
        return "\n".join(lines)

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(not isinstance(item, (dict, list, tuple)) for item in value):
            inline = "[" + ", ".join(repr(item) for item in value) + "]"
            if len(pad) + len(inline) <= 88:
                return inline
        lines = ["["]
        for item in value:
            rendered = _format_py_literal(item, indent + 4)
            lines.append(f"{child_pad}{rendered},")
        lines.append(f"{pad}]")
        return "\n".join(lines)

    if isinstance(value, tuple):
        if not value:
            return "()"
        if len(value) == 1:
            return f"({_format_py_literal(value[0], indent)},)"
        return "(" + ", ".join(_format_py_literal(item, indent) for item in value) + ")"

    return repr(value)


def _format_kwargs_block(items: list[tuple[str, str]], indent: int = 8) -> str:
    """Render keyword argument lines from ``(name, value_expr)`` pairs."""
    pad = " " * indent
    return "\n".join(f"{pad}{name}={expr}," for name, expr in items)


def _build_config_payload(plan: ScriptPlan, cfg: ConfigRecord) -> dict[str, Any]:
    """Build the deterministic literal ``CONFIG`` dict embedded in the script."""
    output_dir = cfg.d.sft.output_dir.rstrip("/\\") or "fineTuningOutput"
    save_model_dir = plan.save_model_dir or f"{output_dir}/final_model"
    return {
        "config_id": plan.config_id,
        "config_hash": plan.config_hash,
        "task_type": plan.task_type,
        "model_id": plan.model_id,
        "tokenizer_id": plan.tokenizer_id,
        "train_data_file": cfg.c.train_data_file,
        "eval_data_file": cfg.c.eval_data_file,
        "output_dir": output_dir,
        "save_model_dir": save_model_dir,
        "include_system_prompt": plan.include_system_prompt,
        "uses_distributed": plan.uses_distributed,
        "fix_mistral_regex": plan.fix_mistral_regex,
        "max_seq_length": plan.max_seq_length,
        "a": cfg.a.model_dump(mode="json"),
        "b": cfg.b.model_dump(mode="json"),
        "c": cfg.c.model_dump(mode="json"),
        "d": cfg.d.model_dump(mode="json"),
        "p": cfg.p.model_dump(mode="json"),
        "h": plan.kwargs,
    }


def _build_model_kwargs(plan: ScriptPlan, cfg: ConfigRecord) -> list[tuple[str, str]]:
    """Build keyword arguments for ``AutoModelForCausalLM.from_pretrained``."""
    h_load_model = dict(plan.kwargs.get("load_model", {}))
    items: list[tuple[str, str]] = [
        ("dtype", repr(cfg.a.dtype)),
        ("local_files_only", repr(cfg.a.local_files_only)),
    ]
    used_keys = {"dtype", "local_files_only"}

    if cfg.a.tp_plan:
        items.append(("tp_plan", repr(cfg.a.tp_plan)))
        used_keys.add("tp_plan")
    elif "device_map" in h_load_model:
        items.append(("device_map", repr(h_load_model["device_map"])))
        used_keys.add("device_map")
    elif cfg.a.device_map:
        items.append(("device_map", repr(cfg.a.device_map)))
        used_keys.add("device_map")

    if "attn_implementation" in h_load_model:
        items.append(("attn_implementation", repr(h_load_model["attn_implementation"])))
        used_keys.add("attn_implementation")
    elif cfg.a.attn_implementation is not None:
        items.append(("attn_implementation", repr(cfg.a.attn_implementation)))
        used_keys.add("attn_implementation")

    trust_remote = h_load_model.get("trust_remote_code", cfg.a.trust_remote_code)
    items.append(("trust_remote_code", repr(trust_remote)))
    used_keys.add("trust_remote_code")

    if "quantization_config" in h_load_model:
        items.append(("quantization_config", repr(h_load_model["quantization_config"])))
        used_keys.add("quantization_config")
    elif cfg.a.quantization is not None:
        items.append(("quantization_config", "quantization_config"))
        used_keys.add("quantization_config")

    for key in sorted(h_load_model):
        if key not in used_keys:
            items.append((key, repr(h_load_model[key])))

    return items


def _build_tokenizer_kwargs(plan: ScriptPlan, cfg: ConfigRecord) -> list[tuple[str, str]]:
    """Build keyword arguments for ``AutoTokenizer.from_pretrained``."""
    h_load_tok = dict(plan.kwargs.get("load_tokenizer", {}))
    items: list[tuple[str, str]] = [
        ("local_files_only", repr(cfg.a.local_files_only)),
        ("padding_side", repr(h_load_tok.get("padding_side", cfg.b.padding_side))),
        ("use_fast", repr(h_load_tok.get("use_fast", cfg.b.use_fast))),
        ("truncation_side", repr(h_load_tok.get("truncation_side", cfg.b.truncation_side))),
    ]
    used_keys = {"local_files_only", "padding_side", "use_fast", "truncation_side"}

    if "model_max_length" in h_load_tok:
        items.append(("model_max_length", repr(h_load_tok["model_max_length"])))
        used_keys.add("model_max_length")
    elif cfg.b.model_max_length is not None:
        items.append(("model_max_length", repr(cfg.b.model_max_length)))
        used_keys.add("model_max_length")

    if "trust_remote_code" in h_load_tok:
        items.append(("trust_remote_code", repr(h_load_tok["trust_remote_code"])))
        used_keys.add("trust_remote_code")
    elif cfg.a.trust_remote_code:
        items.append(("trust_remote_code", repr(cfg.a.trust_remote_code)))
        used_keys.add("trust_remote_code")

    if plan.fix_mistral_regex:
        items.append(("fix_mistral_regex", "True"))
        used_keys.add("fix_mistral_regex")

    for key in sorted(h_load_tok):
        if key not in used_keys:
            items.append((key, repr(h_load_tok[key])))

    return items


def _build_lora_kwargs(cfg: ConfigRecord) -> list[tuple[str, str]]:
    """Build keyword arguments for ``LoraConfig``."""
    lora = cfg.d.lora
    return [
        ("r", repr(lora.r)),
        ("lora_alpha", repr(lora.lora_alpha)),
        ("lora_dropout", repr(lora.lora_dropout)),
        ("bias", repr(lora.bias)),
        ("target_modules", repr(list(lora.target_modules))),
        ("use_dora", repr(lora.use_dora)),
        ("init_lora_weights", repr(lora.init_lora_weights)),
    ]


def _build_sft_kwargs(plan: ScriptPlan, cfg: ConfigRecord) -> list[tuple[str, str]]:
    """Build keyword arguments for ``SFTConfig``."""
    sft = cfg.d.sft
    items: list[tuple[str, str]] = [
        ("output_dir", "output_dir"),
        ("learning_rate", repr(sft.learning_rate)),
        ("weight_decay", repr(sft.weight_decay)),
        ("num_train_epochs", repr(sft.num_train_epochs)),
        ("warmup_steps", repr(sft.warmup_steps)),
        ("optim", repr(sft.optim)),
        ("per_device_train_batch_size", repr(sft.per_device_train_batch_size)),
        ("gradient_accumulation_steps", repr(sft.gradient_accumulation_steps)),
        ("lr_scheduler_type", repr(sft.lr_scheduler_type)),
        ("max_grad_norm", repr(sft.max_grad_norm)),
        ("eval_strategy", repr(sft.eval_strategy)),
        ("save_strategy", repr(sft.save_strategy)),
        ("logging_strategy", repr(sft.logging_strategy)),
        ("logging_steps", repr(sft.logging_steps)),
        ("load_best_model_at_end", repr(sft.load_best_model_at_end)),
        ("metric_for_best_model", repr(sft.metric_for_best_model)),
        ("push_to_hub", repr(sft.push_to_hub)),
        ("bf16", repr(sft.bf16)),
        ("gradient_checkpointing", repr(sft.gradient_checkpointing)),
        ("seed", repr(sft.seed)),
    ]
    if plan.max_seq_length is not None:
        items.append(("max_length", repr(plan.max_seq_length)))
    return items


def _render_quantization_block(quantization: str | None) -> str:
    """Render conditional ``BitsAndBytesConfig`` setup when quantization is enabled."""
    if quantization == "4bit":
        return (
            "    quantization_config = BitsAndBytesConfig(\n"
            "        load_in_4bit=True,\n"
            '        bnb_4bit_quant_type="nf4",\n'
            "        bnb_4bit_use_double_quant=True,\n"
            "        bnb_4bit_compute_dtype=torch.bfloat16,\n"
            "    )\n"
        )
    if quantization == "8bit":
        return (
            "    quantization_config = BitsAndBytesConfig(\n"
            "        load_in_8bit=True,\n"
            "    )\n"
        )
    return ""


def generate_fine_tune_script(plan: ScriptPlan, cfg: ConfigRecord) -> str:
    """Generate a self-contained SFT + LoRA fine-tuning script from ``plan`` and ``cfg``."""
    if plan.task_type != "fine_tune":
        raise ValueError(f"Expected task_type 'fine_tune', got {plan.task_type!r}")

    config_literal = _format_py_literal(_build_config_payload(plan, cfg), indent=0)
    transformers_imports = ["AutoModelForCausalLM", "AutoTokenizer"]
    if cfg.a.quantization is not None:
        transformers_imports.append("BitsAndBytesConfig")
    transformers_imports.append("set_seed")
    transformers_import_line = f"from transformers import {', '.join(transformers_imports)}"

    distributed_init_block = (
        "    if not torch.distributed.is_initialized():\n"
        "        torch.distributed.init_process_group()\n"
        if plan.uses_distributed
        else ""
    )
    distributed_cleanup_block = (
        "        if torch.distributed.is_initialized():\n"
        "            torch.distributed.destroy_process_group()\n"
    ) if plan.uses_distributed else ""

    quantization_block = _render_quantization_block(cfg.a.quantization)
    model_kwargs_block = _format_kwargs_block(_build_model_kwargs(plan, cfg), indent=8)
    tokenizer_kwargs_block = _format_kwargs_block(_build_tokenizer_kwargs(plan, cfg), indent=8)
    lora_kwargs_block = _format_kwargs_block(_build_lora_kwargs(cfg), indent=8)
    sft_kwargs_block = _format_kwargs_block(_build_sft_kwargs(plan, cfg), indent=8)

    chat_template_block = (
        "    if CONFIG['p']['chat_template_override']:\n"
        "        tokenizer.chat_template = CONFIG['p']['chat_template_override']\n"
    )

    if plan.uses_distributed:
        train_eval_save_block = (
            "    try:\n"
            "        model.train()\n"
            "        trainer.train()\n"
            "        if eval_dataset is not None:\n"
            "            trainer.evaluate()\n"
            "        trainer.save_model(save_model_dir)\n"
            '        print("Done Saving the Model : ", datetime.datetime.now())\n'
            "    finally:\n"
            f"{distributed_cleanup_block}"
        )
    else:
        train_eval_save_block = (
            "    model.train()\n"
            "    trainer.train()\n"
            "    if eval_dataset is not None:\n"
            "        trainer.evaluate()\n"
            "    trainer.save_model(save_model_dir)\n"
            '    print("Done Saving the Model : ", datetime.datetime.now())\n'
        )

    lines = [
        f'"""Auto-generated fine-tuning script for {plan.model_id}.',
        "",
        f"Config ID: {plan.config_id}",
        f"Config Hash: {plan.config_hash}",
        f"Script Filename: {plan.script_filename}",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "import datetime",
        "import json",
        "import os",
        "import random",
        "",
        "import numpy as np",
        "import torch",
        "from datasets import Dataset",
        "from peft import LoraConfig",
        transformers_import_line,
        "from trl import SFTConfig, SFTTrainer",
        "",
        f"CONFIG = {config_literal}",
        "",
        "",
        "def set_reproducibility_seed(seed: int) -> None:",
        '    """Set random seeds across stdlib, NumPy, PyTorch, and Transformers."""',
        "    random.seed(seed)",
        "    np.random.seed(seed)",
        "    torch.manual_seed(seed)",
        "    if torch.cuda.is_available():",
        "        torch.cuda.manual_seed_all(seed)",
        "    set_seed(seed)",
        "",
        "",
        "def format_evidence_items(",
        "    evidences: list[dict],",
        "    evidence_answers_used: int = CONFIG['c']['evidence_answers_used'],",
        "    missing_answer_placeholder: str = CONFIG['c']['missing_answer_placeholder'],",
        ") -> str:",
        '    """Format question/answer evidence items while always preserving questions."""',
        "    blocks: list[str] = []",
        "    for evidence in evidences:",
        "        question = str(",
        '            evidence.get("questions")',
        '            if "questions" in evidence',
        '            else evidence.get("question", "")',
        "        )",
        '        docs = evidence.get("top_k_doc") or []',
        "        if evidence_answers_used > 0 and docs:",
        '            answer = "\\n".join(str(doc) for doc in docs[:evidence_answers_used])',
        "        else:",
        "            answer = missing_answer_placeholder",
        '        blocks.append(f"### Question: {question}\\n### Answer: {answer}")',
        '    return "\\n".join(blocks)',
        "",
        "",
        "def _render_user_prompt(",
        "    claim: str,",
        "    evidences: list[dict],",
        "    user_prompt_template: str,",
        "    evidence_answers_used: int,",
        "    missing_answer_placeholder: str,",
        ") -> str:",
        '    """Render the user prompt from ``user_prompt_template`` and evidence items."""',
        "    evidence_text = format_evidence_items(",
        "        evidences,",
        "        evidence_answers_used=evidence_answers_used,",
        "        missing_answer_placeholder=missing_answer_placeholder,",
        "    )",
        '    default_tail = "### Question: {question}\\n### Answer: {answer}"',
        "    if user_prompt_template.endswith(default_tail):",
        "        prefix = user_prompt_template[: -len(default_tail)].replace('{claim}', claim)",
        "        if evidence_text:",
        "            return f'{prefix}{evidence_text}'",
        "        return prefix.rstrip('\\n')",
        "    if '{question}' in user_prompt_template and '{answer}' in user_prompt_template:",
        "        if evidences:",
        "            first = evidences[0]",
        "            first_question = str(",
        "                first.get('questions') if 'questions' in first else first.get('question', '')",
        "            )",
        "            first_docs = first.get('top_k_doc') or []",
        "            if evidence_answers_used > 0 and first_docs:",
        '                first_answer = "\\n".join(str(doc) for doc in first_docs[:evidence_answers_used])',
        "            else:",
        "                first_answer = missing_answer_placeholder",
        "        else:",
        "            first_question = ''",
        "            first_answer = missing_answer_placeholder",
        "        rendered = (",
        "            user_prompt_template",
        "            .replace('{claim}', claim)",
        "            .replace('{question}', first_question)",
        "            .replace('{answer}', first_answer)",
        "        )",
        "        if len(evidences) > 1:",
        "            extra = format_evidence_items(",
        "                evidences[1:],",
        "                evidence_answers_used=evidence_answers_used,",
        "                missing_answer_placeholder=missing_answer_placeholder,",
        "            )",
        "            return f'{rendered}\\n{extra}'",
        "        return rendered",
        "    rendered = user_prompt_template.replace('{claim}', claim)",
        "    if '{evidence}' in rendered:",
        "        return rendered.replace('{evidence}', evidence_text)",
        "    if evidence_text:",
        "        return f'{rendered}\\n{evidence_text}'",
        "    return rendered",
        "",
        "",
        "def build_messages(",
        "    record: dict,",
        "    system_prompt: str = CONFIG['p']['system_prompt'],",
        "    include_system_prompt: bool = CONFIG['include_system_prompt'],",
        "    user_prompt_template: str = CONFIG['p']['user_prompt_template'],",
        "    assistant_prompt: str = CONFIG['p']['assistant_prompt'],",
        "    evidence_answers_used: int = CONFIG['c']['evidence_answers_used'],",
        "    missing_answer_placeholder: str = CONFIG['c']['missing_answer_placeholder'],",
        ") -> list[dict[str, str]]:",
        '    """Build one chat-format training/eval sample consistently for both splits."""',
        "    claim = str(record.get('claim', ''))",
        "    evidences = list(record.get('evidences') or [])",
        "    user_content = _render_user_prompt(",
        "        claim=claim,",
        "        evidences=evidences,",
        "        user_prompt_template=user_prompt_template,",
        "        evidence_answers_used=evidence_answers_used,",
        "        missing_answer_placeholder=missing_answer_placeholder,",
        "    )",
        "    assistant_content = f\"{assistant_prompt}{record.get('label', '')}\"",
        "    messages: list[dict[str, str]] = []",
        "    if include_system_prompt and system_prompt.strip():",
        '        messages.append({"role": "system", "content": system_prompt})',
        '    messages.append({"role": "user", "content": user_content})',
        '    messages.append({"role": "assistant", "content": assistant_content})',
        "    return messages",
        "",
        "",
        "def build_dataset(",
        "    records: list[dict],",
        "    max_samples: int | None = CONFIG['c']['max_samples'],",
        "    system_prompt: str = CONFIG['p']['system_prompt'],",
        "    include_system_prompt: bool = CONFIG['include_system_prompt'],",
        "    user_prompt_template: str = CONFIG['p']['user_prompt_template'],",
        "    assistant_prompt: str = CONFIG['p']['assistant_prompt'],",
        "    evidence_answers_used: int = CONFIG['c']['evidence_answers_used'],",
        "    missing_answer_placeholder: str = CONFIG['c']['missing_answer_placeholder'],",
        ") -> Dataset:",
        '    """Convert raw JSON records into a conversational Hugging Face ``Dataset``."""',
        "    selected = records[:max_samples] if max_samples is not None else records",
        "    samples = [",
        "        build_messages(",
        "            record,",
        "            system_prompt=system_prompt,",
        "            include_system_prompt=include_system_prompt,",
        "            user_prompt_template=user_prompt_template,",
        "            assistant_prompt=assistant_prompt,",
        "            evidence_answers_used=evidence_answers_used,",
        "            missing_answer_placeholder=missing_answer_placeholder,",
        "        )",
        "        for record in selected",
        "    ]",
        '    return Dataset.from_dict({"messages": samples})',
        "",
        "",
        "def parse_args(argv: list[str] | None = None) -> argparse.Namespace:",
        '    """Parse optional CLI overrides for model, dataset, and output paths."""',
        '    parser = argparse.ArgumentParser(description="Fine-tune causal LM with LoRA and SFT.")',
        '    parser.add_argument("--modelId", type=str, default=None, help="Override model ID or local path.")',
        '    parser.add_argument("--trainDataFile", type=str, default=None, help="Override training JSON file path.")',
        '    parser.add_argument("--evalDataFile", type=str, default=None, help="Override evaluation JSON file path.")',
        '    parser.add_argument("--outputDir", type=str, default=None, help="Override trainer output directory.")',
        '    parser.add_argument("--saveModelDir", type=str, default=None, help="Override final adapter save directory.")',
        "    return parser.parse_args(argv)",
        "",
        "",
        "def main() -> None:",
        '    """Run LoRA supervised fine-tuning and save the final adapter."""',
        "    args = parse_args()",
        f"    set_reproducibility_seed({cfg.d.sft.seed!r})",
        "",
        "    if torch.cuda.is_available():",
        "        torch.cuda.empty_cache()",
        "",
        "    model_id = args.modelId if args.modelId is not None else CONFIG['model_id']",
        "    tokenizer_id = (",
        "        args.modelId",
        "        if args.modelId is not None and CONFIG['tokenizer_id'] == CONFIG['model_id']",
        "        else CONFIG['tokenizer_id']",
        "    )",
        "    train_data_file = (",
        "        args.trainDataFile if args.trainDataFile is not None else CONFIG['train_data_file']",
        "    )",
        "    eval_data_file = (",
        "        args.evalDataFile if args.evalDataFile is not None else CONFIG['eval_data_file']",
        "    )",
        "    output_dir = args.outputDir if args.outputDir is not None else CONFIG['output_dir']",
        "    save_model_dir = (",
        "        args.saveModelDir",
        "        if args.saveModelDir is not None",
        '        else os.path.join(output_dir, "final_model")',
        "    )",
        "",
    ]

    if distributed_init_block:
        lines.append(distributed_init_block.rstrip("\n"))
        lines.append("")

    if quantization_block:
        lines.append(quantization_block.rstrip("\n"))
        lines.append("")

    lines.extend(
        [
            '    print("Start - Loading the model - ", datetime.datetime.now())',
            "    model = AutoModelForCausalLM.from_pretrained(",
            "        model_id,",
            model_kwargs_block,
            "    )",
            '    print("End - Loading the model - ", datetime.datetime.now())',
            '    print("Done Loading the Model : ", datetime.datetime.now())',
            "",
            '    print("Start - Loading the Tokenizer : ", datetime.datetime.now())',
            "    tokenizer = AutoTokenizer.from_pretrained(",
            "        tokenizer_id,",
            tokenizer_kwargs_block,
            "    )",
            f"    tokenizer.pad_token = {cfg.b.pad_token!r}",
            chat_template_block.rstrip("\n"),
            '    print("Done Loading the Tokenizer : ", datetime.datetime.now())',
            "",
            "    lora_config = LoraConfig(",
            lora_kwargs_block,
            "    )",
            "    model.add_adapter(lora_config)",
            "",
            "    training_args = SFTConfig(",
            sft_kwargs_block,
            "    )",
            "",
            '    print("Start - Loading the Train Data : ", datetime.datetime.now())',
            '    with open(train_data_file, encoding="utf-8") as train_handle:',
            "        train_records = json.load(train_handle)",
            "    train_dataset = build_dataset(train_records)",
            '    print("Done Loading the Train Data : ", datetime.datetime.now())',
            "",
            "    eval_dataset = None",
            "    if eval_data_file:",
            '        print("Start - Loading the Eval Data : ", datetime.datetime.now())',
            '        with open(eval_data_file, encoding="utf-8") as eval_handle:',
            "            eval_records = json.load(eval_handle)",
            "        eval_dataset = build_dataset(eval_records)",
            '        print("Done Loading the Eval Data : ", datetime.datetime.now())',
            "",
            "    trainer = SFTTrainer(",
            "        model=model,",
            "        args=training_args,",
            "        processing_class=tokenizer,",
            "        train_dataset=train_dataset,",
            "        eval_dataset=eval_dataset,",
            "    )",
            "",
            train_eval_save_block.rstrip("\n"),
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
            "",
        ]
    )

    return "\n".join(lines)


__all__ = ["generate_fine_tune_script"]
