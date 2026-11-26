"""Unit tests for Issue #57: fine-tune script generator (fine_tune_gen.py)."""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.codegen.fine_tune_gen import generate_fine_tune_script
from app.codegen.planner import build_script_plan
from app.parsing.errors import (
    DONE_LOADING_EVAL_DATA,
    DONE_LOADING_MODEL,
    DONE_LOADING_TOKENIZER,
    DONE_LOADING_TRAIN_DATA,
)
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord

_CHECK_NO_CJK_SPEC = importlib.util.spec_from_file_location(
    "check_no_cjk", ROOT / ".github" / "scripts" / "check_no_cjk.py"
)
assert _CHECK_NO_CJK_SPEC is not None and _CHECK_NO_CJK_SPEC.loader is not None
_CHECK_NO_CJK = importlib.util.module_from_spec(_CHECK_NO_CJK_SPEC)
_CHECK_NO_CJK_SPEC.loader.exec_module(_CHECK_NO_CJK)


def _make_cfg(model_id: str, **overrides) -> ConfigRecord:
    payload = defaults_for(model_id, "fine_tune")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            for sub_k, sub_v in value.items():
                if isinstance(sub_v, dict) and isinstance(payload[key].get(sub_k), dict):
                    payload[key][sub_k].update(sub_v)
                else:
                    payload[key][sub_k] = sub_v
        else:
            payload[key] = value
    return ConfigRecord(**payload)


def _exec_generated_namespace(code: str) -> dict:
    """Execute generated script module definitions with lightweight ML stubs."""

    class _FakeDataset:
        def __init__(self, data: dict) -> None:
            self.data = data

        @classmethod
        def from_dict(cls, data: dict) -> _FakeDataset:
            return cls(data)

    stub_modules = {}
    for mod_name in ("numpy", "torch", "datasets", "peft", "transformers", "trl"):
        stub = types.ModuleType(mod_name)
        if mod_name == "datasets":
            stub.Dataset = _FakeDataset  # type: ignore[attr-defined]
        elif mod_name == "peft":
            stub.LoraConfig = object  # type: ignore[attr-defined]
        elif mod_name == "transformers":
            stub.AutoModelForCausalLM = object  # type: ignore[attr-defined]
            stub.AutoTokenizer = object  # type: ignore[attr-defined]
            stub.BitsAndBytesConfig = object  # type: ignore[attr-defined]
            stub.set_seed = lambda seed: None  # type: ignore[attr-defined]
        elif mod_name == "trl":
            stub.SFTConfig = object  # type: ignore[attr-defined]
            stub.SFTTrainer = object  # type: ignore[attr-defined]
        stub_modules[mod_name] = stub

    saved = {name: sys.modules.get(name) for name in stub_modules}
    sys.modules.update(stub_modules)
    try:
        namespace: dict = {"__name__": "generated_finetune_test"}
        exec(compile(code, "generated_finetune.py", "exec"), namespace)
        return namespace
    finally:
        for name, orig in saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig


def _find_call(tree: ast.AST, func_name: str) -> ast.Call:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == func_name:
                return node
            if isinstance(node.func, ast.Attribute) and node.func.attr == func_name:
                return node
    raise AssertionError(f"Call to {func_name!r} not found in AST")


def _find_attr_call(tree: ast.AST, obj_name: str, attr_name: str) -> ast.Call:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr_name
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == obj_name
        ):
            return node
    raise AssertionError(f"Call to {obj_name}.{attr_name} not found in AST")


def _kwarg_map(call: ast.Call) -> dict[str, ast.expr]:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}


@pytest.mark.parametrize("model_id", [MISTRAL_ID, DEEPSEEK_ID])
def test_ast_parse_and_compile_for_both_models(model_id: str) -> None:
    """Generated scripts for both Mistral and DeepSeek parse and compile cleanly."""
    cfg = _make_cfg(model_id)
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)

    tree = ast.parse(code)
    compile(tree, plan.script_filename, "exec")

    assert "CONFIG =" in code
    assert _CHECK_NO_CJK.first_offence(code) is None


@pytest.mark.parametrize("model_id", [MISTRAL_ID, DEEPSEEK_ID])
def test_all_six_defects_absent_and_seed_present(model_id: str) -> None:
    """All 6 reference-script defects are eliminated in the generated script."""
    cfg = _make_cfg(model_id)
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)

    # Defect 2: never <myPadToken>, always validated cfg.b.pad_token
    assert "<myPadToken>" not in code
    assert f"tokenizer.pad_token = {cfg.b.pad_token!r}" in code

    # Defect 3: never post-load tokenizer.fix_mistral_regex = ...
    assert "tokenizer.fix_mistral_regex" not in code
    tok_call = _find_attr_call(ast.parse(code), "AutoTokenizer", "from_pretrained")
    tok_kwargs = _kwarg_map(tok_call)
    if model_id == MISTRAL_ID:
        assert "fix_mistral_regex" in tok_kwargs
        assert ast.literal_eval(tok_kwargs["fix_mistral_regex"]) is True
    else:
        assert "fix_mistral_regex" not in tok_kwargs

    # Never hard-code rank=0 in distributed init
    assert "rank=0" not in code
    assert "init_process_group()" in code
    assert "destroy_process_group()" in code

    # Defect 4: single DRY build_dataset/build_messages path, no inconsistent Response prefix
    assert "def build_messages(" in code
    assert "def build_dataset(" in code
    assert "NOTE TO SELF" not in code
    assert code.count("build_dataset(") >= 3  # 1 def + train + eval calls

    # Defect 5: clean imports and CLI overrides without 0.1 overwrite
    assert "import csv" not in code
    assert "import re\n" not in code
    assert "import pandas" not in code
    assert code.count("import argparse") == 1
    for flag in ("--modelId", "--trainDataFile", "--evalDataFile", "--outputDir", "--saveModelDir"):
        assert flag in code
    assert "model.generation_config.temperature = 0.1" not in code

    # Defect 6: set_reproducibility_seed defined & called, SFTConfig(seed=...)
    assert "def set_reproducibility_seed(" in code
    assert f"set_reproducibility_seed({cfg.d.sft.seed!r})" in code
    sft_call = _find_call(ast.parse(code), "SFTConfig")
    sft_kwargs = _kwarg_map(sft_call)
    assert ast.literal_eval(sft_kwargs["seed"]) == cfg.d.sft.seed


def test_mutations_update_generated_script() -> None:
    """Mutating quantization, pad_token, seed, system_prompt, and h.load_model updates code."""
    cfg_base = _make_cfg(MISTRAL_ID)
    plan_base, _ = build_script_plan(cfg_base)
    code_base = generate_fine_tune_script(plan_base, cfg_base)
    assert "BitsAndBytesConfig" not in code_base

    cfg_mut = _make_cfg(
        MISTRAL_ID,
        a={"quantization": "4bit"},
        b={"pad_token": "</s>"},
        d={"sft": {"seed": 1337}},
        p={"system_prompt": "Custom fact-checking instruction."},
        h={"load_model": {"low_cpu_mem_usage": True, "revision": "v2"}},
    )
    plan_mut, _ = build_script_plan(cfg_mut)
    code_mut = generate_fine_tune_script(plan_mut, cfg_mut)

    assert "BitsAndBytesConfig" in code_mut
    assert "load_in_4bit=True" in code_mut
    assert "tokenizer.pad_token = '</s>'" in code_mut
    assert "set_reproducibility_seed(1337)" in code_mut
    assert "Custom fact-checking instruction." in code_mut

    tree = ast.parse(code_mut)
    model_call = _find_attr_call(tree, "AutoModelForCausalLM", "from_pretrained")
    model_kwargs = _kwarg_map(model_call)
    assert ast.literal_eval(model_kwargs["low_cpu_mem_usage"]) is True
    assert ast.literal_eval(model_kwargs["revision"]) == "v2"
    assert isinstance(model_kwargs["quantization_config"], ast.Name)
    assert model_kwargs["quantization_config"].id == "quantization_config"

    sft_call = _find_call(tree, "SFTConfig")
    assert ast.literal_eval(_kwarg_map(sft_call)["seed"]) == 1337


def test_8bit_quantization_and_device_map_without_tp_plan() -> None:
    """8-bit quantization emits BitsAndBytesConfig; tp_plan=None omits distributed init."""
    cfg = _make_cfg(
        DEEPSEEK_ID,
        a={"quantization": "8bit", "tp_plan": None, "device_map": "auto"},
    )
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)

    assert "BitsAndBytesConfig" in code
    assert "load_in_8bit=True" in code
    assert "init_process_group" not in code
    assert "destroy_process_group" not in code

    model_call = _find_attr_call(ast.parse(code), "AutoModelForCausalLM", "from_pretrained")
    model_kwargs = _kwarg_map(model_call)
    assert "tp_plan" not in model_kwargs
    assert ast.literal_eval(model_kwargs["device_map"]) == "auto"


def test_deepseek_empty_system_prompt_omits_system_role() -> None:
    """DeepSeek with empty system_prompt omits the system role message in build_messages."""
    cfg = _make_cfg(DEEPSEEK_ID, p={"system_prompt": ""})
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)
    ns = _exec_generated_namespace(code)

    sample = {
        "claim": "Water boils at 100C at 1 atm.",
        "label": "True",
        "evidences": [{"questions": "What is the boiling point?", "top_k_doc": ["100C"]}],
    }
    msgs = ns["build_messages"](sample)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "### Response: Verdict: True"


def test_deepseek_nonempty_system_prompt_includes_system_role() -> None:
    """DeepSeek with non-empty system_prompt includes the system role message."""
    cfg = _make_cfg(DEEPSEEK_ID, p={"system_prompt": "Check the numerical claim."})
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)
    ns = _exec_generated_namespace(code)

    sample = {
        "claim": "Water boils at 100C.",
        "label": "True",
        "evidences": [{"questions": "Boiling point?", "top_k_doc": ["100C"]}],
    }
    msgs = ns["build_messages"](sample)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[0]["content"] == "Check the numerical claim."


def test_isolated_execution_empty_top_k_doc_preserves_question() -> None:
    """Executing format_evidence_items/build_messages proves empty top_k_doc keeps question."""
    cfg = _make_cfg(
        MISTRAL_ID,
        c={
            "max_samples": 1,
            "evidence_answers_used": 2,
            "missing_answer_placeholder": "No evidence found.",
        },
        p={
            "assistant_prompt": "### Response: Verdict: ",
            "verdict_parse_marker": "### Response: Verdict: ",
        },
    )
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)
    ns = _exec_generated_namespace(code)

    formatted_empty = ns["format_evidence_items"](
        [{"questions": "Is GDP rising?", "top_k_doc": []}]
    )
    assert "### Question: Is GDP rising?" in formatted_empty
    assert "### Answer: No evidence found." in formatted_empty

    formatted_multi = ns["format_evidence_items"](
        [{"questions": "What are the values?", "top_k_doc": ["Ans A", "Ans B", "Ans C"]}]
    )
    assert "Ans A" in formatted_multi
    assert "Ans B" in formatted_multi
    assert "Ans C" not in formatted_multi

    records = [
        {
            "claim": "The claim text.",
            "label": "False",
            "evidences": [
                {"questions": "Preserved question?", "top_k_doc": []},
            ],
        },
        {
            "claim": "Ignored due to max_samples=1.",
            "label": "True",
            "evidences": [{"questions": "Q2?", "top_k_doc": ["A2"]}],
        },
    ]
    msgs = ns["build_messages"](records[0])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "### Claim: The claim text." in msgs[1]["content"]
    assert "### Question: Preserved question?" in msgs[1]["content"]
    assert "### Answer: No evidence found." in msgs[1]["content"]
    assert msgs[2]["content"] == "### Response: Verdict: False"

    ds = ns["build_dataset"](records)
    assert len(ds.data["messages"]) == 1
    assert ds.data["messages"][0] == msgs


def test_custom_user_prompt_template_and_chat_template_override() -> None:
    """Custom user_prompt_template and chat_template_override are respected."""
    cfg = _make_cfg(
        MISTRAL_ID,
        p={
            "user_prompt_template": "Verify: {claim}\nQ: {question}\nA: {answer}",
            "chat_template_override": "{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}",
        },
    )
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)
    ns = _exec_generated_namespace(code)

    assert "tokenizer.chat_template = CONFIG['p']['chat_template_override']" in code

    msgs = ns["build_messages"](
        {
            "claim": "Inflation was 2%.",
            "label": "Conflicting",
            "evidences": [{"questions": "What was inflation?", "top_k_doc": []}],
        }
    )
    assert msgs[1]["content"] == (
        "Verify: Inflation was 2%.\n"
        "Q: What was inflation?\n"
        "A: No answer could be found !"
    )


def test_lora_sft_save_model_and_phase_markers_present() -> None:
    """Generated script injects LoraConfig, SFTConfig, save_model, and log markers."""
    cfg = _make_cfg(
        MISTRAL_ID,
        d={
            "lora": {"r": 32, "lora_alpha": 64, "use_dora": True},
            "sft": {
                "output_dir": "customOut",
                "bf16": False,
                "gradient_checkpointing": True,
                "seed": 99,
            },
        },
    )
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)

    tree = ast.parse(code)
    lora_kwargs = _kwarg_map(_find_call(tree, "LoraConfig"))
    assert ast.literal_eval(lora_kwargs["r"]) == 32
    assert ast.literal_eval(lora_kwargs["lora_alpha"]) == 64
    assert ast.literal_eval(lora_kwargs["use_dora"]) is True
    assert ast.literal_eval(lora_kwargs["init_lora_weights"]) is True

    sft_kwargs = _kwarg_map(_find_call(tree, "SFTConfig"))
    assert ast.literal_eval(sft_kwargs["bf16"]) is False
    assert ast.literal_eval(sft_kwargs["gradient_checkpointing"]) is True
    assert ast.literal_eval(sft_kwargs["seed"]) == 99
    assert ast.literal_eval(sft_kwargs["max_length"]) == 4096

    assert "model.add_adapter(lora_config)" in code
    assert "trainer.train()" in code
    assert "trainer.evaluate()" in code
    assert 'os.path.join(output_dir, "final_model")' in code
    assert "trainer.save_model(save_model_dir)" in code

    for marker in (
        "Start - Loading the model -",
        "End - Loading the model -",
        f"{DONE_LOADING_MODEL} :",
        f"{DONE_LOADING_TOKENIZER} :",
        f"{DONE_LOADING_TRAIN_DATA} :",
        f"{DONE_LOADING_EVAL_DATA} :",
    ):
        assert marker in code


def test_cli_parse_args_and_pure_import() -> None:
    """CLI overrides parse cleanly and fine_tune_gen imports without nicegui."""
    cfg = _make_cfg(MISTRAL_ID)
    plan, _ = build_script_plan(cfg)
    code = generate_fine_tune_script(plan, cfg)
    ns = _exec_generated_namespace(code)

    parsed = ns["parse_args"](
        [
            "--modelId", "/models/mistral",
            "--trainDataFile", "/data/tr.json",
            "--evalDataFile", "/data/ev.json",
            "--outputDir", "/runs/out",
            "--saveModelDir", "/runs/final",
        ]
    )
    assert parsed.modelId == "/models/mistral"
    assert parsed.trainDataFile == "/data/tr.json"
    assert parsed.evalDataFile == "/data/ev.json"
    assert parsed.outputDir == "/runs/out"
    assert parsed.saveModelDir == "/runs/final"

    with pytest.raises(ValueError):
        inf_cfg = ConfigRecord(**defaults_for(MISTRAL_ID, "inference"))
        inf_plan, _ = build_script_plan(inf_cfg)
        generate_fine_tune_script(inf_plan, inf_cfg)

    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import app.codegen.fine_tune_gen; "
                "assert 'nicegui' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_generated_files_contain_no_cjk() -> None:
    """Neither fine_tune_gen.py nor test_codegen_finetune.py contains CJK characters."""
    for rel in ("app/codegen/fine_tune_gen.py", "tests/test_codegen_finetune.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            offence = _CHECK_NO_CJK.first_offence(line)
            assert offence is None, f"{rel}:{lineno}: U+{offence:04X}"
