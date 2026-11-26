"""Unit tests for Issue #58: inference script generator (app/codegen/inference_gen.py)."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.codegen.inference_gen import generate_inference_script
from app.codegen.planner import INFERENCE_PHASE_MARKERS, build_script_plan
from app.parsing import parse_log_file
from app.parsing.verdicts import extract_verdicts_with_source
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord


def _make_cfg(model_id: str, **overrides: Any) -> ConfigRecord:
    payload = defaults_for(model_id, "inference")
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key].update(value)
        else:
            payload[key] = value
    return ConfigRecord(**payload)


def _generate(model_id: str, **overrides: Any) -> tuple[ConfigRecord, str]:
    cfg = _make_cfg(model_id, **overrides)
    plan, _warnings = build_script_plan(cfg)
    code = generate_inference_script(plan, cfg)
    return cfg, code


def _extract_config_literal(code: str) -> dict[str, Any]:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CONFIG":
                    return ast.literal_eval(node.value)
    raise AssertionError("CONFIG assignment not found in generated script")


def _find_calls(code: str, func_name: str, attr_owner: str | None = None) -> list[ast.Call]:
    tree = ast.parse(code)
    matches: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if attr_owner is None:
            if isinstance(func, ast.Name) and func.id == func_name:
                matches.append(node)
            elif isinstance(func, ast.Attribute) and func.attr == func_name:
                matches.append(node)
        else:
            if (
                isinstance(func, ast.Attribute)
                and func.attr == func_name
                and isinstance(func.value, ast.Name)
                and func.value.id == attr_owner
            ):
                matches.append(node)
    return matches


def _call_kw_names(call: ast.Call) -> set[str]:
    return {kw.arg for kw in call.keywords if kw.arg is not None}


def _call_kw_dict(call: ast.Call) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kw in call.keywords:
        if kw.arg is not None:
            try:
                result[kw.arg] = ast.literal_eval(kw.value)
            except Exception:
                result[kw.arg] = ast.unparse(kw.value)
    return result


class _FakeEncoded(dict):
    def to(self, device: str) -> "_FakeEncoded":
        self["device"] = device
        return self


class _FakeModel:
    last_generate_kwargs: dict[str, Any] | None = None
    eval_called: bool = False

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._cursor = 0

    def eval(self) -> "_FakeModel":
        self.eval_called = True
        return self

    def generate(self, **kwargs: Any) -> list[str]:
        self.last_generate_kwargs = dict(kwargs)
        batch = len(kwargs["input_ids"])
        out: list[str] = []
        for _ in range(batch):
            out.append(self._responses[self._cursor % len(self._responses)])
            self._cursor += 1
        return out


class _FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token: str | None = None
        self.pad_token_id: int = 11
        self.eos_token_id: int = 2
        self.chat_template: str | None = None
        self.applied_messages: list[list[dict[str, str]]] = []
        self.encode_calls: list[dict[str, Any]] = []
        self.decode_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.applied_messages.append(messages)
        rendered = "|".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"<s>{rendered}"

    def __call__(self, prompts: list[str], **kwargs: Any) -> _FakeEncoded:
        self.encode_calls.append({"prompts": list(prompts), **kwargs})
        return _FakeEncoded(
            {
                "input_ids": [[101, idx] for idx in range(len(prompts))],
                "attention_mask": [[1, 1] for _ in prompts],
            }
        )

    def decode(self, seq: Any, **kwargs: Any) -> str:
        self.decode_calls.append(dict(kwargs))
        return str(seq)


def _exec_generated_with_stubs(
    code: str,
    *,
    responses: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute generated script inside an isolated namespace with lightweight ML stubs."""
    fake_model = _FakeModel(responses or ["### Response: Verdict: True"])
    fake_tokenizer = _FakeTokenizer()
    recorder: dict[str, Any] = {
        "model": fake_model,
        "tokenizer": fake_tokenizer,
        "model_from_pretrained": [],
        "tokenizer_from_pretrained": [],
        "peft_from_pretrained": [],
        "bnb_calls": [],
        "dist_init": 0,
        "dist_destroy": 0,
        "seed_calls": [],
    }

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.manual_seed = lambda s: recorder["seed_calls"].append(("torch", s))
    fake_torch.no_grad = contextlib.nullcontext
    fake_torch.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        manual_seed_all=lambda s: recorder["seed_calls"].append(("cuda", s)),
    )
    _dist_state = {"initialized": False}

    def _init_pg() -> None:
        _dist_state["initialized"] = True
        recorder["dist_init"] += 1

    def _destroy_pg() -> None:
        _dist_state["initialized"] = False
        recorder["dist_destroy"] += 1

    fake_torch.distributed = types.SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: _dist_state["initialized"],
        init_process_group=_init_pg,
        destroy_process_group=_destroy_pg,
    )

    fake_np = types.ModuleType("numpy")
    fake_np.random = types.SimpleNamespace(
        seed=lambda s: recorder["seed_calls"].append(("numpy", s))
    )

    fake_tf = types.ModuleType("transformers")

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeModel:
            recorder["model_from_pretrained"].append((model_id, kwargs))
            return fake_model

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id: str, **kwargs: Any) -> _FakeTokenizer:
            recorder["tokenizer_from_pretrained"].append((tokenizer_id, kwargs))
            return fake_tokenizer

    class _FakeBnb:
        def __init__(self, **kwargs: Any) -> None:
            recorder["bnb_calls"].append(kwargs)

    fake_tf.AutoModelForCausalLM = _FakeAutoModel
    fake_tf.AutoTokenizer = _FakeAutoTokenizer
    fake_tf.BitsAndBytesConfig = _FakeBnb
    fake_tf.set_seed = lambda s: recorder["seed_calls"].append(("transformers", s))

    fake_peft = types.ModuleType("peft")

    class _FakePeftModel:
        @staticmethod
        def from_pretrained(model: Any, adapter_path: str, **kwargs: Any) -> _FakeModel:
            recorder["peft_from_pretrained"].append((adapter_path, kwargs))
            return model

    fake_peft.PeftModel = _FakePeftModel

    saved: dict[str, Any] = {}
    stubs = {
        "torch": fake_torch,
        "numpy": fake_np,
        "transformers": fake_tf,
        "peft": fake_peft,
    }
    for name, mod in stubs.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod

    ns: dict[str, Any] = {"__name__": "generated_inference_test"}
    try:
        exec(compile(ast.parse(code), "<generated_inference>", "exec"), ns)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

    return ns, recorder


@pytest.mark.parametrize("model_id", [MISTRAL_ID, DEEPSEEK_ID])
def test_ast_parse_and_compile_both_models(model_id: str) -> None:
    """Generated inference scripts parse and compile cleanly for Mistral and DeepSeek."""
    cfg, code = _generate(model_id)
    tree = ast.parse(code)
    compile(tree, f"<{model_id}>", "exec")

    embedded = _extract_config_literal(code)
    assert embedded["model_id"] == model_id
    assert embedded["f"]["temperature"] == cfg.f.temperature
    assert embedded["seed"] == cfg.d.sft.seed
    assert "<myPadToken>" not in code
    assert "tokenizer.fix_mistral_regex =" not in code
    assert "rank=0" not in code


def test_pure_module_import_without_nicegui() -> None:
    """app.codegen.inference_gen imports cleanly without nicegui."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import app.codegen.inference_gen; "
                "assert 'nicegui' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_all_seven_evidence_modes_and_empty_top_k_doc_preserves_question() -> None:
    """All 7 c.evidence_config modes work and empty top_k_doc preserves the question."""
    _cfg, code = _generate(MISTRAL_ID)
    ns, _rec = _exec_generated_with_stubs(code)

    select_evidence_items = ns["select_evidence_items"]
    format_evidence_items = ns["format_evidence_items"]
    build_user_prompt = ns["build_user_prompt"]

    record = {
        "claim": "Sample claim.",
        "evidences": [
            {"questions": "Q1?", "top_k_doc": []},
            {"questions": "Q2?", "top_k_doc": ["A2 doc1", "A2 doc2"]},
            {"questions": "Q3?", "top_k_doc": ["A3 doc1"]},
        ],
    }
    placeholder = "No answer could be found !"

    # 1. with_evidence (empty top_k_doc on Q1 must preserve Q1!)
    items = select_evidence_items(record, "with_evidence", 1, placeholder, random.Random(42))
    assert items == [
        {"question": "Q1?", "answer": placeholder},
        {"question": "Q2?", "answer": "A2 doc1"},
        {"question": "Q3?", "answer": "A3 doc1"},
    ]
    formatted = format_evidence_items(items)
    assert "### Question: Q1?" in formatted
    assert f"### Answer: {placeholder}" in formatted

    # answers_used=2 joins top 2 docs for Q2
    items_k2 = select_evidence_items(record, "with_evidence", 2, placeholder, random.Random(42))
    assert items_k2[1]["answer"] == "A2 doc1\nA2 doc2"

    # 2. without_evidence
    items_wo = select_evidence_items(record, "without_evidence", 1, placeholder, random.Random(42))
    assert items_wo == []
    prompt_wo = build_user_prompt(
        record,
        "without_evidence",
        1,
        placeholder,
        "### Claim: {claim}\n### Question: {question}\n### Answer: {answer}",
        random.Random(42),
    )
    assert prompt_wo == "### Claim: Sample claim."
    assert "### Question:" not in prompt_wo
    assert "### Answer:" not in prompt_wo

    # 3. random_evidence (deterministic for same seed, supports donor_evidences)
    items_rand_a = select_evidence_items(
        record, "random_evidence", 1, placeholder, random.Random(7)
    )
    items_rand_b = select_evidence_items(
        record, "random_evidence", 1, placeholder, random.Random(7)
    )
    assert items_rand_a == items_rand_b
    assert len(items_rand_a) == 3

    donor = [{"questions": "DonorQ?", "top_k_doc": ["DonorA"]}]
    items_donor = select_evidence_items(
        record, "random_evidence", 1, placeholder, random.Random(7), donor_evidences=donor
    )
    assert items_donor == [{"question": "DonorQ?", "answer": "DonorA"}]

    # 4. question_only
    items_qo = select_evidence_items(record, "question_only", 1, placeholder, random.Random(42))
    assert items_qo == [
        {"question": "Q1?", "answer": None},
        {"question": "Q2?", "answer": None},
        {"question": "Q3?", "answer": None},
    ]
    fmt_qo = format_evidence_items(items_qo)
    assert "### Question: Q1?" in fmt_qo
    assert "### Answer:" not in fmt_qo

    # 5. answer_only
    items_ao = select_evidence_items(record, "answer_only", 1, placeholder, random.Random(42))
    assert items_ao == [
        {"question": None, "answer": placeholder},
        {"question": None, "answer": "A2 doc1"},
        {"question": None, "answer": "A3 doc1"},
    ]
    fmt_ao = format_evidence_items(items_ao)
    assert "### Question:" not in fmt_ao
    assert f"### Answer: {placeholder}" in fmt_ao
    assert "### Answer: A2 doc1" in fmt_ao

    # 6. all_questions_last_answer
    items_aqla = select_evidence_items(
        record, "all_questions_last_answer", 1, placeholder, random.Random(42)
    )
    assert items_aqla == [
        {"question": "Q1?", "answer": None},
        {"question": "Q2?", "answer": None},
        {"question": "Q3?", "answer": "A3 doc1"},
    ]
    # Also verify empty top_k_doc on last item preserves question + placeholder
    rec_empty_last = {
        "claim": "C",
        "evidences": [
            {"questions": "Q1?", "top_k_doc": ["A1"]},
            {"questions": "Q2?", "top_k_doc": []},
        ],
    }
    items_aqla_empty = select_evidence_items(
        rec_empty_last, "all_questions_last_answer", 1, placeholder, random.Random(42)
    )
    assert items_aqla_empty == [
        {"question": "Q1?", "answer": None},
        {"question": "Q2?", "answer": placeholder},
    ]

    # 7. all_questions_all_but_last_answer
    items_aqabla = select_evidence_items(
        record, "all_questions_all_but_last_answer", 1, placeholder, random.Random(42)
    )
    assert items_aqabla == [
        {"question": "Q1?", "answer": placeholder},
        {"question": "Q2?", "answer": "A2 doc1"},
        {"question": "Q3?", "answer": None},
    ]
    fmt_aqabla = format_evidence_items(items_aqabla)
    assert "### Question: Q1?\n### Answer: No answer could be found !" in fmt_aqabla
    assert "### Question: Q3?" in fmt_aqabla
    assert "A3 doc1" not in fmt_aqabla


def test_batched_generation_attention_mask_and_kwargs(tmp_path: Path) -> None:
    """batch_size=4 emits batched generation with attention_mask and h.generate kwargs."""
    cfg, code = _generate(
        MISTRAL_ID,
        f={"batch_size": 4, "max_new_tokens": 256, "temperature": 0.25},
        h={
            "load_model": {"low_cpu_mem_usage": True},
            "load_tokenizer": {"clean_up_tokenization_spaces": True},
            "encode": {"return_token_type_ids": False},
            "generate": {"num_beams": 2, "min_new_tokens": 3},
        },
    )
    assert cfg.f.batch_size == 4

    embedded = _extract_config_literal(code)
    assert embedded["f"]["batch_size"] == 4
    assert embedded["h"]["generate"] == {"num_beams": 2, "min_new_tokens": 3}

    gen_calls = _find_calls(code, "generate", attr_owner="model")
    assert len(gen_calls) == 1
    gen_kw = _call_kw_dict(gen_calls[0])
    assert "attention_mask" in gen_kw
    assert gen_kw["num_beams"] == 2
    assert gen_kw["min_new_tokens"] == 3
    assert gen_kw["max_new_tokens"] == 256

    model_calls = _find_calls(code, "from_pretrained", attr_owner="AutoModelForCausalLM")
    assert len(model_calls) == 1
    assert _call_kw_dict(model_calls[0])["low_cpu_mem_usage"] is True

    tok_calls = _find_calls(code, "from_pretrained", attr_owner="AutoTokenizer")
    assert len(tok_calls) == 1
    tok_kw = _call_kw_dict(tok_calls[0])
    assert tok_kw["fix_mistral_regex"] is True
    assert tok_kw["clean_up_tokenization_spaces"] is True

    # Execute main() end-to-end with 5 records and batch_size=4 -> 2 batches (4 + 1)
    test_file = tmp_path / "test.json"
    out_dir = tmp_path / "out"
    records = [
        {"claim": f"Claim {i}", "evidences": [{"questions": f"Q{i}?", "top_k_doc": [f"A{i}"]}]}
        for i in range(5)
    ]
    test_file.write_text(json.dumps(records), encoding="utf-8")

    ns, rec = _exec_generated_with_stubs(
        code,
        responses=[
            "### Response: Verdict: True",
            "### Response: Verdict: False</s>",
            "No marker output",
            "### Response: Verdict: Conflicting",
            "### Response: Verdict: True",
        ],
    )
    old_argv = sys.argv
    sys.argv = [
        "inference.py",
        "--testDataFile",
        str(test_file),
        "--outputDir",
        str(out_dir),
        "--temperature",
        "0.42",
    ]
    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf):
            ns["main"]()
    finally:
        sys.argv = old_argv

    assert len(rec["tokenizer"].encode_calls) == 2
    assert len(rec["tokenizer"].encode_calls[0]["prompts"]) == 4
    assert len(rec["tokenizer"].encode_calls[1]["prompts"]) == 1
    assert rec["tokenizer"].encode_calls[0]["return_token_type_ids"] is False
    assert rec["model"].eval_called is True
    # CLI --temperature 0.42 must NOT be overwritten by hard-coded 0.1
    assert rec["model"].last_generate_kwargs is not None
    assert rec["model"].last_generate_kwargs["temperature"] == pytest.approx(0.42)
    assert "attention_mask" in rec["model"].last_generate_kwargs
    assert (out_dir / "verdicts.json").exists()

    # Check log parser round-trip on captured stdout!
    log_path = tmp_path / "run.log"
    log_path.write_text(stdout_buf.getvalue(), encoding="utf-8")
    parsed = parse_log_file(log_path)
    assert parsed["family"] == "inference"
    assert parsed["status"] == "success"
    assert parsed["output_format"] == "banner"

    warnings: list[str] = []
    source, verdicts = extract_verdicts_with_source(
        log_path,
        marker=cfg.p.verdict_parse_marker,
        labels=cfg.c.label_values,
        warnings=warnings,
    )
    assert source == "parsed"
    assert warnings == []
    assert [v["kind"] for v in verdicts] == [
        "exact",
        "recovered",
        "empty",
        "exact",
        "exact",
    ]
    assert [v["parsed"] for v in verdicts] == [
        "True",
        "False",
        None,
        "Conflicting",
        "True",
    ]


def test_deepseek_forced_prefix_system_omission_and_no_fix_mistral_regex(tmp_path: Path) -> None:
    """DeepSeek omits empty system prompt, appends forced_prefix, and omits fix_mistral_regex."""
    cfg, code = _generate(DEEPSEEK_ID)
    assert "tokenizer.fix_mistral_regex" not in code
    assert "<myPadToken>" not in code
    assert "<｜end▁of▁sentence｜>" in code
    assert "rank=0" not in code
    assert _extract_config_literal(code)["b"]["fix_mistral_regex"] is False

    tok_calls = _find_calls(code, "from_pretrained", attr_owner="AutoTokenizer")
    assert len(tok_calls) == 1
    assert "fix_mistral_regex" not in _call_kw_names(tok_calls[0])

    test_file = tmp_path / "ds_test.json"
    out_dir = tmp_path / "ds_out"
    test_file.write_text(
        json.dumps([{"claim": "C1", "evidences": [{"questions": "Q1?", "top_k_doc": ["A1"]}]}]),
        encoding="utf-8",
    )

    ns, rec = _exec_generated_with_stubs(
        code,
        responses=["<think>Reasoning</think>\n### Response: Verdict: Conflicting"],
    )
    old_argv = sys.argv
    sys.argv = ["inference.py", "--testDataFile", str(test_file), "--outputDir", str(out_dir)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ns["main"]()
    finally:
        sys.argv = old_argv

    assert rec["tokenizer"].pad_token == cfg.b.pad_token
    assert rec["tokenizer"].applied_messages == [[{"role": "user", "content": "### Claim: C1\n### Question: Q1?\n### Answer: A1"}]]
    encoded_prompt = rec["tokenizer"].encode_calls[0]["prompts"][0]
    assert encoded_prompt.endswith("\n\n")


def test_conditional_quantization_peft_distributed_and_greedy_decoding() -> None:
    """Conditional imports, distributed init/cleanup, and do_sample=False behave cleanly."""
    # 1. Default: no BitsAndBytesConfig, no PeftModel, distributed present (tp_plan='auto')
    _cfg_default, code_default = _generate(MISTRAL_ID)
    assert "BitsAndBytesConfig" not in code_default
    assert "PeftModel" not in code_default
    assert "torch.distributed.init_process_group()" in code_default
    assert "torch.distributed.destroy_process_group()" in code_default

    # 2. Quantization='4bit', tp_plan=None, device_map='auto', do_sample=False, temperature=0.0
    _cfg_q, code_q = _generate(
        MISTRAL_ID,
        a={"quantization": "4bit", "tp_plan": None, "device_map": "auto"},
        f={"do_sample": False, "temperature": 0.0},
    )
    assert "BitsAndBytesConfig" in code_q
    assert "init_process_group" not in code_q
    assert "destroy_process_group" not in code_q
    model_kw = _call_kw_dict(_find_calls(code_q, "from_pretrained", attr_owner="AutoModelForCausalLM")[0])
    assert model_kw["device_map"] == "auto"
    assert "tp_plan" not in model_kw
    assert model_kw["quantization_config"] == "quantization_config"
    gen_kw = _call_kw_dict(_find_calls(code_q, "generate", attr_owner="model")[0])
    assert gen_kw["do_sample"] is False
    assert "temperature" not in gen_kw

    # 3. Adapter path via plan.adapter_path (forward-compatible with Issue #62 a.adapter_path)
    cfg_base = _make_cfg(MISTRAL_ID)
    plan_base, _ = build_script_plan(cfg_base)
    from dataclasses import replace

    plan_with_adapter = replace(plan_base, adapter_path="fineTuningOutput/final_model")
    code_adapter = generate_inference_script(plan_with_adapter, cfg_base)
    assert "from peft import PeftModel" in code_adapter
    assert "PeftModel.from_pretrained" in code_adapter
    compile(ast.parse(code_adapter), "<adapter>", "exec")


def test_required_inference_phase_markers_present() -> None:
    """Every required phase marker and banner literal is present in generated code."""
    _cfg, code = _generate(MISTRAL_ID)
    for marker in INFERENCE_PHASE_MARKERS:
        assert marker in code, f"Missing phase marker: {marker!r}"
    for literal in (
        "Done Loading the Model :",
        "Done Loading the Tokenizer :",
        "Done Preparing the Input :",
        "*****INPUTS*****",
        "*****OUTPUT*****",
        "*****VERDICTS*****",
    ):
        assert literal in code


def test_chat_template_override_and_random_evidence_across_records(tmp_path: Path) -> None:
    """chat_template_override, random_evidence across records, and --modelId override work."""
    custom_tpl = "{{ messages[0]['content'] }}"
    _cfg, code = _generate(
        MISTRAL_ID,
        c={"evidence_config": "random_evidence"},
        p={"chat_template_override": custom_tpl},
    )
    assert 'tokenizer.chat_template = CONFIG["chat_template_override"]' in code

    records = [
        {"claim": "C0", "evidences": [{"questions": "Q0?", "top_k_doc": ["A0"]}]},
        {"claim": "C1", "evidences": [{"questions": "Q1?", "top_k_doc": ["A1"]}]},
        {"claim": "C2", "evidences": [{"questions": "Q2?", "top_k_doc": ["A2"]}]},
    ]
    test_file = tmp_path / "rand.json"
    out_dir = tmp_path / "rand_out"
    test_file.write_text(json.dumps(records), encoding="utf-8")

    ns, rec = _exec_generated_with_stubs(code)
    old_argv = sys.argv
    sys.argv = [
        "inference.py",
        "--modelId",
        "local/custom-mistral",
        "--testDataFile",
        str(test_file),
        "--outputDir",
        str(out_dir),
    ]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ns["main"]()
    finally:
        sys.argv = old_argv

    assert rec["model_from_pretrained"][0][0] == "local/custom-mistral"
    assert rec["tokenizer_from_pretrained"][0][0] == "local/custom-mistral"
    assert rec["tokenizer"].chat_template == custom_tpl
    assert len(rec["tokenizer"].applied_messages) == 3


def test_no_cjk_in_inference_gen_and_tests() -> None:
    """Ensure no CJK characters appear in inference_gen.py or this test file."""
    spec = importlib.util.spec_from_file_location(
        "check_no_cjk", ROOT / ".github" / "scripts" / "check_no_cjk.py"
    )
    assert spec is not None and spec.loader is not None
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)

    for path in (
        ROOT / "app" / "codegen" / "inference_gen.py",
        ROOT / "tests" / "test_codegen_inference.py",
    ):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            offence = checker.first_offence(line)
            assert offence is None, (
                f"CJK codepoint U+{offence:04X} found in {path}:{lineno}: {line}"
            )
