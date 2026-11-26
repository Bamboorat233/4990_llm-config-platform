"""Unit tests for Issue #59: deps_gen, verify engine, reference_sections, and generate_bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from app.codegen import GenerationResult, build_script_plan, generate_bundle
from app.codegen.deps_gen import generate_requirements_txt
from app.codegen.verify import (
    check_known_pitfalls,
    check_parameter_parity,
    check_parser_compatibility,
    check_reference_parity,
    check_syntax,
)
from app.parsing.errors import (
    DONE_LOADING_EVAL_DATA,
    DONE_PREPARING_INPUT,
)
from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
from app.schemas import ConfigRecord

_CHECK_NO_CJK_SPEC = importlib.util.spec_from_file_location(
    "check_no_cjk", ROOT / ".github" / "scripts" / "check_no_cjk.py"
)
assert _CHECK_NO_CJK_SPEC is not None and _CHECK_NO_CJK_SPEC.loader is not None
_CHECK_NO_CJK = importlib.util.module_from_spec(_CHECK_NO_CJK_SPEC)
_CHECK_NO_CJK_SPEC.loader.exec_module(_CHECK_NO_CJK)


def _make_cfg(model_id: str, task_type: str, **overrides: Any) -> ConfigRecord:
    payload = defaults_for(model_id, task_type)  # type: ignore[arg-type]
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


@pytest.mark.parametrize("model_id", [MISTRAL_ID, DEEPSEEK_ID])
@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_generate_bundle_all_four_combinations_pass_all_checks(
    model_id: str,
    task_type: str,
) -> None:
    """All 4 Model x Task combinations pass syntax, parity, pitfall, and marker checks."""
    cfg = _make_cfg(model_id, task_type)
    result = generate_bundle(cfg)

    assert set(result.keys()) == set(GenerationResult.__annotations__.keys())
    assert result["config_id"] == cfg.config_id
    assert result["task_type"] == task_type
    assert result["model_id"] == model_id
    assert result["slurm_script"] is None
    assert result["syntax_ok"] is True
    assert result["syntax_error"] is None
    assert result["parity_ok"] is True
    assert result["parity_diffs"] == []
    assert result["pitfall_issues"] == []
    assert result["marker_issues"] == []
    assert result["script_sha256"] == hashlib.sha256(
        result["script_code"].encode("utf-8")
    ).hexdigest()
    assert json.loads(result["config_json"])["config_id"] == cfg.config_id


@pytest.mark.parametrize("task_type", ["fine_tune", "inference"])
def test_generate_bundle_with_nondefault_options_keeps_parity(task_type: str) -> None:
    """Custom quantization, device_map, greedy decoding, and h kwargs keep parity_ok=True."""
    cfg = _make_cfg(
        MISTRAL_ID,
        task_type,
        a={
            "quantization": "4bit",
            "tp_plan": None,
            "device_map": "auto",
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        },
        f={"do_sample": False, "temperature": 0.0, "batch_size": 4},
        h={
            "load_model": {"low_cpu_mem_usage": True},
            "load_tokenizer": {"clean_up_tokenization_spaces": True},
            "encode": {"return_token_type_ids": False},
            "generate": {"num_beams": 2},
        },
    )
    result = generate_bundle(cfg)
    assert result["syntax_ok"] is True
    assert result["parity_ok"] is True, result["parity_diffs"]
    assert result["pitfall_issues"] == []
    assert result["marker_issues"] == []


def test_negative_mutated_parameter_breaks_parity() -> None:
    """Mutating a generated parameter in script_code causes parity_ok=False."""
    cfg = _make_cfg(MISTRAL_ID, "fine_tune")
    result = generate_bundle(cfg)
    mutated = result["script_code"].replace("learning_rate=0.0002,", "learning_rate=0.999,")
    assert mutated != result["script_code"]

    diffs = check_parameter_parity(cfg, mutated)
    assert len(diffs) >= 1
    assert any(d["field"] == "d.sft.learning_rate" for d in diffs)

    inf_cfg = _make_cfg(DEEPSEEK_ID, "inference")
    inf_result = generate_bundle(inf_cfg)
    inf_mutated = inf_result["script_code"].replace(
        "max_new_tokens=1024,",
        "max_new_tokens=42,",
    )
    inf_diffs = check_parameter_parity(inf_cfg, inf_mutated)
    assert any(d["field"] == "f.max_new_tokens" for d in inf_diffs)


def test_negative_deleted_reference_section_or_unexpected_kwarg_fails_reference_parity() -> None:
    """Deleting a reference call site or injecting an unexpected kwarg fails reference parity."""
    ft_cfg = _make_cfg(MISTRAL_ID, "fine_tune")
    ft_code = generate_bundle(ft_cfg)["script_code"]
    ft_missing_lora = ft_code.replace("lora_config = LoraConfig(", "lora_config = dict(")
    ref_diffs = check_reference_parity(ft_cfg, ft_missing_lora)
    assert any(d["field"] == "lora_config.call_site" for d in ref_diffs)

    inf_cfg = _make_cfg(MISTRAL_ID, "inference")
    inf_code = generate_bundle(inf_cfg)["script_code"]
    inf_missing_decode = inf_code.replace("tokenizer.decode(", "str(")
    inf_diffs = check_reference_parity(inf_cfg, inf_missing_decode)
    assert any(d["field"] == "decoding.call_site" for d in inf_diffs)

    inf_unexpected_kw = inf_code.replace(
        "skip_special_tokens=True,",
        "skip_special_tokens=True,\n                bogus_extra_kwarg=123,",
    )
    kw_diffs = check_reference_parity(inf_cfg, inf_unexpected_kw)
    assert any(d["field"] == "decoding.bogus_extra_kwarg" for d in kw_diffs)


def test_negative_pitfall_regressions_detected() -> None:
    """Injecting <myPadToken>, post-load fix_mistral_regex, or rank=0 triggers pitfall_issues."""
    cfg = _make_cfg(MISTRAL_ID, "inference")
    code = generate_bundle(cfg)["script_code"]

    with_bad_pad = code.replace("'<pad>'", "'<myPadToken>'")
    issues_pad = check_known_pitfalls(cfg, with_bad_pad)
    assert any("<myPadToken>" in item for item in issues_pad)

    with_post_load_regex = code + "\ntokenizer.fix_mistral_regex = True\n"
    issues_regex = check_known_pitfalls(cfg, with_post_load_regex)
    assert any("fix_mistral_regex" in item for item in issues_regex)

    with_rank_zero = code.replace(
        "torch.distributed.init_process_group()",
        "torch.distributed.init_process_group(rank=0)",
    )
    issues_rank = check_known_pitfalls(cfg, with_rank_zero)
    assert any("rank=0" in item for item in issues_rank)


def test_negative_broken_syntax_and_missing_phase_markers() -> None:
    """Syntax errors return syntax_ok=False and missing phase markers trigger marker_issues."""
    ok, err = check_syntax("def broken(:\n    pass\n", filename="broken.py")
    assert ok is False
    assert err is not None
    assert "SyntaxError" in err

    ft_cfg = _make_cfg(MISTRAL_ID, "fine_tune")
    ft_bundle = generate_bundle(ft_cfg)
    ft_no_marker = ft_bundle["script_code"].replace(DONE_LOADING_EVAL_DATA, "Eval Loaded")
    ft_marker_issues = check_parser_compatibility(
        ft_cfg, ft_bundle["script_filename"], ft_no_marker
    )
    assert any(DONE_LOADING_EVAL_DATA in item for item in ft_marker_issues)

    inf_cfg = _make_cfg(MISTRAL_ID, "inference")
    inf_bundle = generate_bundle(inf_cfg)
    inf_no_marker = inf_bundle["script_code"].replace(DONE_PREPARING_INPUT, "Prepared")
    inf_marker_issues = check_parser_compatibility(
        inf_cfg, inf_bundle["script_filename"], inf_no_marker
    )
    assert any(DONE_PREPARING_INPUT in item for item in inf_marker_issues)

    wrong_name_issues = check_parser_compatibility(
        inf_cfg, "fineTune_Mistral24B_12345678.py", inf_bundle["script_code"]
    )
    assert len(wrong_name_issues) == 1
    assert "family" in wrong_name_issues[0]


def test_deps_gen_conditional_requirements() -> None:
    """deps_gen emits base packages and conditional trl/peft/bitsandbytes/mistral-common/flash-attn."""
    base_packages = (
        "torch>=2.4.0",
        "transformers>=4.57.0",
        "accelerate>=1.0.0",
        "datasets>=3.0.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.5.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
        "protobuf>=4.25.0",
    )

    # 1. Default Mistral fine_tune: includes trl, peft, bitsandbytes (adamw_8bit), mistral-common
    ft_mistral = _make_cfg(MISTRAL_ID, "fine_tune")
    reqs_ft_ms = generate_requirements_txt(ft_mistral)
    for pkg in (*base_packages, "trl>=0.12.0", "peft>=0.13.0", "bitsandbytes>=0.44.0", "mistral-common>=1.5.2"):
        assert pkg in reqs_ft_ms
    assert "flash-attn>=2.6.0" not in reqs_ft_ms

    # 2. Default DeepSeek inference: no trl, no peft, no bitsandbytes (even though d.sft.optim is adamw_8bit), no mistral-common
    inf_deepseek = _make_cfg(DEEPSEEK_ID, "inference")
    reqs_inf_ds = generate_requirements_txt(inf_deepseek)
    for pkg in base_packages:
        assert pkg in reqs_inf_ds
    assert "trl>=0.12.0" not in reqs_inf_ds
    assert "peft>=0.13.0" not in reqs_inf_ds
    assert "bitsandbytes>=0.44.0" not in reqs_inf_ds
    assert "mistral-common>=1.5.2" not in reqs_inf_ds
    assert "flash-attn>=2.6.0" not in reqs_inf_ds

    # 3. DeepSeek inference with 4bit quantization + flash_attention_2 + adapter_path
    inf_ds_q = _make_cfg(
        DEEPSEEK_ID,
        "inference",
        a={"quantization": "4bit", "attn_implementation": "flash_attention_2"},
    )
    # Simulate adapter_path (compatible with Issue #62 ModelSection extension)
    inf_ds_q_proxy = SimpleNamespace(
        model_id=inf_ds_q.model_id,
        task_type=inf_ds_q.task_type,
        a=SimpleNamespace(**inf_ds_q.a.model_dump(), adapter_path="fineTuningOutput/final_model"),
        d=inf_ds_q.d,
    )
    reqs_inf_ds_q = generate_requirements_txt(inf_ds_q_proxy)  # type: ignore[arg-type]
    assert "peft>=0.13.0" in reqs_inf_ds_q
    assert "bitsandbytes>=0.44.0" in reqs_inf_ds_q
    assert "flash-attn>=2.6.0" in reqs_inf_ds_q
    assert "trl>=0.12.0" not in reqs_inf_ds_q

    # 4. Fine-tune without 8bit optimizer and without quantization omits bitsandbytes
    ft_ds_no8bit = _make_cfg(
        DEEPSEEK_ID,
        "fine_tune",
        d={"sft": {"optim": "adamw_torch"}},
        a={"quantization": None},
    )
    reqs_no_bnb = generate_requirements_txt(ft_ds_no8bit)
    assert "trl>=0.12.0" in reqs_no_bnb
    assert "peft>=0.13.0" in reqs_no_bnb
    assert "bitsandbytes>=0.44.0" not in reqs_no_bnb


def test_reference_sections_fixture_covers_both_tasks_and_six_defects() -> None:
    """reference_sections.json is valid JSON covering fine_tune, inference, and 6 defects."""
    baseline_path = ROOT / "tests" / "fixtures" / "reference_sections.json"
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert len(data["defects"]) == 6
    assert set(data["tasks"].keys()) == {"fine_tune", "inference"}
    ft_sections = {s["section"] for s in data["tasks"]["fine_tune"]["sections"]}
    inf_sections = {s["section"] for s in data["tasks"]["inference"]["sections"]}
    assert {"model_loading", "tokenizer_loading", "lora_config", "sft_config", "sft_trainer"} <= ft_sections
    assert {"model_loading", "tokenizer_loading", "prompt_building", "token_encoding", "generation", "decoding"} <= inf_sections


def test_pure_modules_import_without_nicegui() -> None:
    """app.codegen.deps_gen, verify, and __init__ import cleanly without nicegui."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
                "import app.codegen, app.codegen.deps_gen, app.codegen.verify; "
                "assert 'nicegui' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_no_cjk_in_new_files() -> None:
    """Ensure no CJK characters exist in deps_gen.py, verify.py, reference_sections.json, or this test."""
    for rel in (
        "app/codegen/__init__.py",
        "app/codegen/deps_gen.py",
        "app/codegen/verify.py",
        "tests/fixtures/reference_sections.json",
        "tests/test_codegen_verify.py",
    ):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            offence = _CHECK_NO_CJK.first_offence(line)
            assert offence is None, f"{rel}:{lineno}: U+{offence:04X}"
