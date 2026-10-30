"""Checks for the registry contract (issue #2).

Runs with either:

    python tests/test_registry.py
    pytest tests/test_registry.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.registry import (  # noqa: E402
    DEEPSEEK_ID,
    MISTRAL_ID,
    MODELS,
    MODEL_IDS,
    defaults_for,
    get_model,
)

PAYLOAD_KEYS = {
    "model_id",
    "task_type",
    "run_status",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "p",
}


def test_registry_contains_both_models() -> None:
    assert set(MODELS) == {MISTRAL_ID, DEEPSEEK_ID}
    assert MODEL_IDS == (MISTRAL_ID, DEEPSEEK_ID)


def test_model_metadata_matches_the_spec() -> None:
    mistral = get_model(MISTRAL_ID)
    assert mistral.architectures == "MistralForCausalLM"
    assert (mistral.pad_token, mistral.pad_token_id) == ("<pad>", 11)
    assert (mistral.eos_token, mistral.eos_token_id) == ("</s>", 2)
    assert mistral.effective_max_seq_len == 32768
    assert mistral.allowed_pad_tokens == ("<pad>", "</s>")
    assert mistral.needs_fix_mistral_regex is True

    deepseek = get_model(DEEPSEEK_ID)
    assert deepseek.architectures == "Qwen2ForCausalLM"
    assert deepseek.pad_token_id == 151643
    assert deepseek.pad_token == deepseek.eos_token == "<｜end▁of▁sentence｜>"
    assert deepseek.tokenizer_model_max_length == 16384
    assert deepseek.effective_max_seq_len == 32768
    assert deepseek.needs_fix_mistral_regex is False


def test_payload_has_exactly_the_contract_keys() -> None:
    for model_id in MODEL_IDS:
        for task_type in ("fine_tune", "inference"):
            payload = defaults_for(model_id, task_type)
            assert set(payload) == PAYLOAD_KEYS
            assert set(payload["d"]) == {"lora", "sft"}


def test_payload_is_json_serialisable() -> None:
    for model_id in MODEL_IDS:
        for task_type in ("fine_tune", "inference"):
            json.dumps(defaults_for(model_id, task_type))


def test_all_four_combinations_fill_task_fields() -> None:
    for model_id in MODEL_IDS:
        fine_tune = defaults_for(model_id, "fine_tune")
        inference = defaults_for(model_id, "inference")

        assert fine_tune["e"]["add_generation_prompt"] is False
        assert inference["e"]["add_generation_prompt"] is True

        assert fine_tune["c"]["train_data_file"] is not None
        assert fine_tune["c"]["eval_data_file"] is not None
        assert fine_tune["c"]["test_data_file"] is None

        assert inference["c"]["train_data_file"] is None
        assert inference["c"]["eval_data_file"] is None
        assert inference["c"]["test_data_file"] is not None

        assert fine_tune["f"]["batch_size"] == 1


def test_mistral_fine_tune_matches_parameters_section_13() -> None:
    payload = defaults_for(MISTRAL_ID, "fine_tune")

    assert payload["a"] == {
        "model_id": MISTRAL_ID,
        "dtype": "auto",
        "local_files_only": True,
        "tp_plan": "auto",
        "device_map": None,
        "attn_implementation": "sdpa",
        "quantization": None,
        "trust_remote_code": False,
    }
    assert payload["b"] == {
        "tokenizer_id": MISTRAL_ID,
        "padding_side": "left",
        "pad_token": "<pad>",
        "pad_token_id": 11,
        "fix_mistral_regex": True,
        "use_fast": True,
        "model_max_length": 32768,
        "truncation_side": "right",
    }
    assert payload["c"] == {
        "train_data_file": "data/trainData.json",
        "eval_data_file": "data/evalData.json",
        "test_data_file": None,
        "data_format": "json",
        "label_values": ["True", "False", "Conflicting"],
        "evidence_answers_used": 1,
        "missing_answer_placeholder": "No answer could be found !",
        "max_seq_length": 4096,
        "max_samples": None,
        "evidence_config": "with_evidence",
    }
    assert payload["d"] == {
        "lora": {
            "r": 64,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "bias": "none",
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "use_dora": False,
            "init_lora_weights": True,
        },
        "sft": {
            "learning_rate": 0.0002,
            "weight_decay": 0.01,
            "num_train_epochs": 2,
            "warmup_steps": 5,
            "optim": "adamw_8bit",
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "lr_scheduler_type": "linear",
            "max_grad_norm": 1.0,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "push_to_hub": False,
            "output_dir": "fineTuningOutput",
            "bf16": True,
            "gradient_checkpointing": False,
            "seed": 42,
        },
    }
    assert payload["e"] == {
        "apply_chat_template": True,
        "add_generation_prompt": False,
        "tokenize": True,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
        "max_length": 4096,
        "device": "cuda",
    }
    assert payload["f"] == {
        "max_new_tokens": 512,
        "temperature": 0.15,
        "top_p": 0.9,
        "top_k": 10,
        "do_sample": True,
        "use_cache": True,
        "num_return_sequences": 1,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "pad_token_id": 11,
        "eos_token_id": 2,
        "forced_prefix": None,
        "batch_size": 1,
    }
    assert payload["g"] == {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False,
        "spaces_between_special_tokens": True,
    }
    assert payload["h"] == {}
    assert payload["p"] == {
        "system_prompt": (
            "You are a fact-checking assistant. Given a numerical claim and "
            "its retrieved evidence, output a verdict."
        ),
        "user_prompt_template": (
            "### Claim: {claim}\n### Question: {question}\n### Answer: {answer}"
        ),
        "assistant_prompt": "### Response: Verdict: ",
        "chat_template_override": None,
        "verdict_parse_marker": "### Response: Verdict: ",
    }


def test_deepseek_inference_matches_issue_2_acceptance() -> None:
    payload = defaults_for(DEEPSEEK_ID, "inference")

    assert payload["b"]["pad_token_id"] == 151643
    assert payload["f"]["pad_token_id"] == 151643
    assert payload["f"]["eos_token_id"] == 151643
    assert payload["f"]["temperature"] == 0.6
    assert payload["f"]["top_p"] == 0.95
    assert payload["f"]["top_k"] is None
    assert payload["f"]["max_new_tokens"] == 1024
    assert payload["f"]["forced_prefix"] == "\n\n"
    assert payload["b"]["pad_token"] == "<｜end▁of▁sentence｜>"
    assert payload["b"]["fix_mistral_regex"] is False
    assert payload["b"]["model_max_length"] == 16384
    assert payload["p"]["system_prompt"] == ""
    assert payload["e"]["add_generation_prompt"] is True


def test_calls_return_fresh_objects() -> None:
    first = defaults_for(MISTRAL_ID, "fine_tune")
    first["c"]["label_values"].append("Unknown")
    first["d"]["lora"]["target_modules"].clear()
    first["h"]["generate"] = {"top_k": 1}

    second = defaults_for(MISTRAL_ID, "fine_tune")
    assert second["c"]["label_values"] == ["True", "False", "Conflicting"]
    assert len(second["d"]["lora"]["target_modules"]) == 7
    assert second["h"] == {}


def test_unknown_model_and_task_raise_value_error() -> None:
    for call in (
        lambda: get_model("nope/model"),
        lambda: defaults_for("nope/model", "fine_tune"),
        lambda: defaults_for(MISTRAL_ID, "train"),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    failures = 0
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:  # noqa: BLE001 - report every failure
                failures += 1
                print(f"FAIL {name}: {error}")
            else:
                print(f"ok   {name}")
    print(f"\n{'FAILED' if failures else 'OK'}: {failures} failure(s)")
    sys.exit(1 if failures else 0)
