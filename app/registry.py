"""Model registry: static model facts and the pre-filled form defaults.

Public API (contract in ``specs/interfaces.md`` section 2):

- :data:`MODELS` — the registry, keyed by Hugging Face repo id
- :class:`ModelMeta` — frozen facts for one model
- :func:`get_model` — look one model up
- :func:`defaults_for` — build a complete, ready-to-save configuration payload
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TaskType = Literal["fine_tune", "inference"]
TASK_TYPES: tuple[TaskType, ...] = ("fine_tune", "inference")

MISTRAL_ID = "mistralai/Mistral-Small-24B-Instruct-2501"
DEEPSEEK_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

DEFAULT_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Given a numerical claim and its "
    "retrieved evidence, output a verdict."
)
DEFAULT_USER_PROMPT_TEMPLATE = (
    "### Claim: {claim}\n### Question: {question}\n### Answer: {answer}"
)
DEFAULT_ASSISTANT_PROMPT = "### Response: Verdict: "
DEFAULT_VERDICT_MARKER = "### Response: Verdict: "

LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

TRAIN_DATA_FILE = "data/trainData.json"
EVAL_DATA_FILE = "data/evalData.json"
TEST_DATA_FILE = "data/testData.json"


@dataclass(frozen=True)
class ModelMeta:
    """Facts about one model, taken from its Model Card and config files.

    ``allowed_pad_tokens`` and ``effective_max_seq_len`` exist so the
    validation rules can enforce their limits without re-reading the spec.
    """

    model_id: str
    display_name: str
    architectures: str
    num_parameters: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    hidden_size: int
    intermediate_size: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    tokenizer_model_max_length: int
    effective_max_seq_len: int
    bos_token: str
    bos_token_id: int
    eos_token: str
    eos_token_id: int
    pad_token: str
    pad_token_id: int
    allowed_pad_tokens: tuple[str, ...]
    allowed_roles: tuple[str, ...]
    system_prompt_policy: Literal["required", "warn"]
    default_temperature: float
    default_top_p: float | None
    default_top_k: int | None
    default_max_new_tokens: int
    default_forced_prefix: str | None
    needs_fix_mistral_regex: bool
    weights_size_gb: float
    min_gpu_memory_bf16_gb: int


MODELS: dict[str, ModelMeta] = {
    MISTRAL_ID: ModelMeta(
        model_id=MISTRAL_ID,
        display_name="Mistral-Small-24B-Instruct-2501",
        architectures="MistralForCausalLM",
        num_parameters="24B (23,572,403,200)",
        num_hidden_layers=40,
        num_attention_heads=32,
        num_key_value_heads=8,
        hidden_size=5120,
        intermediate_size=32768,
        head_dim=128,
        vocab_size=131072,
        max_position_embeddings=32768,
        tokenizer_model_max_length=32768,
        effective_max_seq_len=32768,
        bos_token="<s>",
        bos_token_id=1,
        eos_token="</s>",
        eos_token_id=2,
        pad_token="<pad>",
        pad_token_id=11,
        allowed_pad_tokens=("<pad>", "</s>"),
        allowed_roles=("system", "user", "assistant"),
        system_prompt_policy="required",
        default_temperature=0.15,
        default_top_p=0.9,
        default_top_k=10,
        default_max_new_tokens=512,
        default_forced_prefix=None,
        needs_fix_mistral_regex=True,
        weights_size_gb=47.1,
        min_gpu_memory_bf16_gb=55,
    ),
    DEEPSEEK_ID: ModelMeta(
        model_id=DEEPSEEK_ID,
        display_name="DeepSeek-R1-Distill-Qwen-32B",
        architectures="Qwen2ForCausalLM",
        num_parameters="32.8B (32,763,876,352)",
        num_hidden_layers=64,
        num_attention_heads=40,
        num_key_value_heads=8,
        hidden_size=5120,
        intermediate_size=27648,
        head_dim=128,
        vocab_size=152064,
        max_position_embeddings=131072,
        tokenizer_model_max_length=16384,
        effective_max_seq_len=32768,
        bos_token="<｜begin▁of▁sentence｜>",
        bos_token_id=151646,
        eos_token="<｜end▁of▁sentence｜>",
        eos_token_id=151643,
        pad_token="<｜end▁of▁sentence｜>",
        pad_token_id=151643,
        allowed_pad_tokens=("<｜end▁of▁sentence｜>",),
        allowed_roles=("system", "user", "assistant", "tool"),
        system_prompt_policy="warn",
        default_temperature=0.6,
        default_top_p=0.95,
        default_top_k=None,
        default_max_new_tokens=1024,
        default_forced_prefix="\n\n",
        needs_fix_mistral_regex=False,
        weights_size_gb=65.5,
        min_gpu_memory_bf16_gb=80,
    ),
}

MODEL_IDS: tuple[str, ...] = tuple(MODELS)


def get_model(model_id: str) -> ModelMeta:
    """Return the metadata for ``model_id`` or raise ``ValueError``."""
    try:
        return MODELS[model_id]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise ValueError(
            f"unknown model_id {model_id!r}; expected one of: {known}"
        ) from None


def defaults_for(model_id: str, task_type: TaskType) -> dict[str, Any]:
    """Return a complete configuration payload pre-filled for this combination.

    The result serialises straight to the JSON shape of
    ``specs/parameters.md`` section 13 and can be splatted into
    ``ConfigRecord(**defaults_for(...))``. Every call returns fresh objects.
    """
    meta = get_model(model_id)
    if task_type not in TASK_TYPES:
        accepted = ", ".join(TASK_TYPES)
        raise ValueError(
            f"unknown task_type {task_type!r}; expected one of: {accepted}"
        )

    is_fine_tune = task_type == "fine_tune"

    return {
        "model_id": meta.model_id,
        "task_type": task_type,
        "run_status": "pending",
        "a": {
            "model_id": meta.model_id,
            "dtype": "auto",
            "local_files_only": True,
            "tp_plan": "auto",
            "device_map": None,
            "attn_implementation": "sdpa",
            "quantization": None,
            "trust_remote_code": False,
        },
        "b": {
            "tokenizer_id": meta.model_id,
            "padding_side": "left",
            "pad_token": meta.pad_token,
            "pad_token_id": meta.pad_token_id,
            "fix_mistral_regex": meta.needs_fix_mistral_regex,
            "use_fast": True,
            "model_max_length": meta.tokenizer_model_max_length,
            "truncation_side": "right",
        },
        "c": {
            "train_data_file": TRAIN_DATA_FILE if is_fine_tune else None,
            "eval_data_file": EVAL_DATA_FILE if is_fine_tune else None,
            "test_data_file": None if is_fine_tune else TEST_DATA_FILE,
            "data_format": "json",
            "label_values": ["True", "False", "Conflicting"],
            "evidence_answers_used": 1,
            "missing_answer_placeholder": "No answer could be found !",
            "max_seq_length": 4096,
            "max_samples": None,
            "evidence_config": "with_evidence",
        },
        "d": {
            "lora": {
                "r": 64,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none",
                "target_modules": list(LORA_TARGET_MODULES),
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
        },
        "e": {
            "apply_chat_template": True,
            "add_generation_prompt": not is_fine_tune,
            "tokenize": True,
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
            "max_length": 4096,
            "device": "cuda",
        },
        "f": {
            "max_new_tokens": meta.default_max_new_tokens,
            "temperature": meta.default_temperature,
            "top_p": meta.default_top_p,
            "top_k": meta.default_top_k,
            "do_sample": True,
            "use_cache": True,
            "num_return_sequences": 1,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "pad_token_id": meta.pad_token_id,
            "eos_token_id": meta.eos_token_id,
            "forced_prefix": meta.default_forced_prefix,
            "batch_size": 1,
        },
        "g": {
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
            "spaces_between_special_tokens": True,
        },
        "h": {},
        "p": {
            "system_prompt": (
                DEFAULT_SYSTEM_PROMPT
                if meta.system_prompt_policy == "required"
                else ""
            ),
            "user_prompt_template": DEFAULT_USER_PROMPT_TEMPLATE,
            "assistant_prompt": DEFAULT_ASSISTANT_PROMPT,
            "chat_template_override": None,
            "verdict_parse_marker": DEFAULT_VERDICT_MARKER,
        },
    }


__all__ = [
    "DEEPSEEK_ID",
    "MISTRAL_ID",
    "MODELS",
    "MODEL_IDS",
    "ModelMeta",
    "TASK_TYPES",
    "TaskType",
    "defaults_for",
    "get_model",
]
