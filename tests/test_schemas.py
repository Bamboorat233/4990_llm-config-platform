"""Checks for the configuration schemas (issue #3).

Runs with either:

    python tests/test_schemas.py
    pytest tests/test_schemas.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from app.registry import (  # noqa: E402
    DEEPSEEK_ID,
    MISTRAL_ID,
    MODEL_IDS,
    defaults_for,
)
from app.schemas import (  # noqa: E402
    ConfigRecord,
    DataSection,
    EncodingSection,
    config_fingerprint,
)

TASK_TYPES = ("fine_tune", "inference")

CONFIG_KEYS = {
    "config_id",
    "config_hash",
    "model_id",
    "task_type",
    "run_status",
    "created_at",
    "created_by",
    "notes",
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

FINGERPRINT_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _record(model_id: str = MISTRAL_ID, task_type: str = "fine_tune") -> ConfigRecord:
    return ConfigRecord(**defaults_for(model_id, task_type))


def _reversed_mapping(value: Any) -> Any:
    """Recursively rebuild every mapping with its keys in reverse order."""
    if isinstance(value, dict):
        return {key: _reversed_mapping(value[key]) for key in reversed(list(value))}
    return value


def _expect_validation_error(call: Callable[[], Any]) -> ValidationError:
    try:
        call()
    except ValidationError as error:
        return error
    raise AssertionError("expected ValidationError")


def test_all_four_combinations_build() -> None:
    for model_id in MODEL_IDS:
        for task_type in TASK_TYPES:
            cfg = ConfigRecord(**defaults_for(model_id, task_type))
            dumped = cfg.model_dump(mode="json")
            json.dumps(dumped)
            assert set(dumped) == CONFIG_KEYS
            assert dumped["model_id"] == model_id
            assert dumped["task_type"] == task_type
            assert dumped["run_status"] == "pending"
            assert UUID(dumped["config_id"]).version == 4
            assert dumped["config_hash"] == ""
            assert dumped["created_by"] is None
            assert dumped["notes"] is None


def test_mistral_fine_tune_matches_parameters_section_13() -> None:
    dumped = _record(MISTRAL_ID, "fine_tune").model_dump(mode="json")

    assert dumped["b"]["pad_token"] == "<pad>"
    assert dumped["b"]["pad_token_id"] == 11
    assert dumped["d"]["sft"]["seed"] == 42
    assert dumped["a"]["dtype"] == "auto"
    assert dumped["a"]["attn_implementation"] == "sdpa"
    assert dumped["c"]["label_values"] == ["True", "False", "Conflicting"]
    assert dumped["c"]["max_seq_length"] == 4096
    assert dumped["e"]["add_generation_prompt"] is False
    assert dumped["e"]["max_length"] == 4096
    assert dumped["f"]["temperature"] == 0.15
    assert dumped["f"]["pad_token_id"] == 11
    assert dumped["f"]["eos_token_id"] == 2
    assert dumped["f"]["batch_size"] == 1
    assert dumped["g"]["skip_special_tokens"] is True
    assert dumped["h"] == {}
    assert dumped["p"]["assistant_prompt"] == "### Response: Verdict: "
    assert dumped["p"]["verdict_parse_marker"] == "### Response: Verdict: "

    # Section 13 is abbreviated: every key it shows must be present.
    section_13_keys = {
        "a": {
            "dtype",
            "local_files_only",
            "tp_plan",
            "attn_implementation",
            "quantization",
            "trust_remote_code",
        },
        "b": {
            "padding_side",
            "pad_token",
            "fix_mistral_regex",
            "use_fast",
            "model_max_length",
        },
        "c": {
            "train_data_file",
            "eval_data_file",
            "data_format",
            "label_values",
            "evidence_answers_used",
            "missing_answer_placeholder",
            "max_seq_length",
        },
        "d": {"lora", "sft"},
        "e": {
            "apply_chat_template",
            "add_generation_prompt",
            "tokenize",
            "return_tensors",
            "padding",
            "truncation",
            "max_length",
            "device",
        },
        "f": {
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
            "use_cache",
            "num_return_sequences",
            "pad_token_id",
            "eos_token_id",
            "forced_prefix",
            "batch_size",
        },
        "g": {"skip_special_tokens"},
        "p": {
            "system_prompt",
            "user_prompt_template",
            "assistant_prompt",
            "chat_template_override",
            "verdict_parse_marker",
        },
    }
    for component, keys in section_13_keys.items():
        assert keys <= set(dumped[component]), component


def test_fingerprint_ignores_bookkeeping_fields() -> None:
    cfg = _record()
    before = config_fingerprint(cfg)

    changed = cfg.model_copy(
        update={
            "config_id": str(uuid4()),
            "config_hash": "sha256:deadbeef",
            "run_status": "success",
            "created_at": "2030-01-01T00:00:00Z",
            "created_by": "someone",
            "notes": "a note",
        }
    )
    assert config_fingerprint(changed) == before
    assert config_fingerprint(cfg) == before
    assert cfg.notes is None and cfg.created_by is None
    assert cfg.run_status == "pending"


def test_fingerprint_tracks_temperature() -> None:
    cfg = _record()
    warmer = cfg.model_copy(
        update={"f": cfg.f.model_copy(update={"temperature": 0.9})}
    )
    assert config_fingerprint(warmer) != config_fingerprint(cfg)


def test_fingerprint_tracks_model_and_task_type() -> None:
    cfg = _record(MISTRAL_ID, "fine_tune")
    other_model = cfg.model_copy(update={"model_id": DEEPSEEK_ID})
    other_task = cfg.model_copy(update={"task_type": "inference"})
    assert config_fingerprint(other_model) != config_fingerprint(cfg)
    assert config_fingerprint(other_task) != config_fingerprint(cfg)


def test_fingerprint_format() -> None:
    fingerprint = config_fingerprint(_record())
    assert FINGERPRINT_PATTERN.fullmatch(fingerprint)
    assert len(fingerprint) == len("sha256:") + 64


def test_fingerprint_is_stable_across_calls_and_round_trips() -> None:
    cfg = _record()
    fingerprint = config_fingerprint(cfg)
    assert config_fingerprint(cfg) == fingerprint

    from_dump = ConfigRecord.model_validate(cfg.model_dump(mode="json"))
    assert config_fingerprint(from_dump) == fingerprint

    from_json = ConfigRecord.model_validate_json(cfg.model_dump_json())
    assert config_fingerprint(from_json) == fingerprint

    payload = defaults_for(MISTRAL_ID, "fine_tune")
    reordered = ConfigRecord.model_validate(_reversed_mapping(payload))
    assert config_fingerprint(reordered) == fingerprint


def test_unknown_keys_are_rejected() -> None:
    payload = defaults_for(MISTRAL_ID, "fine_tune")
    payload["a"]["bogus"] = 1
    _expect_validation_error(lambda: ConfigRecord(**payload))

    payload = defaults_for(MISTRAL_ID, "fine_tune")
    payload["bogus"] = 1
    _expect_validation_error(lambda: ConfigRecord(**payload))


def test_invalid_literals_are_rejected() -> None:
    payload = defaults_for(MISTRAL_ID, "fine_tune")
    payload["task_type"] = "train"
    _expect_validation_error(lambda: ConfigRecord(**payload))

    payload = defaults_for(MISTRAL_ID, "fine_tune")
    payload["c"]["evidence_config"] = "everything"
    _expect_validation_error(lambda: ConfigRecord(**payload))


def test_add_generation_prompt_is_required() -> None:
    payload = defaults_for(MISTRAL_ID, "fine_tune")
    del payload["e"]["add_generation_prompt"]
    _expect_validation_error(lambda: ConfigRecord(**payload))
    _expect_validation_error(lambda: EncodingSection(tokenize=True))


def test_mutable_defaults_are_isolated() -> None:
    first = DataSection()
    second = DataSection()
    first.label_values.append("Unknown")
    assert second.label_values == ["True", "False", "Conflicting"]

    payload = defaults_for(MISTRAL_ID, "fine_tune")
    del payload["h"]
    first_record = ConfigRecord(**payload)
    second_record = ConfigRecord(**payload)
    first_record.h["generate"] = {"top_k": 1}
    assert second_record.h == {}


def test_created_at_is_utc_iso8601() -> None:
    created_at = _record().created_at
    parsed = datetime.fromisoformat(created_at)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)


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
