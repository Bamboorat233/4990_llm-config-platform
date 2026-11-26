"""Config-driven requirements.txt generator (issue #59).

This module is pure Python and must never import ``nicegui``.
"""

from __future__ import annotations

from app.registry import MISTRAL_ID
from app.schemas import ConfigRecord

_BASE_REQUIREMENTS: tuple[str, ...] = (
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


def generate_requirements_txt(cfg: ConfigRecord) -> str:
    """Return an execution ``requirements.txt`` tailored to ``cfg``."""
    lines: list[str] = [
        f"# Auto-generated requirements for {cfg.task_type} ({cfg.a.model_id or cfg.model_id})",
        *_BASE_REQUIREMENTS,
    ]

    adapter_path = getattr(cfg.a, "adapter_path", None)
    has_adapter = isinstance(adapter_path, str) and bool(adapter_path.strip())

    if cfg.task_type == "fine_tune":
        lines.append("trl>=0.12.0")
        lines.append("peft>=0.13.0")
    elif has_adapter:
        lines.append("peft>=0.13.0")

    uses_8bit_optim = cfg.task_type == "fine_tune" and "8bit" in (cfg.d.sft.optim or "")
    uses_quantization = cfg.a.quantization in ("4bit", "8bit")
    if uses_8bit_optim or uses_quantization:
        lines.append("bitsandbytes>=0.44.0")

    model_id = cfg.a.model_id or cfg.model_id
    if model_id == MISTRAL_ID:
        lines.append("mistral-common>=1.5.2")

    if cfg.a.attn_implementation == "flash_attention_2":
        lines.append("# Requires CUDA build environment (nvcc)")
        lines.append("flash-attn>=2.6.0")

    return "\n".join(lines) + "\n"


__all__ = ["generate_requirements_txt"]
