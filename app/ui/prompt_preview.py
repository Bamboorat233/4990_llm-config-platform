"""Render the exact text a model receives after its chat template is applied.

The preview mirrors the rendered samples in ``specs/model-reference.md``
sections 2.2 (Mistral) and 3.2 (DeepSeek). It is a read-only aid that lets a
researcher see the effect of the Component p prompt fields without loading
any model weights, so it works offline.

This module does not import NiceGUI, so it can be unit-tested directly.
"""

from __future__ import annotations

import re
from typing import Any

from app.registry import DEEPSEEK_ID, MODELS

# One example record in the QuanTemp format used by the reference scripts.
SAMPLE_VALUES = {
    "claim": "The unemployment rate fell to 3.5% in 2019.",
    "question": "What was the unemployment rate in 2019?",
    "answer": "The rate averaged 3.7% across 2019.",
}
SAMPLE_LABEL = "Conflicting"

# Shown when Mistral has no system prompt: its template then injects a long
# default prompt that contains today's date (see model-reference.md 2.2).
MISTRAL_DEFAULT_SYSTEM = "<default system prompt containing today's date>"

# DeepSeek role tags (fullwidth vertical bar, as in the official template).
DS_USER_TAG = "<\uff5cUser\uff5c>"
DS_ASSISTANT_TAG = "<\uff5cAssistant\uff5c>"

_LINE_PATTERN = re.compile(
    r"^(?:#+\s*)?(claim|question|answer)\s*:\s*(.*)$", re.IGNORECASE
)


class _KeepMissing(dict):
    """Leave unknown ``{placeholders}`` untouched instead of raising."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def autoformat_user_prompt(
    raw_text: str, custom_samples: dict[str, str] | None = None
) -> tuple[str, dict[str, str]]:
    """Format ``Claim: ...`` lines with ``### `` headings without erasing user text.

    - Preserves the user's visible text in the input box while ensuring the
      ``### Claim:``, ``### Question:``, and ``### Answer:`` prefixes are applied.
    - Also extracts custom sample text into ``samples`` for the live preview.
    """
    samples = dict(SAMPLE_VALUES)
    if custom_samples:
        for k, v in custom_samples.items():
            if v:
                samples[k] = v

    if not raw_text:
        return raw_text, samples

    formatted_lines: list[str] = []
    for line in raw_text.splitlines():
        match = _LINE_PATTERN.match(line.strip())
        if match:
            field_key = match.group(1).lower()
            remainder = match.group(2).strip()
            title = field_key.capitalize()
            placeholder = "{" + field_key + "}"
            if not remainder:
                formatted_lines.append(f"### {title}: ")
            elif placeholder in remainder:
                formatted_lines.append(f"### {title}: {remainder}")
            else:
                cleaned = remainder.strip("\"'")
                if cleaned:
                    samples[field_key] = cleaned
                formatted_lines.append(f"### {title}: {cleaned}")
        else:
            formatted_lines.append(line)

    result = "\n".join(formatted_lines)
    if raw_text.endswith("\n"):
        result += "\n"
    return result, samples


def fill_user_template(
    template: str, custom_samples: dict[str, str] | None = None
) -> str:
    """Substitute the sample claim/question/answer into the user template."""
    formatted, samples = autoformat_user_prompt(template, custom_samples)
    try:
        return formatted.format_map(_KeepMissing(samples))
    except (ValueError, IndexError):
        return formatted


def render_prompt_preview(
    form_state: dict[str, Any],
    model_id: str,
    task_type: str,
    custom_samples: dict[str, str] | None = None,
) -> str:
    """Return the rendered prompt string for the current form state.

    - ``fine_tune`` shows a training sample (system + user + assistant target).
    - ``inference`` shows a generation prompt (``add_generation_prompt=True``).
    """
    p = form_state.get("p", {}) or {}
    system = p.get("system_prompt") or ""
    user = fill_user_template(p.get("user_prompt_template") or "", custom_samples)
    assistant = p.get("assistant_prompt") or ""
    meta = MODELS.get(model_id)
    if meta is None:
        return ""

    if model_id == DEEPSEEK_ID:
        text = f"{meta.bos_token}{system}{DS_USER_TAG}{user}{DS_ASSISTANT_TAG}"
        if task_type == "fine_tune":
            return text + f"{assistant}{SAMPLE_LABEL}{meta.eos_token}"
        return text + "<think>\n"

    system_block = system if system else MISTRAL_DEFAULT_SYSTEM
    text = (
        f"{meta.bos_token}[SYSTEM_PROMPT]{system_block}[/SYSTEM_PROMPT]"
        f"[INST]{user}[/INST]"
    )
    if task_type == "fine_tune":
        return text + f"{assistant}{SAMPLE_LABEL}{meta.eos_token}"
    return text
