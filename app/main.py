"""Main NiceGUI entry point: developer workbench for LLM fact-checking configs.

GitHub issues implemented:
- Issue #6: main.py page layout and form skeleton (M0.5 milestone)
- Issue #7: Model / task switching linkage (M0.5 milestone)
- Issue #11: Save, deduplicate, load and duplicate in the UI (M1.0 milestone)
- Issue #12: Component h (**kwargs) dynamic key-value panels (M1.0 milestone)

Task-type visibility rules (specs/parameters.md sections 5, 6, 11 and 14):
- Component d (Optimizer) is shown only for fine_tune.
- c.train_data_file / c.eval_data_file only for fine_tune;
  c.test_data_file / c.evidence_config only for inference.
- p.assistant_prompt only for fine_tune ("not used" for inference).
- e.add_generation_prompt is derived from the task type, so it is read-only.

Run with:
    python -m app.main
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nicegui import binding, ui

from app.registry import DEEPSEEK_ID, MISTRAL_ID, MODELS, defaults_for
from app.schemas import ConfigRecord, config_fingerprint
from app.store import (
    delete_config,
    duplicate_config,
    find_by_hash,
    get_config,
    init_db,
    insert_config,
    list_configs,
)
from app.ui.console import build_console
from app.ui.kwargs_panel import KwargsSubPanel, validate_kwargs_section
from app.ui.prompt_preview import (
    SAMPLE_VALUES,
    autoformat_user_prompt,
    render_prompt_preview,
)
from app.ui.results import build_results_panel
from app.validation import validate_cross_fields

# Options for the model selector (value -> label)
AVAILABLE_MODELS = {
    MISTRAL_ID: "Mistral Small 24B (mistralai/Mistral-Small-24B-Instruct-2501)",
    DEEPSEEK_ID: "DeepSeek R1 Distill 32B (deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)",
}
MODEL_ID_OPTIONS = [MISTRAL_ID, DEEPSEEK_ID]
SHORT_MODEL_NAMES = {MISTRAL_ID: "Mistral-24B", DEEPSEEK_ID: "DeepSeek-32B"}

TASK_TYPE_OPTIONS = {
    "fine_tune": "Fine-Tune",
    "inference": "Inference",
}

EVIDENCE_CONFIG_OPTIONS = {
    "with_evidence": "With Evidence (full)",
    "without_evidence": "Without Evidence",
    "random_evidence": "Random Evidence",
    "question_only": "Question Only",
    "answer_only": "Answer Only",
    "all_questions_last_answer": "All Questions + Last Answer",
    "all_questions_all_but_last_answer": "All Questions + All But Last Answer",
}

DEFAULT_USER_TEMPLATE = (
    "### Claim: {claim}\n### Question: {question}\n### Answer: {answer}\n"
)

MODEL_PROMPTS_CONFIG_PATH = Path(__file__).with_name("model_prompts.json")


def load_model_prompts_config() -> dict[str, dict[str, dict[str, bool]]]:
    """Load per-model, per-task prompt visibility rules from app/model_prompts.json."""
    with MODEL_PROMPTS_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def get_prompt_visibility(model_id: str, task_type: str) -> dict[str, bool]:
    """Return which prompts (system_prompt, user_prompt_template, assistant_prompt) are shown."""
    config = load_model_prompts_config()
    model_cfg = config.get(model_id, {})
    task_cfg = model_cfg.get(task_type, {})
    return {
        "system_prompt": bool(task_cfg.get("system_prompt", True)),
        "user_prompt_template": bool(
            task_cfg.get("user_prompt_template", True)
        ),
        "assistant_prompt": bool(
            task_cfg.get("assistant_prompt", task_type == "fine_tune")
        ),
    }


# Pipeline tabs: (component key, icon, short title)
PIPELINE_TABS = [
    ("a", "dns", "Model"),
    ("b", "text_fields", "Tokenizer"),
    ("c", "dataset", "Data"),
    ("d", "tune", "Optimizer"),
    ("e", "code", "Encoding"),
    ("f", "auto_awesome", "Generation"),
    ("g", "subtitles", "Decoding"),
    ("h", "data_object", "Extra kwargs"),
]

# UI font ('Plus Jakarta Sans' + 'Inter' weights 400-700) + 'JetBrains Mono' (weight 400) for code blocks
FONT_HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&family=Plus+Jakarta+Sans:wght@500;600;700&display=swap" rel="stylesheet">
"""

PAGE_CSS = """
/* === LAYER 1: PRIMITIVE TOKENS === */
:root {
  --font-sans: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;

  /* Strict border-radius scale (ui-ux-pro-max primitive tokens) */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-full: 9999px;

  /* Component-level radius tokens */
  --card-radius: var(--radius-lg);
  --input-radius: var(--radius-md);
  --button-radius: var(--radius-md);
  --badge-radius: var(--radius-full);

  --slate-50: #f8fafc;
  --slate-100: #f1f5f9;
  --slate-200: #e2e8f0;
  --slate-300: #cbd5e1;
  --slate-600: #475569;
  --slate-700: #334155;
  --slate-800: #1e293b;
  --slate-900: #0f172a;

  --zinc-100: #f4f4f5;
  --zinc-300: #d4d4d8;
  --zinc-400: #a1a1aa;
  --zinc-700: #3f3f46;
  --zinc-800: #27272a;
  --zinc-900: #141417;
  --zinc-950: #09090b;
}

/* === LAYER 2: SEMANTIC TOKENS (LIGHT & DARK) === */
:root {
  --bg-canvas: #f3f5f8;
  --bg-surface: #ffffff;
  --bg-subtle: #f8fafc;
  --bg-input: #f8fafc;
  --bg-header: #0f172a;
  --text-primary: var(--slate-900);
  --text-secondary: var(--slate-600);
  --border-default: var(--slate-200);
  --border-input: var(--slate-300);
  --card-shadow: 0 1px 2px 0 rgba(15, 23, 42, 0.03);
}

body.body--dark {
  --bg-canvas: var(--zinc-950);
  --bg-surface: var(--zinc-900);
  --bg-subtle: #1c1c21;
  --bg-input: #1c1c21;
  --bg-header: var(--zinc-950);
  --text-primary: var(--zinc-100);
  --text-secondary: var(--zinc-400);
  --border-default: var(--zinc-800);
  --border-input: var(--zinc-700);
  --card-shadow: none;
}

/* === LAYER 3: GLOBAL OVERRIDES & COMPONENT TOKENS === */
body,
.q-btn,
.q-field,
.q-tab,
.q-item,
.q-uploader,
.q-expansion-item,
.q-card,
.q-badge,
.q-chip,
input,
textarea {
  font-family: var(--font-sans) !important;
  letter-spacing: -0.011em !important;
  font-variant-numeric: tabular-nums;
}

body {
  background-color: var(--bg-canvas) !important;
  color: var(--text-primary) !important;
}

.mono {
  font-family: var(--font-mono) !important;
  font-weight: 400 !important;
}

.surface-header {
  background-color: var(--bg-header) !important;
  border-bottom: 1px solid #1e293b !important;
  color: #f8fafc !important;
}

.surface-card {
  background-color: var(--bg-surface) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--card-radius) !important;
  color: var(--text-primary) !important;
  box-shadow: var(--card-shadow) !important;
}

.surface-subtle {
  background-color: var(--bg-subtle) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: var(--input-radius) !important;
  color: var(--text-primary) !important;
}

.surface-drawer {
  background-color: var(--bg-surface) !important;
  border-right: 1px solid var(--border-default) !important;
  color: var(--text-primary) !important;
}

.border-token {
  border: 1px solid var(--border-default) !important;
}

.divider-b {
  border-bottom: 1px solid var(--border-default) !important;
}

.divider-t {
  border-top: 1px solid var(--border-default) !important;
}

.divider-r {
  border-right: 1px solid var(--border-default) !important;
}

/* Atera-style column header tick mark */
.atera-col-header {
  position: relative;
  padding-left: 8px;
}
.atera-col-header::before {
  content: "";
  position: absolute;
  left: 0;
  top: 2px;
  bottom: 2px;
  width: 1px;
  background-color: var(--border-default);
}

/* Sharp, structured diagnostic callout matching the rest of the UI (no pastel bubble) */
.diagnostic-box {
  background-color: var(--bg-subtle) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: 3px !important;
  color: var(--text-primary) !important;
}
.diagnostic-box-error {
  border-left: 3px solid #dc2626 !important;
}
body.body--dark .diagnostic-box-error {
  border-left: 3px solid #f87171 !important;
}
.diagnostic-box-warning {
  border-left: 3px solid #d97706 !important;
}
body.body--dark .diagnostic-box-warning {
  border-left: 3px solid #fbbf24 !important;
}

.text-secondary-token {
  color: var(--text-secondary) !important;
}

.q-btn:not(.q-btn--round) {
  border-radius: var(--button-radius) !important;
}

/* Clean, grounded inputs with transparent example placeholders */
.q-field--outlined .q-field__control {
  background-color: var(--bg-input) !important;
  border-radius: var(--input-radius) !important;
  min-height: 36px !important;
  padding: 0 10px !important;
}

.q-field--outlined .q-field__control:before {
  border: 1px solid var(--border-default) !important;
  transition: border-color 0.15s ease;
}

.q-field--outlined:hover .q-field__control:before {
  border-color: var(--border-input) !important;
}

.q-field__native,
.q-field__prefix,
.q-field__suffix,
.q-field__input {
  color: var(--text-primary) !important;
  font-size: 13px !important;
  font-weight: 400 !important;
  padding: 4px 0 !important;
}

.q-field__native::placeholder,
.q-field__input::placeholder {
  color: var(--text-secondary) !important;
  opacity: 0.45 !important;
}

/* Transparent example text appearance for untouched default prompt values */
.is-example-text .q-field__native,
.is-example-text .q-field__input {
  color: var(--text-secondary) !important;
  opacity: 0.45 !important;
  transition: opacity 0.15s ease, color 0.15s ease;
}

.is-example-text.q-field--focused .q-field__native,
.is-example-text.q-field--focused .q-field__input {
  color: var(--text-primary) !important;
  opacity: 1 !important;
}

.q-field__marginal {
  height: 36px !important;
  color: var(--text-secondary) !important;
}

/* Animated sliding pill toggle: Fine-Tune (Green) <-> Inference (Blue) with zero bleed */
.task-toggle {
  position: relative;
  background-color: var(--bg-subtle) !important;
  border: 1px solid var(--border-default) !important;
  border-radius: 8px !important;
  padding: 2px !important;
  overflow: hidden;
}

.task-toggle::before {
  content: "";
  position: absolute;
  top: 2px;
  bottom: 2px;
  left: 2px;
  width: calc(50% - 2px);
  background-color: #10b981;
  border-radius: 6px;
  transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.25s ease;
  z-index: 0;
  pointer-events: none;
}

.task-toggle.mode-inference::before {
  transform: translateX(100%);
  background-color: #3b82f6;
}

.task-toggle .q-btn,
.task-toggle .q-btn.bg-primary,
.task-toggle .q-btn::before {
  background: transparent !important;
  background-color: transparent !important;
  box-shadow: none !important;
  z-index: 1;
  min-height: 32px !important;
  font-weight: 500 !important;
  color: var(--text-primary) !important;
  transition: color 0.2s ease !important;
}

.task-toggle:not(.mode-inference) .q-btn:first-child,
.task-toggle.mode-inference .q-btn:last-child {
  color: #ffffff !important;
}

/* Ensure dropdown menus and active items are readable in Dark Mode */
body.body--dark .q-menu {
  background-color: #1c1c21 !important;
  border: 1px solid var(--zinc-700) !important;
  color: var(--zinc-100) !important;
}

body.body--dark .q-item--active {
  color: #60a5fa !important;
  background-color: rgba(59, 130, 246, 0.15) !important;
}

/* Hide +1/-1 spinner arrows on all number inputs */
input[type="number"]::-webkit-outer-spin-button,
input[type="number"]::-webkit-inner-spin-button {
  -webkit-appearance: none !important;
  margin: 0 !important;
}
input[type="number"] {
  -moz-appearance: textfield !important;
}

/* Vertical pipeline tabs */
.pipeline-tabs .q-tab {
  justify-content: flex-start;
  min-height: 40px;
  padding: 0 12px;
}
.pipeline-tabs .q-tab__content {
  flex-direction: row;
  gap: 8px;
}

/* Prompt preview & JSON code blocks: regular weight 400 (no semi-bold bloat) */
.prompt-preview-box pre,
.prompt-preview-box code,
.json-viewer pre,
.json-viewer code,
.json-viewer span,
.q-code pre,
.q-code code,
.q-code span {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-x: hidden !important;
  font-family: var(--font-mono) !important;
  font-size: 12px !important;
  font-weight: 400 !important;
  line-height: 1.5 !important;
}

/* Clean typographic status tags: plain text with little green/yellow/red dot (no highlight box) */
.info-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  font-weight: 600;
  letter-spacing: -0.01em;
  line-height: 1.4;
  white-space: nowrap;
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

.tag-success {
  color: #059669 !important;
}
body.body--dark .tag-success {
  color: #34d399 !important;
}

.tag-info {
  color: #2563eb !important;
}
body.body--dark .tag-info {
  color: #60a5fa !important;
}

.tag-warning,
.tag-terminated {
  color: #d97706 !important;
}
body.body--dark .tag-warning,
body.body--dark .tag-terminated {
  color: #fbbf24 !important;
}

.tag-error {
  color: #dc2626 !important;
}
body.body--dark .tag-error {
  color: #f87171 !important;
}

.tag-neutral {
  color: var(--text-secondary) !important;
  font-weight: 500;
}

/* Text-only green/blue classes for the Saved Configurations drawer model names */
.model-fine-tune {
  color: #059669 !important;
}
body.body--dark .model-fine-tune {
  color: #34d399 !important;
}
.model-inference {
  color: #2563eb !important;
}
body.body--dark .model-inference {
  color: #60a5fa !important;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 9999px;
  display: inline-block;
  flex-shrink: 0;
}
.dot-success { background-color: #10b981; }
.dot-info { background-color: #3b82f6; }
.dot-warning, .dot-terminated { background-color: #f59e0b; }
.dot-error { background-color: #ef4444; }
.dot-neutral { background-color: #94a3b8; }

.phase-row {
  background-color: var(--bg-surface);
  transition: background-color 0.12s ease;
}
.phase-row:hover {
  background-color: var(--bg-subtle);
}

/* Remove Quasar q-uploader's empty white list area; uploaded files render in our custom list inside the dropdown */
.compact-uploader .q-uploader__list {
  display: none !important;
}
.compact-uploader {
  min-height: 0 !important;
  box-shadow: none !important;
  width: 100% !important;
}
.compact-uploader .q-uploader__header {
  border-radius: 0 !important;
}

/* Properly center and pad the Upload Log File expansion header & content away from the border */
.upload-expansion .q-item {
  min-height: 42px !important;
  padding: 8px 14px !important;
  align-items: center !important;
}
.upload-expansion .q-item__section--avatar {
  min-width: 28px !important;
  padding-right: 10px !important;
}
.upload-expansion .q-expansion-item__content {
  padding: 2px 12px 12px 12px !important;
}
"""


def validate_full_config(cfg: ConfigRecord) -> list[dict[str, str]]:
    """Validate cross-field rules 1-10 plus Component h kwargs whitelist."""
    issues = list(validate_cross_fields(cfg))
    issues.extend(validate_kwargs_section(cfg.h))
    return issues


def _apply_theme_defaults() -> None:
    """Vibrant, readable blue/sky highlights that stay clear in both Light and Dark mode."""
    ui.colors(
        primary="#3b82f6",  # blue-500
        secondary="#0ea5e9",  # sky-500
        accent="#6366f1",  # indigo-500
        positive="#10b981",  # emerald-500
        negative="#ef4444",  # red-500
        warning="#f59e0b",  # amber-500
    )
    for element in (ui.input, ui.textarea, ui.number, ui.select):
        element.default_props("outlined dense")


def _field_label(text: str) -> None:
    """External WCAG AA field label above an input control."""
    ui.label(text).classes("text-xs font-medium text-secondary-token mb-1 select-none")


class _PromptTextarea(ui.textarea):
    """Textarea that starts empty (showing placeholder) and populates default on Space."""

    _on_explicit_set: Callable[[Any], None] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "value" and self._on_explicit_set is not None:
            self._on_explicit_set(value)


class _PromptInput(ui.input):
    """Input that starts empty (showing placeholder) and populates default on Space."""

    _on_explicit_set: Callable[[Any], None] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "value" and self._on_explicit_set is not None:
            self._on_explicit_set(value)


class PlatformUI:
    """Manages the platform state, widgets, form linkage, and layout."""

    current_model_id = binding.BindableProperty()
    current_task_type = binding.BindableProperty()
    status_text = binding.BindableProperty()
    saved_count = binding.BindableProperty()

    def __init__(self) -> None:
        init_db()

        self.current_model_id = MISTRAL_ID
        self.current_task_type = "fine_tune"
        self.created_by_val = ""
        self.notes_val = ""
        self.status_text = "Valid"
        self.saved_count = 0

        # Sample claim/question/answer used for live formatting & preview
        self.sample_overrides: dict[str, str] = dict(SAMPLE_VALUES)

        self.form_state = defaults_for(self.current_model_id, self.current_task_type)
        if "h" not in self.form_state or not isinstance(self.form_state["h"], dict):
            self.form_state["h"] = {}

        self.field_widgets: dict[str, Any] = {}
        self.kwargs_panels: dict[str, KwargsSubPanel] = {}
        self.is_batch_updating = False

        # References to dynamically shown/hidden UI elements
        self.optimizer_expansion: Any = (
            None  # Component d panel (checked by test_main.py)
        )
        self.optimizer_tab: Any = None
        self.pipeline_tabs: Any = None
        self.side_tabs: Any = None
        self.data_fine_tune_container: Any = None
        self.data_inference_container: Any = None
        self.system_prompt_container: Any = None
        self.user_prompt_container: Any = None
        self.assistant_prompt_container: Any = None
        self.saved_configs_container: Any = None
        self.console: Any = None
        self.prompt_preview: Any = None
        self.results_panel: Any = None
        self.delete_dialog: Any = None
        self.delete_dialog_label: Any = None

    # ------------------------------------------------------------------
    # Dotted-path helpers
    # ------------------------------------------------------------------
    def _set_path_value(self, path: str, value: Any) -> None:
        """Update a dotted path (e.g., 'd.lora.r') in the nested form_state."""
        parts = path.split(".")
        target = self.form_state
        for segment in parts[:-1]:
            if segment not in target or not isinstance(target[segment], dict):
                target[segment] = {}
            target = target[segment]
        target[parts[-1]] = value

    def _get_path_value(self, path: str, default: Any = None) -> Any:
        """Retrieve a value by dotted path (e.g., 'f.temperature')."""
        target = self.form_state
        for segment in path.split("."):
            if isinstance(target, dict) and segment in target:
                target = target[segment]
            else:
                return default
        return target

    def _refresh_console_soon(self) -> None:
        """Schedule a console refresh on the next event-loop tick.

        NiceGUI suppresses client updates for an element while it is handling
        that element's own ``value`` change, so error props applied
        synchronously from an ``on_value_change`` handler never reach the
        browser. Deferring the refresh by one tick applies the props after the
        handler returns, which makes live field highlighting visible.
        """
        if self.console is None:
            return
        self.refresh_console()
        ui.timer(0.01, self.refresh_console, once=True)

    # ------------------------------------------------------------------
    # Widget registration and field factories
    # ------------------------------------------------------------------
    def _register_input(
        self,
        path: str,
        widget: Any,
        cast_func: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Register a widget in field_widgets and wire its change callback."""
        self.field_widgets[path] = widget

        def on_change(e: Any) -> None:
            if self.is_batch_updating:
                return
            val = e.value if hasattr(e, "value") else widget.value
            if cast_func is not None and val is not None:
                try:
                    val = cast_func(val)
                except (ValueError, TypeError):
                    pass

            # Selecting a different model in a.model_id or b.tokenizer_id switches the platform model
            if (
                path in {"a.model_id", "b.tokenizer_id"}
                and val in AVAILABLE_MODELS
                and val != self.current_model_id
            ):
                from types import SimpleNamespace

                self.on_model_changed(SimpleNamespace(value=val))
                return

            # Pressing Space in an empty prompt field populates its default text into the box
            if path.startswith("p.") and val == " ":
                p_key = path.split(".", 1)[1]
                default_val = (
                    defaults_for(self.current_model_id, self.current_task_type)
                    .get("p", {})
                    .get(p_key)
                )
                if default_val:
                    populated = (
                        default_val.rstrip("\n")
                        if isinstance(default_val, str)
                        else str(default_val)
                    )
                    self.is_batch_updating = True
                    try:
                        widget.value = populated
                    finally:
                        self.is_batch_updating = False
                    self._set_path_value(path, default_val)
                    if path == "p.assistant_prompt":
                        self._set_path_value(
                            "p.verdict_parse_marker", default_val
                        )
                    self._refresh_console_soon()
                    return

            if path == "p.user_prompt_template":
                raw_str = str(val or "").strip()
                if not raw_str:
                    self._set_path_value(path, DEFAULT_USER_TEMPLATE)
                else:
                    formatted, updated_samples = autoformat_user_prompt(
                        str(val), self.sample_overrides
                    )
                    self.sample_overrides = updated_samples
                    self._set_path_value(path, formatted)
                    # When the user presses Enter (newline), auto-format the completed line in-place
                    if str(val).endswith("\n") and formatted != str(val):
                        self.is_batch_updating = True
                        try:
                            widget.value = formatted
                        finally:
                            self.is_batch_updating = False
            else:
                self._set_path_value(path, val)
                if path == "p.assistant_prompt":
                    self._set_path_value("p.verdict_parse_marker", val)
            self._refresh_console_soon()

        widget.on_value_change(on_change)
        if isinstance(widget, (_PromptTextarea, _PromptInput)):

            def _handle_explicit_set(v: Any) -> None:
                if self.is_batch_updating:
                    return
                if v == "":
                    if path == "p.user_prompt_template":
                        self._set_path_value(path, DEFAULT_USER_TEMPLATE)
                    else:
                        self._set_path_value(path, "")
                    self._refresh_console_soon()

            widget._on_explicit_set = _handle_explicit_set
        return widget

    def text_field(
        self,
        path: str,
        label: str,
        *,
        textarea: bool = False,
        readonly: bool = False,
        placeholder: str | None = None,
    ) -> Any:
        """External-label text input (or textarea) bound to `path`."""
        value = self._get_path_value(path)
        if isinstance(value, list):
            value = ", ".join(str(x) for x in value)
        if isinstance(value, str) and textarea:
            value = value.rstrip("\n")
        init_val = "" if value is None else value
        is_prompt = path.startswith("p.")
        if is_prompt:
            # Start visually empty (greyed-out placeholder) when holding the default value;
            # pressing Space inside the box populates the default text for editing.
            p_key = path.split(".", 1)[1]
            default_p = defaults_for(self.current_model_id, self.current_task_type).get(
                "p", {}
            )
            default_val = default_p.get(p_key) or ""
            if str(init_val).strip() == str(default_val).strip():
                init_val = ""
        with ui.column().classes("w-full min-w-0 gap-0"):
            _field_label(label)
            if is_prompt:
                factory = _PromptTextarea if textarea else _PromptInput
            else:
                factory = ui.textarea if textarea else ui.input
            widget = factory(value=init_val, placeholder=placeholder)
            widget.props("outlined dense").classes("w-full")
            if textarea:
                widget.props("autogrow")
            if readonly:
                widget.props("readonly")
        return self._register_input(path, widget)

    def number_field(
        self,
        path: str,
        label: str,
        cast: Callable[[Any], Any] = int,
        *,
        readonly: bool = False,
        fmt: str | None = None,
    ) -> Any:
        """External-label numeric input bound to `path`."""
        kwargs = {"format": fmt} if fmt else {}
        with ui.column().classes("w-full min-w-0 gap-0"):
            _field_label(label)
            widget = ui.number(value=self._get_path_value(path), **kwargs)
            widget.props("outlined dense").classes("w-full")
            if readonly:
                widget.props("readonly")
        return self._register_input(path, widget, cast_func=cast)

    def select_field(
        self,
        path: str,
        label: str,
        options: Any,
        default: Any = None,
    ) -> Any:
        """External-label dropdown bound to `path`."""
        value = self._get_path_value(path)
        with ui.column().classes("w-full min-w-0 gap-0"):
            _field_label(label)
            widget = (
                ui.select(
                    options=options,
                    value=default if value is None else value,
                )
                .props("outlined dense")
                .classes("w-full")
            )
        return self._register_input(path, widget)

    def switch_field(self, path: str, label: str, *, disabled: bool = False) -> Any:
        """Boolean switch bound to `path`."""
        widget = ui.switch(label, value=bool(self._get_path_value(path)))
        widget.props("color=primary dense").classes("text-xs font-medium")
        if disabled:
            widget.props("disable")
        return self._register_input(path, widget)

    def slider_field(
        self, path: str, label: str, min_v: float, max_v: float, step: float
    ) -> Any:
        """Clean slider + number input without cramped border box."""
        with ui.column().classes("w-full min-w-0 gap-1.5 py-1"):
            _field_label(label)
            with ui.row().classes("w-full items-center no-wrap gap-4"):
                number = (
                    ui.number(
                        value=self._get_path_value(path),
                        format="%.2f",
                        min=min_v,
                        max=max_v,
                        step=step,
                    )
                    .props("outlined dense")
                    .classes("w-28 shrink-0")
                )
                ui.slider(min=min_v, max=max_v, step=step).props(
                    "label color=primary dense"
                ).classes("flex-grow").bind_value(
                    number,
                    forward=lambda v: v,
                    backward=lambda v: min_v if v is None else v,
                )
        return self._register_input(path, number, cast_func=float)

    # ------------------------------------------------------------------
    # Auto-formatting for User Prompt Template (Claim / Question / Answer)
    # ------------------------------------------------------------------
    def format_user_prompt_field(self) -> None:
        """Auto-prefix `Claim:` / `Question:` / `Answer:` lines with `### ` and update JSON/preview."""
        widget = self.field_widgets.get("p.user_prompt_template")
        raw = (
            widget.value
            if widget is not None and widget.value is not None
            else (self._get_path_value("p.user_prompt_template", "") or "")
        )
        if not str(raw).strip():
            self._set_path_value("p.user_prompt_template", DEFAULT_USER_TEMPLATE)
            self.refresh_console()
            return
        formatted, updated_samples = autoformat_user_prompt(
            str(raw), self.sample_overrides
        )
        self.sample_overrides = updated_samples
        self._set_path_value("p.user_prompt_template", formatted)
        if widget is not None:
            self.is_batch_updating = True
            try:
                widget.value = formatted.rstrip("\n")
            finally:
                self.is_batch_updating = False
        self.refresh_console()

    # ------------------------------------------------------------------
    # Direct Error Navigation (jumps to tab + scrolls to widget)
    # ------------------------------------------------------------------
    def navigate_to_field(self, field_path: str) -> None:
        """Open the pipeline tab containing `field_path` and scroll/focus the widget."""
        prefix = field_path.split(".")[0] if "." in field_path else ""
        if (
            prefix in {"a", "b", "c", "d", "e", "f", "g", "h"}
            and self.pipeline_tabs is not None
        ):
            if not (prefix == "d" and self.current_task_type != "fine_tune"):
                self.pipeline_tabs.value = prefix

        widget = self.field_widgets.get(field_path)
        if (
            widget is not None
            and hasattr(widget, "id")
            and getattr(ui.context.client, "has_socket_connection", False)
        ):
            wid = widget.id
            ui.run_javascript(
                f"""
                setTimeout(() => {{
                    const el = getElement({wid});
                    if (el && el.$el) {{
                        el.$el.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                        const inp = el.$el.querySelector('input, textarea');
                        if (inp) inp.focus();
                    }}
                }}, 120);
                """
            )
        ui.notify(f"Focused field: {field_path}", type="warning", position="top")

    # ------------------------------------------------------------------
    # Live refresh
    # ------------------------------------------------------------------
    def refresh_console(self) -> None:
        """Re-validate, update the JSON preview and prompt preview."""
        if self.console is None:
            return
        state = self.console.refresh()
        n_err, n_warn = len(state.errors), len(state.warnings)
        if n_err:
            self.status_text = f"{n_err} Error{'s' if n_err > 1 else ''}"
        elif n_warn:
            self.status_text = f"{n_warn} Warning{'s' if n_warn > 1 else ''}"
        else:
            self.status_text = "Valid"

        if self.prompt_preview is not None:
            self.prompt_preview.content = render_prompt_preview(
                self.form_state,
                self.current_model_id,
                self.current_task_type,
                self.sample_overrides,
            )

    def _on_metadata_changed(self, attr: str, value: Any) -> None:
        """Store created_by / notes and refresh the JSON preview."""
        setattr(self, attr, value or "")
        if not self.is_batch_updating:
            self.refresh_console()

    # ------------------------------------------------------------------
    # Config assembly and state loading
    # ------------------------------------------------------------------
    def build_config_record(self) -> ConfigRecord:
        """Assemble the current UI state into a validated Pydantic ConfigRecord."""
        payload = copy.deepcopy(self.form_state)
        payload["model_id"] = self.current_model_id
        payload["task_type"] = self.current_task_type
        payload["created_by"] = self.created_by_val or None
        payload["notes"] = self.notes_val or None

        p = payload.get("p", {})
        if isinstance(p.get("user_prompt_template"), str):
            formatted, updated_samples = autoformat_user_prompt(
                p["user_prompt_template"], self.sample_overrides
            )
            self.sample_overrides = updated_samples
            p["user_prompt_template"] = formatted

        c = payload.get("c", {})
        if isinstance(c.get("label_values"), str):
            c["label_values"] = [
                item.strip() for item in c["label_values"].split(",") if item.strip()
            ]

        d = payload.get("d", {})
        lora = d.get("lora", {})
        if isinstance(lora.get("target_modules"), str):
            lora["target_modules"] = [
                item.strip()
                for item in lora["target_modules"].split(",")
                if item.strip()
            ]

        f = payload.get("f", {})
        if f.get("top_k") == "" or f.get("top_k") == 0:
            if self.current_model_id == DEEPSEEK_ID:
                f["top_k"] = None

        return ConfigRecord(**payload)

    def load_state(self, new_state: dict[str, Any]) -> None:
        """Populate all form widgets from a dictionary state."""
        self.is_batch_updating = True
        try:
            # Mutate the existing dictionaries in place instead of rebinding
            # them: every KwargsSubPanel keeps a reference to the "h" sub-dict
            # captured at construction time, so replacing it would orphan the
            # panel and its writes would never reach the live form state.
            new_state = copy.deepcopy(new_state)
            new_h = new_state.get("h")
            if not isinstance(new_h, dict):
                new_h = {}
            current_h = self.form_state.get("h")
            if not isinstance(current_h, dict):
                current_h = {}
                self.form_state["h"] = current_h
            current_h.clear()
            current_h.update(new_h)
            new_state["h"] = current_h
            self.form_state.clear()
            self.form_state.update(new_state)

            self.current_model_id = new_state.get("model_id", self.current_model_id)
            self.current_task_type = new_state.get("task_type", self.current_task_type)
            self.created_by_val = new_state.get("created_by") or ""
            self.notes_val = new_state.get("notes") or ""

            if "model_id" in self.field_widgets:
                self.field_widgets["model_id"].value = self.current_model_id
            if "task_type" in self.field_widgets:
                self.field_widgets["task_type"].value = self.current_task_type
            if "created_by" in self.field_widgets:
                self.field_widgets["created_by"].value = self.created_by_val
            if "notes" in self.field_widgets:
                self.field_widgets["notes"].value = self.notes_val

            default_p = defaults_for(self.current_model_id, self.current_task_type).get(
                "p", {}
            )
            for path, widget in self.field_widgets.items():
                if path in {
                    "model_id",
                    "task_type",
                    "created_by",
                    "notes",
                } or path.startswith("h."):
                    continue
                val = self._get_path_value(path)
                if val is not None and hasattr(widget, "value"):
                    if isinstance(val, list):
                        widget.value = ", ".join(str(x) for x in val)
                    elif path.startswith("p."):
                        p_key = path.split(".", 1)[1]
                        default_val = default_p.get(p_key) or ""
                        if str(val).strip() == str(default_val).strip():
                            widget.value = ""
                        else:
                            widget.value = val
                    else:
                        widget.value = val

            for stage, panel in self.kwargs_panels.items():
                panel.clear()
                existing_kwargs = self.form_state.get("h", {}).get(stage, {})
                for k, v in existing_kwargs.items():
                    val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    panel.add_row(initial_key=k, initial_val=val_str)

            self._update_visibility()
        finally:
            self.is_batch_updating = False

        self.refresh_console()

    def _update_visibility(self) -> None:
        """Show or hide task-specific fields (see module docstring for rules)."""
        is_fine_tune = self.current_task_type == "fine_tune"
        task_widget = self.field_widgets.get("task_type")
        if task_widget is not None:
            if is_fine_tune:
                task_widget.classes(remove="mode-inference")
            else:
                task_widget.classes(add="mode-inference")

        for element in (
            self.optimizer_expansion,
            self.optimizer_tab,
            self.data_fine_tune_container,
        ):
            if element is not None:
                element.set_visibility(is_fine_tune)
        if self.data_inference_container is not None:
            self.data_inference_container.set_visibility(not is_fine_tune)

        # Apply per-model, per-task prompt visibility from app/model_prompts.json
        prompt_vis = get_prompt_visibility(
            self.current_model_id, self.current_task_type
        )
        if self.system_prompt_container is not None:
            self.system_prompt_container.set_visibility(
                prompt_vis["system_prompt"]
            )
        if self.user_prompt_container is not None:
            self.user_prompt_container.set_visibility(
                prompt_vis["user_prompt_template"]
            )
        if self.assistant_prompt_container is not None:
            self.assistant_prompt_container.set_visibility(
                prompt_vis["assistant_prompt"]
            )

        if (
            not is_fine_tune
            and self.pipeline_tabs is not None
            and self.pipeline_tabs.value == "d"
        ):
            self.pipeline_tabs.value = "a"

    # ------------------------------------------------------------------
    # Model / task switching (Issue #7)
    # ------------------------------------------------------------------
    def _switch_defaults(self) -> None:
        """Reload registry defaults while preserving user metadata."""
        saved_by, saved_notes = self.created_by_val, self.notes_val
        self.load_state(defaults_for(self.current_model_id, self.current_task_type))
        self.created_by_val, self.notes_val = saved_by, saved_notes
        if "created_by" in self.field_widgets:
            self.field_widgets["created_by"].value = saved_by
        if "notes" in self.field_widgets:
            self.field_widgets["notes"].value = saved_notes
        self.refresh_console()

    def on_model_changed(self, e: Any) -> None:
        """Switch model: reload defaults for the new (model, task_type)."""
        new_model = e.value
        if new_model == self.current_model_id:
            return
        self.current_model_id = new_model
        self._switch_defaults()
        ui.notify(
            f"Loaded defaults for {SHORT_MODEL_NAMES.get(new_model, new_model)}",
            type="info",
        )

    def on_task_type_changed(self, e: Any) -> None:
        """Switch task type: reload defaults for the new (model, task_type)."""
        new_task = e.value
        if new_task == self.current_task_type:
            return
        self.current_task_type = new_task
        self._switch_defaults()
        ui.notify(
            f"Loaded defaults for {TASK_TYPE_OPTIONS.get(new_task, new_task)}",
            type="info",
        )

    # ------------------------------------------------------------------
    # Persistence (Issue #11)
    # ------------------------------------------------------------------
    def save_to_db(self, cfg: ConfigRecord) -> None:
        """Persist config to SQLite with SHA-256 hash deduplication."""
        fingerprint = config_fingerprint(cfg)
        existing_id = find_by_hash(fingerprint)
        if existing_id is not None:
            ui.notify(
                f"An identical experiment config already exists (ID: {existing_id[:8]})",
                type="warning",
            )
            return

        cfg.config_hash = fingerprint
        cfg.created_by = self.created_by_val or None
        cfg.notes = self.notes_val or None
        saved_id = insert_config(cfg)
        ui.notify(f"Config successfully saved (ID: {saved_id[:8]})", type="positive")
        self.refresh_saved_configs_list()

    def reset_to_defaults(self) -> None:
        """Reset the form back to registry defaults for current model and task."""
        self.load_state(defaults_for(self.current_model_id, self.current_task_type))
        ui.notify("Form reset to registry defaults.", type="info")

    def load_saved_config(self, config_id: str) -> None:
        """Load a saved config from SQLite into the form."""
        cfg = get_config(config_id)
        if cfg is None:
            ui.notify(f"Config {config_id} not found.", type="negative")
            return
        self.load_state(cfg.model_dump())
        ui.notify(f"Loaded config {config_id[:8]}", type="positive")

    def duplicate_saved_config(self, config_id: str) -> None:
        """Clone a saved config as an unsaved draft with a new ID and timestamp."""
        clone = duplicate_config(config_id)
        self.load_state(clone.model_dump())
        ui.notify(
            f"Duplicated config {config_id[:8]} as new draft ({clone.config_id[:8]}). "
            "Edit and save to persist.",
            type="info",
        )

    def delete_saved_config(self, config_id: str) -> None:
        """Permanently remove a saved config and refresh the drawer list."""
        delete_config(config_id)
        ui.notify(f"Deleted config {config_id[:8]}", type="warning")
        self.refresh_saved_configs_list()

    async def confirm_delete(self, config_id: str) -> None:
        """Ask for confirmation before deleting a saved config."""
        if self.delete_dialog is None:
            self.delete_saved_config(config_id)
            return
        self.delete_dialog_label.text = f"Config {config_id[:8]} will be permanently removed from the SQLite database."
        if await self.delete_dialog:
            self.delete_saved_config(config_id)

    def refresh_saved_configs_list(self) -> None:
        """Re-render the saved-config cards in the left drawer with filled green/blue badges."""
        if self.saved_configs_container is None:
            return
        self.saved_configs_container.clear()
        configs = list_configs()
        self.saved_count = len(configs)
        if self.results_panel is not None:
            self.results_panel.refresh_config_options()

        with self.saved_configs_container:
            if not configs:
                with ui.column().classes(
                    "w-full items-center gap-1 py-8 text-secondary-token"
                ):
                    ui.icon("inventory_2").classes("text-3xl opacity-60")
                    ui.label("No saved configs yet").classes("text-xs font-medium")
                return

            for item in configs:
                cid = item["config_id"]
                task = item["task_type"]
                created_at = item["created_at"][:16].replace("T", " ")
                model_color_cls = (
                    "model-fine-tune" if task == "fine_tune" else "model-inference"
                )

                with (
                    ui.card()
                    .props("flat")
                    .classes("w-full px-3 py-2.5 gap-1 rounded-md surface-subtle")
                ):
                    with ui.row().classes(
                        "w-full items-center justify-between no-wrap"
                    ):
                        ui.label(
                            SHORT_MODEL_NAMES.get(
                                item["model_id"], item["model_id"]
                            )
                        ).classes(f"font-semibold text-xs {model_color_cls}")
                        with ui.row().classes("items-center gap-0.5 no-wrap"):
                            ui.button(
                                icon="file_open",
                                on_click=lambda _, c=cid: self.load_saved_config(c),
                            ).props(
                                "flat round dense size=sm color=primary"
                            ).tooltip("Load")
                            ui.button(
                                icon="content_copy",
                                on_click=lambda _, c=cid: self.duplicate_saved_config(
                                    c
                                ),
                            ).props(
                                "flat round dense size=sm color=secondary"
                            ).tooltip("Duplicate")
                            ui.button(
                                icon="delete_outline",
                                on_click=lambda _, c=cid: self.confirm_delete(c),
                            ).props(
                                "flat round dense size=sm color=negative"
                            ).tooltip("Delete")
                    with ui.row().classes(
                        "w-full items-center justify-between text-[11px] text-secondary-token no-wrap"
                    ):
                        ui.label(cid[:8])
                        ui.label(created_at)
                    if item.get("created_by") or item.get("notes"):
                        ui.label(
                            " · ".join(
                                x
                                for x in (item.get("created_by"), item.get("notes"))
                                if x
                            )
                        ).classes("text-xs text-secondary-token line-clamp-2")


# ======================================================================
# Layout Builders
# ======================================================================
def _section_header(title: str) -> None:
    """Clean typographic card header without decorative icon badges."""
    ui.label(title).classes("text-sm font-semibold tracking-tight")


def _metric_cell(
    platform: PlatformUI, icon: str, label: str, fn: Callable[[str], str]
) -> None:
    """Hardware metric tile with icon bound to current_model_id."""
    with ui.column().classes(
        "flex-1 min-w-[115px] p-2.5 gap-0.5 rounded-md surface-subtle"
    ):
        with ui.row().classes("items-center gap-1 text-secondary-token no-wrap"):
            ui.icon(icon, color="primary").classes("text-xs")
            ui.label(label).classes(
                "text-[10px] font-semibold uppercase tracking-wider"
            )
        ui.label().classes("text-sm font-semibold").bind_text_from(
            platform, "current_model_id", backward=fn
        )


def _build_header(platform: PlatformUI, dark: Any) -> Any:
    """Clean top bar: hamburger menu, title, and dark mode toggle."""
    with ui.header(elevated=False).classes(
        "items-center justify-between px-4 py-2 surface-header"
    ):
        with ui.row().classes("items-center gap-3 no-wrap"):
            menu_btn = ui.button(icon="menu").props("flat dense round color=white")
            ui.label("LLM Fact-Checking Config Platform").classes(
                "text-sm font-semibold tracking-tight text-white"
            )

        ui.button(icon="dark_mode", on_click=dark.toggle).props(
            "flat dense round color=white"
        )
    return menu_btn


def _build_drawer(platform: PlatformUI) -> Any:
    """Left drawer strictly for Saved Configurations (no count badge or refresh button)."""
    with (
        ui.left_drawer(value=False, bordered=False)
        .props("width=310")
        .classes("p-4 gap-3 surface-drawer") as drawer
    ):
        ui.label("Saved Configurations").classes("text-sm font-semibold tracking-tight")

        ui.separator()
        platform.saved_configs_container = ui.column().classes("w-full gap-2")
        platform.refresh_saved_configs_list()

    with (
        ui.dialog() as platform.delete_dialog,
        ui.card()
        .props("flat")
        .classes("p-5 gap-3 min-w-[320px] rounded-lg surface-card"),
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("warning", color="negative").classes("text-xl")
            ui.label("Delete configuration?").classes("text-sm font-semibold")
        platform.delete_dialog_label = ui.label().classes(
            "text-xs text-secondary-token"
        )
        with ui.row().classes("w-full justify-end gap-2 pt-2"):
            ui.button(
                "Cancel", on_click=lambda: platform.delete_dialog.submit(False)
            ).props("flat dense no-caps")
            ui.button(
                "Delete",
                icon="delete",
                on_click=lambda: platform.delete_dialog.submit(True),
            ).props("unelevated dense no-caps color=negative")
    return drawer


def _build_experiment_card(platform: PlatformUI) -> None:
    """Card 1: Model selector, animated execution mode toggle, metric tiles, and metadata."""
    with ui.card().props("flat").classes("w-full p-4 gap-3.5 rounded-lg surface-card"):
        _section_header("Experiment & Target Model")

        # No redundant Target LLM / Execution Mode headings above these two controls
        with ui.grid(columns=2).classes("w-full gap-4 items-center"):
            model_select = (
                ui.select(
                    options=AVAILABLE_MODELS,
                    value=platform.current_model_id,
                    on_change=platform.on_model_changed,
                )
                .props("outlined dense")
                .classes("w-full")
            )
            platform.field_widgets["model_id"] = model_select

            task_toggle = (
                ui.toggle(
                    options=TASK_TYPE_OPTIONS,
                    value=platform.current_task_type,
                    on_change=platform.on_task_type_changed,
                )
                .props("spread no-caps unelevated toggle-color=transparent")
                .classes("task-toggle w-full")
            )
            platform.field_widgets["task_type"] = task_toggle

        with ui.row().classes("w-full gap-2.5"):
            _metric_cell(
                platform,
                "hub",
                "Parameters",
                lambda m: MODELS[m].num_parameters.split(" ")[0],
            )
            _metric_cell(
                platform,
                "straighten",
                "Context",
                lambda m: f"{MODELS[m].effective_max_seq_len:,} tok",
            )
            _metric_cell(
                platform, "layers", "Layers", lambda m: str(MODELS[m].num_hidden_layers)
            )
            _metric_cell(
                platform,
                "memory",
                "Min VRAM (bf16)",
                lambda m: f"{MODELS[m].min_gpu_memory_bf16_gb} GB",
            )

        with ui.grid(columns=2).classes("w-full gap-4"):
            with ui.column().classes("w-full min-w-0 gap-0"):
                _field_label("Researcher / Author")
                created_by_input = (
                    ui.input(value=platform.created_by_val)
                    .props("outlined dense clearable")
                    .classes("w-full")
                )
                created_by_input.on_value_change(
                    lambda e: platform._on_metadata_changed("created_by_val", e.value)
                )
                platform.field_widgets["created_by"] = created_by_input

            with ui.column().classes("w-full min-w-0 gap-0"):
                _field_label("Experiment Notes")
                notes_input = (
                    ui.input(value=platform.notes_val)
                    .props("outlined dense clearable")
                    .classes("w-full")
                )
                notes_input.on_value_change(
                    lambda e: platform._on_metadata_changed("notes_val", e.value)
                )
                platform.field_widgets["notes"] = notes_input


def _build_prompt_card(platform: PlatformUI) -> None:
    """Card 2: Prompts driven by app/model_prompts.json (3 prompts in fine-tune, 2 in inference, per-model)."""
    with ui.card().props("flat").classes("w-full p-4 gap-3.5 rounded-lg surface-card"):
        _section_header("Prompts")

        platform.system_prompt_container = ui.column().classes("w-full min-w-0 gap-0")
        with platform.system_prompt_container:
            platform.text_field(
                "p.system_prompt",
                "System Prompt",
                textarea=True,
                placeholder="You are a fact-checking assistant. Given a numerical claim and its retrieved evidence, output a verdict.",
            )

        platform.user_prompt_container = ui.column().classes("w-full min-w-0 gap-0")
        with platform.user_prompt_container:
            user_tpl_widget = platform.text_field(
                "p.user_prompt_template",
                "User Prompt Template",
                textarea=True,
                placeholder=(
                    "Claim: The unemployment rate fell to 3.5% in 2019.\n"
                    "Question: What was the unemployment rate in 2019?\n"
                    "Answer: The rate averaged 3.7% across 2019."
                ),
            )
            user_tpl_widget.on("blur", lambda _: platform.format_user_prompt_field())

        platform.assistant_prompt_container = ui.column().classes(
            "w-full min-w-0 gap-0"
        )
        with platform.assistant_prompt_container:
            platform.text_field(
                "p.assistant_prompt",
                "Assistant Prompt",
                placeholder="### Response: Verdict: ",
            )


def _build_component_panel(platform: PlatformUI, key: str) -> None:
    """Contents of one pipeline tab panel (Components a-h)."""
    if key == "a":
        with ui.grid(columns=2).classes("w-full gap-4"):
            platform.select_field(
                "a.model_id",
                "Model ID / HuggingFace Path",
                MODEL_ID_OPTIONS,
            )
            platform.select_field(
                "a.dtype",
                "Weight Dtype",
                ["auto", "bfloat16", "float16", "float32"],
            )
            platform.text_field("a.tp_plan", "Tensor Parallel Plan")
            platform.text_field("a.device_map", "Device Map")
            platform.select_field(
                "a.attn_implementation",
                "Attention Kernel",
                ["sdpa", "eager", "flash_attention_2"],
                default="sdpa",
            )
            platform.select_field(
                "a.quantization",
                "Quantization",
                ["", "4bit", "8bit"],
                default="",
            )
        with ui.row().classes("w-full gap-6 pt-1"):
            platform.switch_field("a.local_files_only", "Local Files Only (offline)")
            platform.switch_field("a.trust_remote_code", "Trust Remote Code")

    elif key == "b":
        with ui.grid(columns=2).classes("w-full gap-4"):
            platform.select_field(
                "b.tokenizer_id",
                "Tokenizer ID",
                MODEL_ID_OPTIONS,
            )
            platform.select_field("b.padding_side", "Padding Side", ["left", "right"])
            platform.text_field("b.pad_token", "Pad Token")
            platform.number_field(
                "b.pad_token_id", "Pad Token ID (read-only)", readonly=True
            )
            platform.number_field("b.model_max_length", "Max Context Length")
            platform.select_field(
                "b.truncation_side", "Truncation Side", ["right", "left"]
            )
        with ui.row().classes("w-full gap-6 pt-1"):
            platform.switch_field("b.fix_mistral_regex", "Fix Mistral Regex")
            platform.switch_field("b.use_fast", "Use Fast Tokenizer")

    elif key == "c":
        platform.data_fine_tune_container = ui.column().classes("w-full gap-3")
        with platform.data_fine_tune_container:
            with ui.grid(columns=2).classes("w-full gap-4"):
                platform.text_field("c.train_data_file", "Train Split Path")
                platform.text_field("c.eval_data_file", "Eval Split Path")

        platform.data_inference_container = ui.column().classes("w-full gap-3")
        with platform.data_inference_container:
            with ui.grid(columns=2).classes("w-full gap-4"):
                platform.text_field("c.test_data_file", "Test Split Path")
                platform.select_field(
                    "c.evidence_config",
                    "Evidence Mode",
                    EVIDENCE_CONFIG_OPTIONS,
                )

        with ui.grid(columns=2).classes("w-full gap-4"):
            platform.text_field("c.label_values", "Verdict Labels (comma-separated)")
            platform.text_field(
                "c.missing_answer_placeholder", "Missing Evidence Placeholder"
            )
            platform.number_field("c.evidence_answers_used", "Evidence Answers Used")
            platform.number_field("c.max_seq_length", "Max Sequence Length")

    elif key == "d":
        with ui.row().classes(
            "w-full items-center gap-2 no-wrap rounded-lg px-3 py-2 bg-amber-500/10 border border-amber-500/30"
        ):
            ui.icon("lock", color="warning").classes("text-base")
            ui.label("Fixed LoRA + SFT research preset (read-only).").classes(
                "text-xs font-medium"
            )
        with ui.grid(columns=3).classes("w-full gap-4"):
            platform.number_field("d.lora.r", "LoRA Rank (r)", readonly=True)
            platform.number_field("d.lora.lora_alpha", "LoRA Alpha", readonly=True)
            platform.number_field(
                "d.lora.lora_dropout", "LoRA Dropout", float, readonly=True
            )
        with ui.grid(columns=2).classes("w-full gap-4"):
            platform.text_field(
                "d.lora.target_modules", "Target Modules", readonly=True
            )
            platform.text_field("d.lora.bias", "LoRA Bias", readonly=True)
        with ui.grid(columns=3).classes("w-full gap-4"):
            platform.number_field(
                "d.sft.learning_rate", "Learning Rate", float, readonly=True
            )
            platform.number_field("d.sft.num_train_epochs", "Epochs", readonly=True)
            platform.number_field(
                "d.sft.per_device_train_batch_size", "Batch Size / GPU", readonly=True
            )
            platform.text_field("d.sft.eval_strategy", "Eval Strategy", readonly=True)
            platform.text_field("d.sft.save_strategy", "Save Strategy", readonly=True)
            platform.number_field("d.sft.seed", "Random Seed", readonly=True)

    elif key == "e":
        with ui.grid(columns=3).classes("w-full gap-4"):
            platform.text_field("e.return_tensors", "Return Tensors")
            platform.text_field("e.device", "Target Device")
            platform.number_field("e.max_length", "Encoding Max Length")
        with ui.row().classes("w-full gap-6 pt-1"):
            platform.switch_field("e.apply_chat_template", "Apply Chat Template")
            platform.switch_field(
                "e.add_generation_prompt",
                "Add Generation Prompt (auto)",
                disabled=True,
            )
            platform.switch_field("e.tokenize", "Tokenize")
            platform.switch_field("e.padding", "Padding")
            platform.switch_field("e.truncation", "Truncation")

    elif key == "f":
        platform.slider_field("f.temperature", "Temperature", 0.0, 2.0, 0.05)
        platform.slider_field("f.top_p", "Top-P", 0.0, 1.0, 0.01)
        with ui.grid(columns=3).classes("w-full gap-4"):
            platform.number_field("f.max_new_tokens", "Max New Tokens")
            platform.number_field("f.top_k", "Top-K")
            platform.number_field("f.batch_size", "Batch Size")
            platform.number_field(
                "f.repetition_penalty",
                "Repetition Penalty",
                float,
                fmt="%.2f",
            )
            platform.text_field("f.forced_prefix", "Forced Prefix")
        with ui.row().classes("w-full gap-6 pt-1"):
            platform.switch_field("f.do_sample", "Do Sample")
            platform.switch_field("f.use_cache", "Use KV Cache")

    elif key == "g":
        with ui.column().classes("w-full gap-2.5"):
            platform.switch_field("g.skip_special_tokens", "Skip Special Tokens")
            platform.switch_field(
                "g.clean_up_tokenization_spaces",
                "Clean Up Tokenization Spaces",
            )
            platform.switch_field(
                "g.spaces_between_special_tokens",
                "Spaces Between Special Tokens",
            )

    elif key == "h":
        with ui.column().classes("w-full gap-3"):
            for stage in ["load_model", "load_tokenizer", "encode", "generate"]:
                sub_panel = KwargsSubPanel(
                    stage=stage,
                    h_state=platform.form_state["h"],
                    field_widgets=platform.field_widgets,
                    on_change_callback=platform._refresh_console_soon,
                )
                platform.kwargs_panels[stage] = sub_panel
                sub_panel.build_ui()


def _build_pipeline_card(platform: PlatformUI) -> None:
    """Card 3: Components a-h organized in an animated vertical tab console."""
    with (
        ui.card()
        .props("flat")
        .classes("w-full p-0 gap-0 rounded-lg overflow-hidden surface-card")
    ):
        with ui.row().classes("w-full px-4 py-3 border-b border-token"):
            _section_header("Pipeline Parameters")
        with ui.row().classes("w-full no-wrap gap-0 items-stretch"):
            with (
                ui.tabs(value="a")
                .props("vertical no-caps active-color=primary indicator-color=primary")
                .classes(
                    "pipeline-tabs w-44 shrink-0 py-1.5 surface-subtle border-r border-token"
                ) as tabs
            ):
                tab_refs = {}
                for key, icon, title in PIPELINE_TABS:
                    tab_refs[key] = ui.tab(
                        key, label=f"{key} · {title}", icon=icon
                    ).classes("text-xs")
            platform.pipeline_tabs = tabs
            platform.optimizer_tab = tab_refs["d"]

            with (
                ui.tab_panels(tabs, value="a")
                .props("animated")
                .classes("flex-grow min-w-0 bg-transparent")
            ):
                for key, _, _ in PIPELINE_TABS:
                    with ui.tab_panel(key).classes("gap-3.5 p-4") as panel:
                        _build_component_panel(platform, key)
                    if key == "d":
                        platform.optimizer_expansion = panel


def _build_side_panel(platform: PlatformUI) -> None:
    """Right sticky column: Validation & JSON, Prompt Preview, Run Results, and HPC Terminal."""
    with (
        ui.card()
        .props("flat")
        .classes("w-full p-0 gap-0 rounded-lg overflow-hidden surface-card")
    ):
        with (
            ui.tabs(value="json")
            .props(
                "dense no-caps align=left active-color=primary indicator-color=primary"
            )
            .classes(
                "w-full px-2 pt-1 surface-subtle border-b border-token"
            ) as side_tabs
        ):
            ui.tab("json", label="Validation & JSON", icon="data_object").classes(
                "text-xs font-medium"
            )
            ui.tab("preview", label="Prompt Preview", icon="visibility").classes(
                "text-xs font-medium"
            )
            ui.tab("results", label="Run Results", icon="assessment").classes(
                "text-xs font-medium"
            )
            ui.tab("terminal", label="HPC Terminal", icon="terminal").classes(
                "text-xs font-medium"
            )
        platform.side_tabs = side_tabs

        with (
            ui.tab_panels(side_tabs, value="json")
            .props("animated")
            .classes("w-full bg-transparent")
        ):
            with ui.tab_panel("json").classes("p-4"):
                platform.console = build_console(
                    container=ui.column().classes("w-full"),
                    field_widgets=platform.field_widgets,
                    build_config=platform.build_config_record,
                    validate=validate_full_config,
                    save=platform.save_to_db,
                    reset=platform.reset_to_defaults,
                    on_navigate=platform.navigate_to_field,
                )

            with ui.tab_panel("preview").classes("p-4 gap-3"):
                platform.prompt_preview = ui.code("", language="plaintext").classes(
                    "prompt-preview-box w-full text-xs rounded-md border-token"
                )

            with ui.tab_panel("results").classes("p-4"):
                platform.results_panel = build_results_panel(
                    container=ui.column().classes("w-full"),
                    list_configs_fn=list_configs,
                )

            with ui.tab_panel("terminal").classes("p-4"):
                ui.element("div").classes(
                    "w-full min-h-[480px] rounded-md surface-subtle"
                )


def create_app() -> PlatformUI:
    """Construct the full NiceGUI developer workbench layout."""
    platform = PlatformUI()
    _apply_theme_defaults()
    ui.add_head_html(FONT_HEAD_HTML)
    ui.add_css(PAGE_CSS)
    dark = ui.dark_mode()

    menu_btn = _build_header(platform, dark)
    drawer = _build_drawer(platform)
    menu_btn.on_click(drawer.toggle)

    with ui.column().classes("w-full max-w-[1600px] mx-auto p-4 gap-4"):
        with ui.row().classes("w-full gap-4 items-start flex-wrap lg:flex-nowrap"):
            with ui.column().classes("w-full lg:w-7/12 min-w-0 gap-4"):
                _build_experiment_card(platform)
                _build_prompt_card(platform)
                _build_pipeline_card(platform)
            with ui.column().classes("w-full lg:w-5/12 min-w-0 lg:sticky top-14 gap-4"):
                _build_side_panel(platform)

    platform._update_visibility()
    platform.refresh_console()
    return platform


@ui.page("/")
def index_page() -> None:
    """Index page route."""
    create_app()


def main() -> None:
    """CLI entry point: start the NiceGUI web server."""
    ui.run(
        title="LLM Fact-Checking Platform",
        port=8080,
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
