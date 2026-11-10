"""Main NiceGUI entry point: page layout, form skeleton, and linkage.

GitHub issues implemented:
- Issue #6: main.py page layout and form skeleton (M0.5 milestone)
- Issue #7: Model / task switching linkage (M0.5 milestone)
- Issue #11: Save, deduplicate, load and duplicate in the UI (M1.0 milestone)
- Issue #12: Component h (**kwargs) dynamic key-value panels (M1.0 milestone)

Run with:
    python -m app.main
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.registry import (
    DEEPSEEK_ID,
    MISTRAL_ID,
    defaults_for,
)
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
from app.validation import validate_cross_fields

# Available models for selection
AVAILABLE_MODELS = {
    MISTRAL_ID: f"Mistral Small 24B ({MISTRAL_ID})",
    DEEPSEEK_ID: f"DeepSeek R1 Distill Qwen 32B ({DEEPSEEK_ID})",
}

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


def validate_full_config(cfg: ConfigRecord) -> list[dict[str, str]]:
    """Validate cross-field rules 1-10 plus Component h kwargs whitelist."""
    issues = list(validate_cross_fields(cfg))
    kwargs_issues = validate_kwargs_section(cfg.h)
    issues.extend(kwargs_issues)
    return issues


class PlatformUI:
    """Manages the platform state, widgets, form linkage, and layout.

    Coordinates reactive two-way data binding:
    - User input -> updates `self.form_state` dictionary.
    - Model/task switch -> loads fresh defaults from `registry.py` into `self.form_state`.
    - Live validation -> validates `ConfigRecord` on every change and updates `self.console`.
    - Left drawer -> displays SQLite saved configs with Load, Duplicate, and Delete actions.
    """

    def __init__(self) -> None:
        # Ensure the SQLite database table exists on launch
        init_db()

        # Initial default selections (Mistral + Fine-Tune per design spec)
        self.current_model_id = MISTRAL_ID
        self.current_task_type = "fine_tune"
        self.created_by_val = ""
        self.notes_val = ""

        # Populate initial form dictionary with defaults from registry
        self.form_state = defaults_for(self.current_model_id, self.current_task_type)
        if "h" not in self.form_state or not isinstance(self.form_state["h"], dict):
            self.form_state["h"] = {}

        # Widget registry mapping dotted paths (e.g. "b.pad_token") to NiceGUI widget instances.
        # Used by the validation console to apply red error rings directly on offending inputs.
        self.field_widgets: dict[str, Any] = {}
        self.kwargs_panels: dict[str, KwargsSubPanel] = {}

        # Flag to prevent event loops / re-triggering during programmatic bulk updates
        self.is_batch_updating = False

        # References to dynamically shown/hidden UI containers
        self.optimizer_expansion: Any = None
        self.data_fine_tune_container: Any = None
        self.data_inference_container: Any = None
        self.saved_configs_container: Any = None
        self.console: Any = None

    def _set_path_value(self, path: str, value: Any) -> None:
        """Update a dotted path (e.g., 'd.lora.r') in the nested form_state dictionary."""
        parts = path.split(".")
        target = self.form_state
        for segment in parts[:-1]:
            if segment not in target or not isinstance(target[segment], dict):
                target[segment] = {}
            target = target[segment]
        target[parts[-1]] = value

    def _get_path_value(self, path: str, default: Any = None) -> Any:
        """Retrieve a value by dotted path (e.g., 'f.temperature') from form_state."""
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
        ui.timer(0.01, self.console.refresh, once=True)

    def _register_input(
        self,
        path: str,
        widget: Any,
        cast_func: Callable[[Any], Any] | None = None,
    ) -> Any:
        """Register a widget in field_widgets and set up value change callback.

        - Binds the widget's value changes to update `form_state`.
        - Applies optional type casting (e.g., int, float) to enforce schema types.
        - Automatically triggers a live validation console refresh.
        """
        self.field_widgets[path] = widget

        def on_change(e: Any) -> None:
            # Skip if we are programmatically updating widgets (e.g. during load_state)
            if self.is_batch_updating:
                return
            val = e.value if hasattr(e, "value") else widget.value
            if cast_func is not None and val is not None:
                try:
                    val = cast_func(val)
                except (ValueError, TypeError):
                    pass
            self._set_path_value(path, val)
            # Re-evaluate rules and update live JSON preview on the next tick so
            # the error props are not suppressed by the element's own update.
            self._refresh_console_soon()

        widget.on_value_change(on_change)
        return widget

    def build_config_record(self) -> ConfigRecord:
        """Assemble the current UI state into a validated Pydantic ConfigRecord.

        Handles string-to-list parsing for comma-separated fields (like label_values
        and target_modules) and applies model-specific quirks like DeepSeek top_k nullability.
        """
        payload = copy.deepcopy(self.form_state)
        payload["model_id"] = self.current_model_id
        payload["task_type"] = self.current_task_type
        payload["created_by"] = self.created_by_val or None
        payload["notes"] = self.notes_val or None

        # Parse comma-separated strings into lists expected by Pydantic schemas
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

        # DeepSeek R1 quirk: top_k is disabled / nullable (empty or 0 means None)
        f = payload.get("f", {})
        if f.get("top_k") == "" or f.get("top_k") == 0:
            if self.current_model_id == DEEPSEEK_ID:
                f["top_k"] = None

        return ConfigRecord(**payload)

    def load_state(self, new_state: dict[str, Any]) -> None:
        """Populate all form widgets from a dictionary state.

        Uses `is_batch_updating = True` while populating fields to avoid firing
        redundant individual `on_value_change` events during bulk updates.
        """
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

            # Update top-level control widgets if available
            if "model_id" in self.field_widgets:
                self.field_widgets["model_id"].value = self.current_model_id
            if "task_type" in self.field_widgets:
                self.field_widgets["task_type"].value = self.current_task_type
            if "created_by" in self.field_widgets:
                self.field_widgets["created_by"].value = self.created_by_val
            if "notes" in self.field_widgets:
                self.field_widgets["notes"].value = self.notes_val

            # Update registered leaf inputs (Components a through g and p)
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
                    # Format lists (like target_modules or label_values) into comma-separated text
                    if isinstance(val, list):
                        widget.value = ", ".join(str(x) for x in val)
                    else:
                        widget.value = val

            # Refresh dynamic Component h (**kwargs) sub-panels
            for stage, panel in self.kwargs_panels.items():
                panel.clear()
                existing_kwargs = self.form_state.get("h", {}).get(stage, {})
                for k, v in existing_kwargs.items():
                    val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    panel.add_row(initial_key=k, initial_val=val_str)

            # Re-apply conditional section visibility (e.g. Optimizer hidden during inference)
            self._update_visibility()
        finally:
            self.is_batch_updating = False

        # Refresh console to update JSON preview and re-check rules against newly loaded state
        if self.console is not None:
            self.console.refresh()

    def _update_visibility(self) -> None:
        """Toggle visibility of optimizer and data fields based on task_type.

        Per spec:
        - Optimizer (Component d) is only visible for 'fine_tune'.
        - 'train_data_file' and 'eval_data_file' are only for 'fine_tune'.
        - 'test_data_file' and 'evidence_config' are only for 'inference'.
        """
        is_fine_tune = self.current_task_type == "fine_tune"
        if self.optimizer_expansion is not None:
            self.optimizer_expansion.set_visibility(is_fine_tune)
        if self.data_fine_tune_container is not None:
            self.data_fine_tune_container.set_visibility(is_fine_tune)
        if self.data_inference_container is not None:
            self.data_inference_container.set_visibility(not is_fine_tune)

    def on_model_changed(self, e: Any) -> None:
        """Switch model: reloads defaults for current (model, task_type).

        User-entered metadata (created_by, notes) is preserved across switches
        because those fields are not part of the registry defaults.
        """
        new_model = e.value
        if new_model == self.current_model_id:
            return
        # Preserve user metadata before load_state overwrites them
        saved_by = self.created_by_val
        saved_notes = self.notes_val
        self.current_model_id = new_model
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        # Restore user metadata
        self.created_by_val = saved_by
        self.notes_val = saved_notes
        if "created_by" in self.field_widgets:
            self.field_widgets["created_by"].value = saved_by
        if "notes" in self.field_widgets:
            self.field_widgets["notes"].value = saved_notes
        ui.notify(f"Switched model to {new_model}", type="info")

    def on_task_type_changed(self, e: Any) -> None:
        """Switch task type: reloads defaults for current (model, task_type).

        User-entered metadata (created_by, notes) is preserved across switches
        because those fields are not part of the registry defaults.
        """
        new_task = e.value
        if new_task == self.current_task_type:
            return
        # Preserve user metadata before load_state overwrites them
        saved_by = self.created_by_val
        saved_notes = self.notes_val
        self.current_task_type = new_task
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        # Restore user metadata
        self.created_by_val = saved_by
        self.notes_val = saved_notes
        if "created_by" in self.field_widgets:
            self.field_widgets["created_by"].value = saved_by
        if "notes" in self.field_widgets:
            self.field_widgets["notes"].value = saved_notes
        ui.notify(f"Switched task type to {new_task}", type="info")

    def save_to_db(self, cfg: ConfigRecord) -> None:
        """Persist config to SQLite with SHA256 hash deduplication.

        Computes a SHA-256 fingerprint over the functional experiment parameters.
        If an identical configuration is already saved, blocks insertion with a warning.
        Otherwise, writes to SQLite and refreshes the drawer card feed.
        """
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
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        ui.notify("Form reset to registry defaults.", type="info")

    def load_saved_config(self, config_id: str) -> None:
        """Load a saved config from SQLite by ID into the active editing form."""
        cfg = get_config(config_id)
        if cfg is None:
            ui.notify(f"Config {config_id} not found.", type="negative")
            return
        self.load_state(cfg.model_dump())
        ui.notify(f"Loaded config {config_id[:8]}", type="positive")

    def duplicate_saved_config(self, config_id: str) -> None:
        """Clone an existing config with a new ID and timestamp as an unsaved draft.

        Allows the user to easily branch off an existing experiment, adjust one or more
        hyperparameters (e.g. temperature), and save as a new independent record.
        """
        clone = duplicate_config(config_id)
        self.load_state(clone.model_dump())
        ui.notify(
            f"Duplicated config {config_id[:8]} as new draft ({clone.config_id[:8]}). Edit and save to persist.",
            type="info",
        )

    def delete_saved_config(self, config_id: str) -> None:
        """Permanently remove a saved config from SQLite and refresh the drawer list."""
        delete_config(config_id)
        ui.notify(f"Deleted config {config_id[:8]}", type="warning")
        self.refresh_saved_configs_list()

    def refresh_saved_configs_list(self) -> None:
        """Re-render the list of saved configs in the left drawer."""
        if self.saved_configs_container is None:
            return
        self.saved_configs_container.clear()
        configs = list_configs()

        with self.saved_configs_container:
            if not configs:
                ui.label("No saved configs yet.").classes(
                    "text-sm text-gray-500 italic p-2"
                )
                return

            for item in configs:
                cid = item["config_id"]
                short_id = cid[:8]
                task = item["task_type"]
                model_name = (
                    "Mistral-24B" if "Mistral" in item["model_id"] else "DeepSeek-32B"
                )
                created_at = item["created_at"][:19].replace("T", " ")

                with ui.card().classes(
                    "w-full p-3 mb-2 shadow-sm border border-gray-200"
                ):
                    with ui.row().classes(
                        "w-full items-center justify-between no-wrap"
                    ):
                        ui.label(model_name).classes("font-semibold text-sm")
                        badge_color = "positive" if task == "fine_tune" else "primary"
                        ui.badge(task, color=badge_color).classes("text-xs")
                    with ui.row().classes(
                        "w-full items-center justify-between text-xs text-gray-500 my-1"
                    ):
                        ui.label(f"ID: {short_id}")
                        ui.label(created_at)
                    with ui.row().classes("w-full justify-end gap-1 mt-1"):
                        ui.button(
                            "Load",
                            on_click=lambda _, c=cid: self.load_saved_config(c),
                        ).props("flat dense size=sm")
                        ui.button(
                            "Duplicate",
                            on_click=lambda _, c=cid: self.duplicate_saved_config(c),
                        ).props("flat dense size=sm color=secondary")
                        ui.button(
                            icon="delete",
                            on_click=lambda _, c=cid: self.delete_saved_config(c),
                        ).props("flat dense size=sm color=negative")


def create_app() -> PlatformUI:
    """Construct the full NiceGUI page layout and form skeleton."""
    platform = PlatformUI()

    # Custom page styling
    ui.colors(primary="#1976D2", secondary="#26A69A", accent="#9C27B0")

    # Header
    with ui.header().classes(
        "items-center justify-between px-4 py-2 bg-slate-900 text-white"
    ):
        with ui.row().classes("items-center gap-3"):
            drawer_btn = ui.button(icon="menu").props("flat round color=white")
            ui.label("LLM Fact-Checking Config Platform").classes(
                "text-lg font-bold tracking-wide"
            )
        with ui.row().classes("items-center gap-2"):
            ui.badge("Fir HPC Ready", color="positive").classes("text-xs px-2 py-1")
            # ui.label("Milestone 0.5 & 1.0").classes("text-xs text-slate-400")

    # Left Drawer (saved configs)
    with ui.left_drawer(value=False).classes("bg-slate-50 p-3") as drawer:
        drawer_btn.on_click(drawer.toggle)
        with ui.row().classes("w-full items-center justify-between mb-3"):
            ui.label("Saved Configurations").classes(
                "font-bold text-base text-slate-800"
            )
            ui.button(
                icon="refresh",
                on_click=platform.refresh_saved_configs_list,
            ).props("flat round dense size=sm")
        platform.saved_configs_container = ui.column().classes("w-full gap-2")
        platform.refresh_saved_configs_list()

    # Main Area: Split 7:5 (Left: Form skeleton, Right: Sticky validation console)
    # This ratio is specified in design/Spec_and_roadmap.md to give ample room for
    # prompt templates on the left and a persistent, readable JSON preview on the right.
    with ui.row().classes("w-full p-4 gap-6 no-wrap items-start"):
        # ====================================================================
        # LEFT COLUMN (7/12 width): Form configuration area
        # ====================================================================
        with ui.column().classes("w-7/12 gap-4"):
            # 1. Top-Level Control Card: Model ID, Task Type toggle, Author, and Notes
            with ui.card().classes("w-full shadow-md"):
                ui.label("Experiment Configuration").classes(
                    "text-base font-bold text-slate-800"
                )
                with ui.grid(columns=2).classes("w-full gap-4"):
                    model_select = ui.select(
                        options=AVAILABLE_MODELS,
                        value=platform.current_model_id,
                        label="Target LLM (model_id)",
                        on_change=platform.on_model_changed,
                    ).classes("w-full")
                    platform.field_widgets["model_id"] = model_select

                    task_toggle = ui.toggle(
                        options=TASK_TYPE_OPTIONS,
                        value=platform.current_task_type,
                        on_change=platform.on_task_type_changed,
                    ).classes("self-center")
                    platform.field_widgets["task_type"] = task_toggle

                with ui.grid(columns=2).classes("w-full gap-4 mt-2"):
                    created_by_input = ui.input(
                        label="Created By (Author / Researcher)",
                        value=platform.created_by_val,
                    ).classes("w-full")
                    created_by_input.on_value_change(
                        lambda e: setattr(platform, "created_by_val", e.value)
                    )
                    platform.field_widgets["created_by"] = created_by_input

                    notes_input = ui.textarea(
                        label="Experiment Notes",
                        value=platform.notes_val,
                    ).classes("w-full")
                    notes_input.on_value_change(
                        lambda e: setattr(platform, "notes_val", e.value)
                    )
                    platform.field_widgets["notes"] = notes_input

            # 2. Prompt Section (Component p)
            with ui.card().classes("w-full shadow-md"):
                with ui.row().classes("items-center justify-between w-full"):
                    ui.label("Component p: Prompts & Chat Template").classes(
                        "text-base font-bold text-slate-800"
                    )
                    # ui.badge("Research Critical", color="accent").classes("text-xs")

                sys_prompt = ui.textarea(
                    label="System Prompt (p.system_prompt)",
                    value=platform.form_state["p"]["system_prompt"],
                ).classes("w-full")
                platform._register_input("p.system_prompt", sys_prompt)

                user_prompt = ui.textarea(
                    label="User Prompt Template (p.user_prompt_template)",
                    value=platform.form_state["p"]["user_prompt_template"],
                ).classes("w-full")
                platform._register_input("p.user_prompt_template", user_prompt)

                with ui.grid(columns=2).classes("w-full gap-4"):
                    asst_prompt = ui.input(
                        label="Assistant Prompt (p.assistant_prompt)",
                        value=platform.form_state["p"]["assistant_prompt"],
                    ).classes("w-full")
                    platform._register_input("p.assistant_prompt", asst_prompt)

                    parse_marker = ui.input(
                        label="Verdict Parse Marker (p.verdict_parse_marker)",
                        value=platform.form_state["p"]["verdict_parse_marker"],
                    ).classes("w-full")
                    platform._register_input("p.verdict_parse_marker", parse_marker)

                template_override = ui.input(
                    label="Chat Template Override (p.chat_template_override, optional)",
                    value=platform.form_state["p"].get("chat_template_override") or "",
                ).classes("w-full")
                platform._register_input("p.chat_template_override", template_override)

            # 3. Expansion Panels: Components a through h
            with ui.column().classes("w-full gap-2"):
                # Panel a: Model Loading
                with ui.expansion(
                    "Component a: Loading the Model", icon="memory"
                ).classes("w-full bg-white shadow"):
                    with ui.grid(columns=2).classes("w-full gap-4 p-2"):
                        a_model_id = ui.input(
                            label="Model ID / Path (a.model_id)",
                            value=platform.form_state["a"]["model_id"],
                        )
                        platform._register_input("a.model_id", a_model_id)

                        a_dtype = ui.select(
                            options=["auto", "bfloat16", "float16", "float32"],
                            label="Dtype (a.dtype)",
                            value=platform.form_state["a"]["dtype"],
                        )
                        platform._register_input("a.dtype", a_dtype)

                        a_tp = ui.input(
                            label="Tensor Parallelism Plan (a.tp_plan)",
                            value=platform.form_state["a"]["tp_plan"] or "",
                        )
                        platform._register_input("a.tp_plan", a_tp)

                        a_dev_map = ui.input(
                            label="Device Map (a.device_map)",
                            value=platform.form_state["a"]["device_map"] or "",
                        )
                        platform._register_input("a.device_map", a_dev_map)

                        a_attn = ui.select(
                            options=["sdpa", "eager", "flash_attention_2"],
                            label="Attention Implementation (a.attn_implementation)",
                            value=platform.form_state["a"]["attn_implementation"]
                            or "sdpa",
                        )
                        platform._register_input("a.attn_implementation", a_attn)

                        a_quant = ui.select(
                            options=["", "4bit", "8bit"],
                            label="Quantization (a.quantization)",
                            value=platform.form_state["a"]["quantization"] or "",
                        )
                        platform._register_input("a.quantization", a_quant)

                    with ui.row().classes("w-full gap-6 p-2"):
                        a_local = ui.switch(
                            "Local Files Only (a.local_files_only)",
                            value=platform.form_state["a"]["local_files_only"],
                        )
                        platform._register_input("a.local_files_only", a_local)

                        a_trust = ui.switch(
                            "Trust Remote Code (a.trust_remote_code)",
                            value=platform.form_state["a"]["trust_remote_code"],
                        )
                        platform._register_input("a.trust_remote_code", a_trust)

                # Panel b: Tokenizer
                with ui.expansion(
                    "Component b: Loading the Tokenizer", icon="text_fields"
                ).classes("w-full bg-white shadow"):
                    with ui.grid(columns=2).classes("w-full gap-4 p-2"):
                        b_tok_id = ui.input(
                            label="Tokenizer ID (b.tokenizer_id)",
                            value=platform.form_state["b"]["tokenizer_id"],
                        )
                        platform._register_input("b.tokenizer_id", b_tok_id)

                        b_pad_side = ui.select(
                            options=["left", "right"],
                            label="Padding Side (b.padding_side)",
                            value=platform.form_state["b"]["padding_side"],
                        )
                        platform._register_input("b.padding_side", b_pad_side)

                        b_pad_token = ui.input(
                            label="Pad Token (b.pad_token)",
                            value=platform.form_state["b"]["pad_token"],
                        )
                        platform._register_input("b.pad_token", b_pad_token)

                        b_pad_id = ui.number(
                            label="Pad Token ID (b.pad_token_id, read-only)",
                            value=platform.form_state["b"]["pad_token_id"],
                        ).props("readonly bg-gray-100")
                        platform._register_input(
                            "b.pad_token_id", b_pad_id, cast_func=int
                        )

                        b_max_len = ui.number(
                            label="Model Max Length (b.model_max_length)",
                            value=platform.form_state["b"]["model_max_length"],
                        )
                        platform._register_input(
                            "b.model_max_length", b_max_len, cast_func=int
                        )

                        b_trunc_side = ui.select(
                            options=["right", "left"],
                            label="Truncation Side (b.truncation_side)",
                            value=platform.form_state["b"]["truncation_side"],
                        )
                        platform._register_input("b.truncation_side", b_trunc_side)

                    with ui.row().classes("w-full gap-6 p-2"):
                        b_regex = ui.switch(
                            "Fix Mistral Regex (b.fix_mistral_regex)",
                            value=platform.form_state["b"]["fix_mistral_regex"],
                        )
                        platform._register_input("b.fix_mistral_regex", b_regex)

                        b_fast = ui.switch(
                            "Use Fast Tokenizer (b.use_fast)",
                            value=platform.form_state["b"]["use_fast"],
                        )
                        platform._register_input("b.use_fast", b_fast)

                # Panel c: Data
                with ui.expansion(
                    "Component c: Loading the Data", icon="dataset"
                ).classes("w-full bg-white shadow"):
                    with ui.column().classes("w-full p-2 gap-4"):
                        # Fine-tuning data paths container
                        platform.data_fine_tune_container = ui.column().classes(
                            "w-full gap-2"
                        )
                        with platform.data_fine_tune_container:
                            ui.label("Fine-Tuning Datasets").classes(
                                "text-xs font-semibold text-gray-600"
                            )
                            with ui.grid(columns=2).classes("w-full gap-4"):
                                c_train = ui.input(
                                    label="Train Data File (c.train_data_file)",
                                    value=platform.form_state["c"]["train_data_file"]
                                    or "",
                                )
                                platform._register_input("c.train_data_file", c_train)

                                c_eval = ui.input(
                                    label="Eval Data File (c.eval_data_file)",
                                    value=platform.form_state["c"]["eval_data_file"]
                                    or "",
                                )
                                platform._register_input("c.eval_data_file", c_eval)

                        # Inference data paths & evidence container
                        platform.data_inference_container = ui.column().classes(
                            "w-full gap-2"
                        )
                        with platform.data_inference_container:
                            ui.label("Inference Datasets & Evidence").classes(
                                "text-xs font-semibold text-gray-600"
                            )
                            with ui.grid(columns=2).classes("w-full gap-4"):
                                c_test = ui.input(
                                    label="Test Data File (c.test_data_file)",
                                    value=platform.form_state["c"]["test_data_file"]
                                    or "",
                                )
                                platform._register_input("c.test_data_file", c_test)

                                c_evidence = ui.select(
                                    options=EVIDENCE_CONFIG_OPTIONS,
                                    label="Evidence Configuration (c.evidence_config)",
                                    value=platform.form_state["c"]["evidence_config"],
                                )
                                platform._register_input(
                                    "c.evidence_config", c_evidence
                                )

                        # Common data fields
                        with ui.grid(columns=2).classes("w-full gap-4 mt-2"):
                            c_labels = ui.input(
                                label="Label Values (c.label_values, comma-separated)",
                                value=", ".join(
                                    platform.form_state["c"]["label_values"]
                                ),
                            )
                            platform._register_input("c.label_values", c_labels)

                            c_placeholder = ui.input(
                                label="Missing Answer Placeholder (c.missing_answer_placeholder)",
                                value=platform.form_state["c"][
                                    "missing_answer_placeholder"
                                ],
                            )
                            platform._register_input(
                                "c.missing_answer_placeholder", c_placeholder
                            )

                            c_answers_used = ui.number(
                                label="Evidence Answers Used (c.evidence_answers_used)",
                                value=platform.form_state["c"]["evidence_answers_used"],
                            )
                            platform._register_input(
                                "c.evidence_answers_used", c_answers_used, cast_func=int
                            )

                            c_max_seq = ui.number(
                                label="Max Seq Length (c.max_seq_length)",
                                value=platform.form_state["c"]["max_seq_length"]
                                or 4096,
                            )
                            platform._register_input(
                                "c.max_seq_length", c_max_seq, cast_func=int
                            )

                # Panel d: Optimizer (LoRA + SFT) - Read-only, fine-tune only.
                # In Milestone 0.5/1.0, training hyperparameters are frozen to proven research defaults
                # (learning rate 2e-4, LoRA r=64, alpha=16) to avoid errant cluster crashes.
                platform.optimizer_expansion = ui.expansion(
                    "Component d: Optimizer (LoRA + SFT, Read-Only)",
                    icon="tune",
                ).classes("w-full bg-white shadow")
                with platform.optimizer_expansion:
                    ui.label("LoRA Adapter Settings").classes(
                        "text-xs font-semibold text-gray-600 mt-2 px-2"
                    )
                    with ui.grid(columns=3).classes("w-full gap-3 p-2"):
                        d_r = ui.number(
                            "LoRA Rank (d.lora.r)",
                            value=platform.form_state["d"]["lora"]["r"],
                        ).props("readonly bg-gray-100")
                        platform._register_input("d.lora.r", d_r, cast_func=int)

                        d_alpha = ui.number(
                            "LoRA Alpha (d.lora.lora_alpha)",
                            value=platform.form_state["d"]["lora"]["lora_alpha"],
                        ).props("readonly bg-gray-100")
                        platform._register_input(
                            "d.lora.lora_alpha", d_alpha, cast_func=int
                        )

                        d_dropout = ui.number(
                            "LoRA Dropout (d.lora.lora_dropout)",
                            value=platform.form_state["d"]["lora"]["lora_dropout"],
                        ).props("readonly bg-gray-100")
                        platform._register_input(
                            "d.lora.lora_dropout", d_dropout, cast_func=float
                        )

                    with ui.grid(columns=2).classes("w-full gap-3 px-2"):
                        d_target = ui.input(
                            "Target Modules",
                            value=", ".join(
                                platform.form_state["d"]["lora"]["target_modules"]
                            ),
                        ).props("readonly bg-gray-100")
                        platform._register_input("d.lora.target_modules", d_target)

                        d_bias = ui.input(
                            "LoRA Bias", value=platform.form_state["d"]["lora"]["bias"]
                        ).props("readonly bg-gray-100")
                        platform._register_input("d.lora.bias", d_bias)

                    ui.label("SFT Trainer Settings").classes(
                        "text-xs font-semibold text-gray-600 mt-4 px-2"
                    )
                    with ui.grid(columns=3).classes("w-full gap-3 p-2"):
                        d_lr = ui.number(
                            "Learning Rate",
                            value=platform.form_state["d"]["sft"]["learning_rate"],
                        ).props("readonly bg-gray-100")
                        platform._register_input(
                            "d.sft.learning_rate", d_lr, cast_func=float
                        )

                        d_epochs = ui.number(
                            "Epochs",
                            value=platform.form_state["d"]["sft"]["num_train_epochs"],
                        ).props("readonly bg-gray-100")
                        platform._register_input(
                            "d.sft.num_train_epochs", d_epochs, cast_func=int
                        )

                        d_batch = ui.number(
                            "Train Batch Size",
                            value=platform.form_state["d"]["sft"][
                                "per_device_train_batch_size"
                            ],
                        ).props("readonly bg-gray-100")
                        platform._register_input(
                            "d.sft.per_device_train_batch_size", d_batch, cast_func=int
                        )

                        d_eval_strat = ui.input(
                            "Eval Strategy",
                            value=platform.form_state["d"]["sft"]["eval_strategy"],
                        ).props("readonly bg-gray-100")
                        platform._register_input("d.sft.eval_strategy", d_eval_strat)

                        d_save_strat = ui.input(
                            "Save Strategy",
                            value=platform.form_state["d"]["sft"]["save_strategy"],
                        ).props("readonly bg-gray-100")
                        platform._register_input("d.sft.save_strategy", d_save_strat)

                        d_seed = ui.number(
                            "Seed", value=platform.form_state["d"]["sft"]["seed"]
                        ).props("readonly bg-gray-100")
                        platform._register_input("d.sft.seed", d_seed, cast_func=int)

                # Panel e: Token Encoding
                with ui.expansion("Component e: Token Encoding", icon="code").classes(
                    "w-full bg-white shadow"
                ):
                    with ui.grid(columns=2).classes("w-full gap-4 p-2"):
                        e_ret_tensors = ui.input(
                            label="Return Tensors (e.return_tensors)",
                            value=platform.form_state["e"]["return_tensors"],
                        )
                        platform._register_input("e.return_tensors", e_ret_tensors)

                        e_device = ui.input(
                            label="Device (e.device)",
                            value=platform.form_state["e"]["device"],
                        )
                        platform._register_input("e.device", e_device)

                        e_max_len = ui.number(
                            label="Max Length (e.max_length)",
                            value=platform.form_state["e"]["max_length"],
                        )
                        platform._register_input(
                            "e.max_length", e_max_len, cast_func=int
                        )

                    with ui.row().classes("w-full gap-6 p-2"):
                        e_template = ui.switch(
                            "Apply Chat Template (e.apply_chat_template)",
                            value=platform.form_state["e"]["apply_chat_template"],
                        )
                        platform._register_input("e.apply_chat_template", e_template)

                        e_add_gen = ui.switch(
                            "Add Generation Prompt (e.add_generation_prompt)",
                            value=platform.form_state["e"]["add_generation_prompt"],
                        )
                        platform._register_input("e.add_generation_prompt", e_add_gen)

                        e_tok = ui.switch(
                            "Tokenize (e.tokenize)",
                            value=platform.form_state["e"]["tokenize"],
                        )
                        platform._register_input("e.tokenize", e_tok)

                        e_padding = ui.switch(
                            "Padding (e.padding)",
                            value=platform.form_state["e"]["padding"],
                        )
                        platform._register_input("e.padding", e_padding)

                        e_trunc = ui.switch(
                            "Truncation (e.truncation)",
                            value=platform.form_state["e"]["truncation"],
                        )
                        platform._register_input("e.truncation", e_trunc)

                # Panel f: Text Generation
                with ui.expansion(
                    "Component f: Text Generation", icon="auto_stories"
                ).classes("w-full bg-white shadow"):
                    with ui.grid(columns=3).classes("w-full gap-3 p-2"):
                        f_tokens = ui.number(
                            label="Max New Tokens (f.max_new_tokens)",
                            value=platform.form_state["f"]["max_new_tokens"],
                        )
                        platform._register_input(
                            "f.max_new_tokens", f_tokens, cast_func=int
                        )

                        f_temp = ui.number(
                            label="Temperature (f.temperature)",
                            value=platform.form_state["f"]["temperature"],
                            format="%.2f",
                        )
                        platform._register_input(
                            "f.temperature", f_temp, cast_func=float
                        )

                        f_top_p = ui.number(
                            label="Top P (f.top_p)",
                            value=platform.form_state["f"]["top_p"],
                            format="%.2f",
                        )
                        platform._register_input("f.top_p", f_top_p, cast_func=float)

                        f_top_k = ui.number(
                            label="Top K (f.top_k)",
                            value=platform.form_state["f"]["top_k"],
                        )
                        platform._register_input("f.top_k", f_top_k, cast_func=int)

                        f_batch = ui.number(
                            label="Batch Size (f.batch_size)",
                            value=platform.form_state["f"]["batch_size"],
                        )
                        platform._register_input("f.batch_size", f_batch, cast_func=int)

                        f_rep_pen = ui.number(
                            label="Repetition Penalty (f.repetition_penalty)",
                            value=platform.form_state["f"]["repetition_penalty"],
                            format="%.2f",
                        )
                        platform._register_input(
                            "f.repetition_penalty", f_rep_pen, cast_func=float
                        )

                    with ui.grid(columns=2).classes("w-full gap-3 px-2"):
                        f_forced_prefix = ui.input(
                            label="Forced Prefix (f.forced_prefix)",
                            value=platform.form_state["f"]["forced_prefix"] or "",
                        )
                        platform._register_input("f.forced_prefix", f_forced_prefix)

                    with ui.row().classes("w-full gap-6 p-2"):
                        f_sample = ui.switch(
                            "Do Sample (f.do_sample)",
                            value=platform.form_state["f"]["do_sample"],
                        )
                        platform._register_input("f.do_sample", f_sample)

                        f_cache = ui.switch(
                            "Use Cache (f.use_cache)",
                            value=platform.form_state["f"]["use_cache"],
                        )
                        platform._register_input("f.use_cache", f_cache)

                # Panel g: Token Decoding
                with ui.expansion(
                    "Component g: Token Decoding", icon="translate"
                ).classes("w-full bg-white shadow"):
                    with ui.row().classes("w-full gap-6 p-2"):
                        g_skip = ui.switch(
                            "Skip Special Tokens (g.skip_special_tokens)",
                            value=platform.form_state["g"]["skip_special_tokens"],
                        )
                        platform._register_input("g.skip_special_tokens", g_skip)

                        g_clean = ui.switch(
                            "Clean Up Tokenization Spaces (g.clean_up_tokenization_spaces)",
                            value=platform.form_state["g"][
                                "clean_up_tokenization_spaces"
                            ],
                        )
                        platform._register_input(
                            "g.clean_up_tokenization_spaces", g_clean
                        )

                        g_spaces = ui.switch(
                            "Spaces Between Special Tokens (g.spaces_between_special_tokens)",
                            value=platform.form_state["g"][
                                "spaces_between_special_tokens"
                            ],
                        )
                        platform._register_input(
                            "g.spaces_between_special_tokens", g_spaces
                        )

                # Panel h: Optional kwargs (Issue #12)
                # Allows passing arbitrary additional arguments to underlying Hugging Face methods.
                # Every key is verified against preset method whitelists (e.g. AutoModelForCausalLM).
                with ui.expansion(
                    "Component h: Optional Arguments (**kwargs)", icon="tune"
                ).classes("w-full bg-white shadow"):
                    ui.label(
                        "Forward extra kwargs verified against library whitelists."
                    ).classes("text-xs text-gray-500 p-2")
                    for stage in ["load_model", "load_tokenizer", "encode", "generate"]:
                        sub_panel = KwargsSubPanel(
                            stage=stage,
                            h_state=platform.form_state["h"],
                            field_widgets=platform.field_widgets,
                            on_change_callback=platform._refresh_console_soon,
                        )
                        platform.kwargs_panels[stage] = sub_panel
                        sub_panel.build_ui()

        # ====================================================================
        # RIGHT COLUMN (5/12 width): Sticky live validation console & JSON preview
        # ====================================================================
        # Stays pinned at the top as the user scrolls through the form on the left.
        # Shows real-time error messages, disables saving on violations, and renders the JSON blob.
        with ui.column().classes("w-5/12 sticky top-4"):
            platform.console = build_console(
                container=ui.column().classes("w-full"),
                field_widgets=platform.field_widgets,
                build_config=platform.build_config_record,
                validate=validate_full_config,
                save=platform.save_to_db,
                reset=platform.reset_to_defaults,
            )

    # Initial visibility check
    platform._update_visibility()
    return platform


@ui.page("/")
def index_page() -> None:
    """Index page route."""
    create_app()


def main() -> None:
    """CLI entry point: start NiceGUI web server."""
    ui.run(
        title="LLM Fact-Checking Platform",
        port=8080,
        reload=True,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
