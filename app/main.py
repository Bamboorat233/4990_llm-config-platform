# from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

from nicegui import ui

from app.registry import DEEPSEEK_ID, MISTRAL_ID, defaults_for
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

# from app.ui.kwargs_panel import validate_kwargs_section
# from app.validation import validate_cross_fields

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
    "wihtout_evidence": "Without Evidence",
    "random_evidence": "Random Evidence",
    "question_only": "Question Only",
    "answer_only": "Answer Only",
    "all_question_last_answer": "All Questions + Last Answer",
    "all_questions_last_answer": "All Questions + All But Last Answer",
}


# def validate_full_config(cfg: ConfigRecord) -> list[dict[str, str]]:
#     issues = list(validate_cross_fields(cfg))
#     kwargs_issues = validate_kwargs_section(cfg.h)
#     issues.extend(kwargs_issues)
#     return issues


class PlatformUI:
    # default model is Mistral
    def __init__(self) -> None:
        init_db()

        self.current_model_id = MISTRAL_ID
        self.current_task_type = "fine_tune"
        self.created_by_val = ""
        self.notes_val = ""

        # loading defaults_for func from registry.py
        self.form_state = defaults_for(self.current_model_id, self.current_task_type)
        if "h" not in self.form_state or not isinstance(self.form_state["h"], dict):
            self.form_state["h"] = {}

        self.field_widgets: dict[str, Any] = {}
        # self.kwargs_panels: dict[str, KwargsSubPanel] = {}

        self.is_batch_updating = False

        self.optimizer_expansion: Any = None
        self.data_fine_tune_container: Any = None
        self.data_inference_container: Any = None
        self.saved_configs_container: Any = None
        self.console: Any = None

    # makes the JSON format
    def _set_path_value(self, path: str, value: Any) -> None:
        parts = path.split(".")
        target = self.form_state
        for segment in parts[:-1]:
            if segment not in target or not isinstance(target[segment], dict):
                target[segment] = {}
            target = target[segment]
        target[parts[-1]] = value

    # read the dictionary value | ex: tempature: 0.6
    def _get_path_value(self, path: str, default: Any = None) -> Any:
        target = self.form_state
        for segment in path.split("."):
            if isinstance(target, dict) and segment in target:
                target = target[segment]
            else:
                return default
            return target

    ########################## FUNCIONALITY ###############################

    # NiceUI input, validation check, live refresh func
    def _register_input(
        self,
        path: str,
        widget: Any,
        cast_func: Callable[[Any]] | None = None,
    ) -> Any:
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
            self._set_path_value(path, val)

            # LIVE UPDATE
            if self.console is not None:
                self.console.refresh()

        widget.on_value_change(on_change)
        return widget

    def build_config_record(self) -> ConfigRecord:

        payload = copy.deepcopy(self.form_state)
        payload["model_id"] = self.current_model_id
        payload["task_type"] = self.current_task_type
        payload["created_by"] = self.created_by_val or None
        payload["notes"] = self.notes_val or None

        # clean up whitespaces
        c = payload.get("c", {})
        if isinstance(c.get("label_values"), str):
            c["label_values"] = [
                item.strip() for item in c["label_values"].split(",") if item.strip()
            ]
        # Hugging Face needs a string array
        # d = payload.get("d", {})
        # lora = d.get("lora", {})
        # if isinstance(lora.get("target_modules"), str):
        #     lora["target_modules"] = [
        #         item.strip()
        #         for item in lora["target_modules"].split(",")
        #         if item.strip()
        #     ]

        # sets top_k to 0 when deepseek toggled
        f = payload.get("f", {})
        if f.get("top_k") == "" or f.get("top_k") == 0:  # noqa: SIM102
            if self.current_model_id == DEEPSEEK_ID:
                f["top_k"] = None

        return ConfigRecord(**payload)

    # func for making profiles
    def load_state(self, new_state: dict[str, Any]) -> None:

        self.is_batch_updating = True
        try:
            # makes a save state
            self.form_state = copy.deepcopy(new_state)
            if "h" not in self.form_state or not isinstance(self.form_state["h"], dict):
                self.form_state["h"] = {}

            self.current_model_id = new_state.get("model_id", self.current_model_id)
            self.current_task_type = new_state.get("task_type", self.current_task_type)
            self.created_by_val = new_state.get("created_by") or ""
            self.notes_val = new_state.get("notes") or ""

            # puts the actual values inside the widgets
            if "model_id" in self.field_widgets:
                self.field_widgets["model_id"].value = self.current_model_id
            if "task_type" in self.field_widgets:
                self.field_widgets["task_type"].value = self.current_task_type
            if "created_by" in self.field_widgets:
                self.field_widgets["created_by"].value = self.created_by_val
            if "notes" in self.field_widgets:
                self.field_widgets["notes"].value = self.notes_val

            # loads the profile
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
                    # make lists comma seperated
                    if isinstance(val, list):
                        widget.value = ", ".join(str(x) for x in val)
                    else:
                        widget.value = val

            # refresh hyperparameters and load user arguments
            for stage, panel in self.kwargs_panels.items():
                panel.clear()
                existing_kwargs = self.form_state.get("h", {}).get(stage, {})
                for k, v in existing_kwargs.items():
                    val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                    panel.add_row(initial_key=k, initial_val=val_str)

            # hides the menu that you open
            self._update_visibility()
        finally:
            self.is_batch_updating = False

        if self.console is not None:
            self.console.refresh()

    def _update_visibility(self) -> None:
        is_fine_tune = self.current_task_type == "fine_tune"
        if self.optimizer_expansion is not None:
            self.optimizer_expansion.set_visibility(is_fine_tune)
        if self.data_fine_tune_container is not None:
            self.data_fine_tune_container.set_visibility(is_fine_tune)

    def on_model_changed(self, e: Any) -> None:
        new_model = e.Value
        if new_model == self.current_model_id:
            return

        # this was an earlier bug this is the fix
        # preserve user metadata before load_state
        saved_by = self.created_by_val
        saved_notes = self.notes_val
        self.current_model_id = new_model
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        # restore the metadata
        self.created_by_val = saved_by
        self.notes_val = saved_notes
        if "created_by" in self.field_widgets:
            self.field_widgets["created_by"].value = saved_by
        if "notes" in self.field_widgets:
            self.field_widgets["notes"].value = saved_notes
        # import from NiceUi
        ui.notify(f"Switched model to {new_model}", type="info")

    def on_task_type_changed(self, e: Any) -> None:
        new_task = e.value
        if new_task == self.current_task_type:
            return

        # preserve user metadata before load_state overwrites them
        saved_by = self.created_by_val
        saved_notes = self.notes_val
        self.current_task_type = new_task
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        self.created_by_val = saved_by
        self.notes_val = saved_notes
        if "created_by" in self.field_widgets:
            self.field_widgets["created_by"].value = saved_by
        if "notes" in self.field_widgets:
            self.field_widgets["notes"].value = saved_notes
        ui.notify(f"Switched task type to {new_task}", type="info")

    def save_to_db(self, cfg: ConfigRecord) -> None:
        fingerprint = config_fingerprint(cfg)
        existing_id = find_by_hash(fingerprint)
        if existing_id is not None:
            ui.notify(
                f"An identical expermient config already exsists (ID: {existing_id[:8]})",
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
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        ui.notify("Form reset to registry defaults.", type="info")

    def load_saved_config(self, config_id: str) -> None:
        cfg = get_config(config_id)
        if cfg is None:
            ui.notify(f"Config {config_id} not found.", type="negative")
            return
        self.load_state(cfg.model_dump())
        ui.notify(f"Loaded config {config_id[:8]}", type="positive")

    def duplicate_saved_config(self, config_id: str) -> None:
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

    ######################### DESIGN ######################################

    def refresh_saved_configs_list(self) -> None:
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
    platform = PlatformUI()

    ########################### Style ########################################

    # Page styling
    ui.colors(primary="#1976D2", secondary="#26A69A", accent="#9C27B0")

    # Header
    with (
        ui.header().classes(
            "items-center justify-between px-4 py-2 bg-slate-900 text-white"
        ),
        ui.row().classes("items-center gap-3"),
    ):
        drawer_btn = ui.button(icon="menu").props("flat round color=white")
        ui.label("LLM Fact-Checking Config Platform").classes(
            "text-lg font-bold tracking-wide"
        )
        # with ui.row().classes("items-center gap-2"):
        #     ui.badge("Fir HPC Ready", color="positive").classes("text-xs px-2 py-1")
        #     ui.label("Milestone 0.5 & 1.0").classes("text-xs text-slate-400")

    # hamburger button
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

    ############################ Body ########################################

    with ui.row().classes("w-full p-4 gap-6 no-wrap items-start"):
        with ui.column().classes("w-7/12 gap-4"):
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

            ######################## PROMPT STYLING ##################################

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

            #################### COMPONENTS ################################################

            # PART B
            with ui.column().classes("w-full gap-2"):
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

                # PART B
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

                    # PANEL C
                    # WREWRITE?
                with ui.expansion(
                    "Component c: Loading the Data", icon="dataset"
                ).classes("w-full bg-white shadow"):
                    with ui.column().classes("w-full p-2 gap-4"):
                        # fine-tuning data paths container
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

                        # Inference data paths and evidence container
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

                        # common data fields
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
                                "c.evidence_answers_used",
                                c_answers_used,
                                cast_func=int,
                            )

                            c_max_seq = ui.number(
                                label="Max Seq Length (c.max_seq_length)",
                                value=platform.form_state["c"]["max_seq_length"]
                                or 4096,
                            )
                            platform._register_input(
                                "c.max_seq_length", c_max_seq, cast_func=int
                            )

        ########################### Main #########################################

        # Initial visibility check

        with ui.column().classes("w-5/12 sticky top-4"):
            platform.console = build_console(
                container=ui.column().classes("w-full"),
                field_widgets=platform.field_widgets,
                build_config=platform.build_config_record,
                validate=lambda cfg: [],
                save=platform.save_to_db,
                reset=platform.reset_to_defaults,
            )
    platform._update_visibility()
    return platform


@ui.page("/")
def index_page() -> None:
    """Index page route."""
    create_app()


def main() -> None:
    ui.run(
        title="LLM Fact-Checking Platform",
        port=8080,
        reload=True,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
