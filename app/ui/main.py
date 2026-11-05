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
    insert_config,
)
from app.ui.kwargs_panel import validate_kwargs_section
from app.validation import validate_cross_fields

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


def validate_full_config(cfg: ConfigRecord) -> list[dict[str, str]]:
    issues = list(validate_cross_fields(cfg))
    kwargs_issues = validate_kwargs_section(cfg.h)
    issues.extend(kwargs_issues)
    return issues


class PlatformUI:
    # default model is Mistral
    def __init__(self) -> None:
        # init_db()

        self.current_model_id = MISTRAL_ID
        self.current_task_type = "fine_tune"
        self.create_by_val = ""
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
        payload["created_by"] = self.create_by_val or None
        payload["notes"] = self.notes_val or None

        # clean up whitespaces
        c = payload.get("c", {})
        if isinstance(c.get("label_values"), str):
            c["label_values"] = [
                item.strip() for item in c["label_values"].split(",") if item.strip()
            ]
        # Hugging Face needs a string array
        d = payload.get("d", {})
        lora = d.get("lora", {})
        if isinstance(lora.get("target_modules"), str):
            lora["target_modules"] = [
                item.strip()
                for item in lora["target_modules"].split(",")
                if item.strip()
            ]

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

            self.current_model_id = new_state.ge("model_id", self.current_model_id)
            self.current_task_type = new_state.get("task_type", self.current_task_type)
            self.create_by_val = new_state.get("created_by") or ""
            self.notes_val = new_state.get("notes") or ""

            # puts the actual values inside the widgets
            if "model_id" in self.field_widgets:
                self.field_widgets["model_id"].value = self.current_model_id
            if "task_type" in self.field_widgets:
                self.field_widgets["task_type"].value = self.current_task_type
            if "created_by" in self.field_widgets:
                self.field_widgets["created_by"].value = self.create_by_val
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
        saved_by = self.create_by_val
        saved_notes = self.notes_val
        self.current_model_id = new_model
        defaults = defaults_for(self.current_model_id, self.current_task_type)
        self.load_state(defaults)
        # restore the metadata
        self.create_by_val = saved_by
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
        self.create_by_val = saved_by
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
        cfg.created_by = self.create_by_val or None
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


def create_app() -> PlatformUI:
    platform = PlatformUI()

    ########################### Main #########################################

    # Initial visibility check
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
