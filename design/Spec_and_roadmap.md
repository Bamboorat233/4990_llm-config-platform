## **I. Technical Specification**

### **1. System Architecture and Data Flow**

Drop the traditional three-tier chain of "frontend sends HTTP request -> backend routing -> database" in favor of a **single-process reactive architecture**:

```plaintext
[NiceGUI UI components (Quasar/Vue)]
       ↕ (two-way data binding bind_value / event callbacks on_change)
[Session State (in-memory Python dict)]
       ↕ (real-time conversion / validation)
[Pydantic v2 Schema (ConfigRecord + 10 cross-field rules)]
       ↓ (serialized after passing validation)
[SQLite (store.py: single-table JSON persistence + SHA256 hash deduplication)]
```

### **2. Directory Structure and Module Responsibilities**

```plaintext
4990B/
├── app/
│   ├── __init__.py
│   ├── main.py          # NiceGUI entry point: page layout, two-way binding, linked callbacks, error highlighting
│   ├── registry.py      # Model fact registry: ModelMeta dataclass, static Mistral/DeepSeek parameters and default-config generation
│   ├── schemas.py       # Core data layer: Pydantic models for components a–h/p, ConfigRecord, fingerprint computation and the 10 validation rules
│   └── store.py         # Storage layer: SQLite wrapper (initialization, create/delete/query, lookup by hash, config cloning)
├── data/
│   └── platform.db      # SQLite database auto-generated at runtime (listed in .gitignore)
├── requirements.txt     # Only needs nicegui>=2.0, pydantic>=2.10
└── README.md
```

### **3. Core Module Specifications**

#### **3.1 app/registry.py (Model Registry and Defaults Engine)**

* **Data structure**: Use `@dataclass(frozen=True)` to define `ModelMeta`, strictly recording the vocabulary, special token IDs, context limit, and recommended sampling parameters for Mistral-Small-24B-Instruct-2501 and DeepSeek-R1-Distill-Qwen-32B.
* **Core functions**:
  * `get_model(model_id: str) -> ModelMeta`
  * `defaults_for(model_id: str, task_type: Literal["fine_tune", "inference"]) -> dict`: called whenever the user switches model or task type in the UI; returns the complete a–h and p dictionary. For example:
    * Mistral + fine_tune: `b.pad_token="<pad>"`, `b.pad_token_id=11`, `e.add_generation_prompt=False`, `f.temperature=0.15`.
    * DeepSeek + inference: `b.pad_token="<｜end▁of▁sentence｜>"`, `b.pad_token_id=151643`, `e.add_generation_prompt=True`, `f.temperature=0.6`, `f.max_new_tokens=1024`, `f.forced_prefix="\n\n"`.

#### **3.2 app/schemas.py (Pydantic Models and Cross-Field Validation)**

* **Component models**: Defined strictly per sections 3–11 of `specs/parameters.md`: `ModelSection` (a), `TokenizerSection` (b), `DataSection` (c), `OptimizerSection` (d), `EncodingSection` (e), `GenerationSection` (f), `DecodingSection` (g), `PromptSection` (p), and the top-level `ConfigRecord`.
* **Fingerprint algorithm `config_fingerprint(cfg: ConfigRecord) -> str`**:
  * When exporting to a dict, exclude `config_id`, `config_hash`, `run_status`, `created_at`, `created_by`, `notes`.
  * Serialize with `json.dumps(data, sort_keys=True)`, then compute sha256.
* **Cross-field validator `validate_cross_fields(cfg: ConfigRecord) -> list[ValidationIssue]`**:
  * Returns a structured list: `[{"field": "b.pad_token", "level": "error" | "warning", "msg": "..."}]`, which NiceGUI can use directly to mark the corresponding input fields red.
  * **Full coverage of all 10 rules**:
    1. `(e.max_length or 0) + f.max_new_tokens <= effective_max_seq_len` (MS: 32768; DS: 32768/16384).
    2. When `task_type == "inference"` and `f.batch_size > 1`, enforce `b.padding_side == "left"`.
    3. `b.pad_token` whitelist check (MS allows only `<pad>` or `</s>`; DS allows only `<｜end▁of▁sentence｜>`; reject `<myPadToken>`).
    4. For the MS model, enforce `b.fix_mistral_regex == True`.
    5. `f.temperature == 0` requires `f.do_sample == False`; `f.temperature > 0` requires `f.do_sample == True`.
    6. When `task_type == "fine_tune"` and `d.sft.load_best_model_at_end == True`, require `eval_strategy == save_strategy` and a non-empty `c.eval_data_file`.
    7. Consistency check of `p.assistant_prompt` and `p.verdict_parse_marker` between fine-tuning and downstream inference.
    8. Mutual-exclusion check of `a.tp_plan` and `a.device_map` (they cannot both be non-empty).
    9. For the MS model, return an error if `p.system_prompt` is empty (prevents injection of the default template containing the current date).
    10. For the DS model, return a warning if `p.system_prompt` is non-empty (notes that it will be concatenated right after BOS with no separator).

#### **3.3 app/store.py (SQLite Persistence Layer)**

* **Table schema**:

  ```sql
  CREATE TABLE IF NOT EXISTS configs (
      config_id TEXT PRIMARY KEY,
      config_hash TEXT UNIQUE NOT NULL,
      model_id TEXT NOT NULL,
      task_type TEXT NOT NULL,
      run_status TEXT NOT NULL DEFAULT 'pending',
      created_at TEXT NOT NULL,
      created_by TEXT,
      notes TEXT,
      payload_json TEXT NOT NULL
  );
  ```

* **Interface signatures**:
  * `init_db() -> None`
  * `find_by_hash(config_hash: str) -> str | None` (returns the existing `config_id` if found)
  * `insert_config(cfg: ConfigRecord) -> str`
  * `list_configs() -> list[dict]` (sorted by `created_at DESC`, for display in the left drawer list)
  * `get_config(config_id: str) -> ConfigRecord | None`
  * `duplicate_config(config_id: str) -> ConfigRecord` (generates a new UUID and current timestamp, for further modification)

#### **3.4 app/main.py (NiceGUI Interface and Interaction Spec)**

* **Overall layout**:
  * **Header**: project title + environment status indicator + sidebar collapse button.
  * **Left Drawer**: list of saved configs (card feed showing abbreviated `model_id`, `task_type` badge, first 8 characters of `config_id`, creation time); supports click-to-load and one-click "Duplicate".
  * **Main Split (main workspace, left/right columns at 7:5)**:
    * **Left column (form configuration area)**:
      1. Top-level control card: `model_id` (`ui.select`), `task_type` (`ui.toggle`), `created_by`, `notes`.
      2. Prompt configuration card (p): `system_prompt` (`ui.textarea`), `user_prompt_template`, `assistant_prompt`.
      3. Accordion panel group (`ui.expansion`): holds components a through h in order.
    * **Right column (live preview and validation console, sticky)**:
      1. Validation status panel: shows pass status in real time, or lists currently triggered errors (red) and warnings (yellow) with their exact field paths (e.g. `b.pad_token`).
      2. Live JSON preview: uses `ui.code(language='json')` to render the full JSON serialization of the current form in real time.
      3. Action button group: Validate (manually trigger a deep check), Save to DB (automatically disabled when errors exist), Reset to Defaults.
* **Key UI linkage strategies**:
  * **Component d (Optimizer)**: all inputs set to `.props('readonly bg-grey-2')`; the outer container is bound with `.bind_visibility_from(state, 'task_type', value='fine_tune')` (automatically hidden, or collapsed and marked not applicable, when inference is selected).
  * **Component c (Data)**: `train_data_file` and `eval_data_file` are visible/required only under fine_tune; `test_data_file` and `evidence_config` are visible/required only under inference.
  * **Component h (\*\*kwargs)**: provides sub-panels with dynamically addable/removable key-value rows for `load_model`, `load_tokenizer`, `encode`, and `generate`; key names are validated on the backend via `inspect.signature` or a preset whitelist.

## **II. Technical Approach and Step-by-Step Implementation Plan (Implementation Roadmap)**

Given that today is **October 29**, with only 2 days until the **M0.5 check on October 31 (Saturday)** and 9 days until the **formal M1.0 delivery on November 7**, the technical approach is split into **4 progressive sprints**:

### **Sprint 1: Engineering Foundation and Data Core (complete on October 29)**

**Goal**: Write no UI code yet; first get the data models, default-value dictionaries, and all 10 validation rules running 100% from the command line.

1. **Initialize the repository**:
   * Run `git init` in the 4990B root directory and create `.gitignore` (excluding `data/*.db`, `__pycache__/`, `.venv/`).
   * Install dependencies: `pip install nicegui pydantic`.
2. **Implement app/registry.py**:
   * Write the `MODELS` dictionary and the `defaults_for()` function.
3. **Implement app/schemas.py**:
   * Write the a–h, p, and `ConfigRecord` Pydantic models.
   * Write `validate_cross_fields()` covering all 10 validation rules.
4. **Core smoke test**:
   * Test with a single command: `python -c "from app.registry import defaults_for; from app.schemas import ConfigRecord; print(ConfigRecord(**defaults_for('mistralai/Mistral-Small-24B-Instruct-2501', 'fine_tune')).model_dump_json(indent=2))"`, and confirm the output JSON matches section 13 of `specs/parameters.md` exactly.

### **Sprint 2: NiceGUI Reactive Form and M0.5 Delivery (complete on October 30)**

**Goal**: Build the `app/main.py` interface and implement all interactive demo features required for M0.5.

1. **Set up the state container**:
   * Maintain a reactive config dictionary `current_state` in `main.py`, initially loaded with the Mistral + fine_tune defaults.
2. **Write the form rendering functions**:
   * Write the NiceGUI components in order: top-level selectors -> Prompt section -> a–h accordion panels.
   * Bind `on_change` events on `model_id` and `task_type`: on any switch, call `defaults_for()` to refresh the state, then call `refresh_ui()` to update the interface and the JSON preview.
3. **Wire up live validation and field highlighting**:
   * On every form change, build a `ConfigRecord` and run `validate_cross_fields()`.
   * Store all widget references in a dictionary, e.g. `field_widgets["b.pad_token"] = input_widget`; when a field has an error, call `widget.props('error error-message="specific reason"')` directly to show a precise, field-level red error border.
4. **M0.5 rehearsal (October 31)**:
   * Start the service with `python -m app.main` (runs at http://localhost:8080 by default).
   * Click through the M0.5 demo checklist in `planning/TaskList.md` item by item: switch models -> switch between fine-tuning/inference (show component d read-only and hidden, component e switching automatically) -> deliberately trigger a validation error (e.g. clear Mistral's `system_prompt`) -> show the full JSON generated live on the right.

### **Sprint 3: SQLite Persistence, Deduplication, and Config Management (November 1 – November 3)**

**Goal**: Implement `app/store.py` and connect it to the NiceGUI left drawer, meeting the M1.0 persistence requirements.

1. **Implement app/store.py**:
   * Use Python's built-in `sqlite3` to implement table creation, insertion, lookup by `config_hash`, list queries, and cloning logic.
2. **Connect the "Save" and "Deduplicate" workflow**:
   * When Save to DB is clicked, first compute `config_hash` and call `find_by_hash()`.
   * If a duplicate hash exists, show `ui.notify("An identical experiment config already exists (ID: xxx)", type="warning")` and block the duplicate insert.
   * If the config is new, write it to `data/platform.db`, show a success notification, and refresh the left config list.
3. **Connect "Load" and "Duplicate"**:
   * Clicking a config in the left list: deserialize its `payload_json` and populate the main form with it.
   * Clicking Duplicate: call `store.duplicate_config()` to generate a new `config_id` and load it into the form, so researchers can tweak a single hyperparameter (e.g. change temperature from 0.15 to 0.9) and save it as a new experiment.

### **Sprint 4: Acceptance Checklist Self-Testing and Documentation Wrap-Up (November 4 – November 6)**

**Goal**: Check off each of the M1.0 acceptance items in `planning/TaskList.md`.

1. **Write the automated test script `test_m1_checklist.py`**:
   * Assert all the M1.0 acceptance edge cases directly in code (e.g. rejection of `<myPadToken>`, rejection of the 30000+5000 length overflow, rejection of temperature=0 + do_sample=True, persistence verification after a database restart, etc.).
2. **Complete README.md**:
   * Document the startup steps, the module architecture diagram, and explanations of the 6 fixed defects from the original scripts.

We can go straight into **Sprint 1** now. Would you like me to write the full code for `app/registry.py` and `app/schemas.py` now?
