# Group 15 Project Task List

**Project**: Integration of Large Language Models into Autonomous Fact-Checking Pipelines — Infrastructure and Technical Implementation
**Team**: Daniel Laurin, Kerui Li, Simbarashe Mamvura, Tiziana Nikundana
**Supervisor**: Dr. Aznam Yacoub (Collaborator: Gagandeep Kaur / MAC team)
**Duration**: Oct 25 – Dec 5, 2026

> **How this list is organised**: it follows the three sections of the milestone document, one section per checkpoint, and every item maps back to the wording of the requirement it satisfies. Items that the milestone document does not state but the research design requires are tagged **[research-critical]** (e.g. `evidence_config`, empty-response rate) so their priority can be explained at the progress checks.
>
> **Cadence**: M0.5 / M1.5 / M2.5 are online progress checks (show work achieved so far); M1.0 / M2.0 / M3.0 are formal deliverable deadlines. All six checkpoints fall on a Saturday.

---

## Milestone map

| Checkpoint | Date | Type | Milestone-doc section | Must be demonstrable that day | Lead workstream |
| --- | --- | --- | --- | --- | --- |
| M0.5 | Oct 31 (Sat) | Progress check | `Milestone 0.5 & Milestone 1.0`, first half | Environment ready + clickable parameter form (no DB required) | A |
| M1.0 | Nov 7 (Sat) | Deliverable | `Milestone 0.5 & Milestone 1.0` | Complete UI + configs persisted to the DB as JSON | A + B |
| M1.5 | Nov 14 (Sat) | Progress check | `Milestone 1.5 & Milestone 2.0`, first half | General parser for all six scenarios + UI display | C |
| M2.0 | Nov 21 (Sat) | Deliverable | `Milestone 1.5 & Milestone 2.0` | Successful-inference parser + metrics + persistence | C + B |
| M2.5 | Nov 28 (Sat) | Progress check | `Milestone 2.5 & Milestone 3.0`, first half | Generator emits a runnable script + section-by-section parity passes | D |
| M3.0 | Dec 5 (Sat) | Deliverable | `Milestone 2.5 & Milestone 3.0` | End-to-end integration + wrap-up | D + all |

> **Why M0.5 needs a prototype**: the milestone document describes M0.5 and M1.0 in the same section, and defines a progress check as an opportunity to surface difficulties and *show the work achieved so far*. M0.5 therefore needs the **first half of the M1.0 requirement**, not only account applications and environment setup.

---

## M0.5 — Oct 31 (Sat) · Online progress check

**Goal**: environment and tooling ready; parameter-collection form clickable (not yet wired to the database).

### 0.1 Environment & tooling
- [ ] Apply for and activate CCDB accounts; obtain Fir (AllianceCan) HPC access
- [ ] Set up VS Code remote development (SSH to Fir)
- [ ] Initialise the Git repository (`.gitignore` excludes model weights, output dirs, `__pycache__`, local DB files) and the branch/review workflow
- [ ] Read the reference scripts `reference-scripts/fineTuningMistral_en.py` and `reference-scripts/inferenceMistral_en.py`; understand the full training/inference flow
- [ ] Review the QuanTemp dataset format (numerical claim + 3–5 QA evidence pairs; labels True/False/Conflicting)
- [ ] Confirm both models' chat templates and required parameters (findings already archived in `specs/model-reference.md` and `specs/parameters.md`)

### 0.2 Parameter-collection form prototype (first half of M1.0)
- [ ] Lock the stack (Python + FastAPI + Pydantic + SQLite; a static frontend page is sufficient)
- [ ] Pages 1–3 clickable: select model → select fine-tune / inference → enter the three prompts
- [ ] Parameter form skeleton rendered in groups a–h (fields, types and defaults taken from `specs/parameters.md`)
- [ ] "Preview" shows the full config JSON (held in frontend memory only at this stage, not persisted)

### 0.3 Escalate the blockers (as early as possible)
- [ ] ⚠️ Request the **six sample output files** from the PhD student (hard dependency of M1.5)
- [ ] ⚠️ Request the **sample execution script** from the PhD student (hard dependency of M2.5)
- [ ] Confirm the Fir authorisation process and the size of the first debug dataset

**Oct 31 demo checklist**: screenshots of the SSH session and the Git history + the form prototype driven live + the summary page of script and Model Card findings.

---

## M1.0 — Nov 7 (Sat) · Deliverable: parameter-collection UI + database

### 1.1 User interface (adapts to the task type)

- [ ] Page 1: select an LLM from a list (first batch: Mistral-Small-24B-Instruct-2501; leave room for DeepSeek-R1-Distill-Qwen-32B)
- [ ] Page 2: choose task type — **fine-tune / inference** (the remaining form switches accordingly)
- [ ] Page 3: the three prompts — system / user / assistant (model-dependent; for Mistral the system prompt must be **required**, otherwise the template injects a date-stamped default and the same config is no longer reproducible across days)
- [ ] Page 4: per-component parameters
  - [ ] a. Loading the model (dtype, tp_plan, local_files_only, attn_implementation)
  - [ ] b. Loading the tokenizer (padding_side, pad_token, fix_mistral_regex, model_max_length)
  - [ ] c. Loading the data (train / eval / test paths, label values, missing-answer placeholder)
    - [ ] **[research-critical]** evidence configuration `evidence_config`: full evidence / no evidence / random evidence / question only / answer only / all questions + last answer only / all questions + all but the last answer
  - [ ] d. Optimizer (LoRA + SFT; parameters fixed and displayed **read-only**: r=64, lora_alpha=16, dropout=0.1, lr=2e-4, …)
  - [ ] e. Token encoding (apply_chat_template, add_generation_prompt, return_tensors, padding, truncation, max_length)
  - [ ] f. Text generation (max_new_tokens, temperature, top_p, top_k, do_sample, batch_size)
  - [ ] g. Token decoding (skip_special_tokens, clean_up_tokenization_spaces)
  - [ ] h. Optional arguments `**kwargs` per component (keys must exist in the target library signature; unknown keys must fail at generation time rather than being silently ignored)
- [ ] Implement the validation rules (at least the 10 rules in `specs/parameters.md` §12: length constraint, pad_token validity, eval/save strategy consistency, identical assistant prefix between fine-tune and inference, …)
- [ ] Full preview before submission, with validation errors located to the offending field

### 1.2 Database

- [ ] Schema: model, task type, prompts, per-component parameters, created/updated timestamps, run status, config fingerprint (dedup)
- [ ] Split into a **config table + run table**: one config can have many runs (needed for re-scoring in M2.0 and comparison in M3.0)
- [ ] Persist everything collected by the UI in **JSON format** (field structure identical to the example in `specs/parameters.md` §13)
- [ ] Config read / list / duplicate (duplicate is the main entry point for "change one parameter and run again")
- [ ] Link the fine-tune config to the inference configs that consume its adapter (every inference record must be traceable to the adapter it used)

### 1.3 Acceptance

- [ ] **Nov 7 — submit Milestone 1.0**: working UI + database demo (live or recorded)

---

## M1.5 — Nov 14 (Sat) · Online progress check

**Goal**: a general parser for all six scenarios, displayed in the UI so nobody has to read the log files by hand.

- [ ] Obtain the six sample output files and record each log structure (key lines, error formats, timeout signatures)
- [ ] General parser: detect and classify all six scenarios
  - [ ] ① errant fine-tuning　② successful fine-tuning　③ terminated fine-tuning (e.g. timeout)
  - [ ] ④ errant inference　⑤ successful inference　⑥ terminated inference (e.g. timeout)
- [ ] Structured result per scenario: status, key error messages, progress, duration
- [ ] UI display: select / upload an output file → show the parsed result card
- [ ] Parser unit tests: at least one sample per scenario, plus hand-built edge cases (empty file, truncated mid-run, encoding errors)

**Nov 14 demo checklist**: all six sample files classified as the correct scenario + the UI display.

---

## M2.0 — Nov 21 (Sat) · Deliverable: successful-inference parsing + scoring

- [ ] Dedicated parser for "successful inference": extract the generated text (the reference script uses `rfind("### Response: Verdict: ")`; DeepSeek's template emits a `<think>` block first, so parsing must tolerate it — `startswith` would break)
- [ ] Output cleaning and normalisation
  - [ ] Count empty responses and Incomplete Responses separately
  - [ ] Normalise malformed labels (`Truelicting`, `Falselicting`, `Conf`, …) via an explicit whitelist mapping
  - [ ] Keep the raw text of anything outside the whitelist and tag it `unparsed` — never drop it silently
- [ ] Align cleaned verdicts with ground truth and compute **accuracy, F1, precision, recall** (three classes; document the macro / micro / weighted choice explicitly)
- [ ] **[research-critical]** report the **number and share of empty responses** alongside the four metrics (reducing empty responses to zero is a standalone goal in the research design and is otherwise unverifiable)
- [ ] Confusion matrix + per-class P/R/F1; support grouping by `evidence_config`
- [ ] Persist parsed results and scores to the database; visualise them in the UI (first step in replacing Excel logging)
- [ ] **Nov 21 — submit Milestone 2.0: parser + scoring demo**

---

## M2.5 — Nov 28 (Sat) · Online progress check

**Goal**: the generator produces **syntactically valid, parameter-consistent** training / inference scripts from a stored config.

- [ ] Finalise the generator design around the PhD student's sample execution script (templates + parameter injection, not string concatenation)
- [ ] Generate a Python training or inference script **automatically** from the M1.0 config
- [ ] Generated script includes: all imports, injected hyperparameters, data loading, LoRA/SFT setup (training), generation + verdict extraction (inference)
- [ ] Additionally generate `requirements.txt` (the milestone text requires "all the necessary libraries and dependencies")
- [ ] Validate against the reference script section by section: parameter parity (automated diff script) + syntax check (`ast.parse` / `py_compile`)
- [ ] Fix the known pitfalls (the generator must not copy them)
  - [ ] Operator-precedence bug: `"...Answer: " + doc[0] if doc else ...` (when `top_k_doc` is empty, even the Question is dropped)
  - [ ] `pad_token = "<myPadToken>"` is not a real token → Mistral: `<pad>` (id 11) or `</s>`; DeepSeek: keep its EOS
  - [ ] `fix_mistral_regex` must be passed to `from_pretrained()`; assigning it afterwards has no effect
  - [ ] Assistant prefix differs between the train and eval sets (the eval set is missing `Verdict: `)
  - [ ] All paths injected from config; no hard-coding
  - [ ] `seed` must be part of the config (absent in the reference script, which breaks reproducibility)

**Nov 28 demo checklist**: training script and inference script generated from the same config + syntax check passes + parameter parity diff is empty.

---

## M3.0 — Dec 5 (Sat) · Deliverable: end-to-end chain + wrap-up

- [ ] End-to-end integration: DB config → generated script → (remote) execution → output file → parser → UI display
- [ ] Multi-config comparison (starting with two configs): metrics side by side + highlighted differences
  - [ ] If this is confirmed out of scope for this term, agree it with Gagandeep **before M2.5** so the research side does not expect it
- [ ] Review and adjust the M1.0 implementation as needed (explicitly allowed by the milestone document)
- [ ] Technical documentation: environment setup, DB schema, generation logic, user manual
- [ ] Confirm every item on the known-pitfall list is fixed
- [ ] **Dec 5 — submit Milestone 3.0 + project wrap-up**

---

## Cross-cutting tasks (entire duration)

- [ ] Disciplined Git commits and code review (MAC team rule: zero copy-paste of unverified AI output; review before merge)
- [ ] Fir compliance: no persistent daemons / web services; asynchronous push-pull sync only; supervisor authorisation before deployment
- [ ] Keep the technical documentation current as milestones land (do not leave it all to the final week)
- [ ] Weekly team sync; reach out to the PhD student / supervisor anytime outside official checkpoints
- [ ] Update the dependency table below weekly; escalate any unresolved blocker at the team sync

## Dependencies & blockers

| Dependency | Provider | Blocks | Needed by | Status |
| --- | --- | --- | --- | --- |
| Fir (AllianceCan) HPC access | CCDB / supervisor authorisation | Remote-execution integration at M3.0 | Oct 31 | To confirm |
| Six sample output files | PhD student | M1.5, M2.0 | Nov 14 (target Oct 31) | To chase |
| Sample execution script | PhD student | M2.5, M3.0 | Nov 28 (target Nov 7) | To chase |
| Dataset train / eval / test split | PhD student | Path-config demo at M1.0 | Nov 7 | To confirm |

## Suggested allocation (4 people × 4 workstreams)

| Workstream | Milestones covered | Scope | Owner |
| --- | --- | --- | --- |
| A. Frontend / forms | M0.5, M1.0, M2.0 (display) | Pages, validation, result visualisation | TBD |
| B. Data layer / Pydantic | M1.0, M2.0 (persistence) | Schema, JSON storage, config fingerprint, config↔run linking | TBD |
| C. Log parsing / scoring | M1.5, M2.0 | Six-scenario parsing, cleaning, metrics | TBD |
| D. Script generation / integration | M2.5, M3.0 | Template generation, parity checks, end-to-end integration | TBD |

> Convention: the lead workstream for each milestone owns the demo on that date, and one other member cross-reviews; nothing merges to main without at least one review.

## Key risks

| Risk | Mitigation |
| --- | --- |
| Violating Fir allocation rules (persistent processes) → resource lockout | Rigorous local testing; supervisor authorisation before Fir deployment |
| AI-assisted debugging introduces vulnerabilities | Mandatory peer review; zero copy-paste of unverified AI output |
| Execution time may have a strict lower bound → thousands of runs infeasible | Optimise and cache a single run first; confirm expectations with the PhD student |
| Unpredictable LLM output types → scoring script hard to generalise | Whitelist + normalisation in the parser; keep feeding anomalous samples back |
| Sample files / sample script arrive late → M1.5 and M2.5 have no input | Chase weekly from M0.5; use self-generated small samples meanwhile and build the parser around an adapter layer, so a new format only needs a new adapter |
| The M1.5 and M2.5 checkpoints get skipped → problems only surface on the delivery date | Give each checkpoint its own demo checklist; escalate at the team sync if unmet |
