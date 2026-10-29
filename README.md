# 4990_llm-config-platform

**Group 15 · University of Windsor 4990B**
**Project**: Integration of Large Language Models into Autonomous Fact-Checking Pipelines — Infrastructure and Technical Implementation
**Team**: Daniel Laurin, Kerui Li, Simbarashe Mamvura, Tiziana Nikundana (mentored by the MAC team; supervised by Dr. Aznam Yacoub)
**Duration**: 2026-10-25 → 2026-12-05

**Checkpoints**: M0.5 `10/31` · M1.0 `11/7` · M1.5 `11/14` · M2.0 `11/21` · M2.5 `11/28` · M3.0 `12/5` (all Saturdays; 0.5 / 1.5 / 2.5 are online progress checks, 1.0 / 2.0 / 3.0 are formal deliverables)

---

## Reading order

| Order | File | Why read it first |
| --- | --- | --- |
| 1 | `doc/Project Milestones - 4990.docx` | The acceptance criteria as written. All functional requirements for the three delivery points live here |
| 2 | `planning/TaskList.md` | Breaks the acceptance criteria into checkable tasks across the six checkpoints |
| 3 | `specs/parameters.md` | Every parameter field the platform exposes: type, default, validation rules |
| 4 | `specs/model-reference.md` | Chat templates, special tokens and reference-script defects for both models |
| 5 | `reference-scripts/*.py` | Reference training / inference scripts supplied by the PhD student (line-by-line comments) |
| 6 | `design/Spec_and_roadmap.md` | Technical approach and sprint plan (currently the NiceGUI route) |

> Files under `doc/` are **not in this repository**; they are shared inside the team via OneDrive only (see the next section).

---

## Repository scope

This repository is **public**, so it contains only assets that can be published:

| Content | In repo | Reason |
| --- | --- | --- |
| `app/` (platform code), `README.md`, `.gitignore` | ✅ | The platform implementation itself |
| `specs/` (parameter spec, model facts) | ✅ | Compiled from public HF Model Cards; nothing sensitive |
| `planning/` (task list) | ✅ | Internal team planning; no student IDs or emails |
| `design/` (technical approach and sprint plan) | ✅ | Team-authored plan |
| `doc/` | ❌ | Contains **student IDs**, the supervisor's mass **email**, and **unpublished** research-group briefing PDFs |
| `reference-scripts/` | ❌ | The PhD student's research code, not cleared for publication |
| `data/`, model weights, run artefacts | ❌ | Large or regenerable; see `.gitignore` |

> If `doc/` and `reference-scripts/` should be version-controlled, the right move is to **make the repository private first**, then remove the two corresponding lines from `.gitignore`. A public repository cannot be truly un-published once leaked: it may already be cached or indexed.

---

## Repository language policy

Everything committed to this repository is **written in English**. No Chinese (or other CJK) characters are permitted in tracked files, including comments, docs and filenames. `.github/workflows/language-policy.yml` fails the build if a CJK character appears in a tracked file; the only exception is the DeepSeek special-token notation (`<｜end▁of▁sentence｜>`), which uses fullwidth vertical bars and is therefore not treated as CJK.

---

## Directory structure

```plaintext
4990B/
├── README.md                    ← this file
├── TODO.md                      milestone checklist stub
├── doc/                         source documents (from supervisor / research team; not tracked)
│   ├── Project Milestones - 4990.docx      acceptance criteria (internally word/document2.xml)
│   ├── Automated Fact-Checking with LLMs.pdf  research-side briefing
│   ├── Group 15 project description.docx/.pdf project brief
│   ├── Email.md                 supervisor's mass email (roles, red lines, Fir rules)
│   └── other/                   course documents unrelated to the project (archived)
├── planning/                    planning
│   └── TaskList.md              task list
├── specs/                       specifications (single source of truth)
│   ├── parameters.md            parameter spec: fields / types / required / defaults / 10 validation rules
│   └── model-reference.md       Model Card facts, chat-template rendering, script defects
├── design/                      technical approach
│   └── Spec_and_roadmap.md      technical spec and sprint plan
├── reference-scripts/           reference scripts from the PhD student (read-only; do not edit)
│   ├── fineTuningMistral_en.py
│   └── inferenceMistral_en.py
├── .github/workflows/           CI (language policy)
└── app/                         ← to be created. Platform implementation goes here
```

The expected structure of `app/` is described in section 2 of `design/Spec_and_roadmap.md`.

---

## Current status

- **No code has landed yet.** `app/` existed on 10/28 (`main.py`, `schemas.py`, `store.py`, `registry.py`, `static/index.html`, `data/platform.db`) and was then deleted without preserving its contents. If the OneDrive recycle bin can recover it, reuse it rather than rewriting.
- **The Git repository is initialised** and mirrors to GitHub at `https://github.com/Bamboorat233/4990_llm-config-platform`.
- **Open blockers**: the six sample output files, the sample execution script, and the dataset train/eval/test split are all still awaited from the PhD student (blocking M1.5 and M2.5 respectively; neither blocks M1).
- Local environment: Python 3.13.3, `pydantic 2.12.5` installed (FastAPI `0.136.1` is also available).

## Decision pending

**Two mutually exclusive platform approaches exist**; one must be chosen before implementation starts:

| Approach | Source | Characteristics |
| --- | --- | --- |
| **FastAPI + static HTML** | M1 handover document (removed from the tree; recoverable from git history) | Frontend/backend separated; a JSON API in M2.0 follows naturally; continues the shape of the deleted prototype |
| **NiceGUI (single-process reactive)** | `design/Spec_and_roadmap.md` | Pure-Python two-way binding; form linkage and error highlighting need less code; no separate frontend |

Open items and blockers are tracked in `planning/TaskList.md` and `TODO.md`.

## Collaboration constraints (from the supervisor's email and the project brief)

- **No web services, cron jobs, daemons or long-running processes on Fir.** The platform runs locally / on campus only; all synchronisation with Fir is asynchronous push-pull. Written supervisor authorisation is required before deployment.
- **AI-generated code must never be copy-pasted.** Every line must be understood and pass peer review before it is merged.
- Local / Colab use is for debugging only; never fine-tune on the full dataset on campus machines.
