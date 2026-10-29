# Consolidated Parameter Specification

Single source of truth for every parameter exposed in the platform UI (Milestone 1.0), persisted to the database as JSON, and used by the generated execution script (Milestone 3.0).

Values are verified against the two reference scripts, both Hugging Face `tokenizer_config.json`, `config.json`, `generation_config.json`, and the published Model Cards (checked 2026-10-26, `transformers` 4.57.6). Chat-template rendering and the script-bug analysis live in [model-reference.md](model-reference.md).

**Legend**

| Column | Meaning |
|---|---|
| Type | `str` / `int` / `float` / `bool` / `list` / `null` |
| Required | 🔴 mandatory · 🟡 required in practice · ⚪ optional |
| Mistral / DeepSeek | recommended value for that model (`—` = not applicable) |

**Model identifiers used below**

- `MS` = `mistralai/Mistral-Small-24B-Instruct-2501`
- `DS` = `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`

---

## 1. Configuration record (top level)

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `config_id` | str | 🔴 | generated | UUID of the configuration record |
| `config_hash` | str | 🔴 | derived | Hash of all functional parameters → dedup / "same config already run" detection |
| `model_id` | str | 🔴 | — | `MS` or `DS`; drives every downstream default |
| `task_type` | str | 🔴 | — | `fine_tune` \| `inference` (Milestone 1.0 step 2) |
| `run_status` | str | 🔴 | `pending` | `pending` \| `running` \| `success` \| `error` \| `terminated` |
| `created_at` | str (ISO 8601) | 🔴 | now | |
| `created_by` | str | ⚪ | — | team member |
| `notes` | str | ⚪ | — | free text for the researcher |

---

## 2. Model identity (read-only metadata, shown next to the form)

| Field | MS | DS |
|---|---|---|
| `architectures` | `MistralForCausalLM` | `Qwen2ForCausalLM` |
| `num_parameters` | 24B (23,572,403,200) | 32.8B (32,763,876,352) |
| `num_hidden_layers` / `num_attention_heads` / `num_key_value_heads` | 40 / 32 / 8 | 64 / 40 / 8 |
| `hidden_size` / `intermediate_size` | 5120 / 32768 | 5120 / 27648 |
| `head_dim` | 128 | 128 |
| `vocab_size` | 131,072 | 152,064 |
| `max_position_embeddings` | 32,768 | 131,072 |
| `tokenizer.model_max_length` | 32,768 | **16,384** ⚠️ lower than the model limit |
| `effective_max_seq_len` (recommended) | 32,768 | 32,768 (card's serving value) |
| `torch_dtype` | bfloat16 | bfloat16 |
| `weights_size` | ~47.1 GB | ~65.5 GB (~61 GiB) |
| `min_gpu_memory_bf16` | ~55 GB (1×80GB, or TP≥2) | ~80 GB, or `--tp 2` |
| `bos_token` (`id`) | `<s>` (1) | `<｜begin▁of▁sentence｜>` (151646) |
| `eos_token` (`id`) | `</s>` (2) | `<｜end▁of▁sentence｜>` (151643) |
| `pad_token` (`id`) | not set; `<pad>` available (11) | `<｜end▁of▁sentence｜>` (151643) |
| `add_bos_token` | `True` | `True` |
| `allowed_roles` | `system`, `user`, `assistant` | `system`, `user`, `assistant`, `tool` |
| `system_prompt_policy` | **recommended** | **avoid** (raw concatenation after BOS) |
| `generation_config.temperature` | 0.15 | 0.6 |
| `generation_config.top_p` | — | 0.95 |

---

## 3. Component a — Loading the model

| Parameter | Type | Required | Default | MS | DS | Notes |
|---|---|---|---|---|---|---|
| `a.dtype` | str | 🔴 | `auto` | `auto` → bf16 | `auto` → bf16 | `auto` \| `bfloat16` \| `float16` \| `float32` |
| `a.local_files_only` | bool | 🔴 | `true` | `true` | `true` | Must be `true` on Fir compute nodes (no network) |
| `a.tp_plan` | str\|null | ⚪ | `auto` | `auto` | `auto` | Tensor parallelism; needs ≥2 GPUs |
| `a.device_map` | str\|null | ⚪ | `null` | `auto` on a single GPU | `auto` | Alternative to `tp_plan` |
| `a.attn_implementation` | str\|null | ⚪ | `null` | `sdpa` | `sdpa` | `eager` \| `sdpa` \| `flash_attention_2` |
| `a.quantization` | str\|null | ⚪ | `null` | `null` | `null` | `4bit` \| `8bit` via `BitsAndBytesConfig`; imported but unused in the reference script |
| `a.trust_remote_code` | bool | ⚪ | `false` | `false` | `false` | Not needed — both are native architectures |
| `a.model_id` | str | 🔴 | from §1 | — | — | Local path or hub id |

---

## 4. Component b — Loading the tokenizer

| Parameter | Type | Required | Default | MS | DS | Notes |
|---|---|---|---|---|---|---|
| `b.tokenizer_id` | str | ⚪ | `= model_id` | `= model_id` | `= model_id` | |
| `b.padding_side` | str | 🔴 | `left` | `left` | `left` | **`left` is required** for batched decoder-only generation |
| `b.pad_token` | str | 🔴 | model-specific | `<pad>` (id 11) | `<｜end▁of▁sentence｜>` | Fallback for MS: `</s>`. Never use an out-of-vocabulary literal |
| `b.pad_token_id` | int | 🔴 (derived) | resolved | 11 | 151643 | Show as read-only |
| `b.fix_mistral_regex` | bool | 🟡 | `true` | `true` | — | **Must be passed to `from_pretrained()`**; setting it afterwards has no effect |
| `b.use_fast` | bool | ⚪ | `true` | `true` | `true` | |
| `b.model_max_length` | int | ⚪ | from §2 | 32,768 | 16,384 ⚠️ | Override to 32,768 for DS if longer prompts are expected |
| `b.truncation_side` | str | ⚪ | `right` | `right` | `right` | |

---

## 5. Component c — Loading the data

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `c.train_data_file` | str (path) | 🔴 for `fine_tune` | — | JSON list |
| `c.eval_data_file` | str (path) | 🔴 for `fine_tune` | — | Required when `load_best_model_at_end=true` |
| `c.test_data_file` | str (path) | 🔴 for `inference` | — | e.g. `data/testData.json` |
| `c.data_format` | str | 🔴 | `json` | `[{ "claim": str, "evidences": [{ "questions": str, "top_k_doc": [str] }], "label": str }]` |
| `c.label_values` | list[str] | 🔴 | `["True","False","Conflicting"]` | Three-way classification |
| `c.evidence_answers_used` | int | ⚪ | `1` | Reference script reads `top_k_doc[0]` only |
| `c.missing_answer_placeholder` | str | ⚪ | `"No answer could be found !"` | Used when `top_k_doc` is empty |
| `c.max_seq_length` | int\|null | 🟡 | `null` | SFTConfig value; unset in the reference script → silent default truncation |
| `c.max_samples` | int\|null | ⚪ | `null` | For quick smoke tests |

### 5.1 Evidence configurations (inference only — from the research design)

| Parameter | Type | Default | Allowed values |
|---|---|---|---|
| `c.evidence_config` | str | `with_evidence` | `with_evidence` \| `without_evidence` \| `random_evidence` \| `question_only` \| `answer_only` \| `all_questions_last_answer` \| `all_questions_all_but_last_answer` |

---

## 6. Component d — Optimizer (LoRA + SFT) — **fixed, read-only in the UI**

### 6.1 LoRA

| Parameter | Type | Value | Notes |
|---|---|---|---|
| `d.lora.r` | int | 64 | |
| `d.lora.lora_alpha` | int | 16 | |
| `d.lora.lora_dropout` | float | 0.1 | |
| `d.lora.bias` | str | `none` | |
| `d.lora.target_modules` | list[str] | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | All 7 projections; same names in both architectures |
| `d.lora.use_dora` | bool | `false` | |
| `d.lora.init_lora_weights` | bool | `true` | |

### 6.2 SFT / training arguments

| Parameter | Type | Value | Notes |
|---|---|---|---|
| `d.sft.learning_rate` | float | 2e-4 | |
| `d.sft.weight_decay` | float | 0.01 | |
| `d.sft.num_train_epochs` | int | 2 | |
| `d.sft.warmup_steps` | int | 5 | |
| `d.sft.optim` | str | `adamw_8bit` | |
| `d.sft.per_device_train_batch_size` | int | 2 | |
| `d.sft.gradient_accumulation_steps` | int | 1 | Default |
| `d.sft.lr_scheduler_type` | str | `linear` | Default |
| `d.sft.max_grad_norm` | float | 1.0 | Default |
| `d.sft.eval_strategy` | str | `epoch` | |
| `d.sft.save_strategy` | str | `epoch` | Must match `eval_strategy` when `load_best_model_at_end=true` |
| `d.sft.logging_strategy` | str | `steps` | |
| `d.sft.logging_steps` | int | 1 | |
| `d.sft.load_best_model_at_end` | bool | `true` | |
| `d.sft.metric_for_best_model` | str | `eval_loss` | |
| `d.sft.push_to_hub` | bool | `false` | |
| `d.sft.output_dir` | str | `fineTuningOutput` | |
| `d.sft.fp16` / `d.sft.bf16` | bool | `bf16=true` | bf16 matches the model dtype |
| `d.sft.gradient_checkpointing` | bool | `false` | Set `true` if memory-bound |
| `d.sft.seed` | int | 42 | ⚠️ **not set in the reference script** — add it for reproducibility |

---

## 7. Component e — Encoding the tokens

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `e.apply_chat_template` | bool | 🔴 | `true` | Must be `true` — never tokenize the rendered string by hand |
| `e.add_generation_prompt` | bool | 🔴 | auto | `false` for fine-tuning, `true` for inference |
| `e.tokenize` | bool | ⚪ | `true` | `apply_chat_template(tokenize=True)` internally uses `add_special_tokens=False` → no double BOS |
| `e.return_tensors` | str | 🔴 | `pt` | |
| `e.padding` | bool | 🔴 | `true` | Requires `padding_side="left"` for generation |
| `e.truncation` | bool | ⚪ | `true` | Only meaningful with `max_length` |
| `e.max_length` | int\|null | ⚪ | `null` | Must satisfy `max_length + max_new_tokens ≤ effective_max_seq_len` |
| `e.device` | str | 🔴 | `cuda` | |

---

## 8. Component f — Generating the text

| Parameter | Type | Required | Default | MS | DS | Notes |
|---|---|---|---|---|---|---|
| `f.max_new_tokens` | int | 🔴 | 500 | 512 (card) | ≥ 1024 | DS emits a `<think>` block first — 500 is likely too small |
| `f.temperature` | float | 🔴 | model default | **0.15** (card) / 0.9 (script) | **0.6** (card range 0.5–0.7) | Must be persisted — 0.15 vs 0.9 are incomparable runs |
| `f.top_p` | float\|null | 🟡 | — | 0.9 (script), unset by card | **0.95** | |
| `f.top_k` | int\|null | ⚪ | — | 10 (script) | `null` | |
| `f.do_sample` | bool | 🔴 | `true` | `true` | `true` | |
| `f.use_cache` | bool | ⚪ | `true` | `true` | `true` | |
| `f.num_return_sequences` | int | ⚪ | 1 | 1 | card: sample multiple responses and average |
| `f.repetition_penalty` | float | ⚪ | 1.0 | 1.0 | 1.0 | |
| `f.no_repeat_ngram_size` | int | ⚪ | 0 | 0 | 0 | |
| `f.pad_token_id` | int | 🔴 | from §4 | 11 | 151643 | Required for batched generation |
| `f.eos_token_id` | int | 🔴 | from §2 | 2 | 151643 | |
| `f.forced_prefix` | str\|null | ⚪ | `null` | `null` | `"\n\n"` | DS card: force the answer to start reasoning instead of skipping it |
| `f.batch_size` | int | ⚪ | 1 | 1 | 1 | The reference script loops one prompt at a time — batching is the main lever for scaling to 2,000+ runs |

---

## 9. Component g — Decoding the tokens

| Parameter | Type | Default | Notes |
|---|---|---|---|
| `g.skip_special_tokens` | bool | `true` | Strips `<s>`, `</s>`, `[INST]`, `<｜end▁of▁sentence｜>` |
| `g.clean_up_tokenization_spaces` | bool | `false` | Leave default |
| `g.spaces_between_special_tokens` | bool | `true` | Leave default |

---

## 10. Component h — `**kwargs` (optional arguments)

Free-form extra arguments forwarded to any component a–g, stored as:

```json
"h": {
  "load_model": { "...": "..." },
  "load_tokenizer": { "...": "..." },
  "encode": { "...": "..." },
  "generate": { "...": "..." }
}
```

Rules: keys must exist in the target library signature; unknown keys are rejected at generation time (fail fast rather than silently ignored).

---

## 11. Prompt parameters

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `p.system_prompt` | str | 🔴 | — | MS: **always set** (otherwise the template injects a date-stamped default → non-reproducible prompts). DS: officially discouraged — merge into the user prompt instead |
| `p.user_prompt_template` | str | 🔴 | see below | Must be model-appropriate; checked against the Model Card |
| `p.assistant_prompt` | str | 🔴 for `fine_tune` | `"### Response: Verdict: "` | Prepended to the gold label as the training target |
| `p.chat_template_override` | str\|null | ⚪ | `null` | Advanced: replace the model's `chat_template` |

### 11.1 Prompt-format tokens (must stay consistent between training and inference)

| Parameter | Value |
|---|---|
| `p.claim_prefix` | `"### Claim: "` |
| `p.question_prefix` | `"### Question: "` |
| `p.answer_prefix` | `"### Answer: "` |
| `p.no_answer_placeholder` | `"No answer could be found !"` |
| `p.response_prefix` | `"### Response: Verdict: "` |
| `p.verdict_parse_marker` | `"### Response: Verdict: "` (parsed with `rfind`) |

---

## 12. Validation rules (enforce in the form and in the generator)

1. `max_length + max_new_tokens ≤ effective_max_seq_len` (§2).
2. `padding_side` must be `left` whenever `task_type = inference` and `batch_size > 1`.
3. `pad_token` must resolve to a real vocabulary entry: MS → `<pad>` or `</s>`; DS → `<｜end▁of▁sentence｜>`.
4. `fix_mistral_regex` must be emitted as a `from_pretrained()` argument, never as a post-assignment.
5. `temperature > 0` requires `do_sample = true`; `temperature = 0` requires greedy decoding.
6. `load_best_model_at_end = true` requires `eval_strategy == save_strategy` and an `eval_data_file`.
7. `assistant_prompt` must be identical between the fine-tune config and the inference config that consumes that fine-tuned model — otherwise the verdict marker moves and parsing breaks.
8. `task_type = inference` on a fine-tuned adapter requires the matching fine-tune `config_id` (link the two records).
9. `system_prompt` empty + MS → reject or warn: the template silently injects a date-dependent default.
10. `system_prompt` non-empty + DS → warn: the template concatenates it with no delimiter (card advises against it).

---

## 13. Full JSON example (as stored in the database)

```json
{
  "config_id": "3f9c1a7e-2b41-4d8e-9c02-7ab5e6d10f34",
  "config_hash": "sha256:9d1f...c4a2",
  "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
  "task_type": "fine_tune",
  "run_status": "pending",
  "created_at": "2026-11-05T14:22:31Z",
  "created_by": "kerui.li",

  "a": {
    "dtype": "auto",
    "local_files_only": true,
    "tp_plan": "auto",
    "attn_implementation": "sdpa",
    "quantization": null,
    "trust_remote_code": false
  },
  "b": {
    "padding_side": "left",
    "pad_token": "<pad>",
    "fix_mistral_regex": true,
    "use_fast": true,
    "model_max_length": 32768
  },
  "c": {
    "train_data_file": "data/trainData.json",
    "eval_data_file": "data/evalData.json",
    "data_format": "json",
    "label_values": ["True", "False", "Conflicting"],
    "evidence_answers_used": 1,
    "missing_answer_placeholder": "No answer could be found !",
    "max_seq_length": 4096
  },
  "d": {
    "lora": {
      "r": 64,
      "lora_alpha": 16,
      "lora_dropout": 0.1,
      "bias": "none",
      "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    },
    "sft": {
      "learning_rate": 0.0002,
      "weight_decay": 0.01,
      "num_train_epochs": 2,
      "warmup_steps": 5,
      "optim": "adamw_8bit",
      "per_device_train_batch_size": 2,
      "eval_strategy": "epoch",
      "save_strategy": "epoch",
      "logging_strategy": "steps",
      "logging_steps": 1,
      "load_best_model_at_end": true,
      "metric_for_best_model": "eval_loss",
      "push_to_hub": false,
      "output_dir": "fineTuningOutput",
      "bf16": true,
      "seed": 42
    }
  },
  "e": {
    "apply_chat_template": true,
    "add_generation_prompt": false,
    "tokenize": true,
    "return_tensors": "pt",
    "padding": true,
    "truncation": true,
    "max_length": 4096,
    "device": "cuda"
  },
  "f": {
    "max_new_tokens": 512,
    "temperature": 0.15,
    "top_p": 0.9,
    "top_k": 10,
    "do_sample": true,
    "use_cache": true,
    "num_return_sequences": 1,
    "pad_token_id": 11,
    "eos_token_id": 2,
    "forced_prefix": null,
    "batch_size": 1
  },
  "g": {
    "skip_special_tokens": true
  },
  "h": {},
  "p": {
    "system_prompt": "You are a fact-checking assistant. Given a numerical claim and its retrieved evidence, output a verdict.",
    "user_prompt_template": "### Claim: {claim}\n### Question: {question}\n### Answer: {answer}",
    "assistant_prompt": "### Response: Verdict: ",
    "chat_template_override": null,
    "verdict_parse_marker": "### Response: Verdict: "
  }
}
```

---

## 14. Defaults cheat sheet (pre-filled form values)

| Field | `task_type = fine_tune` | `task_type = inference` |
|---|---|---|
| `a.dtype` | `auto` | `auto` |
| `a.tp_plan` | `auto` | `auto` |
| `b.padding_side` | `left` | `left` |
| `b.pad_token` | `<pad>` (MS) / `<｜end▁of▁sentence｜>` (DS) | same |
| `c.*_data_file` | train + eval required | test required |
| `e.add_generation_prompt` | `false` | `true` |
| `f.max_new_tokens` | 512 (MS) / 1024 (DS) | same |
| `f.temperature` | 0.15 (MS) / 0.6 (DS) | 0.15 (MS) / 0.6 (DS) |
| `f.top_p` | 0.9 (MS) / 0.95 (DS) | same |
| `f.top_k` | 10 (MS) / `null` (DS) | same |
| `f.batch_size` | 2 (training batch) | 1 → raise for throughput |
| `p.system_prompt` | required (MS) / warned (DS) | same |
| `p.assistant_prompt` | `### Response: Verdict: ` | not used |
| `p.chat_template_override` | `null` | `null` |
