# Model Reference — Mistral-Small-24B-Instruct-2501 & DeepSeek-R1-Distill-Qwen-32B

Verified against Hugging Face `tokenizer_config.json` / `chat_template`, `config.json`, `generation_config.json` and the published Model Cards (checked 2026-10-26, `transformers` 4.57.6).

Both chat templates below were **rendered locally** with the exact message format used by `fineTuningMistral.txt` / `inferenceMistral.txt`, so the strings are the real output, not a paraphrase.

---

## 1. Side-by-side summary

| | Mistral-Small-24B-Instruct-2501 | DeepSeek-R1-Distill-Qwen-32B |
|---|---|---|
| HF repo | `mistralai/Mistral-Small-24B-Instruct-2501` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| Architecture | `MistralForCausalLM` (24B) | `Qwen2ForCausalLM` (32.8B) |
| Layers / heads / KV heads | 40 / 32 / 8, head_dim 128 | 64 / 40 / 8 |
| Hidden size | 5120 | 5120 |
| Vocab | 131,072 (Tekken) | 152,064 |
| Context | 32,768 (`max_position_embeddings`) | 131,072 in `config.json`; **tokenizer `model_max_length` = 16,384**; card recommends `--max-model-len 32768` |
| dtype | bfloat16 | bfloat16 |
| Weights size | ~47.1 GB (~43.9 GiB) | ~65.5 GB (~61 GiB) |
| Min GPU (bf16) | ~55 GB (single 80GB card, or 2×48GB with TP) | 80GB single card, or `--tp 2` as documented |
| BOS / EOS | `<s>` (id 1) / `</s>` (id 2) | `<｜begin▁of▁sentence｜>` (151646) / `<｜end▁of▁sentence｜>` (151643) |
| PAD | **not set** in `tokenizer_config.json`; `<pad>` exists at **id 11** | already set = EOS `<｜end▁of▁sentence｜>` |
| `add_bos_token` | `True` | `True` |
| Chat template | `[SYSTEM_PROMPT]` / `[INST]` tags | `<｜User｜>` / `<｜Assistant｜>` fullwidth-pipe tags |
| System prompt | **Supported and recommended** | Format supports it, but card says *"Avoid adding a system prompt; all instructions should be contained within the user prompt"* |
| Default sampling | **temperature 0.15** (`generation_config.json`) | **temperature 0.6, top_p 0.95** (`generation_config.json`) |
| Special tooling | `mistral-common >= 1.5.2`, vLLM `>= 0.6.4` (`--tokenizer_mode mistral`) | `--tensor-parallel-size 2 --max-model-len 32768 --enforce-eager` |

---

## 2. Mistral-Small-24B-Instruct-2501

### 2.1 Chat template (verbatim from `tokenizer_config.json`)

```
{%- set today = strftime_now("%Y-%m-%d") %}
{%- set default_system_message = "You are Mistral Small 3, ... Your knowledge base was last
updated on 2023-10-01. The current date is " + today + ". ..." %}
{{- bos_token }}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content'] %}
    {%- set loop_messages = messages[1:] %}
{%- else %}
    {%- set system_message = default_system_message %}
    {%- set loop_messages = messages %}
{%- endif %}
{{- '[SYSTEM_PROMPT]' + system_message + '[/SYSTEM_PROMPT]' }}
{%- for message in loop_messages %}
    {%- if message['role'] == 'user' %}
        {{- '[INST]' + message['content'] + '[/INST]' }}
    {%- elif message['role'] == 'system' %}
        {{- '[SYSTEM_PROMPT]' + message['content'] + '[/SYSTEM_PROMPT]' }}
    {%- elif message['role'] == 'assistant' %}
        {{- message['content'] + eos_token }}
    {%- else %}
        {{- raise_exception('Only user, system and assistant roles are supported!') }}
    {%- endif %}
{%- endfor %}
```

### 2.2 Rendered output (with the project's own message format)

Training sample (system + user + assistant target):

```
<s>[SYSTEM_PROMPT]Instruction for the LLM[/SYSTEM_PROMPT][INST]### Claim: X
### Question: Q1
### Answer: A1[/INST]### Response: Verdict: True</s>
```

Inference sample (`add_generation_prompt=True`):

```
<s>[SYSTEM_PROMPT]Instruction for the LLM[/SYSTEM_PROMPT][INST]### Claim: X
### Question: Q1
### Answer: A1[/INST]
```

Key facts:

- Only **three roles** are legal (`system`, `user`, `assistant`). Any other role raises inside the template — the UI's role selector should be limited to exactly these three.
- The `system` message is consumed **only if it is `messages[0]`**. Put it first or it becomes an inline `[SYSTEM_PROMPT]` block mid-conversation.
- If no system message is supplied, the template injects a long **default system prompt that contains today's date** (`strftime_now`). Consequence for this project: the same config produces **different tokens on different days** unless you always set a system prompt. Set it explicitly, otherwise config-hash based de-duplication is not reproducible.
- Assistant turns are terminated with `</s>`; that is what the model learns to emit at the end of `### Response: Verdict: ...`.

### 2.3 Required parameters

Not repeated here — the full parameter set (types, defaults, ranges, per-model values) is consolidated in **[parameters.md](parameters.md)** §4 (tokenizer) and §8 (generation).

Model-specific values worth remembering: `dtype="auto"` (bf16), `local_files_only=True` on Fir, `tp_plan="auto"`, `padding_side="left"`, `pad_token` must be set explicitly to `<pad>` (id 11), `fix_mistral_regex=True` **passed into `from_pretrained()`**, temperature **0.15** (card) vs 0.9 (reference script).

---

## 3. DeepSeek-R1-Distill-Qwen-32B

### 3.1 Chat template (verbatim, abbreviated to the non-tool paths)

```
{{bos_token}}{{ns.system_prompt}}
{%- for message in messages %}
  {%- if role == 'user' %}      {{'<｜User｜>' + message['content']}}
  {%- if role == 'assistant' and content is not none %}
       {% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}
       {{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}
  {%- if role == 'tool' %} ... {%- endif %}
{%- endfor %}
{%- if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\n'}}{% endif %}
```

### 3.2 Rendered output (same message format)

Training sample:

```
<｜begin▁of▁sentence｜>Instruction for the LLM<｜User｜>### Claim: X
### Question: Q1
### Answer: A1<｜Assistant｜>### Response: Verdict: True<｜end▁of▁sentence｜>
```

Inference sample (`add_generation_prompt=True`):

```
<｜begin▁of▁sentence｜>Instruction for the LLM<｜User｜>### Claim: X
### Question: Q1
### Answer: A1<｜Assistant｜><think>
```

Key facts:

- The system message is concatenated **raw, directly after BOS, with no delimiter tag** — this is exactly why the card discourages system prompts for this model: the boundary between system and first user turn is ambiguous. **Put all instructions into the user prompt** for DeepSeek.
- Generation prompt forces a **`<think>` block** (this is the R1 reasoning format). Downstream parsing must tolerate reasoning text appearing before `### Response: Verdict: ...`; the reference script's `rfind("### Response: Verdict: ")` happens to survive this, but `startswith`-based parsing would not.
- During **training**, if an assistant message contains `</think>`, the template silently **keeps only the text after the last `</think>`**. The project's labels (`### Response: Verdict: True`) contain no think block, so nothing is stripped — but any future data with reasoning traces will be truncated in the target.
- `pad_token == eos_token == <｜end▁of▁sentence｜>`.
- `tokenizer.model_max_length` is 16,384 while `config.json` allows 131,072 — set the length limit explicitly rather than trusting the tokenizer default.

### 3.3 Required parameters

Not repeated here — see **[parameters.md](parameters.md)** §4 and §8 for the full set.

Model-specific values worth remembering: `padding_side="left"`, `pad_token` is already `<｜end▁of▁sentence｜>` (**do not overwrite it**), temperature **0.6** with `top_p=0.95`, and `max_new_tokens` must be large enough to cover the forced `<think>` block.

---

## 4. Findings that directly affect our scripts and the M3.0 generator

1. **`tokenizer.fix_mistral_regex = True` in `fineTuningMistral.txt` has no effect.** The regex repair is applied inside `AutoTokenizer.from_pretrained()` (`_patch_mistral_regex`); assigning the attribute afterwards only sets a flag. Correct form:
   `AutoTokenizer.from_pretrained(modelId, local_files_only=True, padding_side="left", fix_mistral_regex=True)`
   Without it, Mistral tokenization is **incorrect** (upstream issue on Mistral-Small-3/3.1 tokenizers).
2. **`tokenizer.pad_token = "<myPadToken>"` in `inferenceMistral.txt` is not a real token.** It is neither in the Mistral vocab nor the DeepSeek vocab. Use `<pad>` (Mistral, id 11) or leave DeepSeek's EOS pad untouched.
3. **Mistral `pad_token` is unset by default** — a tokenizer used for batched generation without setting pad will warn/behave unpredictably. The UI should surface `pad_token` as a required field (component **b**).
4. **`temperature=0.9` in the inference script is 6× the card's recommendation (0.15).** Not a bug per se, but it is a real config dimension: the platform's config schema should capture it so 0.15 vs 0.9 runs are distinguishable and comparable.
5. **DeepSeek's system prompt is undelimited** — if the UI lets users fill a system prompt for DeepSeek, the generated script should concatenate it into the user prompt instead (card guidance), or at minimum warn.
6. **Role set is model-specific**: Mistral accepts exactly `system/user/assistant`; DeepSeek additionally has `tool`. The UI's prompt section should be driven by the model card metadata, not hardcoded.
7. **Never tokenize the rendered template string yourself.** `apply_chat_template(tokenize=True)` already suppresses special-token insertion; calling `tokenizer(text)` on the rendered string would double the BOS.

---

## 5. UI field mapping (Milestone 1.0 components a–g)

The UI-component ↔ parameter mapping, the validation rules, the full stored-JSON example and a pre-filled defaults cheat sheet are all in **[parameters.md](parameters.md)** §12–§14. This file keeps only the model-card facts, the rendered chat templates and the script findings.
