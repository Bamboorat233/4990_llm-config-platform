# Milestone 1 交接文档（M0.5 + M1.0）

> **本文档用途**:交给另一个 session 的 LLM 作为唯一上下文来源。文档自包含,不需要访问其他对话历史。
> **权威性**:里程碑要求以导师下发的 `doc/Project Milestones - 4990.docx` 为准;参数取值以本文档第 5 节为准;模型事实以第 8 节为准。仓库里的 `specs/parameters.md` / `specs/model-reference.md` 是本文档的展开版。
> **生成时间**:2026-10-28。当前日期距 M0.5 检查 3 天。

---

## 1. 项目背景(30 秒版)

**项目**:Integration of Large Language Models into Autonomous Fact-Checking Pipelines — Infrastructure and Technical Implementation(University of Windsor 4990B,Group 15)。

**要解决的问题**:研究团队(博士生 Gagandeep Kaur 主导)正在做**数值型事实核查**实验,要在 Fir(AllianceCan)超算上跑 **2000+ 次**模型实验,横向比较 zero-shot 与 LoRA 微调、不同证据组合、不同采样参数下的 accuracy / F1。目前他们靠手工写脚本 + 手工读日志 + 用 Excel 记分数,不可扩展。

**我们要交付的东西**:一个本地托管的 Web 编排平台,流程是

```
配置参数 → 生成执行脚本 → (远端)提交执行 → 解析输出日志 → 计算指标 → 可视化对比
```

我们 4 人负责技术实现(BSc 4990),MAC/ARP 学生负责技术方向与评审,Dr. Aznam Yacoub 是主管。

**分工**:A 前端/表单、B 数据层/Pydantic、C 日志解析/评分、D 脚本生成/联调(见末尾分工表)。

---

## 2. Milestone 1 的验收要求(原文要点)

里程碑文档把 **M0.5 与 M1.0 写在同一节**,检查点定位是「看团队有没有理解或实现上的困难,并展示目前已完成的成果」。因此 **M0.5 需要交的是 M1.0 需求的前半部分**,不是只有账号和环境。

### 2.1 必须实现的 UI 功能(逐条对应原文)

| # | 原文要求 | 实现要点 |
|---|---|---|
| 1 | Asks the user to select a Large-Language Model from a list | 模型列表,首批 `Mistral-Small-24B-Instruct-2501`,结构上预留 `DeepSeek-R1-Distill-Qwen-32B` |
| 2 | Asks the user if s/he wants to train the model (fine-tune) or run inference | 二选一,`fine_tune` / `inference`;**后续表单随选择变化** |
| 3 | Asks the user to enter the system prompt / user prompt / assistant prompt | 三个 prompt,内容随模型而定(必须查 Model Card,见第 8 节) |
| 4 | Asks the user to enter Parameters/Hyperparameters for different components (a–g) | 见第 5 节完整字段表 |
| 4a | Loading the model | 组件 a |
| 4b | Loading the tokenizer | 组件 b |
| 4c | Loading the data | 组件 c |
| 4d | Loading the optimizer (LoRA and/or SFT) — assume the parameters to be fixed | 组件 d,**只读展示,不可编辑** |
| 4e | Encoding the tokens | 组件 e |
| 4f | Generating the text | 组件 f |
| 4g | Decoding the tokens | 组件 g |
| 4h | Optional arguments — denoted by `**kwargs` (applicable to all from a to g) | 组件 h,每组件的自由键值对 |
| 5 | Store all the collected information in the database in json format | 配置以 JSON 形式落库 |

> 原文特别注明:「the components in the execution script might be different depending upon the purpose, i.e., training or inference」——所以 a–h 的**可见性/必填性**要按 task_type 切换。

### 2.2 两个检查点各自的交付物

| 检查点 | 日期 | 性质 | 必须能演示 |
|---|---|---|---|
| **M0.5** | **10/31(周六)** | 线上进度检查 | 环境就绪(SSH 到 Fir、Git 提交记录)+ **表单原型可点通**(页面 1–3 + 参数骨架 + JSON 预览,不要求落库) |
| **M1.0** | **11/7(周六)** | 正式交付 | 完整 UI + 配置以 JSON 存入数据库 + 配置列表/读取/复制 |

---

## 3. 当前仓库状态

工作目录:`C:\Users\26773\OneDrive\桌面\26Fall\4990B`

| 路径 | 作用 | 状态 |
|---|---|---|
| `doc/Project Milestones - 4990.docx` | **验收标准**。注意:内部是 `word/document2.xml` 而非 `document.xml`,标准 docx 阅读器会报 `corrupt_file`,需解压读 XML | 参考 |
| `doc/Automated Fact-Checking with LLMs.pdf` | 研究侧汇报(背景、数据集、模型、痛点) | 参考 |
| `doc/Group 15 project description.docx`、`doc/Email.md` | 立项书与导师邮件(约束、红线) | 参考 |
| `doc/other/` | 与项目无直接关系的课程/其他文档(反思模板、研讨课报告) | 归档 |
| `specs/parameters.md` | 参数规格单一事实源(14 节) | 可用 |
| `specs/model-reference.md` | 两个模型的 chat template、token id、脚本缺陷分析 | 可用 |
| `planning/TaskList.md` / `planning/任务列表-中文.md` | 按检查点拆分的任务清单(内容对等) | 可用 |
| `planning/M1-context-handoff.md` | 本文档 | 可用 |
| `design/Spec_and_roadmap.md`(英文) / `.zh.md`(中文) | 另一 session 产出的技术方案与 Sprint 计划。**采用 NiceGUI,与本文档第 4 节的 FastAPI 不一致,需先决策** | 待决策 |
| `reference-scripts/fineTuningMistral_en.py` | 参考微调脚本,逐行英文注释 | 可用 |
| `reference-scripts/inferenceMistral_en.py` | 参考推理脚本,逐行英文注释 | 可用 |
| `reference-scripts/fineTuningMistral_cn.py` / `inferenceMistral_cn.py` | 同逻辑,中文注释 | 可用 |

### 3.1 重要:曾存在过一个 app/ 雏形,已被删除

10/28 10:26 时目录下存在:

```
app/main.py          3,841 B
app/registry.py      8,921 B
app/schemas.py      11,851 B
app/store.py         4,374 B
app/static/index.html 26,458 B
app/data/platform.db 20,480 B
```

约 10:31 整个 `app/` 与 `__pycache__/` 消失(疑似 OneDrive 同步)。**内容未留存,本文档不含其实现细节。** 如果回收站能找回,优先复用;否则按第 4 节的骨架重写。

### 3.2 其他状态

- **不是 git 仓库**(`fatal: not a git repository`)——Phase 0 要求建仓库,这是 M0.5 的展示项之一。
- 本地 Python **3.13.3**,已装 **fastapi 0.136.1**、**pydantic 2.12.5**。
- 尚未拿到:博士生提供的**六种场景样例输出文件**、**样例执行脚本**、数据集 train/eval/test 划分。这些分别阻塞 M1.5 与 M2.5,不影响 M1。

---

## 4. 技术栈与实现骨架(含函数名)

### 4.1 选型(⚠️ 存在未决冲突:本文档为 FastAPI,`design/Spec_and_roadmap.md` 为 NiceGUI)

| 层 | 选择 | 理由 |
|---|---|---|
| Web 框架 | **FastAPI** | 轻量,自动生成 OpenAPI,便于后续 M2.0 出 JSON 接口 |
| 数据校验 | **Pydantic v2** | 导师邮件明确要求用 Pydantic 替代裸 JSON/SQL 结构 |
| 存储 | **SQLite**(单文件) | 满足"存进数据库"且零部署;配置以 JSON 列/文本存储 |
| 前端 | **静态 HTML + 原生 JS**(`static/index.html`) | 无需构建链,M0.5 三天内可交付 |

> 注意:里程碑原文只要求"store in the database in json format"。用 SQLite + 一列存 JSON 文本即满足;不要为此引入 Postgres 或复杂 ORM。

### 4.2 目录与文件职责

```
app/
  __init__.py
  main.py        FastAPI 应用与路由
  registry.py    模型注册表(模型事实与默认值)
  schemas.py     Pydantic 模型(配置结构 + 校验)
  store.py       SQLite 持久化
  static/
    index.html   单页表单
  data/
    platform.db  SQLite 数据库(不提交到 git)
```

### 4.3 建议实现的函数/类(签名)

**`app/schemas.py`** —— 每个组件一个模型,与里程碑 a–h 一一对应:

```python
class ModelSection(BaseModel):        # a
    dtype: Literal["auto","bfloat16","float16","float32"] = "auto"
    local_files_only: bool = True
    tp_plan: str | None = "auto"
    device_map: str | None = None
    attn_implementation: Literal["eager","sdpa","flash_attention_2"] | None = None
    quantization: Literal["4bit","8bit"] | None = None
    trust_remote_code: bool = False
    model_id: str

class TokenizerSection(BaseModel):    # b
    tokenizer_id: str | None = None
    padding_side: Literal["left","right"] = "left"
    pad_token: str
    pad_token_id: int | None = None       # 只读,由 registry 解析
    fix_mistral_regex: bool = True
    use_fast: bool = True
    model_max_length: int | None = None
    truncation_side: Literal["left","right"] = "right"

class DataSection(BaseModel):         # c
    train_data_file: str | None = None
    eval_data_file: str | None = None
    test_data_file: str | None = None
    data_format: Literal["json"] = "json"
    label_values: list[str] = ["True","False","Conflicting"]
    evidence_answers_used: int = 1
    missing_answer_placeholder: str = "No answer could be found !"
    max_seq_length: int | None = None
    max_samples: int | None = None
    evidence_config: Literal[
        "with_evidence","without_evidence","random_evidence",
        "question_only","answer_only",
        "all_questions_last_answer","all_questions_all_but_last_answer"
    ] = "with_evidence"

class OptimizerSection(BaseModel):    # d —— 固定值,前端只读
    lora: LoraSection
    sft: SftSection

class EncodingSection(BaseModel):     # e
    apply_chat_template: bool = True
    add_generation_prompt: bool          # fine_tune=False, inference=True
    tokenize: bool = True
    return_tensors: Literal["pt"] = "pt"
    padding: bool = True
    truncation: bool = True
    max_length: int | None = None
    device: str = "cuda"

class GenerationSection(BaseModel):   # f
    max_new_tokens: int = 512
    temperature: float = 0.15
    top_p: float | None = 0.9
    top_k: int | None = 10
    do_sample: bool = True
    use_cache: bool = True
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    pad_token_id: int | None = None
    eos_token_id: int | None = None
    forced_prefix: str | None = None
    batch_size: int = 1

class DecodingSection(BaseModel):     # g
    skip_special_tokens: bool = True
    clean_up_tokenization_spaces: bool = False
    spaces_between_special_tokens: bool = True

class PromptSection(BaseModel):       # p
    system_prompt: str
    user_prompt_template: str
    assistant_prompt: str = "### Response: Verdict: "
    chat_template_override: str | None = None

class ConfigRecord(BaseModel):        # 顶层
    config_id: str
    config_hash: str
    model_id: str
    task_type: Literal["fine_tune","inference"]
    run_status: Literal["pending","running","success","error","terminated"] = "pending"
    created_at: str
    created_by: str | None = None
    notes: str | None = None
    a: ModelSection; b: TokenizerSection; c: DataSection
    d: OptimizerSection; e: EncodingSection
    f: GenerationSection; g: DecodingSection
    h: dict[str, dict] = {}           # 各组件的 **kwargs

def validate_cross_fields(cfg: ConfigRecord) -> list[dict]: ...   # 实现第 6 节的规则
def config_fingerprint(cfg: ConfigRecord) -> str: ...             # 排除 config_id/created_at/notes 后做 sha256
```

**`app/registry.py`** —— 模型事实与默认值,UI 与校验器共用:

```python
@dataclass(frozen=True)
class ModelMeta:
    model_id: str; display_name: str; num_parameters: str
    num_hidden_layers: int; num_attention_heads: int; num_key_value_heads: int
    hidden_size: int; intermediate_size: int; head_dim: int
    vocab_size: int; max_position_embeddings: int; tokenizer_model_max_length: int
    bos_token: str; bos_token_id: int
    eos_token: str; eos_token_id: int
    pad_token: str; pad_token_id: int
    allowed_roles: tuple[str, ...]
    system_prompt_policy: Literal["required","warn"]
    default_temperature: float; default_top_p: float; default_max_new_tokens: int
    needs_fix_mistral_regex: bool
    min_gpu_memory_bf16_gb: int; weights_size_gb: float

MODELS: dict[str, ModelMeta] = {
    "mistralai/Mistral-Small-24B-Instruct-2501": ModelMeta(...),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": ModelMeta(...),
}

def get_model(model_id: str) -> ModelMeta: ...
def defaults_for(model_id: str, task_type: str) -> dict: ...   # 返回预填表单的完整字典
```

**`app/store.py`** —— 持久化:

```python
def init_db() -> None: ...
def insert_config(cfg: ConfigRecord) -> str: ...        # 返回 config_id
def get_config(config_id: str) -> ConfigRecord | None: ...
def list_configs() -> list[dict]: ...                    # 摘要列表,供 UI 侧栏
def duplicate_config(config_id: str) -> ConfigRecord: ...  # 复制后 config_id/config_hash 重新生成
def find_by_hash(config_hash: str) -> str | None: ...   # 去重:同配置重复提交时提示
```

**`app/main.py`** —— 路由:

```
GET  /                              -> static/index.html
GET  /api/models                    -> 模型列表(含只读元数据)
GET  /api/models/{model_id}/defaults?task_type=...  -> 预填值
POST /api/configs                   -> 校验 + 入库,返回 config_id
GET  /api/configs                   -> 配置列表
GET  /api/configs/{config_id}       -> 单条配置
POST /api/configs/{config_id}/duplicate
GET  /api/health                    -> {"status":"ok"}
```

> `app/main.py:1` 这类形式可放在提交说明里;实现时把 `init_db()` 放在 FastAPI 的 `lifespan` 里,不要用已废弃的 `@app.on_event("startup")`。

---

## 5. 参数规格(UI 字段定义)

**图例**:必填 🔴 / 实践上必填 🟡 / 可选 ⚪;**MS** = `mistralai/Mistral-Small-24B-Instruct-2501`;**DS** = `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`;`—` = 不适用。

### 5.1 顶层配置记录

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `config_id` | str | 🔴 | 生成 | UUID |
| `config_hash` | str | 🔴 | 派生 | 全部功能参数的 hash → 去重/识别重复运行 |
| `model_id` | str | 🔴 | — | 驱动所有下游默认值 |
| `task_type` | str | 🔴 | — | `fine_tune` \| `inference` |
| `run_status` | str | 🔴 | `pending` | `pending` \| `running` \| `success` \| `error` \| `terminated` |
| `created_at` | str(ISO 8601) | 🔴 | now | |
| `created_by` | str | ⚪ | — | |
| `notes` | str | ⚪ | — | 研究者的备注 |

### 5.2 组件 a — Loading the model

| 字段 | 类型 | 必填 | 默认 | MS | DS | 说明 |
|---|---|---|---|---|---|---|
| `a.dtype` | str | 🔴 | `auto` | `auto`(→bf16) | `auto` | `auto`\|`bfloat16`\|`float16`\|`float32` |
| `a.local_files_only` | bool | 🔴 | `true` | `true` | `true` | Fir 计算节点无网络,必须为 true |
| `a.tp_plan` | str\|null | ⚪ | `auto` | `auto` | `auto` | 张量并行,需 ≥2 GPU |
| `a.device_map` | str\|null | ⚪ | `null` | 单卡时 `auto` | `auto` | 与 tp_plan 互斥 |
| `a.attn_implementation` | str\|null | ⚪ | `null` | `sdpa` | `sdpa` | `eager`\|`sdpa`\|`flash_attention_2` |
| `a.quantization` | str\|null | ⚪ | `null` | `null` | `null` | 参考脚本 import 了 BitsAndBytesConfig 但未使用 |
| `a.trust_remote_code` | bool | ⚪ | `false` | `false` | `false` | 两个模型都是原生架构 |
| `a.model_id` | str | 🔴 | 取 §5.1 | — | — | 本地路径或 hub id |

### 5.3 组件 b — Loading the tokenizer

| 字段 | 类型 | 必填 | 默认 | MS | DS | 说明 |
|---|---|---|---|---|---|---|
| `b.tokenizer_id` | str | ⚪ | `= model_id` | 同左 | 同左 | |
| `b.padding_side` | str | 🔴 | `left` | `left` | `left` | 解码器批量生成必须 left |
| `b.pad_token` | str | 🔴 | 按模型 | `<pad>`(id 11) | `<｜end▁of▁sentence｜>` | MS 退路 `</s>`;**禁止不在词表里的字面量** |
| `b.pad_token_id` | int | 🔴 派生 | 解析 | 11 | 151643 | **只读展示** |
| `b.fix_mistral_regex` | bool | 🟡 | `true` | `true` | — | **必须作为 `from_pretrained()` 参数传入**,事后赋值无效 |
| `b.use_fast` | bool | ⚪ | `true` | `true` | `true` | |
| `b.model_max_length` | int | ⚪ | 见 §8 | 32768 | 16384 ⚠️ | DS 建议显式覆盖为 32768 |
| `b.truncation_side` | str | ⚪ | `right` | `right` | `right` | |

### 5.4 组件 c — Loading the data

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `c.train_data_file` | str(路径) | 🔴 仅 fine_tune | — | JSON list |
| `c.eval_data_file` | str(路径) | 🔴 仅 fine_tune | — | `load_best_model_at_end=true` 时必需 |
| `c.test_data_file` | str(路径) | 🔴 仅 inference | — | |
| `c.data_format` | str | 🔴 | `json` | 结构见第 7 节 |
| `c.label_values` | list[str] | 🔴 | `["True","False","Conflicting"]` | 三分类 |
| `c.evidence_answers_used` | int | ⚪ | `1` | 参考脚本只读 `top_k_doc[0]` |
| `c.missing_answer_placeholder` | str | ⚪ | `"No answer could be found !"` | `top_k_doc` 为空时使用 |
| `c.max_seq_length` | int\|null | 🟡 | `null` | SFTConfig 值;参考脚本未设 → 静默截断 |
| `c.max_samples` | int\|null | ⚪ | `null` | 冒烟测试用 |
| `c.evidence_config` | str | ⚪ | `with_evidence` | 见下 |

**`c.evidence_config`(仅 inference,来自研究设计,必须支持)**:
`with_evidence` \| `without_evidence` \| `random_evidence` \| `question_only` \| `answer_only` \| `all_questions_last_answer` \| `all_questions_all_but_last_answer`

### 5.5 组件 d — Optimizer(LoRA + SFT)—— **固定值,前端只读**

LoRA:

| 字段 | 值 |
|---|---|
| `d.lora.r` | 64 |
| `d.lora.lora_alpha` | 16(有效缩放 = 16/64 = 0.25) |
| `d.lora.lora_dropout` | 0.1 |
| `d.lora.bias` | `none` |
| `d.lora.target_modules` | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]`(两架构同名) |
| `d.lora.use_dora` | `false` |
| `d.lora.init_lora_weights` | `true` |

SFT:

| 字段 | 值 | 说明 |
|---|---|---|
| `d.sft.learning_rate` | 2e-4 | |
| `d.sft.weight_decay` | 0.01 | |
| `d.sft.num_train_epochs` | 2 | |
| `d.sft.warmup_steps` | 5 | |
| `d.sft.optim` | `adamw_8bit` | 需 bitsandbytes |
| `d.sft.per_device_train_batch_size` | 2 | |
| `d.sft.gradient_accumulation_steps` | 1 | |
| `d.sft.lr_scheduler_type` | `linear` | |
| `d.sft.max_grad_norm` | 1.0 | |
| `d.sft.eval_strategy` | `epoch` | |
| `d.sft.save_strategy` | `epoch` | 与 eval_strategy 必须一致 |
| `d.sft.logging_strategy` / `logging_steps` | `steps` / 1 | |
| `d.sft.load_best_model_at_end` | `true` | |
| `d.sft.metric_for_best_model` | `eval_loss` | |
| `d.sft.push_to_hub` | `false` | |
| `d.sft.output_dir` | `fineTuningOutput` | |
| `d.sft.bf16` | `true` | 与模型 dtype 一致 |
| `d.sft.gradient_checkpointing` | `false` | 显存紧张时改 true |
| `d.sft.seed` | 42 | ⚠️ 参考脚本缺失,必须补上以保证可复现 |

### 5.6 组件 e — Encoding the tokens

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|---|---|---|
| `e.apply_chat_template` | bool | 🔴 | `true` | 必须 true,禁止手工 tokenize 渲染后的字符串 |
| `e.add_generation_prompt` | bool | 🔴 | 按任务 | `fine_tune=false` / `inference=true` |
| `e.tokenize` | bool | ⚪ | `true` | `apply_chat_template(tokenize=True)` 内部 `add_special_tokens=False`,避免双 BOS |
| `e.return_tensors` | str | 🔴 | `pt` | |
| `e.padding` | bool | 🔴 | `true` | 生成场景需 `padding_side="left"` |
| `e.truncation` | bool | ⚪ | `true` | 仅在设了 `max_length` 时有意义 |
| `e.max_length` | int\|null | ⚪ | `null` | 须满足 `max_length + max_new_tokens ≤ 有效上限` |
| `e.device` | str | 🔴 | `cuda` | |

### 5.7 组件 f — Generating the text

| 字段 | 类型 | 必填 | 默认 | MS | DS | 说明 |
|---|---|---|---|---|---|---|
| `f.max_new_tokens` | int | 🔴 | 500 | 512 | ≥1024 | DS 会先输出 `<think>` 块,500 可能不够 |
| `f.temperature` | float | 🔴 | 按模型 | **0.15**(卡片)/0.9(脚本) | **0.6**(卡片 0.5–0.7) | 必须落库,0.15 与 0.9 不可比 |
| `f.top_p` | float\|null | 🟡 | — | 0.9 | **0.95** | |
| `f.top_k` | int\|null | ⚪ | — | 10 | `null` | |
| `f.do_sample` | bool | 🔴 | `true` | `true` | `true` | |
| `f.use_cache` | bool | ⚪ | `true` | `true` | `true` | |
| `f.num_return_sequences` | int | ⚪ | 1 | 1 | 卡片建议多次采样取平均 | |
| `f.repetition_penalty` | float | ⚪ | 1.0 | 1.0 | 1.0 | |
| `f.no_repeat_ngram_size` | int | ⚪ | 0 | 0 | 0 | |
| `f.pad_token_id` | int | 🔴 | 取 §5.3 | 11 | 151643 | 批量生成必需 |
| `f.eos_token_id` | int | 🔴 | 取 §8 | 2 | 151643 | |
| `f.forced_prefix` | str\|null | ⚪ | `null` | `null` | `"\n\n"` | DS 卡片:迫使模型开始推理 |
| `f.batch_size` | int | ⚪ | 1 | 1 | 1 | 参考脚本逐条循环;批量是扩到 2000+ 次的主要杠杆 |

### 5.8 组件 g — Decoding the tokens

| 字段 | 默认 | 说明 |
|---|---|---|
| `g.skip_special_tokens` | `true` | 去掉 `<s>`、`</s>`、`[INST]`、`<｜end▁of▁sentence｜>` |
| `g.clean_up_tokenization_spaces` | `false` | 保持默认 |
| `g.spaces_between_special_tokens` | `true` | 保持默认 |

### 5.9 组件 h — `**kwargs`

自由键值对,按组件分组存放:

```json
"h": {
  "load_model": { "...": "..." },
  "load_tokenizer": { "...": "..." },
  "encode": { "...": "..." },
  "generate": { "...": "..." }
}
```

规则:键必须存在于目标库函数的签名中;**未知键在生成/校验阶段报错,不要静默忽略**(fail fast)。

### 5.10 组件 p — Prompt 参数

| 字段 | 必填 | 默认 | 说明 |
|---|---|---|---|
| `p.system_prompt` | 🔴 | — | MS 必须设置(否则模板注入带**当天日期**的默认 prompt,同配置不同天不可复现);DS 官方不建议用,应并入 user prompt |
| `p.user_prompt_template` | 🔴 | 见下 | 须与模型匹配,查 Model Card |
| `p.assistant_prompt` | 🔴 仅 fine_tune | `"### Response: Verdict: "` | 拼在 gold label 前作为训练目标 |
| `p.chat_template_override` | ⚪ | `null` | 高级:替换模型 chat_template |

格式常量(训练与推理必须一致):

| 常量 | 值 |
|---|---|
| `p.claim_prefix` | `"### Claim: "` |
| `p.question_prefix` | `"### Question: "` |
| `p.answer_prefix` | `"### Answer: "` |
| `p.no_answer_placeholder` | `"No answer could be found !"` |
| `p.response_prefix` | `"### Response: Verdict: "` |
| `p.verdict_parse_marker` | `"### Response: Verdict: "`(`rfind` 解析) |

---

## 6. 交叉校验规则(表单与生成器都要实现)

1. `max_length + max_new_tokens ≤ effective_max_seq_len`(见 §8)。→ 违反时报错,给出上限。
2. `task_type = inference` 且 `batch_size > 1` 时,`padding_side` 必须为 `left`。
3. `pad_token` 必须解析到真实词表项:MS → `<pad>` 或 `</s>`;DS → `<｜end▁of▁sentence｜>`。不在词表内直接拒绝。
4. `fix_mistral_regex` 必须作为 `from_pretrained()` 参数输出,不能事后赋值。
5. `temperature > 0` 要求 `do_sample = true`;`temperature = 0` 要求贪心解码(`do_sample=false`)。
6. `load_best_model_at_end = true` 要求 `eval_strategy == save_strategy` 且提供了 `eval_data_file`。
7. `assistant_prompt` 在微调配置与其下游推理配置之间必须一致,否则 verdict marker 位移、解析失败。
8. 基于微调 adapter 的推理配置必须关联对应的微调 `config_id`。
9. `system_prompt` 为空 + MS → 拒绝或强警告(模板会注入日期相关的默认值)。
10. `system_prompt` 非空 + DS → 警告(模板会把 system 原样拼在 BOS 后,无分隔符;卡片建议避免)。

**前端表现**:校验失败时错误必须定位到具体字段(如 `b.pad_token`),而不是顶部一条笼统提示;提交前提供完整 JSON 预览。

---

## 7. 数据集与大致的存储 JSON

### 7.1 输入数据格式(QuanTemp 风格)

```json
[
  {
    "claim": "数值型陈述",
    "evidences": [
      { "questions": "用于核查的问题", "top_k_doc": ["最相关文档", "次相关文档"] }
    ],
    "label": "True"
  }
]
```

- 每条记录 1 个 claim + **3~5 组** question/answer 对。
- `label ∈ {"True","False","Conflicting"}`。
- 参考脚本只使用 `top_k_doc[0]` 作为 answer。

### 7.2 配置入库的 JSON 形态

```json
{
  "config_id": "3f9c1a7e-2b41-4d8e-9c02-7ab5e6d10f34",
  "config_hash": "sha256:9d1f...c4a2",
  "model_id": "mistralai/Mistral-Small-24B-Instruct-2501",
  "task_type": "fine_tune",
  "run_status": "pending",
  "created_at": "2026-11-05T14:22:31Z",
  "created_by": "kerui.li",
  "a": { "dtype": "auto", "local_files_only": true, "tp_plan": "auto",
         "attn_implementation": "sdpa", "quantization": null, "trust_remote_code": false },
  "b": { "padding_side": "left", "pad_token": "<pad>", "fix_mistral_regex": true,
         "use_fast": true, "model_max_length": 32768 },
  "c": { "train_data_file": "data/trainData.json", "eval_data_file": "data/evalData.json",
         "data_format": "json", "label_values": ["True","False","Conflicting"],
         "evidence_answers_used": 1, "missing_answer_placeholder": "No answer could be found !",
         "max_seq_length": 4096, "evidence_config": "with_evidence" },
  "d": {
    "lora": { "r": 64, "lora_alpha": 16, "lora_dropout": 0.1, "bias": "none",
              "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] },
    "sft": { "learning_rate": 0.0002, "weight_decay": 0.01, "num_train_epochs": 2,
             "warmup_steps": 5, "optim": "adamw_8bit", "per_device_train_batch_size": 2,
             "eval_strategy": "epoch", "save_strategy": "epoch", "logging_strategy": "steps",
             "logging_steps": 1, "load_best_model_at_end": true,
             "metric_for_best_model": "eval_loss", "push_to_hub": false,
             "output_dir": "fineTuningOutput", "bf16": true, "seed": 42 }
  },
  "e": { "apply_chat_template": true, "add_generation_prompt": false, "tokenize": true,
         "return_tensors": "pt", "padding": true, "truncation": true,
         "max_length": 4096, "device": "cuda" },
  "f": { "max_new_tokens": 512, "temperature": 0.15, "top_p": 0.9, "top_k": 10,
         "do_sample": true, "use_cache": true, "num_return_sequences": 1,
         "pad_token_id": 11, "eos_token_id": 2, "forced_prefix": null, "batch_size": 1 },
  "g": { "skip_special_tokens": true },
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

**表结构建议**:`configs(config_id PK, config_hash UNIQUE, model_id, task_type, run_status, created_at, created_by, notes, payload_json)` + `runs(run_id PK, config_id FK, started_at, finished_at, status, output_path)`。一份配置可对应多次运行(M2.0 的重复评分、M3.0 的对比都需要)。

---

## 8. 模型事实(两个 Model Card 的精简版)

### 8.1 对照表

| 项目 | Mistral-Small-24B-Instruct-2501 | DeepSeek-R1-Distill-Qwen-32B |
|---|---|---|
| HF repo | `mistralai/Mistral-Small-24B-Instruct-2501` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` |
| 架构 | `MistralForCausalLM`,24B(23,572,403,200) | `Qwen2ForCausalLM`,32.8B(32,763,876,352) |
| 层 / 头 / KV 头 | 40 / 32 / 8,head_dim 128 | 64 / 40 / 8 |
| hidden / intermediate | 5120 / 32768 | 5120 / 27648 |
| 词表 | 131,072(Tekken) | 152,064 |
| 上下文 | `max_position_embeddings` 32,768 | `config.json` 131,072;**tokenizer `model_max_length` 仅 16,384**;卡片建议 `--max-model-len 32768` |
| dtype | bfloat16 | bfloat16 |
| 权重体积 | ~47.1 GB(~43.9 GiB) | ~65.5 GB(~61 GiB) |
| 最小显存(bf16) | ~55 GB(单卡 80GB,或 TP≥2) | ~80 GB 单卡,或 `--tp 2` |
| BOS / EOS | `<s>`(1)/ `</s>`(2) | `<｜begin▁of▁sentence｜>`(151646)/ `<｜end▁of▁sentence｜>`(151643) |
| PAD | 未设置;`<pad>` 存在于 id **11** | = EOS(151643) |
| `add_bos_token` | `True` | `True` |
| 允许角色 | `system` / `user` / `assistant`(其他角色模板内直接 raise) | `system` / `user` / `assistant` / `tool` |
| system prompt | **支持且推荐** | **不建议**(原样拼在 BOS 后,无分隔符) |
| `generation_config` 默认 | temperature **0.15** | temperature **0.6**,top_p **0.95** |
| 其他工具 | `mistral-common >= 1.5.2`;vLLM `--tokenizer_mode mistral` | vLLM `--tensor-parallel-size 2 --max-model-len 32768 --enforce-eager` |

### 8.2 Mistral 的 chat template(实际渲染结果)

训练样本(system + user + assistant):

```
<s>[SYSTEM_PROMPT]Instruction for the LLM[/SYSTEM_PROMPT][INST]### Claim: X
### Question: Q1
### Answer: A1[/INST]### Response: Verdict: True</s>
```

推理样本(`add_generation_prompt=True`):

```
<s>[SYSTEM_PROMPT]Instruction for the LLM[/SYSTEM_PROMPT][INST]### Claim: X
### Question: Q1
### Answer: A1[/INST]
```

要点:
- system 消息**只在 `messages[0]` 被消费**;放后面会变成对话中间的 `[SYSTEM_PROMPT]` 块。
- 不给 system 时,模板注入**含当天日期**的默认 prompt(`strftime_now`)→ 破坏 config_hash 的可复现性。**UI 必须把 system prompt 设为必填。**
- assistant 轮以 `</s>` 结束,这是模型学到的收尾。

### 8.3 DeepSeek 的 chat template(实际渲染结果)

训练样本:

```
<｜begin▁of▁sentence｜>Instruction for the LLM<｜User｜>### Claim: X
### Question: Q1
### Answer: A1<｜Assistant｜>### Response: Verdict: True<｜end▁of▁sentence｜>
```

推理样本(`add_generation_prompt=True`):

```
<｜begin▁of▁sentence｜>Instruction for the LLM<｜User｜>### Claim: X
### Question: Q1
### Answer: A1<｜Assistant｜><think>
```

要点:
- system 内容**原样拼在 BOS 之后,没有任何分隔标签** → 卡片因此建议不要用 system prompt,把所有指令放进 user prompt。
- 生成提示会强制插入 `<think>` 块(R1 的推理格式)。**下游解析必须容忍 verdict 之前有推理文本**;`rfind("### Response: Verdict: ")` 恰好能穿过,`startswith` 不行。
- 训练时若 assistant 内容含 `</think>`,模板**只保留最后一个 `</think>` 之后的文本**。当前标签不含 think,不受影响。
- `pad_token == eos_token == <｜end▁of▁sentence｜>`,**不要覆盖**。

---

## 9. 参考脚本的 API 调用与已知缺陷

参考脚本是**线性脚本,没有自定义函数**——全部逻辑写在模块顶层。这本身就是 M3.0 要修的问题(不可参数化、不可复用)。下面列出它们实际调用的 API,以及生成器必须避开的坑。

### 9.1 `fineTuningMistral_en.py` 的关键调用

```python
torch.cuda.empty_cache()
torch.distributed.init_process_group(rank=0)          # 张量并行前必须初始化
AutoModelForCausalLM.from_pretrained(modelId, dtype="auto",
    local_files_only=True, tp_plan="auto")
model.generation_config.temperature = 0.1
model.generation_config.do_sample = True
AutoTokenizer.from_pretrained(modelId, local_files_only=True, padding_side="left")
tokenizer.fix_mistral_regex = True                    # ← 位置错误,见 9.3
tokenizer.pad_token = tokenizer.eos_token
SFTConfig(output_dir=..., learning_rate=2e-4, weight_decay=0.01,
    num_train_epochs=2, warmup_steps=5, optim="adamw_8bit",
    per_device_train_batch_size=2, eval_strategy="epoch",
    logging_strategy="steps", logging_steps=1, save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
    push_to_hub=False)
LoraConfig(r=64, target_modules=[...7 个...], lora_alpha=16,
    lora_dropout=0.1, bias="none")
model.add_adapter(loraConfig)
Dataset.from_dict({"messages": [[{role,content}, ...], ...]})
SFTTrainer(model=model, args=trainingArguments,
    processing_class=tokenizer,
    train_dataset=trainingDataset, eval_dataset=evalDataset)
trainer.train(); trainer.evaluate()
torch.distributed.destroy_process_group()
trainer.save_model("Mistral-Small-24B-Instruct-2501-FineTuned")
```

命令行参数只有三个:`--trainDataFile`、`--evalDataFile`、`--temperature`(**后者解析后从未使用**)。启动方式:`torchrun --nproc_per_node=<N> fineTuningMistral_en.py ...`。

### 9.2 `inferenceMistral_en.py` 的关键调用

```python
AutoModelForCausalLM.from_pretrained(modelId, local_files_only=True,
    torch_dtype="auto", tp_plan="auto")               # 注意用的是 torch_dtype 而非 dtype
AutoTokenizer.from_pretrained(modelId, local_files_only=True, padding_side="left")
tokenizer.pad_token = "<myPadToken>"                  # ← 非法 token,见 9.3
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokenizer.encode(prompt, return_tensors="pt", padding=True).to("cuda")
model.eval()
model.generate(input_ids=..., use_cache=True, max_new_tokens=500,
    temperature=0.9, top_p=0.9, top_k=10, pad_token_id=tokenizer.pad_token_id)
tokenizer.decode(outputs[0], skip_special_tokens=True)
decodedOutput.rfind("### Response: Verdict: ")         # -1 → "No response"
torch.distributed.destroy_process_group()
```

路径全部**硬编码**(`modelId = "...-FineTuned"`,`testDataFile = "data/testData.json"`),无命令行参数。**只 print 判定,不计算任何指标**(虽然 import 了 sklearn 的四个指标函数)。

### 9.3 六个必须修掉的坑(生成器不得照抄)

1. **运算符优先级 bug**
   ```python
   userPromptStr += "\n### Question: " + q + "\n### Answer: " + doc[0] if doc else "No answer could be found !"
   ```
   `if/else` 作用于整个右侧表达式,因此 `top_k_doc` 为空时**连 Question 都不拼接**。正确写法是先判断再拼接。
2. **`tokenizer.pad_token = "<myPadToken>"`** —— 该字符串不在任何模型的词表里。Mistral 用 `<pad>`(id 11)或 `</s>`;DeepSeek 保持 EOS。
3. **`tokenizer.fix_mistral_regex = True` 写在加载之后无效** —— 修复逻辑发生在 `AutoTokenizer.from_pretrained()` 内部,必须作为参数传入;否则 Mistral 分词错误。
4. **训练集与验证集的 assistant 前缀不一致** —— 训练用 `"### Response: Verdict: "`,验证集写成 `"### Response: "`(漏了 `Verdict: `),会污染 `eval_loss` 与最佳模型选择。
5. **`max_new_tokens=500` 对 DeepSeek 偏小** —— 模板强制先输出 `<think>` 块。
6. **`temperature=0.9` 是卡片推荐值(0.15)的 6 倍** —— 不是 bug,但必须作为配置维度落库,否则 0.15 与 0.9 的实验不可比。

另外:脚本缺 `seed`(可复现性)、`SFTConfig` 未设 `max_seq_length`(静默截断)、数据加载两段代码完全重复(作者自己留了 `NOTE TO SELF :: DRY`)。

### 9.4 已知执行成本(用于给用户提示)

| 任务 | 集群 | 耗时 |
|---|---|---|
| DeepSeek-32B zero-shot 推理 | Narval | 9–12 小时 |
| Mistral-24B zero-shot 推理 | Fir | ~1 小时 |
| Mistral-24B 微调 | Fir | 6–7 小时 |
| Mistral-24B 微调后推理 | Fir | ~1 小时 |

空响应条数(研究侧的关键痛点):DeepSeek 最多 60、Mistral zero-shot 最多 55、Mistral 微调后最多 8;**研究目标是降到 0**。

---

## 10. M1 的交付物与验收标准

### 10.1 文件清单

```
app/main.py          FastAPI 入口 + 7 条路由
app/schemas.py       ConfigRecord 及各组件 Pydantic 模型 + 10 条校验规则
app/registry.py      两个模型的 ModelMeta 与默认值
app/store.py         SQLite:init/insert/get/list/duplicate/find_by_hash
app/static/index.html  单页表单:选模型 → 选任务 → 3 个 prompt → a–h → JSON 预览 → 保存
app/data/platform.db   运行后生成(git 忽略)
.gitignore           排除 *.db、__pycache__、模型权重、输出目录
README.md            启动方式、目录说明、已知限制
```

### 10.2 验收(自测清单)

- [ ] 选 Mistral + fine_tune → 表单自动预填 b/d/e/f 的默认值;`d` 分组为只读
- [ ] 选 inference → `e.add_generation_prompt` 自动变 `true`,`c` 只要求 `test_data_file`,组件 `d` 隐藏或标注不适用
- [ ] 清空 `p.system_prompt` 并选 Mistral → 表单报错(规则 9)
- [ ] 把 `b.pad_token` 改成 `<myPadToken>` → 报错(规则 3)
- [ ] 设 `max_length=30000`,`max_new_tokens=5000`(Mistral)→ 报错(规则 1)
- [ ] 设 `temperature=0` 且 `do_sample=true` → 报错(规则 5)
- [ ] 提交后能在 API 查到完整 JSON,字段结构与第 7.2 节一致
- [ ] 同一配置提交两次 → 第二次给出「已存在相同配置」提示(靠 `config_hash`)
- [ ] 复制配置 → 生成新 `config_id`,其余字段不变
- [ ] 重启服务后配置仍在

### 10.3 演示脚本(M0.5 用)

`GET /api/health` → `GET /api/models` → 页面上依次点通三步 → 展开 a–h → 点「预览 JSON」→ 展示生成的结构。**不需要数据库。**

---

## 11. 硬约束(不可协商)

1. **Fir 上不允许任何 Web 服务、cron、daemon 或常驻进程。** 平台必须跑在本地/校内机器,与 Fir 之间只能做**异步 push-pull**(推配置、拉日志)。部署前必须拿到导师书面授权。
2. **AI 生成的代码禁止直接复制粘贴。** 导师邮件与项目书均明文要求:逐行理解 + 同行评审后才能进主分支。
3. **本地/Colab 只用于调试**,不允许用全量数据在校园机器上微调。
4. **技术栈偏好**:Pydantic(数据校验)、MLflow(实验追踪)、R(可视化)是导师点名的方向。M1 阶段**只需 Pydantic**,MLflow/R 属于后续阶段的加分项,不要提前引入。
5. 使用 Maunzer 的校内机器前需先确认可行性(MAC 团队负责协调)。

---

## 12. 未决问题(交给新 session 时需向人类确认)

| # | 问题 | 影响 |
|---|---|---|
| 1 | `app/` 旧代码能否从 OneDrive 回收站找回? | 能找回则省 1–2 天;找不到按第 4 节重写 |
| 2 | 四人如何对应 A/B/C/D 工作流? | 决定谁写 main.py / schemas / registry / store |
| 3 | 数据集 train/eval/test 的实际路径与文件是否已提供? | 影响 M1.0 演示能否用真实数据 |
| 4 | 是否需要支持 DeepSeek 的完整表单(M1 只要求"选一个模型")? | 建议 registry 里先放全量元数据,UI 只放开 Mistral |
| 5 | `evidence_config` 的 7 种取值是否要在 M1 就全部实现生成逻辑? | M1 只需在表单和 schema 里能表达,M3.0 才真正生效 |

---

## 13. 参考文件索引

| 需要什么 | 看哪里 |
|---|---|
| 全部参数的类型/默认值/双模型取值 | `specs/parameters.md`(本文档第 5 节的展开版) |
| chat template 原文、token id、脚本缺陷分析 | `specs/model-reference.md`(第 8、9 节的展开版) |
| 参考脚本(带逐行英文注释) | `reference-scripts/fineTuningMistral_en.py`、`reference-scripts/inferenceMistral_en.py` |
| 检查点排期与任务拆分 | `planning/TaskList.md` / `planning/任务列表-中文.md` |
| 技术方案与 Sprint 计划(NiceGUI 路线) | `design/Spec_and_roadmap.md` / `design/Spec_and_roadmap.zh.md` |
| 验收标准原文 | `doc/Project Milestones - 4990.docx`(需解压读 `word/document2.xml`) |
| 研究背景与痛点 | `doc/Automated Fact-Checking with LLMs.pdf` |
| 项目定位与红线 | `doc/Group 15 project description.docx`、`doc/Email.md` |
| 文件夹总览 | 根目录 `README.md` |
