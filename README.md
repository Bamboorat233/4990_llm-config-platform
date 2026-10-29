# 4990_llm-config-platform

**Group 15 · University of Windsor 4990B**
**项目**:Integration of Large Language Models into Autonomous Fact-Checking Pipelines — Infrastructure and Technical Implementation
**团队**:Daniel Laurin, Kerui Li, Simbarashe Mamvura, Tiziana Nikundana(MAC 团队指导,Dr. Aznam Yacoub 主管)
**周期**:2026-10-25 → 2026-12-05

**检查点**:M0.5 `10/31` · M1.0 `11/7` · M1.5 `11/14` · M2.0 `11/21` · M2.5 `11/28` · M3.0 `12/5`(均为周六;0.5/1.5/2.5 是线上进度检查,1.0/2.0/3.0 是正式交付)

---

## 按顺序读

| 顺序 | 文件 | 为什么先看它 |
| --- | --- | --- |
| 1 | `doc/Project Milestones - 4990.docx` | 验收标准原文。三个交付节点的功能要求都在这里 |
| 2 | `planning/任务列表-中文.md`(= `planning/TaskList.md`) | 把验收标准拆成六个检查点的可勾选任务 |
| 3 | `specs/parameters.md` | 平台要暴露的每一个参数字段、类型、默认值、校验规则 |
| 4 | `specs/model-reference.md` | 两个模型的 chat template、特殊 token、参考脚本的缺陷 |
| 5 | `reference-scripts/*.py` | 博士生提供的参考训练/推理脚本(带逐行注释) |
| 6 | `design/Spec_and_roadmap.md` | 技术方案与 Sprint 计划(当前采用 NiceGUI 路线) |
| — | `planning/M1-context-handoff.md` | 给 M1 阶段的完整交接文档(自包含,可整份喂给 AI 助手) |

> 标了 `doc/` 的文件**不在本仓库中**,仅通过 OneDrive 在团队内共享(原因见下节)。

---

## 仓库范围

本仓库是 **public** 的,因此只收录可以公开的技术资产:

| 内容 | 是否入库 | 原因 |
| --- | --- | --- |
| `app/`(平台代码)、`README.md`、`.gitignore` | ✅ | 平台实现本身 |
| `specs/`(参数规格、模型事实) | ✅ | 基于公开的 HF Model Card 整理,无敏感信息 |
| `planning/`(任务清单、M1 交接文档) | ✅ | 团队内部计划,不含学号/邮件 |
| `design/`(技术方案与 Sprint 计划) | ✅ | 团队自研方案 |
| `doc/` | ❌ | 含**学号**、导师群发**邮件**原文、研究组**未发表**的汇报 PDF |
| `reference-scripts/` | ❌ | 博士生的研究代码,未获公开授权 |
| `data/`、模型权重、运行产物 | ❌ | 体积大或可再生成,见 `.gitignore` |

> 如果希望把 `doc/` 与 `reference-scripts/` 也纳入版本管理,更合适的做法是**先把仓库设为 private**,再删除 `.gitignore` 中对应的两行。公开仓库一旦泄露无法真正撤回(可能已被缓存或索引)。

---

## 目录结构

```plaintext
4990B/
├── README.md                    ← 本文件
├── doc/                         原始来源文档(导师/研究团队提供)
│   ├── Project Milestones - 4990.docx      验收标准(内部为 word/document2.xml)
│   ├── Automated Fact-Checking with LLMs.pdf  研究侧汇报
│   ├── Group 15 project description.docx/.pdf 立项书
│   ├── Email.md                 导师群发邮件(分工、红线、Fir 规则)
│   └── other/                   与项目无直接关系的课程文档(归档)
├── planning/                   计划与交接
│   ├── TaskList.md              任务清单(英文)
│   ├── 任务列表-中文.md          任务清单(中文,内容对等)
│   └── M1-context-handoff.md    Milestone 1 交接文档
├── specs/                      规格(单一事实源)
│   ├── parameters.md            参数规格:字段/类型/必填/默认值/10 条校验规则
│   └── model-reference.md       Model Card 事实、chat template 渲染、脚本缺陷
├── design/                     技术方案
│   ├── Spec_and_roadmap.md      英文版(较新)
│   └── Spec_and_roadmap.zh.md   中文版
├── reference-scripts/          博士生提供的参考脚本(只读,不要直接改)
│   ├── fineTuningMistral_en.py / _cn.py
│   └── inferenceMistral_en.py / _cn.py
└── app/                        ← 待创建。平台的实现代码放这里
```

`app/` 的预期结构见 `planning/M1-context-handoff.md` 第 4 节。

---

## 当前状态

- **代码尚未落地**。`app/` 曾在 10/28 存在(含 `main.py`、`schemas.py`、`store.py`、`registry.py`、`static/index.html`、`data/platform.db`),随后被删除且内容未留存。若 OneDrive 回收站可找回,优先复用。
- **尚未初始化 git 仓库** —— 这是 M0.5 的展示项之一,应尽快完成。
- **已到手的阻塞项**:六种场景的样例输出文件、样例执行脚本、数据集 train/eval/test 划分,均待博士生提供(分别阻塞 M1.5 与 M2.5,不影响 M1)。
- 本地环境:Python 3.13.3,已装 `pydantic 2.12.5`(FastAPI 侧 `0.136.1` 亦可用)。

## 待决策

**平台技术路线存在两份互斥方案**,需要先定一个再动手:

| 方案 | 出处 | 特点 |
| --- | --- | --- |
| **FastAPI + 静态 HTML** | `planning/M1-context-handoff.md` 第 4 节 | 前后端分离,后续 M2.0 出 JSON 接口顺理成章;承接被删雏形的形态 |
| **NiceGUI(单进程响应式)** | `design/Spec_and_roadmap.md` | 纯 Python 双向绑定,表单联动与错误高亮更省代码;无独立前端 |

其余待确认事项见 `planning/M1-context-handoff.md` 第 12 节。

## 协作约束(来自导师邮件与项目书)

- **Fir 上禁止任何 Web 服务、cron、daemon 或常驻进程**;平台只跑本地/校内,与 Fir 之间仅做异步 push-pull;部署前须取得导师书面授权。
- **AI 生成的代码禁止直接复制粘贴**,必须逐行理解并经过同行评审后合入。
- 本地/Colab 仅用于调试,不得在校园机器上用全量数据微调。
