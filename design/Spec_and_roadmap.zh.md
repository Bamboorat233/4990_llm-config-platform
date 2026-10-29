## **一、 技术规格书 (Technical Specification)**

### **1\. 系统架构与数据流**

摒弃传统的“前端发 HTTP 请求 \-\> 后端路由 \-\> 数据库”三层链路，改为**单进程响应式架构**：

Plaintext  
\[NiceGUI 界面组件 (Quasar/Vue)\]   
       ↕ (双向数据绑定 bind\_value / 事件回调 on\_change)  
\[Session State (Python 内存字典)\]  
       ↕ (实时转换 / 校验)  
\[Pydantic v2 Schema (ConfigRecord \+ 10条跨字段规则)\]  
       ↓ (通过校验后序列化)  
\[SQLite (store.py: 单表 JSON 持久化 \+ SHA256 哈希去重)\]

### **2\. 目录结构与模块职责**

Plaintext  
4990B/  
├── app/  
│   ├── \_\_init\_\_.py  
│   ├── main.py          \# NiceGUI 入口：页面布局、双向绑定、联动回调、错误高亮  
│   ├── registry.py      \# 模型事实库：ModelMeta 数据类、Mistral/DeepSeek 静态参数与默认配置生成  
│   ├── schemas.py       \# 核心数据层：Pydantic a-h/p 组件模型、ConfigRecord、指纹计算与 10 条校验规则  
│   └── store.py         \# 存储层：SQLite 封装（初始化、增删查、按 Hash 查重、克隆配置）  
├── data/  
│   └── platform.db      \# 运行时自动生成的 SQLite 库（列入 .gitignore）  
├── requirements.txt     \# 仅需 nicegui\>=2.0, pydantic\>=2.10  
└── README.md

### **3\. 核心模块规格定义**

#### **3.1 app/registry.py（模型注册表与默认值引擎）**

* **数据结构**：使用 @dataclass(frozen=True) 定义 ModelMeta，严格收录 Mistral-Small-24B-Instruct-2501 和 DeepSeek-R1-Distill-Qwen-32B 的词表、特殊 Token ID、上下文上限及推荐采样参数。  
* **核心函数**：  
  * get\_model(model\_id: str) \-\> ModelMeta  
  * defaults\_for(model\_id: str, task\_type: Literal\["fine\_tune", "inference"\]) \-\> dict：当用户在 UI 切换模型或任务类型时调用，返回完整的 a–h 及 p 字典。例如：  
    * 选 Mistral \+ fine\_tune：b.pad\_token="\<pad\>", b.pad\_token\_id=11, e.add\_generation\_prompt=False, f.temperature=0.15。  
    * 选 DeepSeek \+ inference：b.pad\_token="\<｜end▁of▁sentence｜\>", b.pad\_token\_id=151643, e.add\_generation\_prompt=True, f.temperature=0.6, f.max\_new\_tokens=1024, f.forced\_prefix="\\n\\n"。

#### **3.2 app/schemas.py（Pydantic 模型与交叉校验）**

* **组件模型**：严格按照交接文档第 4.3 与第 5 节定义 ModelSection (a), TokenizerSection (b), DataSection (c), OptimizerSection (d), EncodingSection (e), GenerationSection (f), DecodingSection (g), PromptSection (p) 及顶层 ConfigRecord。  
* **指纹算法 config\_fingerprint(cfg: ConfigRecord) \-\> str**：  
  * 导出字典时排除 config\_id, config\_hash, run\_status, created\_at, created\_by, notes。  
  * 使用 json.dumps(data, sort\_keys=True) 序列化后计算 sha256。  
* **交叉校验器 validate\_cross\_fields(cfg: ConfigRecord) \-\> list\[ValidationIssue\]**：  
  * 返回结构化列表：\[{"field": "b.pad\_token", "level": "error" \| "warning", "msg": "..."}\]，供 NiceGUI 直接将对应输入框标红。  
  * **10 条规则全覆盖**：  
    1. (e.max\_length or 0\) \+ f.max\_new\_tokens \<= effective\_max\_seq\_len（MS: 32768, DS: 32768/16384）。  
    2. task\_type \== "inference" 且 f.batch\_size \> 1 时，强制 b.padding\_side \== "left"。  
    3. b.pad\_token 白名单校验（MS 仅允许 \<pad\> 或 \</s\>；DS 仅允许 \<｜end▁of▁sentence｜\>，拦截 \<myPadToken\>）。  
    4. MS 模型强制 b.fix\_mistral\_regex \== True。  
    5. f.temperature \== 0 要求 f.do\_sample \== False；f.temperature \> 0 要求 f.do\_sample \== True。  
    6. task\_type \== "fine\_tune" 且 d.sft.load\_best\_model\_at\_end \== True 时，要求 eval\_strategy \== save\_strategy 且 c.eval\_data\_file 非空。  
    7. 微调与下游推理的 p.assistant\_prompt 与 p.verdict\_parse\_marker 一致性检查。  
    8. a.tp\_plan 与 a.device\_map 互斥检查（不可同时非空）。  
    9. MS 模型下 p.system\_prompt 为空时返回 error（防止注入含当天日期的默认模板）。  
    10. DS 模型下 p.system\_prompt 非空时返回 warning（提示会无分隔符拼接到 BOS 后）。

#### **3.3 app/store.py（SQLite 持久化层）**

* **表结构设计**：  
  SQL  
  CREATE TABLE IF NOT EXISTS configs (  
      config\_id TEXT PRIMARY KEY,  
      config\_hash TEXT UNIQUE NOT NULL,  
      model\_id TEXT NOT NULL,  
      task\_type TEXT NOT NULL,  
      run\_status TEXT NOT NULL DEFAULT 'pending',  
      created\_at TEXT NOT NULL,  
      created\_by TEXT,  
      notes TEXT,  
      payload\_json TEXT NOT NULL  
  );

* **接口签名**：  
  * init\_db() \-\> None  
  * find\_by\_hash(config\_hash: str) \-\> str \| None（若存在返回已有的 config\_id）  
  * insert\_config(cfg: ConfigRecord) \-\> str  
  * list\_configs() \-\> list\[dict\]（按 created\_at DESC 排序，供左侧抽屉列表展示）  
  * get\_config(config\_id: str) \-\> ConfigRecord \| None  
  * duplicate\_config(config\_id: str) \-\> ConfigRecord（生成新 UUID 与当前时间戳，供二次修改）

#### **3.4 app/main.py（NiceGUI 界面与交互规范）**

* **整体布局**：  
  * **Header**：项目标题 \+ 环境状态指示灯 \+ 侧边栏折叠按钮。  
  * **Left Drawer（左侧栏）**：已保存的配置列表（卡片流展示 model\_id 简写、task\_type 徽章、config\_id 前 8 位、创建时间），支持点击加载与一键“Duplicate”。  
  * **Main Split（主工作区，左右分栏 7:5）**：  
    * **左栏（表单配置区）**：  
      1. 顶层控制卡：model\_id (ui.select)、task\_type (ui.toggle)、created\_by、notes。  
      2. Prompt 配置卡 (p)：system\_prompt (ui.textarea)、user\_prompt\_template、assistant\_prompt。  
      3. 手风琴折叠面板组 (ui.expansion)：依次承载组件 a 到 h。  
    * **右栏（实时预览与校验台，Sticky 固定跟随）**：  
      1. 校验状态面板：实时显示通过状态，或以列表展示当前触发的 error（红）与 warning（黄）及具体字段路径（如 b.pad\_token）。  
      2. JSON 实时预览器：使用 ui.code(language='json') 实时渲染当前表单序列化后的完整 JSON。  
      3. 操作按钮组：Validate（手动触发深度检查）、Save to DB（存在 error 时自动 disable）、Reset to Defaults。  
* **关键 UI 联动策略**：  
  * **组件 d (Optimizer)**：所有输入框全部设置 .props('readonly bg-grey-2')；且外层容器绑定 .bind\_visibility\_from(state, 'task\_type', value='fine\_tune')（选 inference 时自动隐藏或折叠标注不适用）。  
  * **组件 c (Data)**：train\_data\_file 和 eval\_data\_file 仅在 fine\_tune 下可见/必填；test\_data\_file 和 evidence\_config 仅在 inference 下可见/必填。  
  * **组件 h (\*\*kwargs)**：为 load\_model、load\_tokenizer、encode、generate 提供动态增删键值对行的子面板，并在后端用 inspect.signature 或预设白名单校验合法键名。

## **二、 技术路径与分步实施计划 (Implementation Roadmap)**

考虑到当前是 **10月29日**，距离 **10月31日（周六）M0.5 检查** 只有 2 天，距离 **11月7日 M1.0 正式交付** 有 9 天，我们将技术路径切分为 **4 个渐进式 Sprint**：

### **Sprint 1：工程基建与数据内核（10月29日 当天完成）**

**目标**：不写任何 UI 代码，先让数据模型、默认值字典和 10 条校验规则在命令行下 100% 跑通。

> 1. **初始化仓库**：  
   * 在 4990B 根目录执行 git init，创建 .gitignore（排除 data/\*.db, \_\_pycache\_\_/, .venv/）。  
   * 安装依赖：pip install nicegui pydantic。  
> 2. **实现 app/registry.py**：  
   * 写入 MODELS 字典与 defaults\_for() 函数。  
> 3. **实现 app/schemas.py**：  
   * 编写 a–h、p 及 ConfigRecord Pydantic 模型。  
   * 编写 validate\_cross\_fields()，覆盖全部 10 条校验规则。  
> 4. **内核冒烟测试**：  
   * 用一行命令测试：python \-c "from app.registry import defaults\_for; from app.schemas import ConfigRecord; print(ConfigRecord(\*\*defaults\_for('mistralai/Mistral-Small-24B-Instruct-2501', 'fine\_tune')).model\_dump\_json(indent=2))"，确认输出的 JSON 与交接文档 §7.2 完全一致。

### **Sprint 2：NiceGUI 响应式表单与 M0.5 交付（10月30日 完成）**

**目标**：完成 app/main.py 界面搭建，实现 M0.5 要求的全部交互演示功能。

> 1. **搭建状态容器**：  
   * 在 main.py 中维护一个响应式配置字典 current\_state，初始加载 Mistral \+ fine\_tune 的默认值。  
> 2. **编写表单渲染函数**：  
   * 按顶层选择器 \-\> Prompt 区 \-\> a–h 折叠面板的顺序编写 NiceGUI 组件。  
   * 绑定 model\_id 和 task\_type 的 on\_change 事件：一旦切换，调用 defaults\_for() 刷新状态，并调用 refresh\_ui() 更新界面及 JSON 预览器。  
> 3. **接入实时校验与字段标红**：  
   * 每次表单变动时，构建 ConfigRecord 并运行 validate\_cross\_fields()。  
   * 将所有控件引用存入字典 field\_widgets\["b.pad\_token"\] \= input\_widget；当某字段报错时，直接调用 widget.props('error error-message="具体原因"')，实现精准到字段级的红框报错提示。  
> 4. **M0.5 演练（10月31日）**：  
   * 启动服务 python \-m app.main（默认运行在 http\://localhost:8080）。  
   * 按 §10.3 演示脚本逐项点击：切换模型 \-\> 切换微调/推理（展示组件 d 只读与隐藏、组件 e 自动切换）-\> 故意触发校验错误（如清空 Mistral 的 system\_prompt）-\> 展示右侧实时生成的完整 JSON。

### **Sprint 3：SQLite 落库、去重与配置管理（11月1日 – 11月3日）**

**目标**：实现 app/store.py 并与 NiceGUI 左侧抽屉打通，满足 M1.0 落库要求。

> 1. **实现 app/store.py**：  
   * 使用 Python 内置 sqlite3 实现建表、插入、按 config\_hash 查重、列表查询与克隆逻辑。  
> 2. **打通“保存”与“去重”工作流**：  
   * 点击 Save to DB 时，先计算 config\_hash 并调用 find\_by\_hash()。  
   * 若存在重复 Hash，弹出 ui.notify("已存在完全相同的实验配置 (ID: xxx)", type="warning") 并阻止重复入库。  
   * 若为新配置，写入 data/platform.db，弹出成功提示并刷新左侧配置列表\[cite: 1\]。  
> 3. **打通“读取”与“复制（Duplicate）”**：  
   * 点击左侧列表某条配置：将其 payload\_json 反序列化并回填至主表单\[cite: 1\]。  
   * 点击 Duplicate：调用 store.duplicate\_config() 生成新 config\_id 加载到表单，方便研究者微调单个超参数（例如将 temperature 从 0.15 改为 0.9）后另存为新实验\[cite: 1\]。

### **Sprint 4：验收清单自测与文档收尾（11月4日 – 11月6日）**

**目标**：逐条勾掉交接文档 §10.2 的 10 项自测清单\[cite: 1\]。

> 1. **编写自动化测试脚本 test\_m1\_checklist.py**：  
   * 用代码直接断言 §10.2 的全部边界情况（如 \<myPadToken\> 拦截、30000+5000 长度超限拦截、temperature=0 \+ do\_sample=True 拦截、数据库重启后持久化验证等）\[cite: 1\]。  
> 2. **完善 README.md**：  
   * 记录启动步骤、模块架构图、6 个已修复的原始脚本缺陷说明\[cite: 1\]。

我们可以现在直接进入 **Sprint 1**。需要我现在为你编写 app/registry.py 和 app/schemas.py 的完整代码吗？