# Deep Research Agent 设计文档

## 概述

一个 Web 端的深度调研 AI Agent，用户输入问题后，agent 通过 ReAct 循环自主完成搜索、阅读、分析、交叉验证，最终生成带来源引用的调研报告。全程不依赖 LangChain、LlamaIndex 等主流 agent 框架，核心模块自研。

**技术栈**：Python + FastAPI + Vue 3 + SQLite

---

## 一、整体架构

```
用户 (Browser) ←→ FastAPI (SSE) ←→ Agent Engine (自研) ←→ LLM API / Search API
```

- 前端通过 SSE 实时接收 agent 每一步的思考、行动和观察
- 后端核心为自研 Agent Engine，包含 5 个模块
- 外部依赖仅 LLM API 和 Search API，均以接口形式接入

---

## 二、核心模块

### 1. ReAct Loop（循环引擎）

Thought → Action → Observation 循环，直到 LLM 输出 FINAL_ANSWER 或达到最大轮数（10 轮）。

每轮将上下文发给 LLM，要求按固定格式输出：

```
THOUGHT: <分析当前已知，判断是否足够，决定下一步>
ACTION: <工具名>|<参数JSON>
```

执行工具后将 Observation 拼回上下文继续下一轮。

### 2. Tool Registry（工具注册中心）

```python
class Tool:
    name: str
    description: str
    parameters: dict       # JSON Schema
    handler: Callable
```

内置工具：
- **web_search** — 搜索关键词，返回标题+摘要+URL
- **fetch_page** — 抓取 URL 正文（HTML→纯文本）
- **cross_check** — 对已收集事实调用 LLM 交叉验证

### 3. Memory Manager（记忆管理）

- 保存完整 Action-Observation 历史
- 窗口超限时压缩旧轮次为摘要注入
- 维护「关键事实清单」防止信息丢失

### 4. Output Parser（输出解析器）

从 LLM 原始输出中解析 THOUGHT / ACTION / FINAL_ANSWER 结构化块，处理格式偏差、JSON 不合法等异常。

### 5. Research Orchestrator（调研编排器）

在标准 ReAct 基础上加入调研专用逻辑：
- 开局生成调研计划，将问题拆为 2-4 个子问题
- 每个子问题至少需要 1 个可信来源
- 回答前强制执行 1 次 cross_check

---

## 三、API 与数据流

### SSE 事件流

```
POST /api/research  { "query": "..." }

event: plan         → {"sub_questions": [...]}
event: step         → {"type": "thought|action|observation", "round": N, ...}
event: cross_check  → {"consistency": "high|medium|low", "conflicts": [...], "verified_facts": [...]}
event: report       → {"markdown": "...", "sources": [...]}
event: done
```

### REST 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/research` | 发起调研，返回 SSE 流 |
| GET | `/api/history` | 获取历史调研记录列表 |
| GET | `/api/history/{id}` | 获取某次调研完整步骤和报告 |

### 前端状态机

```
空闲 → [输入问题] → 调研中(实时步骤展示) → 完成(报告渲染) 或 出错(展示已完成步骤)
```

- 左侧：对话面板 + 最终报告
- 右侧：步骤卡片流（不同颜色区分 Thought/Action/Observation）
- 报告区：Markdown 渲染 + 来源引用列表

---

## 四、技术栈

| 层 | 选型 |
|----|------|
| 后端框架 | FastAPI |
| 前端 | Vue 3 |
| LLM API | OpenAI 兼容接口（设计为可替换） |
| 搜索 API | SerpAPI 或 Bing Search API |
| 网页抓取 | httpx + beautifulsoup4 |
| 数据库 | SQLite + aiosqlite |
| 前端 Markdown | marked.js |

### YAGNI（不做）

- 用户认证/登录
- 多模型同时对比
- 任务队列/后台执行
- PDF 导出

---

## 五、项目结构

```
deep-research-agent/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── agent/
│   │   ├── loop.py
│   │   ├── parser.py
│   │   ├── memory.py
│   │   ├── orchestrator.py
│   │   └── tools/
│   │       ├── registry.py
│   │       ├── web_search.py
│   │       ├── fetch_page.py
│   │       └── cross_check.py
│   ├── llm/
│   │   └── client.py
│   ├── models/
│   │   └── schemas.py
│   ├── routes/
│   │   └── research.py
│   └── db/
│       ├── connection.py
│       └── repository.py
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── ChatPanel.vue
│       │   ├── StepTimeline.vue
│       │   └── ReportView.vue
│       ├── composables/
│       │   └── useSSE.js
│       ├── App.vue
│       └── main.js
├── tests/
│   ├── test_loop.py
│   ├── test_parser.py
│   └── test_tools.py
└── docs/
```

---

## 六、测试策略

| 层级 | 内容 | 工具 |
|------|------|------|
| 单元测试 | parser（多种 LLM 输出格式）、memory（压缩+事实保留）、registry（注册+执行） | pytest |
| 集成测试 | loop（mock LLM 和 search，验证终止条件） | pytest + mock |
| E2E | 完整调研流程（手动，需 API key） | 手动 |

---

## 七、容错设计

| 场景 | 处理 |
|------|------|
| LLM 输出格式无法解析 | 让 LLM 重试一次（附格式提示） |
| Search API 超时/限流 | 告知 LLM 搜索不可用，基于已有信息继续 |
| 网页抓取失败 | 降级为搜索摘要，标注"未能获取全文" |
| LLM 持续不终止 | 硬上限 10 轮后强制生成摘要 |
| 上下文超限 | 压缩前 N 轮为摘要 |
| 用户断开 SSE | 已完成步骤保留到历史记录 |

## 八、前端状态覆盖

| 状态 | UI |
|------|-----|
| 空闲 | 输入框 + 示例问题卡片 |
| 调研中 | 步骤面板实时刷新，loading 指示器 |
| 完成 | 左侧 Markdown 报告，右侧完整步骤回顾 |
| 出错 | 错误提示 + 展示已完成步骤 |
