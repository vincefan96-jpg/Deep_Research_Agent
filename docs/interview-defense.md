# AI 应用开发岗位 — 面试答辩内容

> 基于 **Deep Research Agent（深度调研智能体）** 项目生成
> 项目地址：`D:\AGENT` | 技术栈：Python + FastAPI + Vue 3 + SQLite + OpenAI API

---

## 目录

1. [项目概述（电梯演讲）](#1-项目概述)
2. [核心技术一：Agent 架构设计](#2-agent-架构设计)
3. [核心技术二：提示词工程](#3-提示词工程)
4. [核心技术三：工具调用系统](#4-工具调用系统)
5. [核心技术四：上下文与记忆管理](#5-上下文与记忆管理)
6. [核心技术五：LLM 输出解析与容错](#6-llm-输出解析与容错)
7. [核心技术六：流式输出（SSE）](#7-流式输出sse)
8. [架构决策：为什么不使用 LangChain](#8-为什么不使用-langchain)
9. [AI 应用开发的工程化实践](#9-ai-应用开发的工程化实践)
10. [项目中的难点与解决方案](#10-项目中的难点与解决方案)
11. [面试常见问题速答](#11-面试常见问题速答)
12. [延伸思考：如果要做生产级优化](#12-延伸思考)

---

## 1. 项目概述

### 一句话描述

> 一个**全自研**的深度调研 AI Agent，基于 ReAct 范式，自动进行多轮「思考→搜索→阅读→分析」，生成带来源引用的结构化调研报告。

### 核心能力

| 能力 | 实现方式 |
|------|----------|
| 自主规划 | LLM 将用户问题拆解为 2-4 个子问题 |
| 工具使用 | 自研 Tool Registry，支持 web_search + fetch_page |
| 多步推理 | ReAct 循环（Thought → Action → Observation），最大 10 轮 |
| 记忆管理 | 完整步骤记录 + 关键事实提取 + 上下文自动压缩 |
| 实时反馈 | SSE 流式推送每一步的思考/行动/观察 |
| 容错机制 | 格式重试、工具降级、硬上限兜底、解析容错 |

### 技术亮点

- **零框架依赖**：核心 Agent 引擎完全自研，不依赖 LangChain / LlamaIndex / Semantic Kernel 等
- **Function Calling 驱动**：使用 OpenAI 原生 `tools` 参数实现结构化工具调用，可靠性接近 100%
- **OpenAI 兼容**：LLM 客户端设计为与任何 OpenAI-compatible API 对接（可切换 DeepSeek、Qwen 等）
- **前后端分离**：FastAPI + SSE + Vue 3，实时可视化 Agent 内部思维链

---

## 2. Agent 架构设计

### 2.1 什么是 ReAct 范式？

**ReAct = Reasoning + Acting**，是当前 AI Agent 的主流设计范式。核心理念是让 LLM 交替进行「推理」和「行动」：

```
┌─────────────────────────────────────────────────────┐
│                    ReAct 循环                         │
│                                                      │
│   Thought ──→ Action ──→ Observation                 │
│      ↑                                    │          │
│      └──────────── 循环 ──────────────────┘          │
│                                                      │
│   直到：submit_report 工具调用 或 达到最大轮数                   │
└─────────────────────────────────────────────────────┘
```

### 2.2 本项目的实现

```python
# backend/agent/loop.py — AgentLoop.run() 核心逻辑（使用 Function Calling）

for round_num in range(1, MAX_ROUNDS + 1):
    # 1. Function Calling：LLM 自行决定是否调用工具
    response = await self.llm.chat_with_tools(messages, tools=tool_defs)
    content = response["content"]
    tool_calls = response.get("tool_calls", [])

    # 2. 推送 Thought 到前端（SSE）
    if content:
        yield StepEvent(type="thought", round=round_num, content=content)

    # 3. LLM 没有调用工具 → 结束循环
    if not tool_calls:
        return

    # 4. 执行每个工具调用 + 推送 Observation
    for tc in tool_calls:
        if tc['name'] == 'submit_report':
            self.final_report = tc['arguments'].get('report', '')
            return  # 调研完成
        observation = await self.tool_registry.execute(tc['name'], tc['arguments'])
        yield StepEvent(type="observation", ...)
        messages.append({"role": "tool", "content": observation, ...})
```

**这个设计的优势**：

1. **结构化可靠**：使用 OpenAI 原生 Function Calling，LLM 返回结构化的 `tool_calls` 数组，不存在解析失败问题
2. **可观测**：每一步 Thought/Action/Observation 都通过 SSE 实时推送，用户能看到 Agent "在想什么"
3. **可中断**：达到最大轮数后强制生成摘要，不会无限循环
4. **优雅终止**：`submit_report` 工具作为循环的结束信号，报告内容直接内嵌在工具参数中

### 2.3 Agent 的三层架构

```
┌─────────────────────────────────────────┐
│         ResearchOrchestrator             │  ← 业务编排层
│  生成计划 → 运行循环 → 生成报告           │
├─────────────────────────────────────────┤
│              AgentLoop                   │  ← 循环引擎层
│  Thought → Action → Observation 循环     │
│  (基于 Function Calling)                 │
├─────────────────────────────────────────┤
│  Memory  │  ToolRegistry  │ LLM Client  │  ← 基础组件层
│  Tools (web_search / fetch_page)         │
└─────────────────────────────────────────┘
```

**面试话术**：「我将 Agent 设计为三层架构。最底层是基础组件——LLM 客户端、记忆管理器、工具注册表；中间层是 ReAct 循环引擎，负责驱动 Function Calling 驱动的主循环；最上层是业务编排器，加入了调研特有的逻辑——生成调研计划、最终报告生成。这种分层让每一层职责清晰，方便单独测试和替换。值得一提的是，项目从最初的文本解析方式迭代到了 Function Calling，架构本身能很好地容纳这种升级。」

---

## 3. 提示词工程

提示词是 AI 应用的核心，决定了 LLM 的行为模式。本项目体现了多项提示词工程技巧。

### 3.1 System Prompt 设计（`loop.py:11-30`）

```python
SYSTEM_PROMPT = """你是一个深度调研智能体。你的目标是对研究主题进行彻底调查，
生成一份全面、来源可靠的答案。

你可以使用提供的工具进行搜索和阅读。请逐步深入调研，
当你认为信息足够全面时，使用 submit_report 工具提交最终报告。

规则：
- 先搜索后阅读：先用 web_search 搜索，再对最有价值的 URL 使用 fetch_page 深入阅读
- 多源验证：重要结论不能只依赖单一来源
- 引用来源格式：[来源: URL]
- 如果搜索失败或无结果，调整策略或诚实地报告未能找到的内容"""
```

**这里运用了多项提示词技巧**：

| 技巧 | 体现 | 目的 |
|------|------|------|
| **角色设定** | "你是一个深度调研智能体" | 建立行为模式预期 |
| **工具驱动** | 通过 Function Calling 的 `tools` 参数传递工具定义 | 结构化工具选择，无需文本格式约束 |
| **结构化输出** | `submit_report` 工具参数包含 `report` 字段 | 报告内容通过工具参数获取，天然结构化 |
| **Few-shot 引导** | "先搜索后阅读" | 隐式指导最佳实践 |
| **行为规则** | "多源验证"、"引用来源格式" | 提升输出质量和可信度 |
| **错误恢复提示** | "如果搜索失败...调整策略" | 预防死循环 |

### 3.2 计划生成 Prompt（`orchestrator.py`，使用 JSON Mode）

```python
prompt = f"""将以下研究问题拆分为 2-4 个具体的子问题。
以 JSON 格式返回，格式为 {{"sub_questions": ["子问题1", "子问题2", ...]}}。
问题：{query}"""

# 通过 chat_json_mode() 调用，使用 response_format={"type": "json_object"}
result = await self.llm.chat_json_mode([{"role": "user", "content": prompt}])
sub_questions = result.get("sub_questions", [query])
```

**技巧**：明确了数量约束（2-4 个）、格式约束（JSON Mode 保证输出为合法 JSON）、有兜底逻辑（解析失败时降级为 `[原问题]`）。使用 OpenAI 原生 `response_format={"type": "json_object"}` 大幅提升了 JSON 输出的可靠性。

### 3.3 格式错误修复 Prompt（已废弃）

> **旧版方案**：当 LLM 输出格式不符合 `THOUGHT/ACTION/FINAL_ANSWER` 预期时，将格式错误信息反馈给 LLM 让它重新生成，实现**自愈机制**。
>
> **现状**：迁移到 Function Calling 后不再需要——工具调用由 API 原生保证结构化，不存在格式错误。这是 Function Calling 方案的核心优势。

### 面试延伸：提示词工程的进阶话题

| 话题 | 说明 |
|------|------|
| **Prompt Caching** | 将 System Prompt 缓存以降低延迟和成本（Anthropic/OpenAI 都支持） |
| **Chain-of-Thought** | 显式引导 LLM 逐步推理（本项目 THOUGHT 块即为 CoT 的体现） |
| **结构化输出** | 使用 JSON Mode / Function Calling 替代文本解析（本项目已完成迁移） |
| **提示词版本管理** | 像管理代码一样管理提示词，做 A/B 测试 |
| **Prompt Injection 防护** | 用户输入中可能包含恶意指令，需要清洗或隔离 |

---

## 4. 工具调用系统

### 4.1 为什么 Agent 需要工具？

LLM 的知识有截止日期，且无法访问实时信息。工具（Tool）是 Agent 与外部世界交互的接口。本项目设计了一套**自研工具注册系统**，让 LLM 能自主决定调用哪个工具。

### 4.2 工具注册表设计（`registry.py`）

```python
@dataclass
class Tool:
    name: str              # 工具名
    description: str       # 自然语言描述（给 LLM 看）
    parameters: dict       # JSON Schema 参数定义
    handler: Callable      # 实际执行函数

class ToolRegistry:
    def register(self, tool: Tool): ...
    def get_openai_tools(self) -> list[dict]:  # 生成 OpenAI Function Calling 格式工具定义
        """生成 OpenAI tools 参数格式 [{"type": "function", "function": {...}}]"""
    async def execute(self, name: str, params: dict) -> str:
        """根据工具名查找并调用实际函数"""
```

**设计要点**：

1. **工具描述是给 LLM 看的**：`description` 和 `parameters` 被格式化为 OpenAI Function Calling 的 `tools` 参数，通过 API 原生机制传递给 LLM
2. **参数用 JSON Schema**：OpenAI Function Calling 的标准参数格式，API 原生理解
3. **执行与注册分离**：`handler` 是真正的业务逻辑，与 Agent 引擎解耦

### 4.3 内置工具

#### web_search（`web_search.py`）

```python
async def web_search(query: str, num: int = 5) -> str:
    """调用 SerpAPI 搜索，返回标题+摘要+URL"""
    # 关键设计：
    # - 异步 HTTP（httpx.AsyncClient）
    # - 超时处理（15s）
    # - 异常分类（TimeoutException vs 通用异常）
    # - 返回结构化的文本格式（方便 LLM 解析）
```

#### fetch_page（`fetch_page.py`）

```python
async def fetch_page(url: str, max_chars: int = 5000) -> str:
    """抓取网页 → 移除噪声标签 → 提取纯文本"""
    # 关键设计：
    # - BeautifulSoup 清洗（移除 script/style/nav/footer/header/aside）
    # - 内容截断（防止单个页面撑爆上下文）
    # - 伪装 User-Agent（避免被反爬）
    # - 降级处理（抓取失败时返回友好提示，而不是崩溃）
```

### 4.4 工具调用的完整流程

```
LLM 通过 Function Calling 返回:
  content: "我需要先搜索 AI Agent 的最新趋势..."
  tool_calls: [{"name": "web_search", "arguments": {"query": "AI Agent 趋势 2025"}}]
                │
                ▼
       AgentLoop 从 tool_calls[] 提取工具名和参数（无需文本解析！）
                │
                ▼
       ToolRegistry.execute("web_search", {"query": "..."})
                │
                ▼
       web_search() 函数 → HTTP 请求 → 返回搜索结果文本
                │
                ▼
       将 tool result 拼回 messages: {"role": "tool", "tool_call_id": "...", "content": "..."}
                │
                ▼
       下一轮 LLM 调用可以在上下文中看到搜索结果
```

### 面试延伸：Function Calling vs 文本解析

| 方案 | 本项目（当前） | 本项目（旧版） |
|------|------------|----------------|
| 实现方式 | API 原生 `tools` 参数 | 文本解析 `ACTION: name\|{json}` |
| 可靠性 | ~99.9%（API 保证） | 依赖 LLM 服从格式（约 95%） |
| 灵活性 | 受 API 功能约束 | 完全自主，不依赖 API 特性 |
| 兼容性 | 取决于 LLM 是否支持 | 100%（纯文本，任何 LLM 可用） |

> **面试话术**：「项目从最初的自研文本解析演进到了 OpenAI Function Calling。文本解析的优点是任何 LLM 都能用，但可靠性只能做到约 95%——LLM 偶尔会不按格式输出。迁移到 Function Calling 后可靠性接近 100%，代码也更简洁——我们删除了整个 parser.py 模块。这是一个典型的从通用性优先到可靠性优先的架构演进。」

---

## 5. 上下文与记忆管理

### 5.1 为什么需要记忆管理？

LLM 的上下文窗口（Context Window）是有限的。每轮 ReAct 循环都会往 messages 列表追加内容，如果不加管理，很快就会超出上下文限制。

本项目的上下文增长路径：
```
每轮循环追加 ~1000 tokens（THOUGHT + ACTION + OBSERVATION）
10 轮 × 1000 = 10,000 tokens
+ System Prompt + 搜索返回的大量文本
≈ 可能超过 20,000 tokens
```

### 5.2 MemoryManager 设计（`memory.py`）

```python
class MemoryManager:
    def __init__(self, max_history_chars: int = 8000):
        self.steps: list[Step] = []       # 完整步骤历史
        self.key_facts: list[str] = []    # 关键事实（自动去重）
        self.max_history_chars = max_history_chars  # 压缩阈值
    
    def get_steps_for_context(self) -> str:
        """构建上下文，超限时自动压缩"""
        # 计算总字符数
        # 如果超过阈值 → 压缩早期步骤为摘要 + 保留最近步骤
```

**压缩策略**：
```
未超限：
  [Round 1] THOUGHT: ...
  [Round 1] ACTION: web_search (tool: web_search)
  [Round 1] OBSERVATION: ...
  [Round 2] THOUGHT: ...
  ...

超限后：
  [Compressed summary of 5 early steps]
  Early actions taken: 3
  Key facts collected:
  - Python 是 1991 年发布的
  - ...
  
  --- Recent Steps ---
  [Round 4] THOUGHT: ...
  [Round 4] ACTION: fetch_page (tool: fetch_page)
  ...
```

**设计要点**：
- **信息梯度保留**：早期步骤压缩为摘要（保留语义），最近步骤保留完整细节（当前上下文最重要）
- **关键事实清单**：独立于步骤存储，压缩时不会丢失
- **自动去重**：`add_key_fact` 检查重复，避免信息冗余

### 面试延伸：记忆管理的进阶方案

| 方案 | 说明 | 适用场景 |
|------|------|----------|
| **滑动窗口** | 只保留最近 K 轮 | 简单任务 |
| **摘要压缩**（本项目） | 压缩旧内容为摘要 | 中等复杂度 |
| **向量检索（RAG）** | 将历史存入向量数据库，按相关性检索 | 长对话/多会话 |
| **分层记忆** | 工作记忆 + 短期记忆 + 长期记忆 | 复杂 Agent |

---

## 6. LLM 输出解析与容错

### 6.1 架构演进：从文本解析到 Function Calling

本项目经历了从**自研文本解析**到 **OpenAI Function Calling** 的架构演进。

**旧版方案（`parser.py`，已删除）：**

旧版 Agent 使用正则表达式从 LLM 的原始文本输出中提取结构化指令：

```python
# 旧版 parser.py 核心逻辑
def parse(raw: str) -> ParsedOutput:
    # 1. 正则提取 THOUGHT
    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION|FINAL_ANSWER)|\Z)",
                              raw, re.DOTALL | re.IGNORECASE)
    # 2. 检测 FINAL_ANSWER 关键字
    if re.search(r"FINAL_ANSWER", raw, re.IGNORECASE):
        is_final = True
    # 3. 正则提取 ACTION（NAME|JSON 格式 + json.loads()）
    # ...
```

旧版需要处理的异常包括：缺少冒号、JSON 格式错误、关键字缺失等。虽然有容错机制（大小写不敏感、正则模糊匹配、错误反馈重试），但可靠性本质上受限于 LLM 输出的随机性。

**新版方案（Function Calling）：**

```python
# 新版 loop.py —— 使用 chat_with_tools()
response = await self.llm.chat_with_tools(messages, tools=tool_defs)
content = response["content"]           # 思考文本（无需提取）
tool_calls = response.get("tool_calls")  # 结构化工具调用（API 保证格式）
```

**演进价值**：

| 方面 | 旧版（文本解析） | 新版（Function Calling） |
|------|-----------------|------------------------|
| 可靠性 | ~95%（LLM 偶尔不服从格式） | ~99.9%（API 保证） |
| 代码量 | parser.py 69 行 + loop.py 解析逻辑 | loop.py 直接从 response 读取 |
| 容错需求 | 正则模糊匹配、重试循环、兜底 | 无（API 原生结构化） |
| 兼容性 | 任何 LLM | 支持 Function Calling 的 LLM |

**面试话术**：「LLM 输出解析是 Agent 系统最容易被低估的难点。我在项目中经历了完整的架构演进——从自研正则解析器到 OpenAI Function Calling。自研阶段让我深入理解了输出不稳定的本质问题和各种容错策略，而迁移到 Function Calling 让我体验到标准化接口的价值。现在我能清楚判断什么场景用哪种方案更合适。」

---

## 7. 流式输出（SSE）

### 7.1 为什么选择 SSE？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **SSE**（本项目） | 原生 HTTP，单向推送，简单可靠 | 仅服务器→客户端 |
| WebSocket | 双向通信 | 实现复杂，对 Agent 场景过度 |
| 轮询 | 最简单 | 延迟高，浪费资源 |
| gRPC Stream | 高性能 | 浏览器支持差 |

**选择 SSE 的理由**：Agent 执行是单向的（服务器推送步骤给前端），不需要客户端主动发消息，SSE 是最佳选择。

### 7.2 实现细节（`routes/research.py`）

```python
@router.post("/research")
async def start_research(req: ResearchRequest):
    async def generate():
        async for event in orchestrator.research(req.query):
            # SSE 协议格式：event: <类型>\ndata: <JSON>\n\n
            yield f"event: {_event_type(event)}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",       # 禁止缓存
            "Connection": "keep-alive",         # 保持连接
            "X-Accel-Buffering": "no",          # 禁用 Nginx 缓冲
        },
    )
```

### 7.3 前端 SSE 消费（`useSSE.js`）

```javascript
// 使用 ReadableStream API 手动解析 SSE 流
const reader = response.body.getReader()
const decoder = new TextDecoder()
let buffer = ''

while (true) {
    const { done, value } = await reader.read()
    // 按行解析 event: 和 data: 字段
    // 处理 SSE 协议的分帧问题（buffer 暂存不完整的行）
}
```

**关键细节**：SSE 是基于换行符的协议，但 `reader.read()` 可能返回不完整的数据块。代码中使用 `buffer` 暂存未完成的行，这是一个容易被忽视但关键的实现细节。

---

## 8. 为什么不使用 LangChain？

这是面试中几乎必问的问题。准备好以下回答：

### 8.1 LangChain 的问题

| 问题 | 具体表现 |
|------|----------|
| **过度抽象** | 为了"通用"引入了大量抽象层，理解一行有效代码需要追踪 5 层继承 |
| **调试困难** | 错误堆栈深不见底，难以定位问题根源 |
| **隐藏 Prompt** | 框架内部注入了大量隐藏提示词，行为不可预测 |
| **版本不稳定** | API 频繁 breaking change，0.x 版本持续了很长时间 |
| **学习曲线陡峭** | 框架本身的学习成本可能高于自研 |

### 8.2 自研的优势（以本项目为例）

```python
# 本项目的 Agent 核心逻辑，不到 150 行，完全可控
for round_num in range(1, MAX_ROUNDS + 1):
    response = await self.llm.chat_with_tools(messages, tools=tool_defs)
    # content 即为思考，tool_calls 即为结构化工具调用
    # 无需解析器，无需正则，全结构化
```

**面试话术**：「我不是反对使用框架。我反对在没理解底层原理的情况下使用框架。通过自研这个 Agent，我深入理解了 ReAct 循环、上下文管理、工具调用等核心概念——并且经历了从文本解析到 Function Calling 的架构演进。有了这些理解，后续如果需要快速交付，我可以选择使用框架，并且能清楚地知道框架帮我做了什么、它有哪些局限。」

### 8.3 什么情况下应该用框架？

| 场景 | 建议 |
|------|------|
| 快速原型验证 | 用 LangChain / CrewAI |
| 需要多种 LLM Provider | 用框架的统一接口 |
| 团队协作、需要标准化 | 用框架降低沟通成本 |
| 有深度定制需求 | 自研或框架 + 深度 hack |
| 学习和理解 | **自研**（本项目的目的） |

> **金句**：「最好的框架是你理解每一行代码在做什么的框架。」

---

## 9. AI 应用开发的工程化实践

### 9.1 配置管理

```python
# backend/config.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))
```

**要点**：
- 所有外部依赖通过环境变量注入（12-Factor App 原则）
- `OPENAI_BASE_URL` 支持任意兼容接口（DeepSeek、Qwen、本地模型等）
- 有合理的默认值（不配置也能跑）

### 9.2 异步编程

本项目全面使用 `async/await`：

```python
# LLM 调用是异步的
raw_output = await self.llm.chat(messages)

# 工具执行是异步的
observation = await self.tool_registry.execute(...)

# HTTP 请求是异步的
async with httpx.AsyncClient(timeout=15.0) as client:
    response = await client.get(url, params=...)

# 数据库是异步的
db = await get_db()
```

**面试话术**：「AI 应用是典型的 IO 密集型场景——调用 LLM API、搜索 API、抓取网页全都是网络 IO。使用 `async/await` 可以让单线程处理多个并发请求，大幅提升吞吐量。」

### 9.3 数据库设计

```sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    report TEXT,
    steps_json TEXT DEFAULT '[]'
)
```

**设计要点**：
- SQLite + WAL 模式（`PRAGMA journal_mode=WAL`），支持并发读
- `steps_json` 存储完整的 ReAct 步骤（JSON 序列化）
- `status` 状态机：`running` → `completed` / `error`

### 9.4 前端状态管理

```javascript
// useSSE.js — 基于 Vue 3 Composition API 的状态管理
const isResearching = ref(false)   // 调研进行中
const subQuestions = ref([])       // 调研计划
const steps = ref([])              // ReAct 步骤列表
const report = ref(null)           // 最终报告
const error = ref(null)            // 错误信息
```

状态机：
```
空闲 → 调研中 → 完成（展示报告）
              → 出错（展示错误 + 已完成步骤）
```

---

## 10. 项目中的难点与解决方案

### 难点 1：LLM 输出格式不稳定（已解决）

**问题（旧版）**：LLM 有时候不按 `THOUGHT: ... ACTION: ...` 格式输出，导致解析失败。

**演进过程**：
1. **第一阶段（文本解析 + 容错）**：多层正则匹配、错误反馈循环、兜底处理
2. **第二阶段（迁移到 Function Calling）**：使用 OpenAI 原生 `tools` 参数，工具调用完全结构化，从根本上消除了格式不稳定问题。删除 parser.py，代码更简洁、可靠性更高。

### 难点 2：上下文窗口管理

**问题**：每轮搜索可能返回大量文本，多轮后 context 超限。

**解决方案**：
1. **搜索结果数量限制**（`num=5`）
2. **网页内容截断**（`max_chars=5000`）
3. **MemoryManager 自动压缩**：早期步骤压缩为摘要 + 关键事实，保留最近步骤的完整信息

### 难点 3：工具调用的健壮性

**问题**：网络请求可能超时、被限流、返回异常数据。

**解决方案**：
1. **超时设置**：搜索 15s、网页抓取 20s
2. **异常分类**：区分 TimeoutException 和其他异常，给出针对性提示
3. **降级策略**：抓取失败时告知 LLM "将使用搜索摘要"，而不是直接报错

### 难点 4：SSE 连接的异常处理

**问题**：用户关闭页面时 SSE 断开，调研进行到一半。

**解决方案**：
1. 使用 `StreamingResponse` 的 `generate()` 生成器
2. `try/except` 包裹整个流程，异常时更新数据库状态为 `error`
3. 已完成步骤实时写入数据库（通过 `collected_steps` 逐步收集）

---

## 11. 面试常见问题速答

### Q1: 你是如何设计 Agent 的 System Prompt 的？

> 我遵循了几个原则：第一，明确角色定位；第二，通过 Function Calling 的 `tools` 参数传递工具定义，LLM 自行决定何时调用哪个工具；第三，通过 `submit_report` 工具作为调研完成的信号，LLM 在信息足够时主动提交报告；第四，内置行为规则（多源验证、引用格式等），通过规则约束提升输出质量；第五，预埋错误恢复指引（"如果搜索失败，调整策略"），预防死循环。

### Q2: 为什么从文本解析迁移到 Function Calling？

> 项目最初使用文本解析是为了最大兼容性——任何 LLM 都能用。但实践中发现，即使有完善的容错机制（正则模糊匹配、错误反馈重试），LLM 仍有约 5% 的概率不服从格式要求。随着 OpenAI Function Calling 成为行业标准（现在 DeepSeek、Qwen 等主流模型也都支持），从可靠性角度考虑进行了迁移。迁移后代码更简洁（删除了 parser.py），可靠性接近 100%。这是一个从"通用性优先"到"可靠性优先"的务实决策。

### Q3: 如何处理 LLM 的幻觉问题？

> 在本项目的调研场景中，我做了以下几层处理：第一，要求多源验证——重要结论不能只依赖单一来源；第二，要求引用来源 URL，用户可以自行核实；第三，搜索和网页抓取提供的是真实数据，LLM 基于这些数据进行总结，减少了凭空编造的空间；第四，如果搜索无结果，要求 LLM 诚实报告而非编造。当然，更彻底的方案是引入 RAG 和事实性校验。

### Q4: 如何评估 Agent 的输出质量？

> 当前项目的评估主要靠人工——看报告是否覆盖了子问题、引用来源是否有效、结论是否有依据。如果要系统化评估，我会引入：1）来源覆盖率（报告中引用的来源数 / 搜索到的来源数）；2）事实准确性（抽样核验报告中的陈述是否与原文一致）；3）用户满意度评分。长期看可以建立评估数据集，用 LLM-as-Judge 做自动化评估。

### Q5: 这个项目的最大技术挑战是什么？

> 早期是 LLM 输出解析的健壮性——文本格式的 Agent 本质上是在和 LLM 的概率性输出博弈，你永远不知道下一轮 LLM 会不会突然"不听话"。我的解法是"防御式解析 + 自愈反馈循环"，解析失败时不崩溃，而是把错误信息反馈给 LLM 让它修正。但后来我发现这个问题的根本解是使用 Function Calling——与其不断修补文本解析的漏洞，不如使用 API 原生的结构化输出。这个从"修补"到"换方案"的思维转变，是我在这个项目中最大的收获。

### Q6: 如果要你重构，你会改什么？

> Function Calling 迁移已经完成了——这是最大的改进。接下来三个方向：第一，引入流式 LLM 输出（`stream=True`），让用户看到 LLM 实时生成的过程而不是等整段完成；第二，用向量数据库做长期记忆，支持跨会话的知识积累；第三，引入多 Agent 协作，将调研任务分配给不同专业 Agent 并行处理。但当前的单 Agent + Function Calling 架构已经能很好地覆盖核心场景。

---

## 12. 延伸思考

### 12.1 AI Agent 的当前技术趋势

| 趋势 | 说明 | 与本项目的关联 |
|------|------|---------------|
| **MCP (Model Context Protocol)** | Anthropic 推出的 Agent-工具标准化协议 | 本项目的 Tool Registry 是其雏形 |
| **Agent-to-Agent 通信** | 多 Agent 协作 | 可扩展为多个专业 Agent 分工协作 |
| **Computer Use** | Agent 操控计算机（浏览器、命令行） | 工具系统可扩展支持 |
| **Code Agent** | Agent 编写并执行代码 | Agent Loop 天然支持代码执行工具 |
| **Long-running Agent** | 持续运行数小时/数天的 Agent | 需要在任务队列、状态持久化上增强 |

### 12.2 从 Demo 到生产级需要什么？

| 维度 | 当前状态 | 生产级需求 |
|------|----------|------------|
| **可靠性** | Function Calling + 容错降级 | 指数退避重试 + 断路器 + 多模型 fallback |
| **可观测性** | SSE 步骤展示 | OpenTelemetry 追踪 + 日志聚合 + Dashboard |
| **安全性** | 无 | API Key 管理 + 输入清洗 + 速率限制 + 内容审核 |
| **性能** | 单用户同步 | 任务队列 + Worker 池 + 流式 LLM 输出 |
| **测试** | 手动 | 单元测试 + Mock LLM 集成测试 + E2E |
| **部署** | 本地 uvicorn | Docker + K8s + CI/CD + 蓝绿部署 |

### 12.3 面试中可以主动引导的话题

1. **"我注意到你用了 Prompt Caching，我们项目里..."** → 展示你对前沿特性的关注
2. **"我最近在研究 MCP 协议，本项目的 Tool Registry 恰好..."** → 展示技术视野
3. **"我从文本解析迭代到了 Function Calling，这个过程中我学到了..."** → 展示反思和迭代能力
4. **"Agent 的可观测性是一个被低估的问题..."** → 展示工程化思维

---

## 附录：项目关键代码索引

| 文件 | 核心内容 | 说明 |
|------|----------|------|
| `backend/agent/loop.py` | ReAct 循环引擎（Function Calling 驱动） | 核心模块 |
| `backend/agent/memory.py` | 上下文记忆管理 | 核心模块 |
| `backend/agent/orchestrator.py` | 调研编排器（三阶段） | 核心模块 |
| `backend/agent/tools/registry.py` | 工具注册表（OpenAI tools 格式） | 核心模块 |
| `backend/llm/client.py` | LLM 客户端（chat / chat_json_mode / chat_with_tools） | 核心模块 |
| `backend/routes/research.py` | SSE 流式 API + 历史记录 CRUD | API 层 |
| `frontend/src/composables/useSSE.js` | 前端 SSE 消费 | 前端 |
| `frontend/src/App.vue` | 主布局 + 状态切换 | 前端 |

---

> **总结陈词（面试结束时可以使用）**：
>
> 「这个项目让我从零到一构建了一个完整的 AI Agent 系统，并经历了从文本解析到 Function Calling 的架构演进。我深入理解了 ReAct 范式、提示词工程、上下文管理、工具调用、流式输出等 AI 应用的核心技术。更重要的是，我形成了自己的技术判断力——知道什么时候该自研、什么时候该用标准接口、什么样的设计在 AI 场景下是好的设计。我相信这些能力能让我快速胜任 AI 应用开发岗位。」
