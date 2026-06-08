# 流式数据传递全链路

## 整体架构

```
LLM API (OpenAI)
    │
    ▼
LLMClient.chat()           ← backend/llm/client.py
    │
    ▼
AgentLoop.run()            ← backend/agent/loop.py (ReAct 循环)
    │
    ▼
ResearchOrchestrator.research()  ← backend/agent/orchestrator.py (三阶段)
    │
    ▼
generate()                 ← backend/routes/research.py (SSE 格式化)
    │
    ▼
StreamingResponse          ← FastAPI (HTTP 流式响应)
    │
    ▼  ═══════ text/event-stream ═══════
    │
    ▼
fetch() + ReadableStream   ← frontend/src/composables/useSSE.js
    │
    ▼
Vue 响应式状态              ← subQuestions / steps / report
    │
    ▼
UI 组件渲染
```

---

## 一、LLM 调用层 — `backend/llm/client.py`

```python
class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )

    async def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        """通用对话（非流式）"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    async def chat_json_mode(self, messages: list[dict], max_tokens: int = 4096) -> dict:
        """JSON Mode 调用，返回 dict（用于计划生成）"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    async def chat_with_tools(self, messages: list[dict], tools: list[dict],
                              max_tokens: int = 4096) -> dict:
        """Function Calling 调用，返回 {role, content, tool_calls}（用于 ReAct 循环）"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                })
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
        }
```

- 使用 OpenAI 兼容的 `AsyncOpenAI` 客户端
- `chat()` 是**非流式**调用，等待完整响应后返回文本
- `chat_json_mode()` 使用 `response_format={"type": "json_object"}` 获取结构化 JSON
- `chat_with_tools()` 使用 OpenAI 原生 `tools` 参数实现 Function Calling
- 流式发生在更上层（生成器 `yield` 机制）

---

## 二、ReAct 循环层 — `backend/agent/loop.py`

```python
class AgentLoop:
    async def run(self, query: str, tool_defs: list[dict] = None) -> AsyncGenerator[StepEvent, None]:
        for round_num in range(1, MAX_ROUNDS + 1):
            # 1. Function Calling（LLM 自行决定是否调用工具）
            response = await self.llm.chat_with_tools(messages, tools=tool_defs)
            content = response["content"]
            tool_calls = response.get("tool_calls", [])

            # 2. yield THOUGHT（LLM 的 content 即为思考文本）
            if content:
                yield StepEvent(type="thought", round=round_num, content=content)

            # 3. LLM 没有调用工具 → 结束循环
            if not tool_calls:
                return

            # 4. 执行每个工具调用
            for tc in tool_calls:
                yield StepEvent(type="action", round=round_num,
                    content=f"调用 {tc['name']}",
                    tool_name=tc['name'], tool_params=tc['arguments'])

                observation = await self.tool_registry.execute(tc['name'], tc['arguments'])

                yield StepEvent(type="observation", round=round_num,
                    content=observation[:2000])

            # 5. submit_report 工具调用 → 结束循环
            if any(tc['name'] == 'submit_report' for tc in tool_calls):
                self.final_report = ...  # 保存报告内容
                return
```

**核心机制**：`yield` 是 Python 异步生成器的关键字——产出一条数据就立即向上游推送，不会等所有轮次执行完。`chat_with_tools()` 利用 OpenAI Function Calling，LLM 返回结构化的 `tool_calls` 数组，不再需要文本解析。

每轮循环产出事件：
| 顺序 | 事件类型 | 含义 |
|------|----------|------|
| 1 | `thought` | LLM 的思考过程（`content` 字段） |
| 2 | `action` | 调用的工具名和参数（从 `tool_calls[]` 提取） |
| 3 | `observation` | 工具返回的结果（截断到 2000 字符） |

---

## 三、编排器层 — `backend/agent/orchestrator.py`

```python
class ResearchOrchestrator:
    async def research(self, query: str) -> AsyncGenerator[dict, None]:
        # 阶段一：生成调研计划（JSON Mode）
        plan_prompt = [{"role": "user", "content": f"将以下研究问题拆分为 2-4 个子问题...\n问题：{query}"}]
        plan_result = await self.llm.chat_json_mode(plan_prompt)
        sub_questions = plan_result.get("sub_questions", [query])
        yield PlanEvent(sub_questions=sub_questions).model_dump()
        # → {"sub_questions": ["什么是 X？", "X 的主要应用？", ...]}

        # 阶段二：运行 ReAct 循环（逐条转发）
        tool_defs = self.registry.get_openai_tools()
        loop = AgentLoop(self.registry, self.llm)
        async for step in loop.run(query, tool_defs=tool_defs):
            yield step.model_dump()
        # → {"type": "thought", "round": 1, "content": "..."}
        # → {"type": "action", "round": 1, "content": "...", "tool_name": "web_search"}
        # → {"type": "observation", "round": 1, "content": "..."}
        # → ... 多轮 ...

        # 阶段三：生成最终报告（来自 loop.final_report）
        final_report = loop.final_report
        sources = self._extract_sources(final_report)
        yield ReportEvent(markdown=final_report, sources=sources).model_dump()
        # → {"markdown": "# 调研报告\n\n...", "sources": ["https://..."]}
```

三个阶段的产出对象：

| 阶段 | 模型 | 字段 |
|------|------|------|
| 计划 | `PlanEvent` | `sub_questions: list[str]` |
| 步骤 | `StepEvent` | `type`, `round`, `content`, `tool_name?`, `tool_params?` |
| 报告 | `ReportEvent` | `markdown: str`, `sources: list[str]` |

`.model_dump()` 是 Pydantic 方法，将模型转为普通 dict。

---

## 四、SSE 格式化 + HTTP 响应 — `backend/routes/research.py`

### 4.1 SSE 格式化

```python
async def generate():
    async for event in orchestrator.research(req.query):
        yield f"event: {_event_type(event)}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
```

`_event_type()` 根据字典内容推断事件类型：

```python
def _event_type(event: dict) -> str:
    if "sub_questions" in event:    return "plan"
    if "markdown" in event:         return "report"
    if event.get("type") in ("thought", "action", "observation"):
                                    return "step"
    return "message"
```

产出的 SSE 文本格式：

```
event: plan
data: {"sub_questions": ["什么是 AI？", "AI 的应用场景？"]}

event: step
data: {"type": "thought", "round": 1, "content": "需要先搜索基本概念..."}

event: step
data: {"type": "action", "round": 1, "content": "调用 web_search", "tool_name": "web_search"}

event: step
data: {"type": "observation", "round": 1, "content": "搜索结果: ..."}

event: report
data: {"markdown": "# 报告...", "sources": ["https://..."]}
```

**SSE 协议格式**：`event: <类型>\ndata: <JSON>\n\n`，每对事件数据以空行分隔。

### 4.2 StreamingResponse 配置

```python
return StreamingResponse(
    generate(),
    media_type="text/event-stream",
    headers={
        "Cache-Control": "no-cache",       # 禁止浏览器/代理缓存
        "Connection": "keep-alive",        # 保持 TCP 连接
        "X-Accel-Buffering": "no",         # 禁用 Nginx 缓冲
    },
)
```

`StreamingResponse` 接收异步生成器，每次 `yield` 立即将数据写入 HTTP 响应体，不会等待生成器执行完毕。

### 4.3 会话持久化

```python
async def generate():
    collected_steps = []
    report = None
    try:
        async for event in orchestrator.research(req.query):
            # 收集步骤用于最终持久化
            if event.get("type") in ("thought", "action", "observation"):
                collected_steps.append(event)
            if "markdown" in event:
                report = event["markdown"]
            yield f"event: ..."

        # 正常完成 → status = "completed"
        await update_session(session_id, "completed", report, json.dumps(collected_steps))
    except Exception as e:
        # 异常 → status = "error"，保留已执行的步骤
        yield f"event: error\ndata: ..."
        await update_session(session_id, "error", steps_json=json.dumps(collected_steps))
```

---

## 五、前端接收层 — `frontend/src/composables/useSSE.js`

```javascript
export function useSSE() {
  // 四个响应式状态
  const subQuestions = ref([])   // 子问题列表
  const steps = ref([])          // ReAct 步骤数组
  const report = ref(null)       // 最终报告
  const error = ref(null)        // 错误信息

  function startResearch(query) {
    fetch('/api/research', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query }),
    }).then(async (response) => {
      const reader = response.body.getReader()      // 获取 ReadableStream reader
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read() // 逐块读取字节
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''  // 保留不完整的行到下一次

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()       // 提取事件类型
          } else if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6))    // 解析 JSON
            handleEvent(currentEvent, data)
          }
        }
      }
    })
  }

  function handleEvent(event, data) {
    switch (event) {
      case 'plan':   subQuestions.value = data.sub_questions || []; break
      case 'step':   steps.value.push(data);                        break
      case 'report': report.value = data;                           break
      case 'error':  error.value = data.message;                    break
    }
  }

  return { isResearching, subQuestions, steps, report, error, startResearch }
}
```

### 5.1 ReadableStream 分块读取

```
response.body.getReader()  → 浏览器原生 API，获取流式读取器
reader.read()              → 返回 { done: false, value: Uint8Array }，逐块读取
TextDecoder.decode()       → 将字节数组解码为 UTF-8 字符串（stream: true 表示流模式）
```

### 5.2 缓冲处理（防拆包）

```
网络包可能在任意位置断开：

第 1 个 chunk:  "event: step\ndata: {\"type\":\"tho"   ← 不完整
第 2 个 chunk:  "ught\",\"round\":1}\n\nevent: st"     ← 拼接后完整

buffer 的作用：
  buffer += chunk
  lines = buffer.split('\n')
  buffer = lines.pop()  ← 最后一行可能不完整，保留到下次拼接
```

### 5.3 四个响应式变量的更新时机

| 变量 | 类型 | 更新方式 | 触发条件 |
|------|------|----------|----------|
| `subQuestions` | `ref([])` | 整体赋值 | 收到 `event: plan` 时（仅一次） |
| `steps` | `ref([])` | `push` 追加 | 每收到一次 `event: step` |
| `report` | `ref(null)` | 整体赋值 | 收到 `event: report` 时（仅一次） |
| `error` | `ref(null)` | 整体赋值 | 收到 `event: error` 或 fetch 异常 |

---

## 六、流式传输的关键设计

```
Python async generator (yield)
  → FastAPI StreamingResponse (逐块写入 HTTP 响应)
    → HTTP 长连接 (Content-Type: text/event-stream, Connection: keep-alive)
      → 浏览器 ReadableStream (response.body.getReader)
        → 缓冲拼接 + SSE 协议解析 (event:/data: 行)
          → Vue 响应式状态 → UI 逐条渲染
```

**流式的核心是 `yield`**——Python 异步生成器产出一条就发送一条，前端 `ReadableStream` 收到一条就处理一条，两端配合实现"LLM 思考一步，前端显示一步"的实时效果。

---

## 七、相关文件索引

| 文件 | 职责 |
|------|------|
| [backend/routes/research.py](../backend/routes/research.py) | FastAPI 路由，SSE 端点 + 历史记录 CRUD |
| [backend/agent/orchestrator.py](../backend/agent/orchestrator.py) | 三阶段编排：计划（JSON Mode） → ReAct 循环（Function Calling） → 报告 |
| [backend/agent/loop.py](../backend/agent/loop.py) | ReAct 循环实现：Function Calling 驱动 |
| [backend/llm/client.py](../backend/llm/client.py) | OpenAI API 调用封装（chat / chat_json_mode / chat_with_tools） |
| [backend/agent/tools/registry.py](../backend/agent/tools/registry.py) | 工具注册中心（OpenAI tools 格式） |
| [backend/models/schemas.py](../backend/models/schemas.py) | Pydantic 数据模型 |
| [backend/db/repository.py](../backend/db/repository.py) | SQLite 数据库操作 |
| [frontend/src/composables/useSSE.js](../frontend/src/composables/useSSE.js) | 前端 SSE 接收 + 解析 |

---

## 八、完整调用时序图

```
前端 useSSE           FastAPI 路由        Orchestrator        AgentLoop          LLMClient         OpenAI API
    │                     │                    │                   │                  │                │
    │──POST /api/research─→│                    │                   │                  │                │
    │                     │──research(query)──→│                   │                  │                │
    │                     │                    │──chat_json_mode()──────────────────────────────────→│
    │                     │                    │←──{"sub_questions"}─────────────────────────────────│
    │                     │←──yield plan───────│                   │                  │                │
    │←SSE: plan           │                    │                   │                  │                │
    │                     │                    │──loop.run(q, defs)─→│                  │                │
    │                     │                    │                   │──chat_with_tools()             │
    │                     │                    │                   │  (msg, tools=tool_defs)───────→│
    │                     │                    │                   │←──content + tool_calls[]──────│
    │                     │                    │←──yield thought───│                  │                │
    │←SSE: step(thought)  │                    │                   │                  │                │
    │                     │                    │←──yield action────│                  │                │
    │←SSE: step(action)   │                    │                   │                  │                │
    │                     │                    │                   │──[执行工具]      │                │
    │                     │                    │←──yield observation│                 │                │
    │←SSE: step(obs)      │                    │                   │                  │                │
    │                     │                    │                   │  (多轮循环...)   │                │
    │                     │                    │                   │──submit_report──→│                │
    │                     │                    │←──loop.final_report──                 │                │
    │                     │←──yield report─────│                   │                  │                │
    │←SSE: report         │                    │                   │                  │                │
    │                     │──update_session()  │                   │                  │                │
    │←──连接关闭──────────│                    │                   │                  │                │
```
