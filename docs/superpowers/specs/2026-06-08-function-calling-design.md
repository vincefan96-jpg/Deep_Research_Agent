# Function Calling 替代文本解析 — 设计文档

## 概述

将 ReAct Agent 中基于正则表达式的 LLM 输出解析替换为 OpenAI 兼容的 function calling（tool use）机制，消除自定义文本格式的脆弱性，同时适配国产模型。

**技术决策**：分阶段混合方案

| 阶段 | 机制 | 原因 |
|------|------|------|
| 生成调研计划 | JSON Mode (`response_format`) | Plan 是固定结构化输出，非工具选择场景；国产模型对 JSON Mode 支持更稳定 |
| 工具执行 | Function Calling (`tool_calls`) | 工具选择是 function calling 的核心场景 |
| 提交报告 | Function Calling (`submit_report` tool) | 统一 loop 终止机制 |

---

## 一、核心改动

### 1. LLM Client（`llm/client.py`）

`chat()` 拆分为三种调用模式：

```python
class LLMClient:
    # 模式1：纯文本 — summarize 等简单场景保留
    async def chat(self, messages, max_tokens=4096) -> str

    # 模式2：JSON Mode — Plan 阶段
    async def chat_json_mode(self, messages, max_tokens=2048) -> dict

    # 模式3：Function Calling — ReAct 循环
    async def chat_with_tools(
        self, messages, tools: list[dict], tool_choice="auto"
    ) -> dict  # {"role": "assistant", "content": "...", "tool_calls": [...]}
```

**国产模型适配**：
- `tool_choice="auto"` 而非 `"required"`，避免部分模型在 required 下行为异常
- 返回完整 message 对象，`content` 可能为空，loop 层兼容处理

### 2. Tool Registry（`agent/tools/registry.py`）

- 新增 `get_openai_tools() -> list[dict]`：将 Tool 转为 OpenAI function calling 格式
- 删除 `get_schema_for_llm()`：工具描述不再以文本形式注入 system prompt

注册的工具：

| 工具 | 参数 | 用途 |
|------|------|------|
| `web_search` | `{"query": {"type": "string"}}` | 搜索关键词 |
| `fetch_page` | `{"url": {"type": "string"}}` | 抓取网页全文 |
| `submit_report` | `{"markdown": {"type": "string"}, "sources": {"type": "array", "items": {"type": "string"}}}` | 提交最终报告，结束循环 |

`submit_report` 的 handler 为一个空操作（返回确认字符串），loop 层检测到该工具名时将报告内容提取为最终输出并终止循环。

### 3. ReAct Loop（`agent/loop.py`）— 重写核心逻辑

```
当前：LLM文本 → parse(正则) → 提取 ACTION: tool|{json} → 执行
改为：LLM + tools → response.tool_calls → 直接执行
```

关键变化：
- 消息结构从纯文本变为 `assistant(tool_calls)` + `tool(result)` —— 符合 OpenAI function calling 协议
- Observation 使用 `role: "tool"` 而非 `role: "user"`
- 终止条件：`submit_report` 被调用，或 LLM 返回无 tool_calls 时取其 content 作为最终答案
- 每轮限制执行单个 tool_call（架构上按列表迭代，预留并行扩展）
- 每轮 `content` 作为 thought yeld 到 SSE，前端步骤时间线不变
- 解析失败不再需要重试——function calling 天然结构化

### 4. Parser（`agent/parser.py`）— 删除

Function calling 从根本上消除了文本解析需求，`parse()` 函数和 `ParsedOutput` 数据类一并移除。

### 5. Orchestrator（`agent/orchestrator.py`）

- 阶段1：用 `chat_json_mode()` 替代 `generate_plan()`，`LLMClient.generate_plan()` 移除
- 阶段2：调用 `registry.get_openai_tools()` 获取工具定义，传入 `loop.run()`
- 阶段3：最终报告由 loop 末轮（`submit_report`）产出，不再单独调 LLM

### 6. System Prompt（`loop.py` 中的 `SYSTEM_PROMPT`）

移除格式指令（THOUGHT/Action/FINAL_ANSWER 格式说明），简化为 Agent 角色描述和工具使用指引。工具定义由 function calling 的 `tools` 参数提供。

---

## 二、不变部分

| 模块 | 说明 |
|------|------|
| `agent/memory.py` | 接口无需改动，继续记录 Step |
| `agent/tools/web_search.py` | 工具实现不变 |
| `agent/tools/fetch_page.py` | 工具实现不变 |
| ` models/schemas.py` | StepEvent 结构保持兼容 |
| `db/` | SQLite 层不变 |
| `routes/research.py` | SSE 端点不变 |
| `config.py` | 配置不变 |
| 所有前端文件 | 事件类型兼容，无需任何改动 |

---

## 三、SSE 事件流

保持当前事件类型不变：

```
POST /api/research  { "query": "..." }

event: plan         → {"sub_questions": [...]}
event: step         → {"type": "thought|action|observation", "round": N, "content": "...", "tool_name": "..."}
event: report       → {"markdown": "...", "sources": [...]}
event: done
event: error        → {"message": "..."}
```

loop 内部新增错误包装：未捕获异常统一 yield 为 `error` 事件后终止。

---

## 四、容错设计

| 场景 | 处理 |
|------|------|
| LLM 返回无 tool_calls 也无 content | 取上轮 observation 作为上下文，追加 "请继续调研" 提示重试 1 次 |
| tool_call 解析失败（JSON 异常） | 返回错误 observation，让 LLM 感知并重试 |
| Search API 超时/限流 | 同当前：告知 LLM 搜索不可用 |
| 网页抓取失败 | 同当前：降级为搜索摘要 |
| LLM 持续不终止 | `MAX_ROUNDS` 硬上限，到期强制执行 `summarize()` 生成报告 |
| 上下文超限 | 同当前：MemoryManager 压缩早期步骤 |
| `submit_report` 参数不合规 | 校验 markdown 和 sources，缺失字段用默认值 |
| 国产模型 function calling 不返回 `tool_call_id` | loop 层自动生成 fallback ID |

---

## 五、文件变更清单

| 文件 | 动作 | 说明 |
|------|------|------|
| `backend/llm/client.py` | ✏️ 修改 | 新增 `chat_json_mode()`、`chat_with_tools()`；删除 `generate_plan()` |
| `backend/agent/loop.py` | ✏️ 重写 | 文本消息循环 → tool_calls 循环；更新 SYSTEM_PROMPT |
| `backend/agent/orchestrator.py` | ✏️ 修改 | Plan 用 JSON Mode；传入 tool_defs；报告由 loop 产出 |
| `backend/agent/tools/registry.py` | ✏️ 修改 | 新增 `get_openai_tools()`；删除 `get_schema_for_llm()` |
| `backend/agent/parser.py` | 🗑️ 删除 | 不再需要 |
| `backend/agent/tools/cross_check.py` | 🗑️ 删除（如存在） | 此前已移除交叉验证模块 |
| `backend/models/schemas.py` | ✏️ 微调 | 确保 event 模型兼容 |
| `frontend/` | ✅ 不变 | 无改动 |
| `backend/db/` | ✅ 不变 | 无改动 |
| `backend/routes/` | ✅ 不变 | 无改动 |
| `backend/config.py` | ✅ 不变 | 无改动 |

## 六、测试策略

| 层级 | 内容 | 工具 |
|------|------|------|
| 单元测试 | `registry.get_openai_tools()` 格式正确性 | pytest |
| 单元测试 | `LLMClient.chat_json_mode()` JSON 解析和降级 | pytest + mock |
| 单元测试 | `LLMClient.chat_with_tools()` 响应提取逻辑 | pytest + mock |
| 集成测试 | Loop 终止条件：submit_report、无 tool_calls、MAX_ROUNDS | pytest + mock |
| 集成测试 | 消息格式符合 OpenAI function calling 协议 | pytest |
| 手动测试 | 完整调研流程（需 API key） | 浏览器 + 后端 |
