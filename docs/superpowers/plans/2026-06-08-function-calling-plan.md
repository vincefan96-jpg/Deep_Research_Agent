# Function Calling 替代文本解析 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 ReAct Agent 的文本解析替换为 OpenAI 兼容的 function calling，消除正则解析的脆弱性。

**Architecture:** 分阶段混合方案 — Plan 用 JSON Mode，工具调用用 function calling，报告提交用 `submit_report` tool。删除 `parser.py`，重写 `loop.py`，LLMClient 支持三种调用模式。

**Tech Stack:** Python 3.10+, OpenAI SDK, pytest 9.0 + pytest-asyncio

---

### Task 1: Tool Registry — 新增 `get_openai_tools()`，删除 `get_schema_for_llm()`

**Files:**
- Modify: `backend/agent/tools/registry.py`

- [ ] **Step 1: Replace `get_schema_for_llm()` with `get_openai_tools()`**

```python
# backend/agent/tools/registry.py
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema properties, e.g. {"url": {"type": "string", "description": "..."}}
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_openai_tools(self) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": list(tool.parameters.keys()),
                }
            }
        } for tool in self._tools.values()]

    async def execute(self, name: str, params: dict) -> str:
        tool = self.get(name)
        if tool is None:
            return f"错误：工具 '{name}' 未找到。可用工具：{list(self._tools.keys())}"
        try:
            result = await tool.handler(**params)
            return str(result)
        except Exception as e:
            return f"执行 '{name}' 出错：{e}"
```

- [ ] **Step 2: Verify module loads without error**

```bash
cd d:/AGENT/backend && python -c "from agent.tools.registry import ToolRegistry; r = ToolRegistry(); print('get_openai_tools:', r.get_openai_tools()); print('has get_schema_for_llm:', hasattr(r, 'get_schema_for_llm'))"
```

Expected: `get_openai_tools: []`, `has get_schema_for_llm: False`

- [ ] **Step 3: Commit**

```bash
git add backend/agent/tools/registry.py
git commit -m "feat: registry 新增 get_openai_tools()，移除 get_schema_for_llm()"
```

---

### Task 2: LLM Client — 新增 `chat_json_mode()` 和 `chat_with_tools()`

**Files:**
- Modify: `backend/llm/client.py`

- [ ] **Step 1: Rewrite `backend/llm/client.py`**

```python
import json
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.model = OPENAI_MODEL

    async def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    async def chat_json_mode(self, messages: list[dict], max_tokens: int = 2048) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_choice: str = "auto"
    ) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
            tools=tools,
            tool_choice=tool_choice,
        )
        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    pass
                tool_calls.append({
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "parsed_args": args,
                })
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
        }

    async def summarize(self, text: str, max_length: int = 500) -> str:
        prompt = f"请将以下内容总结在 {max_length} 字以内：\n\n{text}"
        return await self.chat([{"role": "user", "content": prompt}])
```

- [ ] **Step 2: Verify module loads**

```bash
cd d:/AGENT/backend && python -c "from llm.client import LLMClient; c = LLMClient(); print('chat:', hasattr(c, 'chat')); print('chat_json_mode:', hasattr(c, 'chat_json_mode')); print('chat_with_tools:', hasattr(c, 'chat_with_tools')); print('generate_plan:', hasattr(c, 'generate_plan'))"
```

Expected: `chat: True`, `chat_json_mode: True`, `chat_with_tools: True`, `generate_plan: False`

- [ ] **Step 3: Commit**

```bash
git add backend/llm/client.py
git commit -m "feat: llm client 新增 chat_json_mode 和 chat_with_tools，移除 generate_plan"
```

---

### Task 3: ReAct Loop — 重写为 function calling 循环

**Files:**
- Modify: `backend/agent/loop.py`

- [ ] **Step 1: Rewrite `backend/agent/loop.py`**

```python
import json
from typing import AsyncGenerator
from config import MAX_ROUNDS
from models.schemas import Step, StepType, StepEvent
from agent.tools.registry import ToolRegistry
from agent.memory import MemoryManager
from llm.client import LLMClient


SYSTEM_PROMPT = """你是一个深度调研智能体。你的目标是对研究主题进行彻底调查，生成一份全面、来源可靠的答案。

你可以使用提供的工具进行搜索和抓取网页。当信息收集充分时，调用 submit_report 提交最终报告。

规则：
- 先搜索后阅读：先用 web_search 搜索，再对最有价值的 URL 使用 fetch_page 深入阅读
- 多源验证：重要结论不能只依赖单一来源
- 引用来源格式：[来源: URL]
- 如果搜索失败或无结果，调整策略或诚实地报告未能找到的内容"""


class AgentLoop:
    def __init__(self, tool_registry: ToolRegistry, llm: LLMClient):
        self.tool_registry = tool_registry
        self.llm = llm
        self.memory = MemoryManager()
        self.final_report = ""

    async def run(self, query: str, tool_defs: list[dict]) -> AsyncGenerator[StepEvent, None]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"研究问题：{query}"},
        ]

        for round_num in range(1, MAX_ROUNDS + 1):
            response = await self.llm.chat_with_tools(messages, tools=tool_defs)

            thought = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if thought:
                evt = StepEvent(type="thought", round=round_num, content=thought)
                self.memory.add_step(Step(round=round_num, type=StepType.THOUGHT, content=thought))
                yield evt

            if not tool_calls:
                self.final_report = thought or "未能生成报告。"
                yield StepEvent(type="observation", round=round_num, content=self.final_report)
                return

            # Build clean assistant message for API (strip parsed_args)
            clean_msg = {"role": "assistant", "content": thought}
            clean_msg["tool_calls"] = [
                {k: v for k, v in tc.items() if k != "parsed_args"}
                for tc in tool_calls
            ]
            messages.append(clean_msg)

            for i, tc in enumerate(tool_calls):
                tool_name = tc["function"]["name"]
                params = tc.get("parsed_args", {})
                tc_id = tc.get("id") or f"call_{round_num}_{i}"

                if tool_name == "submit_report":
                    markdown = params.get("markdown") or thought or "未能生成报告。"
                    self.final_report = markdown
                    yield StepEvent(type="observation", round=round_num, content=markdown)
                    return

                # Yield action
                action_evt = StepEvent(
                    type="action", round=round_num,
                    content=f"调用 {tool_name}",
                    tool_name=tool_name, tool_params=params,
                )
                self.memory.add_step(Step(
                    round=round_num, type=StepType.ACTION,
                    content=f"调用 {tool_name}，参数 {json.dumps(params, ensure_ascii=False)}",
                    tool_name=tool_name, tool_params=params,
                ))
                yield action_evt

                observation = await self.tool_registry.execute(tool_name, params)

                # Yield observation
                obs_evt = StepEvent(
                    type="observation", round=round_num,
                    content=observation[:2000],
                )
                self.memory.add_step(Step(
                    round=round_num, type=StepType.OBSERVATION,
                    content=observation[:2000],
                ))
                yield obs_evt

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": observation,
                })

                if len(observation) > 200:
                    self.memory.add_key_fact(observation[:300])

        # MAX_ROUNDS reached — force summary
        self.final_report = await self.llm.chat([
            {
                "role": "user",
                "content": f"根据以下调研结果撰写一份调研报告，包含引用来源。\n\n"
                           f"调研结果：\n{self.memory.get_steps_for_context()}"
            }
        ])
        yield StepEvent(type="observation", round=MAX_ROUNDS, content=self.final_report)
```

- [ ] **Step 2: Verify module loads**

```bash
cd d:/AGENT/backend && python -c "from agent.loop import AgentLoop; print('AgentLoop imported')"
```

Expected: `AgentLoop imported`

- [ ] **Step 3: Commit**

```bash
git add backend/agent/loop.py
git commit -m "feat: 重写 ReAct loop 为 function calling 驱动"
```

---

### Task 4: Orchestrator — JSON Mode Plan + 注册 submit_report

**Files:**
- Modify: `backend/agent/orchestrator.py`

- [ ] **Step 1: Rewrite `backend/agent/orchestrator.py`**

```python
from typing import AsyncGenerator
from models.schemas import PlanEvent, ReportEvent
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry, Tool
from agent.tools.web_search import web_search
from agent.tools.fetch_page import fetch_page
from llm.client import LLMClient


async def _submit_report_handler(**kwargs):
    return "报告已提交。"


class ResearchOrchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.registry = ToolRegistry()
        self._setup_tools()

    def _setup_tools(self):
        self.registry.register(Tool(
            name="web_search",
            description="在互联网上搜索信息。返回标题、摘要和 URL 列表。",
            parameters={"query": {"type": "string", "description": "搜索关键词"}},
            handler=web_search,
        ))
        self.registry.register(Tool(
            name="fetch_page",
            description="抓取并提取指定 URL 的文本内容。用于深入阅读文章。",
            parameters={"url": {"type": "string", "description": "要抓取的完整 URL"}},
            handler=fetch_page,
        ))
        self.registry.register(Tool(
            name="submit_report",
            description="提交最终调研报告。当收集到足够信息后调用此工具结束调研。",
            parameters={
                "markdown": {
                    "type": "string",
                    "description": "格式为 Markdown 的综合调研报告，包含引用来源 [来源: URL]",
                },
                "sources": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "报告中引用的所有来源 URL 列表",
                },
            },
            handler=_submit_report_handler,
        ))

    async def research(self, query: str) -> AsyncGenerator[dict, None]:
        # 阶段一：JSON Mode 生成调研计划
        plan_data = await self.llm.chat_json_mode([
            {
                "role": "system",
                "content": (
                    "你是一个研究计划者。将用户的研究问题拆分为 2-4 个具体的子问题。"
                    "只返回一个 JSON 对象，格式：{\"sub_questions\": [\"子问题1\", \"子问题2\"]}"
                ),
            },
            {"role": "user", "content": query},
        ])
        sub_questions = plan_data.get("sub_questions", [query])
        if not isinstance(sub_questions, list) or len(sub_questions) == 0:
            sub_questions = [query]
        yield PlanEvent(sub_questions=sub_questions).model_dump()

        # 阶段二：ReAct 循环
        tool_defs = self.registry.get_openai_tools()
        loop = AgentLoop(self.registry, self.llm)
        async for step in loop.run(query, tool_defs):
            yield step.model_dump()

        # 阶段三：最终报告事件
        sources = self._extract_sources(loop.final_report)
        yield ReportEvent(markdown=loop.final_report, sources=sources).model_dump()

    def _extract_sources(self, report: str) -> list[str]:
        import re
        urls = re.findall(r'https?://[^\s\)\]]+', report)
        return list(dict.fromkeys(urls))
```

- [ ] **Step 2: Verify module loads and tools registered**

```bash
cd d:/AGENT/backend && python -c "from agent.orchestrator import ResearchOrchestrator; o = ResearchOrchestrator(); tools = o.registry.get_openai_tools(); print('Tools:', [t['function']['name'] for t in tools])"
```

Expected: `Tools: ['web_search', 'fetch_page', 'submit_report']`

- [ ] **Step 3: Commit**

```bash
git add backend/agent/orchestrator.py
git commit -m "feat: orchestrator 改用 JSON Mode plan，注册 submit_report 工具"
```

---

### Task 5: 删除 parser.py

**Files:**
- Delete: `backend/agent/parser.py`

- [ ] **Step 1: Delete the file**

```bash
rm d:/AGENT/backend/agent/parser.py
```

- [ ] **Step 2: Verify no import errors**

```bash
cd d:/AGENT/backend && python -c "from agent.loop import AgentLoop; from agent.orchestrator import ResearchOrchestrator; print('All imports clean')"
```

Expected: `All imports clean`

- [ ] **Step 3: Commit**

```bash
git rm backend/agent/parser.py
git commit -m "chore: 删除 parser.py，function calling 消除文本解析需求"
```

---

### Task 6: 测试 — Registry、LLM Client、Loop

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_registry.py`
- Create: `tests/test_client.py`
- Create: `tests/test_loop.py`

- [ ] **Step 1: Create test directory and `tests/__init__.py`**

```bash
mkdir -p d:/AGENT/tests
```

```python
# d:/AGENT/tests/__init__.py (empty)
```

- [ ] **Step 2: Write `tests/test_registry.py`**

```python
import pytest
from agent.tools.registry import ToolRegistry, Tool


async def dummy_handler(**kwargs):
    return str(kwargs)


def test_get_openai_tools_format():
    registry = ToolRegistry()
    registry.register(Tool(
        name="web_search",
        description="搜索互联网",
        parameters={"query": {"type": "string", "description": "搜索关键词"}},
        handler=dummy_handler,
    ))
    registry.register(Tool(
        name="fetch_page",
        description="抓取网页",
        parameters={"url": {"type": "string", "description": "页面URL"}},
        handler=dummy_handler,
    ))

    tools = registry.get_openai_tools()

    assert len(tools) == 2
    for tool in tools:
        assert tool["type"] == "function"
        fn = tool["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]
        assert "required" in fn["parameters"]
        assert set(fn["parameters"]["required"]) == set(fn["parameters"]["properties"].keys())


def test_get_openai_tools_empty():
    registry = ToolRegistry()
    assert registry.get_openai_tools() == []


def test_no_get_schema_for_llm():
    registry = ToolRegistry()
    assert not hasattr(registry, "get_schema_for_llm")


@pytest.mark.asyncio
async def test_execute_success():
    registry = ToolRegistry()
    registry.register(Tool(
        name="test_tool",
        description="test",
        parameters={"x": {"type": "string"}},
        handler=dummy_handler,
    ))
    result = await registry.execute("test_tool", {"x": "hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_execute_not_found():
    registry = ToolRegistry()
    result = await registry.execute("nonexistent", {})
    assert "未找到" in result
```

- [ ] **Step 3: Run registry tests**

```bash
cd d:/AGENT && PYTHONPATH=backend python -m pytest tests/test_registry.py -v
```

Expected: 5 tests pass

- [ ] **Step 4: Write `tests/test_client.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from llm.client import LLMClient


@pytest.mark.asyncio
async def test_chat_json_mode_returns_dict():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"sub_questions": ["a", "b"]}'))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat_json_mode([{"role": "user", "content": "test"}])

    assert result == {"sub_questions": ["a", "b"]}
    call_kwargs = client.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_chat_json_mode_fallback_on_invalid_json():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="not valid json {{{"))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat_json_mode([{"role": "user", "content": "test"}])

    assert result == {}


@pytest.mark.asyncio
async def test_chat_json_mode_fallback_on_empty():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=None))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat_json_mode([{"role": "user", "content": "test"}])

    assert result == {}


@pytest.mark.asyncio
async def test_chat_with_tools_returns_structured_response():
    client = LLMClient()
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "web_search"
    mock_tc.function.arguments = '{"query": "test query"}'

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(
            content="Let me search for that",
            tool_calls=[mock_tc],
        ))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    tools = [{"type": "function", "function": {"name": "ws", "description": "s", "parameters": {}}}]
    result = await client.chat_with_tools([{"role": "user", "content": "test"}], tools)

    assert result["role"] == "assistant"
    assert result["content"] == "Let me search for that"
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["id"] == "call_123"
    assert result["tool_calls"][0]["function"]["name"] == "web_search"
    assert result["tool_calls"][0]["parsed_args"] == {"query": "test query"}


@pytest.mark.asyncio
async def test_chat_with_tools_no_tool_calls():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Here is the answer", tool_calls=None))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat_with_tools([{"role": "user", "content": "test"}], [])

    assert result["role"] == "assistant"
    assert result["content"] == "Here is the answer"
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_chat_with_tools_passes_tool_choice():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="ok", tool_calls=None))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    await client.chat_with_tools([], [], tool_choice="auto")

    call_kwargs = client.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tool_choice"] == "auto"
    assert call_kwargs["tools"] == []


@pytest.mark.asyncio
async def test_chat_returns_string():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="plain text response"))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat([{"role": "user", "content": "hello"}])

    assert result == "plain text response"


@pytest.mark.asyncio
async def test_chat_empty_content_returns_empty_string():
    client = LLMClient()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=None))
    ]
    client.client.chat.completions.create = AsyncMock(return_value=mock_response)

    result = await client.chat([{"role": "user", "content": "hello"}])

    assert result == ""
```

- [ ] **Step 5: Run client tests**

```bash
cd d:/AGENT && PYTHONPATH=backend python -m pytest tests/test_client.py -v
```

Expected: 8 tests pass

- [ ] **Step 6: Write `tests/test_loop.py`**

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry, Tool
from llm.client import LLMClient


@pytest.fixture
def mock_llm():
    return MagicMock(spec=LLMClient)


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(Tool(
        name="web_search",
        description="搜索互联网",
        parameters={"query": {"type": "string", "description": "搜索关键词"}},
        handler=AsyncMock(return_value="搜索结果：关于 AI 的最新进展..."),
    ))
    reg.register(Tool(
        name="fetch_page",
        description="抓取网页",
        parameters={"url": {"type": "string", "description": "URL"}},
        handler=AsyncMock(return_value="页面内容..."),
    ))
    reg.register(Tool(
        name="submit_report",
        description="提交报告",
        parameters={
            "markdown": {"type": "string", "description": "报告"},
            "sources": {"type": "array", "items": {"type": "string"}, "description": "来源"},
        },
        handler=AsyncMock(return_value="报告已提交。"),
    ))
    return reg


@pytest.mark.asyncio
async def test_loop_submit_report_terminates(mock_llm, registry):
    mock_llm.chat_with_tools = AsyncMock(return_value={
        "role": "assistant",
        "content": "调研完成，提交报告。",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "submit_report", "arguments": '{"markdown": "# 报告", "sources": ["http://ex.com"]}'},
            "parsed_args": {"markdown": "# 报告", "sources": ["http://ex.com"]},
        }],
    })

    tool_defs = registry.get_openai_tools()
    loop = AgentLoop(registry, mock_llm)

    events = []
    async for evt in loop.run("测试", tool_defs):
        events.append(evt)

    assert loop.final_report == "# 报告"
    types = [e.type for e in events]
    assert "thought" in types
    assert "observation" in types


@pytest.mark.asyncio
async def test_loop_no_tool_calls_terminates(mock_llm, registry):
    mock_llm.chat_with_tools = AsyncMock(return_value={
        "role": "assistant",
        "content": "答案是 42。",
        "tool_calls": [],
    })

    tool_defs = registry.get_openai_tools()
    loop = AgentLoop(registry, mock_llm)

    events = []
    async for evt in loop.run("简单问题", tool_defs):
        events.append(evt)

    assert loop.final_report == "答案是 42。"
    types = [e.type for e in events]
    assert "thought" in types
    assert "observation" in types


@pytest.mark.asyncio
async def test_loop_executes_web_search_then_submit(mock_llm, registry):
    call_count = [0]

    async def side_effect(messages, tools, tool_choice="auto"):
        call_count[0] += 1
        if call_count[0] == 1:
            return {
                "role": "assistant",
                "content": "让我搜索。",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query": "AI"}'},
                    "parsed_args": {"query": "AI"},
                }],
            }
        return {
            "role": "assistant",
            "content": "搜索完毕。",
            "tool_calls": [{
                "id": "call_2",
                "type": "function",
                "function": {"name": "submit_report", "arguments": '{"markdown": "# AI", "sources": ["http://ex.com"]}'},
                "parsed_args": {"markdown": "# AI", "sources": ["http://ex.com"]},
            }],
        }

    mock_llm.chat_with_tools = AsyncMock(side_effect=side_effect)
    tool_defs = registry.get_openai_tools()
    loop = AgentLoop(registry, mock_llm)

    events = []
    async for evt in loop.run("AI 进展", tool_defs):
        events.append(evt)

    types = [e.type for e in events]
    assert types.count("thought") >= 2
    assert types.count("action") >= 1
    assert types.count("observation") >= 2
    assert loop.final_report == "# AI"


@pytest.mark.asyncio
async def test_loop_observation_uses_tool_role(mock_llm, registry):
    captured = []

    async def capture(messages, tools, tool_choice="auto"):
        captured.clear()
        captured.extend(messages)
        return {
            "role": "assistant",
            "content": "searching",
            "tool_calls": [{
                "id": "call_t",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"query": "test"}'},
                "parsed_args": {"query": "test"},
            }],
        }

    mock_llm.chat_with_tools = AsyncMock(side_effect=capture)
    tool_defs = registry.get_openai_tools()
    loop = AgentLoop(registry, mock_llm)

    async for _ in loop.run("test", tool_defs):
        pass

    tool_msgs = [m for m in captured if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1
    for tm in tool_msgs:
        assert "tool_call_id" in tm
        assert "content" in tm
```

- [ ] **Step 7: Run loop tests**

```bash
cd d:/AGENT && PYTHONPATH=backend python -m pytest tests/test_loop.py -v
```

Expected: 4 tests pass

- [ ] **Step 8: Run all tests**

```bash
cd d:/AGENT && PYTHONPATH=backend python -m pytest tests/ -v
```

Expected: ~17 tests all pass

- [ ] **Step 9: Commit**

```bash
git add tests/__init__.py tests/test_registry.py tests/test_client.py tests/test_loop.py
git commit -m "test: 添加 function calling 相关测试 (registry, client, loop)"
```

---

### Task 7: 手动验证 — 启动应用进行端到端测试

**Files:** None

- [ ] **Step 1: Start the backend**

```bash
cd d:/AGENT/backend && uvicorn main:app --reload --port 8000
```

- [ ] **Step 2: Start the frontend** (separate terminal)

```bash
cd d:/AGENT/frontend && npm run dev
```

- [ ] **Step 3: Test a research query**

Open `http://localhost:5173`, enter a question (e.g. "Python 3.13 有哪些新特性"), verify:
- [ ] Plan event shows sub-questions in frontend
- [ ] Thought steps appear with reasoning
- [ ] Action steps show tool names (web_search / fetch_page)
- [ ] Observation steps show search results
- [ ] Final report renders with markdown formatting
- [ ] Sources are extracted and displayed
- [ ] No parser-related errors in backend console

- [ ] **Step 4: Verify history persistence**

Refresh the page, confirm the history list shows the completed session, open detail view and confirm all steps are preserved correctly.
```
