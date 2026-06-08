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
    http_source = "http://source.com"

    async def capture(messages, tools, tool_choice="auto"):
        captured.clear()
        captured.extend(messages)
        # Return web_search on first call to trigger tool execution
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

    # On second call, return submit_report to terminate the loop
    async def side_effect(messages, tools, tool_choice="auto"):
        captured.clear()
        captured.extend(messages)
        if not any(m.get("role") == "tool" for m in messages):
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
        return {
            "role": "assistant",
            "content": "完成",
            "tool_calls": [{
                "id": "call_r",
                "type": "function",
                "function": {"name": "submit_report", "arguments": f'{{"markdown": "# R", "sources": ["{http_source}"]}}'},
                "parsed_args": {"markdown": "# R", "sources": [http_source]},
            }],
        }

    mock_llm.chat_with_tools = AsyncMock(side_effect=side_effect)
    tool_defs = registry.get_openai_tools()
    loop = AgentLoop(registry, mock_llm)

    async for _ in loop.run("test", tool_defs):
        pass

    tool_msgs = [m for m in captured if m.get("role") == "tool"]
    assert len(tool_msgs) >= 1
    for tm in tool_msgs:
        assert "tool_call_id" in tm
        assert "content" in tm
