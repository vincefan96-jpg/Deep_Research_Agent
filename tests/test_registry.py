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
