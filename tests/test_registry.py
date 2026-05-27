import pytest
from agent.tools.registry import Tool, ToolRegistry


async def echo_handler(text: str = "") -> str:
    return f"echo: {text}"


def test_register_and_get():
    reg = ToolRegistry()
    tool = Tool(name="echo", description="Echoes text", parameters={"text": {"type": "string"}}, handler=echo_handler)
    reg.register(tool)
    assert reg.get("echo") is tool
    assert reg.get("nonexistent") is None


def test_schema_generation():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="Echoes text", parameters={"text": {"type": "string"}}, handler=echo_handler))
    schema = reg.get_schema_for_llm()
    assert "echo" in schema
    assert "Echoes text" in schema


@pytest.mark.asyncio
async def test_execute():
    reg = ToolRegistry()
    reg.register(Tool(name="echo", description="Echoes text", parameters={"text": {"type": "string"}}, handler=echo_handler))
    result = await reg.execute("echo", {"text": "hello"})
    assert result == "echo: hello"


@pytest.mark.asyncio
async def test_execute_unknown_tool():
    reg = ToolRegistry()
    result = await reg.execute("unknown", {})
    assert "未找到" in result
