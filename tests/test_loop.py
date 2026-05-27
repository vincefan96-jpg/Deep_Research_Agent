import pytest
from unittest.mock import AsyncMock
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry, Tool


async def dummy_search(query: str = "") -> str:
    return f"Search results for: {query}"


@pytest.mark.asyncio
async def test_loop_stops_on_final_answer():
    reg = ToolRegistry()
    reg.register(Tool(name="web_search", description="Search", parameters={"query": {"type": "string"}}, handler=dummy_search))

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = [
        'THOUGHT: searching\nACTION: web_search|{"query": "test"}',
        "THOUGHT: done\nFINAL_ANSWER: The report content",
    ]

    loop = AgentLoop(reg, mock_llm)
    events = []
    async for event in loop.run("test query"):
        events.append(event)

    assert len(events) >= 3  # thought, action, observation, thought, final
    assert events[-1].type == "observation"


@pytest.mark.asyncio
async def test_loop_parse_error_retry():
    reg = ToolRegistry()
    reg.register(Tool(name="web_search", description="Search", parameters={"query": {"type": "string"}}, handler=dummy_search))

    mock_llm = AsyncMock()
    mock_llm.chat.side_effect = [
        "garbled output without proper format",
        "THOUGHT: retry\nFINAL_ANSWER: Fixed report",
    ]

    loop = AgentLoop(reg, mock_llm)
    events = []
    async for event in loop.run("test query"):
        events.append(event)

    assert mock_llm.chat.call_count >= 2
