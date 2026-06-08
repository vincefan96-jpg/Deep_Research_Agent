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
