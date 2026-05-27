import pytest
from agent.parser import parse


def test_parse_thought_and_action():
    raw = 'THOUGHT: need to search\nACTION: web_search|{"query": "test"}'
    out = parse(raw)
    assert out.thought == "need to search"
    assert out.action == "web_search"
    assert out.action_params == {"query": "test"}
    assert not out.is_final
    assert out.parse_error is None


def test_parse_final_answer():
    raw = 'THOUGHT: done researching\nFINAL_ANSWER: Here is the report...'
    out = parse(raw)
    assert out.is_final
    assert "Here is the report" in out.final_answer


def test_parse_bad_json_params():
    raw = 'THOUGHT: test\nACTION: web_search|{bad json}'
    out = parse(raw)
    assert out.action == "web_search"
    assert out.parse_error is not None


def test_parse_unparseable():
    raw = "just some random text without proper format"
    out = parse(raw)
    assert out.parse_error is not None


def test_parse_lowercase():
    raw = 'thought: need to search\naction: web_search|{"query": "test"}'
    out = parse(raw)
    assert out.action == "web_search"


def test_parse_multiline_thought():
    raw = 'THOUGHT: line one\nline two\nline three\nACTION: web_search|{"q": "x"}'
    out = parse(raw)
    assert "line one" in out.thought
    assert out.action == "web_search"
