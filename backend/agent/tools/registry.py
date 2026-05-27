from dataclasses import dataclass, field
from typing import Callable, Any
import json


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schema_for_llm(self) -> str:
        """Generate a text description of all tools for the LLM prompt."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            lines.append(f"  参数：{json.dumps(tool.parameters, ensure_ascii=False)}")
        return "\n".join(lines)

    async def execute(self, name: str, params: dict) -> str:
        tool = self.get(name)
        if tool is None:
            return f"错误：工具 '{name}' 未找到。可用工具：{list(self._tools.keys())}"
        try:
            result = await tool.handler(**params)
            return str(result)
        except Exception as e:
            return f"执行 '{name}' 出错：{e}"
