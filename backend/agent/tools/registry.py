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
