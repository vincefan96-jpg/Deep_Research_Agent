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
