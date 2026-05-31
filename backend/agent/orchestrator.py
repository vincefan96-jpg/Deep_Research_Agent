from typing import AsyncGenerator
from models.schemas import PlanEvent, ReportEvent
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry
from agent.tools.web_search import web_search
from agent.tools.fetch_page import fetch_page
from llm.client import LLMClient


class ResearchOrchestrator:
    def __init__(self):
        self.llm = LLMClient()
        self.registry = ToolRegistry()
        self._setup_tools()

    def _setup_tools(self):
        from agent.tools.registry import Tool
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

    async def research(self, query: str) -> AsyncGenerator[dict, None]:
        # 阶段一：生成调研计划
        sub_questions = await self.llm.generate_plan(query) #返回值是 list[str]，例如 ["什么是 X？", "X 的主要应用场景？", "X 的优缺点？"]
        yield PlanEvent(sub_questions=sub_questions).model_dump()

        # 阶段二：运行 ReAct 循环
        loop = AgentLoop(self.registry, self.llm)

        async for step in loop.run(query): #loop.run(query) 返回一个异步生成器，每轮产出 StepEvent（可能是 thought、action 或 observation）
            yield step.model_dump()

        # 阶段三：生成最终报告
        observations = "\n".join([
            s.content for s in loop.memory.steps
            if s.type.value == "observation"
        ][-5:])
        facts = observations if observations else "未收集到信息"
        final_report = await self.llm.chat([
            {
                "role": "system",
                "content": "根据调研结果撰写一份结构良好的 Markdown 调研报告。包含标题、章节标题、内联引用 [来源: URL]，以及文末的来源列表。",
            },
            {
                "role": "user",
                "content": f"研究问题：{query}\n\n子问题：{', '.join(sub_questions)}\n\n收集到的信息：\n{facts}",
            },
        ])

        sources = self._extract_sources(final_report)
        yield ReportEvent(markdown=final_report, sources=sources).model_dump()

    def _extract_sources(self, report: str) -> list[str]:
        import re
        urls = re.findall(r'https?://[^\s\)\]]+', report) #用正则匹配所有 http:// 或 https:// 开头的 URL
        return list(dict.fromkeys(urls))  # 去重保留顺序
