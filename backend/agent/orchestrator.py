from typing import AsyncGenerator
from models.schemas import StepEvent, PlanEvent, CrossCheckEvent, ReportEvent
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry
from agent.tools.web_search import web_search
from agent.tools.fetch_page import fetch_page
from agent.tools.cross_check import cross_check
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
        self.registry.register(Tool(
            name="cross_check",
            description="对已收集的事实进行交叉验证，检查一致性。在撰写最终答案前使用。",
            parameters={"facts": {"type": "string", "description": "待验证的所有收集事实"}},
            handler=cross_check,
        ))

    async def research(self, query: str) -> AsyncGenerator[dict, None]:
        # 阶段一：生成调研计划
        sub_questions = await self.llm.generate_plan(query)
        yield PlanEvent(sub_questions=sub_questions).model_dump()

        # 阶段二：运行 ReAct 循环
        loop = AgentLoop(self.registry, self.llm)
        all_observations = []

        async for step in loop.run(query):
            if step.type == "observation":
                all_observations.append(step.content)
            yield step.model_dump()

        # 阶段三：交叉验证
        facts = "\n".join(all_observations[-5:])  # 最近 5 条观察
        cross_result = await cross_check(facts)
        cross_event = CrossCheckEvent(
            consistency="medium",
            conflicts=[],
            verified_facts=[],
        )
        yield cross_event.model_dump()

        # 阶段四：生成最终报告
        final_report = await self.llm.chat([
            {
                "role": "system",
                "content": "根据调研结果撰写一份结构良好的 Markdown 调研报告。包含标题、章节标题、内联引用 [来源: URL]，以及文末的来源列表。",
            },
            {
                "role": "user",
                "content": f"研究问题：{query}\n\n子问题：{', '.join(sub_questions)}\n\n收集到的信息：\n{facts}\n\n交叉验证结果：\n{cross_result}",
            },
        ])

        sources = self._extract_sources(final_report)
        yield ReportEvent(markdown=final_report, sources=sources).model_dump()

    def _extract_sources(self, report: str) -> list[str]:
        import re
        urls = re.findall(r'https?://[^\s\)\]]+', report)
        return list(dict.fromkeys(urls))  # 去重保留顺序
