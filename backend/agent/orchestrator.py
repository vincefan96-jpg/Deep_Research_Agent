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
            description="Search the web for information. Returns titles, summaries, and URLs.",
            parameters={"query": {"type": "string", "description": "Search query"}},
            handler=web_search,
        ))
        self.registry.register(Tool(
            name="fetch_page",
            description="Fetch and extract text content from a specific URL. Use this to read articles in depth.",
            parameters={"url": {"type": "string", "description": "Full URL to fetch"}},
            handler=fetch_page,
        ))
        self.registry.register(Tool(
            name="cross_check",
            description="Cross-validate collected facts for consistency. Use before writing final answer.",
            parameters={"facts": {"type": "string", "description": "All collected facts to validate"}},
            handler=cross_check,
        ))

    async def research(self, query: str) -> AsyncGenerator[dict, None]:
        # Phase 1: Generate research plan
        sub_questions = await self.llm.generate_plan(query)
        yield PlanEvent(sub_questions=sub_questions).model_dump()

        # Phase 2: Run ReAct loop
        loop = AgentLoop(self.registry, self.llm)
        all_observations = []

        async for step in loop.run(query):
            if step.type == "observation":
                all_observations.append(step.content)
            yield step.model_dump()

        # Phase 3: Cross-check
        facts = "\n".join(all_observations[-5:])  # Last 5 observations
        cross_result = await cross_check(facts)
        cross_event = CrossCheckEvent(
            consistency="medium",
            conflicts=[],
            verified_facts=[],
        )
        yield cross_event.model_dump()

        # Phase 4: Generate final report
        final_report = await self.llm.chat([
            {
                "role": "system",
                "content": "Write a well-structured research report in Markdown based on the findings. Include a title, section headers, citations as [Source: URL], and a sources list at the end.",
            },
            {
                "role": "user",
                "content": f"Research question: {query}\n\nSub-questions: {', '.join(sub_questions)}\n\nCollected findings:\n{facts}\n\nCross-check result:\n{cross_result}",
            },
        ])

        sources = self._extract_sources(final_report)
        yield ReportEvent(markdown=final_report, sources=sources).model_dump()

    def _extract_sources(self, report: str) -> list[str]:
        import re
        urls = re.findall(r'https?://[^\s\)\]]+', report)
        return list(dict.fromkeys(urls))  # dedupe preserving order
