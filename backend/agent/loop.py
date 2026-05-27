import json
from typing import AsyncGenerator
from config import MAX_ROUNDS
from models.schemas import Step, StepType, StepEvent
from agent.tools.registry import ToolRegistry
from agent.parser import parse
from agent.memory import MemoryManager
from llm.client import LLMClient


SYSTEM_PROMPT = """You are a deep research agent. Your goal is to thoroughly investigate a topic and produce a comprehensive, well-sourced answer.

For each round, respond in this exact format:

THOUGHT: <Analyze what you know so far. Is the information sufficient? What is the next logical step to deepen the research?>

Then either:
ACTION: <tool_name>|<json_params>

Or when you have enough information:
FINAL_ANSWER: <Your comprehensive research report in Markdown format, with citations>

Available tools:
{tool_descriptions}

Rules:
- Search before reading: use web_search first, then fetch_page for the most promising URLs
- Verify across sources: don't rely on a single source for important claims
- Cross-check facts before concluding
- Cite sources inline like [Source: URL]
- If search fails or returns nothing useful, adapt your strategy or report honestly what you couldn't find"""


class AgentLoop:
    def __init__(self, tool_registry: ToolRegistry, llm: LLMClient):
        self.tool_registry = tool_registry
        self.llm = llm
        self.memory = MemoryManager()

    async def run(self, query: str) -> AsyncGenerator[StepEvent, None]:
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    tool_descriptions=self.tool_registry.get_schema_for_llm()
                ),
            },
            {"role": "user", "content": f"Research question: {query}"},
        ]

        for round_num in range(1, MAX_ROUNDS + 1):
            # Get LLM response
            raw_output = await self.llm.chat(messages)

            # Parse
            parsed = parse(raw_output)

            # Emit thought
            if parsed.thought:
                thought_step = StepEvent(
                    type="thought",
                    round=round_num,
                    content=parsed.thought,
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.THOUGHT,
                    content=parsed.thought,
                ))
                yield thought_step

            # Handle parse error: retry once
            if parsed.parse_error and not parsed.is_final:
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": f"Your output format was invalid: {parsed.parse_error}. "
                               f"Please follow the format exactly: THOUGHT, then ACTION or FINAL_ANSWER.",
                })
                continue

            # Execute action
            if parsed.action:
                action_step = StepEvent(
                    type="action",
                    round=round_num,
                    content=f"Calling {parsed.action}",
                    tool_name=parsed.action,
                    tool_params=parsed.action_params,
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.ACTION,
                    content=f"Calling {parsed.action} with {json.dumps(parsed.action_params or {})}",
                    tool_name=parsed.action,
                    tool_params=parsed.action_params,
                ))
                yield action_step

                # Execute tool
                observation = await self.tool_registry.execute(
                    parsed.action,
                    parsed.action_params or {},
                )

                obs_step = StepEvent(
                    type="observation",
                    round=round_num,
                    content=observation[:2000],
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.OBSERVATION,
                    content=observation[:2000],
                ))
                yield obs_step

                # Append to messages
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}"})

                # Extract potential key facts
                if len(observation) > 200:
                    self.memory.add_key_fact(observation[:300])

            # Handle final answer
            if parsed.is_final:
                final_content = parsed.final_answer or parsed.thought
                yield StepEvent(
                    type="observation",
                    round=round_num,
                    content=final_content,
                )
                return

        # Max rounds reached: force summary
        final_content = await self.llm.chat([
            {"role": "user", "content": f"Write a research report based on the following findings. Include citations.\n\nFindings:\n{self.memory.get_steps_for_context()}"}
        ])
        yield StepEvent(
            type="observation",
            round=MAX_ROUNDS,
            content=final_content,
        )
