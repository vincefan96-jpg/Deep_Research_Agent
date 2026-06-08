import json
from typing import AsyncGenerator
from config import MAX_ROUNDS
from models.schemas import Step, StepType, StepEvent
from agent.tools.registry import ToolRegistry
from agent.memory import MemoryManager
from llm.client import LLMClient


SYSTEM_PROMPT = """你是一个深度调研智能体。你的目标是对研究主题进行彻底调查，生成一份全面、来源可靠的答案。

你可以使用提供的工具进行搜索和抓取网页。当信息收集充分时，调用 submit_report 提交最终报告。

规则：
- 先搜索后阅读：先用 web_search 搜索，再对最有价值的 URL 使用 fetch_page 深入阅读
- 多源验证：重要结论不能只依赖单一来源
- 引用来源格式：[来源: URL]
- 如果搜索失败或无结果，调整策略或诚实地报告未能找到的内容"""


class AgentLoop:
    def __init__(self, tool_registry: ToolRegistry, llm: LLMClient):
        self.tool_registry = tool_registry
        self.llm = llm
        self.memory = MemoryManager()
        self.final_report = ""

    async def run(self, query: str, tool_defs: list[dict]) -> AsyncGenerator[StepEvent, None]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"研究问题：{query}"},
        ]

        for round_num in range(1, MAX_ROUNDS + 1):
            response = await self.llm.chat_with_tools(messages, tools=tool_defs)

            thought = response.get("content", "")
            tool_calls = response.get("tool_calls", [])

            if thought:
                evt = StepEvent(type="thought", round=round_num, content=thought)
                self.memory.add_step(Step(round=round_num, type=StepType.THOUGHT, content=thought))
                yield evt

            if not tool_calls:
                self.final_report = thought or "未能生成报告。"
                yield StepEvent(type="observation", round=round_num, content=self.final_report)
                return

            # Build clean assistant message for API (strip parsed_args)
            clean_msg = {"role": "assistant", "content": thought}
            clean_msg["tool_calls"] = [
                {k: v for k, v in tc.items() if k != "parsed_args"}
                for tc in tool_calls
            ]
            messages.append(clean_msg)

            for i, tc in enumerate(tool_calls):
                tool_name = tc["function"]["name"]
                params = tc.get("parsed_args", {})
                tc_id = tc.get("id") or f"call_{round_num}_{i}"

                if tool_name == "submit_report":
                    markdown = params.get("markdown") or thought or "未能生成报告。"
                    self.final_report = markdown
                    yield StepEvent(type="observation", round=round_num, content=markdown)
                    return

                # Yield action
                action_evt = StepEvent(
                    type="action", round=round_num,
                    content=f"调用 {tool_name}",
                    tool_name=tool_name, tool_params=params,
                )
                self.memory.add_step(Step(
                    round=round_num, type=StepType.ACTION,
                    content=f"调用 {tool_name}，参数 {json.dumps(params, ensure_ascii=False)}",
                    tool_name=tool_name, tool_params=params,
                ))
                yield action_evt

                observation = await self.tool_registry.execute(tool_name, params)

                # Yield observation
                obs_evt = StepEvent(
                    type="observation", round=round_num,
                    content=observation[:2000],
                )
                self.memory.add_step(Step(
                    round=round_num, type=StepType.OBSERVATION,
                    content=observation[:2000],
                ))
                yield obs_evt

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": observation,
                })

                if len(observation) > 200:
                    self.memory.add_key_fact(observation[:300])

        # MAX_ROUNDS reached — force summary
        self.final_report = await self.llm.chat([
            {
                "role": "user",
                "content": f"根据以下调研结果撰写一份调研报告，包含引用来源。\n\n"
                           f"调研结果：\n{self.memory.get_steps_for_context()}"
            }
        ])
        yield StepEvent(type="observation", round=MAX_ROUNDS, content=self.final_report)
