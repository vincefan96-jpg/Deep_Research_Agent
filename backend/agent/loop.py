import json
from typing import AsyncGenerator
from config import MAX_ROUNDS
from models.schemas import Step, StepType, StepEvent
from agent.tools.registry import ToolRegistry
from agent.parser import parse
from agent.memory import MemoryManager
from llm.client import LLMClient


SYSTEM_PROMPT = """你是一个深度调研智能体。你的目标是对研究主题进行彻底调查，生成一份全面、来源可靠的答案。

每轮请按以下格式回复：

THOUGHT: <分析目前已知信息，是否充分？下一步深入调研的方向是什么？>

然后选择：
ACTION: <工具名>|<JSON参数>

或者当信息足够时：
FINAL_ANSWER: <用 Markdown 格式编写的综合调研报告，包含引用来源>

可用工具：
{tool_descriptions}

规则：
- 先搜索后阅读：先用 web_search 搜索，再对最有价值的 URL 使用 fetch_page 深入阅读
- 多源交叉验证：重要结论不能只依赖单一来源
- 生成最终答案前必须对事实进行交叉验证
- 引用来源格式：[来源: URL]
- 如果搜索失败或无结果，调整策略或诚实地报告未能找到的内容"""


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
            {"role": "user", "content": f"研究问题：{query}"},
        ]

        for round_num in range(1, MAX_ROUNDS + 1):
            raw_output = await self.llm.chat(messages)

            parsed = parse(raw_output)

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

            # 解析失败时重试一次
            if parsed.parse_error and not parsed.is_final:
                messages.append({"role": "assistant", "content": raw_output})
                messages.append({
                    "role": "user",
                    "content": f"你的输出格式不正确：{parsed.parse_error}。"
                               f"请严格按照格式：先写 THOUGHT，然后写 ACTION 或 FINAL_ANSWER。",
                })
                continue

            # 执行工具
            if parsed.action:
                action_step = StepEvent(
                    type="action",
                    round=round_num,
                    content=f"调用 {parsed.action}",
                    tool_name=parsed.action,
                    tool_params=parsed.action_params,
                )
                self.memory.add_step(Step(
                    round=round_num,
                    type=StepType.ACTION,
                    content=f"调用 {parsed.action}，参数 {json.dumps(parsed.action_params or {}, ensure_ascii=False)}",
                    tool_name=parsed.action,
                    tool_params=parsed.action_params,
                ))
                yield action_step

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

                messages.append({"role": "assistant", "content": raw_output})
                messages.append({"role": "user", "content": f"OBSERVATION:\n{observation}"})

                if len(observation) > 200:
                    self.memory.add_key_fact(observation[:300])

            # 处理最终答案
            if parsed.is_final:
                final_content = parsed.final_answer or parsed.thought
                yield StepEvent(
                    type="observation",
                    round=round_num,
                    content=final_content,
                )
                return

        # 达到最大轮数：强制生成摘要
        final_content = await self.llm.chat([
            {"role": "user", "content": f"根据以下调研结果撰写一份调研报告，包含引用来源。\n\n调研结果：\n{self.memory.get_steps_for_context()}"}
        ])
        yield StepEvent(
            type="observation",
            round=MAX_ROUNDS,
            content=final_content,
        )
