from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.model = OPENAI_MODEL

    #通用对话方法。使用 temperature=0.3（低温度保证输出稳定性），发送消息列表并返回 LLM 的文本响应。所有 LLM 调用最终都通过此方法。
    async def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""
    #生成调研计划。发送特定 prompt 要求 LLM 将问题拆分为 2-4 个子问题，返回 JSON 字符串数组。如果 JSON 解析失败，降级为返回 [原问题]
    async def generate_plan(self, query: str) -> list[str]:
        prompt = f"""将以下研究问题拆分为 2-4 个具体的子问题。
只返回一个 JSON 字符串数组，不要包含其他内容。
问题：{query}"""
        raw = await self.chat([{"role": "user", "content": prompt}])
        import json
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [query]

    async def summarize(self, text: str, max_length: int = 500) -> str:
        prompt = f"请将以下内容总结在 {max_length} 字以内：\n\n{text}"
        return await self.chat([{"role": "user", "content": prompt}])
