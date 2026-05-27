from openai import AsyncOpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


class LLMClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
        )
        self.model = OPENAI_MODEL

    async def chat(self, messages: list[dict], max_tokens: int = 4096) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    async def generate_plan(self, query: str) -> list[str]:
        prompt = f"""Break the following research question into 2-4 specific sub-questions.
Return ONLY a JSON array of strings, no other text.
Question: {query}"""
        raw = await self.chat([{"role": "user", "content": prompt}])
        import json
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return [query]

    async def summarize(self, text: str, max_length: int = 500) -> str:
        prompt = f"Summarize the following in under {max_length} characters:\n\n{text}"
        return await self.chat([{"role": "user", "content": prompt}])
