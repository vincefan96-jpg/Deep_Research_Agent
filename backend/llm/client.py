import json
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

    async def chat_json_mode(self, messages: list[dict], max_tokens: int = 2048) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    async def chat_with_tools(
        self, messages: list[dict], tools: list[dict], tool_choice: str = "auto"
    ) -> dict:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.3,
            tools=tools,
            tool_choice=tool_choice,
        )
        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = {}
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    pass
                tool_calls.append({
                    "id": tc.id or "",
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "parsed_args": args,
                })
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
        }

    async def summarize(self, text: str, max_length: int = 500) -> str:
        prompt = f"请将以下内容总结在 {max_length} 字以内：\n\n{text}"
        return await self.chat([{"role": "user", "content": prompt}])
