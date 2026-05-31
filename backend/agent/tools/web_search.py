import httpx
from config import SEARCH_API_KEY, SEARCH_API_URL


async def web_search(query: str, num: int = 5) -> str:
    """在互联网上搜索并返回格式化的结果。"""
    if not SEARCH_API_KEY:
        return "搜索 API key 未配置。"

    try:#用 async with 而非直接 await client.get()，确保无论成功还是异常，连接都能被正确释放
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                SEARCH_API_URL,
                params={
                    "q": query,
                    "api_key": SEARCH_API_KEY,
                    "num": num,
                    "engine": "google",
                },
            )
            response.raise_for_status()
            data = response.json()

        results = []
        organic = data.get("organic_results", [])
        for i, r in enumerate(organic[:num], 1):
            title = r.get("title", "无标题")
            snippet = r.get("snippet", "无描述")
            link = r.get("link", "无链接")
            results.append(f"[{i}] {title}\n    摘要：{snippet}\n    链接：{link}")

        if not results:
            return f"未找到与「{query}」相关的结果。"

        return "\n\n".join(results)

    except httpx.TimeoutException: #TimeoutException — 精确捕获超时，给出针对性建议（缩短搜索词），而不是模糊的"失败了
        return f"搜索「{query}」超时，请尝试更精确的搜索词。"
    except Exception as e:
        return f"搜索「{query}」失败：{e}" #Exception — 兜底捕获所有其他异常
