import httpx
from bs4 import BeautifulSoup


async def fetch_page(url: str, max_chars: int = 5000) -> str:
    """抓取网页并提取纯文本内容。"""
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)",
                    "Accept": "text/html",
                },
                follow_redirects=True,
            )
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        cleaned = "\n".join(lines)
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + f"\n\n[内容已截断，超过 {max_chars} 字符]"

        title = soup.title.string.strip() if soup.title else url
        return f"标题：{title}\n链接：{url}\n\n{cleaned if cleaned else '未提取到文本内容。'}"

    except httpx.TimeoutException:
        return f"抓取 {url} 超时，将使用搜索摘要。"
    except Exception as e:
        return f"抓取 {url} 失败：{e}，将使用搜索摘要。"
