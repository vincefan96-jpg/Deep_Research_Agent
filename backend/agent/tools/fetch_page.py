import httpx
from bs4 import BeautifulSoup


async def fetch_page(url: str, max_chars: int = 5000) -> str:
    """Fetch a web page and return extracted text content."""
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
            cleaned = cleaned[:max_chars] + f"\n\n[Truncated at {max_chars} characters]"

        title = soup.title.string.strip() if soup.title else url
        return f"Title: {title}\nURL: {url}\n\n{cleaned if cleaned else 'No text content extracted.'}"

    except httpx.TimeoutException:
        return f"Failed to fetch {url}: request timed out. Using search snippet instead."
    except Exception as e:
        return f"Failed to fetch {url}: {e}. Using search snippet instead."
