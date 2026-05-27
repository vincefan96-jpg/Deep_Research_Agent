import httpx
from config import SEARCH_API_KEY, SEARCH_API_URL


async def web_search(query: str, num: int = 5) -> str:
    """Search the web and return formatted results."""
    if not SEARCH_API_KEY:
        return "Search API key not configured."

    try:
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
            title = r.get("title", "No title")
            snippet = r.get("snippet", "No description")
            link = r.get("link", "No link")
            results.append(f"[{i}] {title}\n    Summary: {snippet}\n    URL: {link}")

        if not results:
            return f"No results found for '{query}'."

        return "\n\n".join(results)

    except httpx.TimeoutException:
        return f"Search timed out for query '{query}'. Please try a narrower query."
    except Exception as e:
        return f"Search failed for '{query}': {e}"
