"""Web search provider integrating DuckDuckGo, Tavily, and Wikipedia fallback."""
import logging
import time
from typing import List, Optional
from ..config import config
from ..models import SearchResultItem

logger = logging.getLogger(__name__)


class WebSearchTool:
    """Multi-provider search tool with automatic fallback."""

    def __init__(self, max_results: Optional[int] = None):
        self.max_results = max_results or config.MAX_SEARCH_RESULTS_PER_QUERY
        self.tavily_client = None

        if config.TAVILY_API_KEY:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)
            except Exception as e:
                logger.warning(f"Could not initialize Tavily client: {e}")

    def search(self, query: str) -> List[SearchResultItem]:
        """Execute search using best available provider."""
        query = query.strip()
        if not query:
            return []

        # 1. Try Tavily if configured
        if self.tavily_client:
            try:
                results = self._search_tavily(query)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}. Falling back to DuckDuckGo.")

        # 2. Try DuckDuckGo / ddgs
        try:
            results = self._search_duckduckgo(query)
            if results:
                return results
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")

        # 3. Fallback: Wikipedia search for grounding
        try:
            results = self._search_wikipedia(query)
            if results:
                return results
        except Exception as e:
            logger.warning(f"Wikipedia search fallback error: {e}")

        return []

    def _search_tavily(self, query: str) -> List[SearchResultItem]:
        response = self.tavily_client.search(
            query=query,
            max_results=self.max_results,
            search_depth="advanced",
            include_raw_content=False
        )
        items = []
        for r in response.get("results", []):
            items.append(SearchResultItem(
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                source_query=query
            ))
        return items

    def _search_duckduckgo(self, query: str) -> List[SearchResultItem]:
        items: List[SearchResultItem] = []
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                for r in results:
                    title = r.get("title", "")
                    url = r.get("href", "")
                    snippet = r.get("body", "")
                    if title and url:
                        items.append(SearchResultItem(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source_query=query
                        ))
        except Exception:
            # Try alternate import if needed
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                for r in results:
                    title = r.get("title", "")
                    url = r.get("href", "")
                    snippet = r.get("body", "")
                    if title and url:
                        items.append(SearchResultItem(
                            title=title,
                            url=url,
                            snippet=snippet,
                            source_query=query
                        ))
        return items

    def _search_wikipedia(self, query: str) -> List[SearchResultItem]:
        import wikipedia
        items: List[SearchResultItem] = []
        search_hits = wikipedia.search(query, results=min(self.max_results, 3))
        for hit in search_hits:
            try:
                page = wikipedia.page(hit, auto_suggest=False)
                items.append(SearchResultItem(
                    title=f"{page.title} (Wikipedia)",
                    url=page.url,
                    snippet=page.summary[:500] + "...",
                    source_query=query
                ))
            except Exception:
                continue
        return items
