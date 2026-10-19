"""Researcher Agent: Executes search queries, scrapes web pages, and compiles evidence."""
import logging
from typing import List, Dict, Any, Set
from ..models import SearchResultItem
from ..tools.search import WebSearchTool
from ..tools.scraper import WebScraperTool

logger = logging.getLogger(__name__)


def conduct_research(
    queries: List[str],
    existing_evidence: List[Dict[str, Any]],
    search_tool: WebSearchTool,
    scraper_tool: WebScraperTool,
    max_queries_per_step: int = 4
) -> List[Dict[str, Any]]:
    """Execute search queries, scrape content, and deduplicate findings."""
    known_urls: Set[str] = {
        item.get("url") for item in existing_evidence if item.get("url")
    }
    new_evidence: List[Dict[str, Any]] = list(existing_evidence)

    # Limit queries per batch to avoid rate limiting and excessive requests
    selected_queries = queries[:max_queries_per_step]

    for query in selected_queries:
        logger.info(f"Searching web for: {query}")
        hits: List[SearchResultItem] = search_tool.search(query)

        for hit in hits:
            if not hit.url or hit.url in known_urls:
                continue

            known_urls.add(hit.url)

            # Attempt full-text scraping
            scraped_content = scraper_tool.scrape_url(hit.url)
            hit_data = hit.model_dump()
            if scraped_content:
                hit_data["content"] = scraped_content
            else:
                hit_data["content"] = hit.snippet

            new_evidence.append(hit_data)

    return new_evidence
