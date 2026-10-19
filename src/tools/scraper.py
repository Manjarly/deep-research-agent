"""Resilient web scraper and text extraction tool."""
import logging
import re
from typing import Optional
import requests
from bs4 import BeautifulSoup
import html2text

from ..config import config

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


class WebScraperTool:
    """Fetches and extracts clean, readable text from web pages."""

    def __init__(self, max_chars: Optional[int] = None, timeout: int = 8):
        self.max_chars = max_chars or config.MAX_SCRAPE_CHARS
        self.timeout = timeout
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        self.h2t.ignore_images = True
        self.h2t.ignore_emphasis = False
        self.h2t.body_width = 0

    def scrape_url(self, url: str) -> Optional[str]:
        """Fetch and extract clean markdown/text content from a URL."""
        if not url or not url.startswith(("http://", "https://")):
            return None

        # Skip binaries, pdfs, audio, images
        lower_url = url.lower()
        if any(lower_url.endswith(ext) for ext in [".pdf", ".jpg", ".png", ".mp4", ".zip", ".exe"]):
            return None

        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=self.timeout,
                allow_redirects=True,
                stream=True
            )
            # Avoid downloading massive non-text files
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type and "text/plain" not in content_type:
                return None

            # Read first 1MB max
            content_bytes = response.raw.read(1024 * 1024)
            html_text = content_bytes.decode(response.encoding or "utf-8", errors="replace")

            soup = BeautifulSoup(html_text, "html.parser")

            # Remove clutter elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "svg", "noscript"]):
                tag.decompose()

            # Prefer main article container if available
            main_container = soup.find(["article", "main"]) or soup.find(id=re.compile(r"content|main|article", re.I))
            target_html = str(main_container) if main_container else str(soup.body or soup)

            # Convert to markdown text
            markdown_content = self.h2t.handle(target_html)

            # Clean extra whitespaces and empty lines
            cleaned = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()

            if len(cleaned) > self.max_chars:
                cleaned = cleaned[:self.max_chars] + "\n... [Content truncated for context window]"

            return cleaned if len(cleaned) > 100 else None

        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            return None
