"""
Web Tools Module

Tools for fetching, parsing, and analyzing web content.

CAPABILITIES:
- Fetch URLs (HTTP/HTTPS)
- Parse HTML into structured data
- Extract links, text, metadata
- Basic web search functionality

LEARNING POINTS:
- Tools are stateless functions
- They handle errors gracefully
- They return structured data for agents to process
- Production versions might use Selenium, Playwright for JS-heavy sites
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
import re
import logging
import json

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("BeautifulSoup not installed. HTML parsing will be limited.")


@dataclass
class WebPage:
    """
    Represents a fetched web page.

    Contains both raw content and parsed metadata.
    """
    url: str
    status_code: int
    content: str
    content_type: str
    title: Optional[str] = None
    links: List[str] = None
    text: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.links is None:
            self.links = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """
    A single search result.
    """
    title: str
    url: str
    snippet: str
    rank: int


class WebFetcher:
    """
    Async web content fetcher with caching and rate limiting.

    FEATURES:
    - Async HTTP requests
    - Request caching
    - Rate limiting
    - Retry logic
    - User agent rotation

    USAGE:
        fetcher = WebFetcher()
        page = await fetcher.fetch("https://example.com")
        print(page.title, page.text[:100])
    """

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        cache_enabled: bool = True,
        rate_limit: float = 1.0  # seconds between requests to same domain
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_enabled = cache_enabled
        self.rate_limit = rate_limit

        self._cache: Dict[str, WebPage] = {}
        self._last_request: Dict[str, float] = {}  # domain -> timestamp
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.DEFAULT_HEADERS
            )
        return self._session

    async def close(self) -> None:
        """Close the session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _respect_rate_limit(self, url: str) -> None:
        """Wait if needed to respect rate limiting."""
        domain = urlparse(url).netloc
        import time

        if domain in self._last_request:
            elapsed = time.time() - self._last_request[domain]
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)

        self._last_request[domain] = time.time()

    async def fetch(self, url: str, parse: bool = True) -> WebPage:
        """
        Fetch a URL and optionally parse its content.

        Args:
            url: The URL to fetch
            parse: Whether to parse HTML content

        Returns:
            WebPage object with content and metadata
        """
        # Check cache
        if self.cache_enabled and url in self._cache:
            logger.debug(f"Cache hit: {url}")
            return self._cache[url]

        # Rate limiting
        await self._respect_rate_limit(url)

        # Fetch with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                session = await self._get_session()
                async with session.get(url) as response:
                    content = await response.text()
                    content_type = response.headers.get("Content-Type", "")

                    page = WebPage(
                        url=url,
                        status_code=response.status,
                        content=content,
                        content_type=content_type
                    )

                    # Parse HTML if requested and content is HTML
                    if parse and "html" in content_type.lower():
                        page = self._parse_html(page)

                    # Cache the result
                    if self.cache_enabled:
                        self._cache[url] = page

                    return page

            except Exception as e:
                last_error = e
                logger.warning(f"Fetch attempt {attempt + 1} failed for {url}: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed
        return WebPage(
            url=url,
            status_code=0,
            content="",
            content_type="",
            metadata={"error": str(last_error)}
        )

    def _parse_html(self, page: WebPage) -> WebPage:
        """Parse HTML content to extract structured data."""
        if not HAS_BS4:
            # Fallback: basic regex parsing
            return self._parse_html_basic(page)

        soup = BeautifulSoup(page.content, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        page.title = title_tag.get_text(strip=True) if title_tag else None

        # Extract all links
        page.links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            # Convert relative URLs to absolute
            absolute_url = urljoin(page.url, href)
            page.links.append(absolute_url)

        # Extract main text content
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        page.text = soup.get_text(separator="\n", strip=True)

        # Extract metadata
        page.metadata = {}
        for meta in soup.find_all("meta"):
            name = meta.get("name") or meta.get("property", "")
            content = meta.get("content", "")
            if name and content:
                page.metadata[name] = content

        return page

    def _parse_html_basic(self, page: WebPage) -> WebPage:
        """Basic HTML parsing without BeautifulSoup."""
        content = page.content

        # Extract title
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
        page.title = title_match.group(1).strip() if title_match else None

        # Extract links
        link_pattern = r'href=["\']([^"\']+)["\']'
        page.links = [
            urljoin(page.url, href)
            for href in re.findall(link_pattern, content)
        ]

        # Extract text (very basic - remove all tags)
        text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        page.text = text

        return page

    def clear_cache(self) -> None:
        """Clear the URL cache."""
        self._cache.clear()


# ============================================================================
# STANDALONE FUNCTIONS (for simple use cases)
# ============================================================================

async def fetch_url(url: str, timeout: int = 30) -> WebPage:
    """
    Simple function to fetch a URL.

    For more control, use WebFetcher class directly.
    """
    fetcher = WebFetcher(timeout=timeout, cache_enabled=False)
    try:
        return await fetcher.fetch(url)
    finally:
        await fetcher.close()


def parse_html(html_content: str, base_url: str = "") -> Dict[str, Any]:
    """
    Parse HTML content into structured data.

    Args:
        html_content: Raw HTML string
        base_url: Base URL for resolving relative links

    Returns:
        Dictionary with title, text, links, metadata
    """
    page = WebPage(
        url=base_url,
        status_code=200,
        content=html_content,
        content_type="text/html"
    )

    fetcher = WebFetcher()
    parsed = fetcher._parse_html(page)

    return {
        "title": parsed.title,
        "text": parsed.text,
        "links": parsed.links,
        "metadata": parsed.metadata
    }


def extract_links(html_content: str, base_url: str = "") -> List[str]:
    """
    Extract all links from HTML content.

    Args:
        html_content: Raw HTML string
        base_url: Base URL for resolving relative links

    Returns:
        List of absolute URLs
    """
    pattern = r'href=["\']([^"\']+)["\']'
    links = re.findall(pattern, html_content)
    return [urljoin(base_url, link) for link in links]


def extract_text(html_content: str) -> str:
    """
    Extract plain text from HTML content.

    Removes scripts, styles, and HTML tags.
    """
    # Remove script and style content
    text = re.sub(r"<script[^>]*>.*?</script>", "", html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


async def search_web(
    query: str,
    engine: str = "duckduckgo",
    num_results: int = 10
) -> List[SearchResult]:
    """
    Search the web using a search engine.

    NOTE: This is a simplified implementation.
    Production use would require API keys for search engines
    or use a search API service.

    Args:
        query: Search query
        engine: Search engine to use ("duckduckgo", "google", etc.)
        num_results: Maximum number of results

    Returns:
        List of SearchResult objects
    """
    # DuckDuckGo HTML search (no API key required)
    if engine == "duckduckgo":
        return await _search_duckduckgo(query, num_results)

    # For other engines, you'd need API keys
    logger.warning(f"Search engine '{engine}' not implemented, using DuckDuckGo")
    return await _search_duckduckgo(query, num_results)


async def _search_duckduckgo(query: str, num_results: int) -> List[SearchResult]:
    """
    Search using DuckDuckGo HTML interface.

    This scrapes the HTML results page - for production use,
    consider using the DuckDuckGo API or a search API service.
    """
    import urllib.parse

    encoded_query = urllib.parse.quote(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    fetcher = WebFetcher()
    try:
        page = await fetcher.fetch(url)

        if not page.content:
            return []

        results = []

        if HAS_BS4:
            soup = BeautifulSoup(page.content, "html.parser")
            for i, result_div in enumerate(soup.select(".result")):
                if i >= num_results:
                    break

                title_elem = result_div.select_one(".result__title")
                link_elem = result_div.select_one(".result__url")
                snippet_elem = result_div.select_one(".result__snippet")

                if title_elem and link_elem:
                    # DuckDuckGo uses redirect URLs, extract actual URL
                    href = title_elem.find("a", href=True)
                    if href:
                        actual_url = href.get("href", "")
                        # Parse the redirect URL
                        if "uddg=" in actual_url:
                            actual_url = urllib.parse.unquote(
                                actual_url.split("uddg=")[1].split("&")[0]
                            )

                        results.append(SearchResult(
                            title=title_elem.get_text(strip=True),
                            url=actual_url,
                            snippet=snippet_elem.get_text(strip=True) if snippet_elem else "",
                            rank=i + 1
                        ))
        else:
            # Basic regex fallback
            pattern = r'class="result__title"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>([^<]+)'
            matches = re.findall(pattern, page.content, re.DOTALL)
            for i, (url, title) in enumerate(matches[:num_results]):
                results.append(SearchResult(
                    title=title.strip(),
                    url=url,
                    snippet="",
                    rank=i + 1
                ))

        return results

    finally:
        await fetcher.close()


# ============================================================================
# BROWSER AUTOMATION (Placeholder for advanced features)
# ============================================================================

class BrowserAutomation:
    """
    Placeholder for browser automation capabilities.

    For full browser automation (JavaScript rendering, form filling, etc.),
    you would integrate:
    - Playwright (recommended)
    - Selenium
    - Puppeteer (via pyppeteer)

    EXAMPLE (if Playwright were installed):

        from playwright.async_api import async_playwright

        class PlaywrightBrowser(BrowserAutomation):
            async def get_page(self, url: str) -> str:
                async with async_playwright() as p:
                    browser = await p.chromium.launch()
                    page = await browser.new_page()
                    await page.goto(url)
                    content = await page.content()
                    await browser.close()
                    return content
    """

    async def navigate(self, url: str) -> WebPage:
        """Navigate to a URL and get the rendered content."""
        # Default implementation uses simple HTTP fetch
        return await fetch_url(url)

    async def click(self, selector: str) -> bool:
        """Click an element (requires browser automation)."""
        logger.warning("click() requires browser automation library")
        return False

    async def fill_form(self, selector: str, value: str) -> bool:
        """Fill a form field (requires browser automation)."""
        logger.warning("fill_form() requires browser automation library")
        return False

    async def screenshot(self, path: str) -> bool:
        """Take a screenshot (requires browser automation)."""
        logger.warning("screenshot() requires browser automation library")
        return False
