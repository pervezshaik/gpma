"""
Web Browser Agent

An agent specialized in fetching and parsing web content.

CAPABILITIES:
- Fetch web pages by URL
- Extract text content from HTML
- Extract links from pages
- Parse structured data (JSON, tables)

LEARNING POINTS:
- This agent focuses on a single responsibility: web content
- It uses tools (WebFetcher) to do the actual work
- The agent adds intelligence on top of the tools
"""

from typing import Any, Dict, List

from ..core.base_agent import BaseAgent, AgentCapability, TaskResult, Tool
from ..tools.web_tools import WebFetcher, WebPage, fetch_url, extract_links, extract_text

import logging

logger = logging.getLogger(__name__)


class WebBrowserAgent(BaseAgent):
    """
    Agent for web browsing operations.

    SUPPORTED ACTIONS:
    - fetch: Get content from a URL
    - extract_text: Get plain text from a page
    - extract_links: Get all links from a page
    - analyze: Combine fetch with analysis

    EXAMPLE USAGE:
        agent = WebBrowserAgent()
        result = await agent.run_task({
            "action": "fetch",
            "input": "https://example.com"
        })
        print(result.data["title"])
    """

    def __init__(self, name: str = None):
        super().__init__(name or "WebBrowser")
        self._fetcher = WebFetcher(cache_enabled=True)

        # Register tools
        self.register_tool(Tool(
            name="fetch_url",
            description="Fetch content from a URL",
            function=self._fetch_url,
            parameters={"url": "URL to fetch"}
        ))

        self.register_tool(Tool(
            name="extract_links",
            description="Extract all links from a page",
            function=self._extract_links,
            parameters={"url": "URL to analyze"}
        ))

        self.register_tool(Tool(
            name="search",
            description="Search the web",
            function=self._search,
            parameters={"query": "Search query"}
        ))

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="web_fetch",
                description="Fetch and parse web pages",
                keywords=["url", "webpage", "website", "fetch", "browse", "http", "link"],
                priority=2
            ),
            AgentCapability(
                name="web_extract",
                description="Extract content from web pages",
                keywords=["extract", "parse", "content", "text", "html"],
                priority=1
            ),
            AgentCapability(
                name="web_search",
                description="Search the web for information",
                keywords=["search", "find", "lookup", "google", "query"],
                priority=2
            )
        ]

    async def initialize(self) -> None:
        """Initialize the web fetcher."""
        await super().initialize()
        logger.info(f"WebBrowserAgent {self.name} initialized with caching enabled")

    async def shutdown(self) -> None:
        """Clean up the web fetcher."""
        await self._fetcher.close()
        await super().shutdown()

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a web-related task.

        The task dictionary should contain:
        - action: What to do (fetch, extract_text, extract_links, search)
        - input: The URL or search query
        - options: Additional options (optional)
        """
        action = task.get("action", "fetch")
        input_data = task.get("input", "")
        options = task.get("options", {})

        try:
            if action == "fetch":
                return await self._handle_fetch(input_data, options)
            elif action == "extract_text":
                return await self._handle_extract_text(input_data)
            elif action == "extract_links":
                return await self._handle_extract_links(input_data)
            elif action == "search":
                return await self._handle_search(input_data, options)
            elif action == "analyze":
                return await self._handle_analyze(input_data, options)
            else:
                # Try to infer action from input
                if input_data.startswith(("http://", "https://")):
                    return await self._handle_fetch(input_data, options)
                else:
                    return await self._handle_search(input_data, options)

        except Exception as e:
            logger.error(f"WebBrowserAgent error: {e}")
            return TaskResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def _handle_fetch(self, url: str, options: Dict) -> TaskResult:
        """Fetch a URL and return parsed content."""
        page = await self._fetcher.fetch(url)

        if page.status_code == 0:
            return TaskResult(
                success=False,
                data=None,
                error=page.metadata.get("error", "Failed to fetch URL")
            )

        return TaskResult(
            success=True,
            data={
                "url": page.url,
                "title": page.title,
                "text": page.text[:5000] if page.text else "",  # Limit text size
                "text_length": len(page.text) if page.text else 0,
                "links_count": len(page.links),
                "links": page.links[:20],  # First 20 links
                "metadata": page.metadata
            },
            metadata={"status_code": page.status_code}
        )

    async def _handle_extract_text(self, url: str) -> TaskResult:
        """Extract just the text content from a URL."""
        page = await self._fetcher.fetch(url)

        if page.status_code == 0:
            return TaskResult(
                success=False,
                data=None,
                error="Failed to fetch URL"
            )

        return TaskResult(
            success=True,
            data={
                "url": page.url,
                "title": page.title,
                "text": page.text
            }
        )

    async def _handle_extract_links(self, url: str) -> TaskResult:
        """Extract all links from a URL."""
        page = await self._fetcher.fetch(url)

        if page.status_code == 0:
            return TaskResult(
                success=False,
                data=None,
                error="Failed to fetch URL"
            )

        # Categorize links
        internal_links = []
        external_links = []
        from urllib.parse import urlparse
        base_domain = urlparse(url).netloc

        for link in page.links:
            link_domain = urlparse(link).netloc
            if link_domain == base_domain:
                internal_links.append(link)
            else:
                external_links.append(link)

        return TaskResult(
            success=True,
            data={
                "url": page.url,
                "total_links": len(page.links),
                "internal_links": internal_links[:50],
                "external_links": external_links[:50]
            }
        )

    async def _handle_search(self, query: str, options: Dict) -> TaskResult:
        """Search the web."""
        from ..tools.web_tools import search_web

        num_results = options.get("num_results", 10)
        results = await search_web(query, num_results=num_results)

        return TaskResult(
            success=True,
            data={
                "query": query,
                "num_results": len(results),
                "results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "snippet": r.snippet,
                        "rank": r.rank
                    }
                    for r in results
                ]
            }
        )

    async def _handle_analyze(self, url: str, options: Dict) -> TaskResult:
        """Fetch and provide a comprehensive analysis of a page."""
        page = await self._fetcher.fetch(url)

        if page.status_code == 0:
            return TaskResult(
                success=False,
                data=None,
                error="Failed to fetch URL"
            )

        # Basic analysis
        text = page.text or ""
        word_count = len(text.split())

        # Extract key information
        analysis = {
            "url": page.url,
            "title": page.title,
            "word_count": word_count,
            "link_count": len(page.links),
            "has_metadata": bool(page.metadata),
            "content_preview": text[:1000],
            "metadata": page.metadata,
            "top_links": page.links[:10]
        }

        return TaskResult(
            success=True,
            data=analysis
        )

    # Tool implementations
    async def _fetch_url(self, url: str) -> Dict[str, Any]:
        """Tool: Fetch a URL."""
        page = await self._fetcher.fetch(url)
        return {
            "title": page.title,
            "text": page.text[:3000] if page.text else "",
            "success": page.status_code != 0
        }

    async def _extract_links(self, url: str) -> List[str]:
        """Tool: Extract links from a URL."""
        page = await self._fetcher.fetch(url)
        return page.links[:50]

    async def _search(self, query: str) -> List[Dict]:
        """Tool: Search the web."""
        from ..tools.web_tools import search_web
        results = await search_web(query, num_results=5)
        return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
