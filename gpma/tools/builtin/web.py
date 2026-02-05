"""
Web Tools

Web fetching and parsing tools.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..base import BaseTool, ToolCategory, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..registry import ToolRegistry

logger = logging.getLogger(__name__)


async def fetch_url(url: str, extract_text: bool = True) -> str:
    """
    Fetch and extract content from a webpage.

    Args:
        url: URL to fetch
        extract_text: Whether to extract text only (default: True)

    Returns:
        Page content or error message
    """
    try:
        # Try to use the existing web tools
        try:
            from ..web_tools import fetch_url as fetch_url_impl
            page = await fetch_url_impl(url)

            if page.status_code == 0:
                return f"Error fetching URL: {page.metadata.get('error', 'Unknown error')}"

            text = page.text or page.content[:3000]
            return f"Content from {url}:\n\n{text[:5000]}{'...' if len(text) > 5000 else ''}"

        except ImportError:
            # Fallback to basic aiohttp fetch
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        return f"Error: HTTP {response.status}"

                    content = await response.text()

                    if extract_text:
                        # Basic text extraction
                        try:
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(content, 'html.parser')

                            # Remove script and style elements
                            for element in soup(['script', 'style', 'nav', 'footer']):
                                element.decompose()

                            text = soup.get_text(separator='\n', strip=True)
                        except ImportError:
                            # No BeautifulSoup, return raw content
                            text = content

                        return f"Content from {url}:\n\n{text[:5000]}{'...' if len(text) > 5000 else ''}"
                    else:
                        return f"Raw HTML from {url}:\n\n{content[:5000]}{'...' if len(content) > 5000 else ''}"

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return f"Error fetching URL: {str(e)}"


async def extract_links(url: str) -> str:
    """
    Extract all links from a webpage.

    Args:
        url: URL to extract links from

    Returns:
        List of links or error message
    """
    try:
        # Try to use the existing web tools
        try:
            from ..web_tools import fetch_url as fetch_url_impl, extract_links as extract_links_impl
            page = await fetch_url_impl(url)

            if page.status_code == 0:
                return f"Error fetching URL: {page.metadata.get('error', 'Unknown error')}"

            links = extract_links_impl(page.content, url)
            if not links:
                return f"No links found on {url}"

            output = f"Links found on {url}:\n"
            for link in links[:30]:  # Limit to 30 links
                output += f"  - {link}\n"

            if len(links) > 30:
                output += f"  ... and {len(links) - 30} more links"

            return output

        except ImportError:
            # Fallback to basic extraction
            import aiohttp
            from urllib.parse import urljoin, urlparse

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    content = await response.text()

            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')

                links = []
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    full_url = urljoin(url, href)
                    if full_url.startswith(('http://', 'https://')):
                        links.append(full_url)

                if not links:
                    return f"No links found on {url}"

                output = f"Links found on {url}:\n"
                for link in links[:30]:
                    output += f"  - {link}\n"

                if len(links) > 30:
                    output += f"  ... and {len(links) - 30} more links"

                return output

            except ImportError:
                return "Error: BeautifulSoup not installed for link extraction"

    except Exception as e:
        logger.error(f"Link extraction failed: {e}")
        return f"Error extracting links: {str(e)}"


def create_fetch_url_tool() -> BaseTool:
    """Create the fetch URL tool."""
    return BaseTool(
        name="fetch_url",
        description="Fetch and extract text content from a webpage URL.",
        parameters=[
            ToolParameter.string(
                name="url",
                description="URL to fetch",
                required=True
            ),
            ToolParameter.boolean(
                name="extract_text",
                description="Extract text only, removing HTML (default: true)",
                required=False,
                default=True
            )
        ],
        function=fetch_url,
        category=ToolCategory.WEB,
        timeout=30.0,
        retry_count=1,
        tags=["web", "fetch", "url", "http"]
    )


def create_extract_links_tool() -> BaseTool:
    """Create the extract links tool."""
    return BaseTool(
        name="extract_links",
        description="Extract all links from a webpage.",
        parameters=[
            ToolParameter.string(
                name="url",
                description="URL to extract links from",
                required=True
            )
        ],
        function=extract_links,
        category=ToolCategory.WEB,
        timeout=30.0,
        tags=["web", "links", "extract"]
    )


def register_web_tools(registry: "ToolRegistry") -> None:
    """Register all web tools in the registry."""
    registry.register(create_fetch_url_tool())
    registry.register(create_extract_links_tool())
    logger.debug("Registered web tools")
