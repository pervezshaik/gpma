"""
Search Tools

Knowledge base and web search tools.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import BaseTool, ToolCategory, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..registry import ToolRegistry

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Extensible knowledge base for search operations.

    Supports:
    - In-memory knowledge entries
    - Keyword-based search
    - Category filtering
    """

    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load_default_knowledge()

    def _load_default_knowledge(self):
        """Load default programming knowledge."""
        self._entries = {
            "python": {
                "title": "Python Programming Language",
                "content": (
                    "Python is a high-level, interpreted programming language known for its "
                    "clear syntax and readability. Created by Guido van Rossum in 1991, it "
                    "emphasizes code readability and supports multiple programming paradigms "
                    "including procedural, object-oriented, and functional programming. "
                    "Python is widely used for web development, data science, AI/ML, "
                    "automation, and scripting."
                ),
                "keywords": ["python", "programming", "language", "guido", "interpreted"],
                "category": "programming_language"
            },
            "javascript": {
                "title": "JavaScript Programming Language",
                "content": (
                    "JavaScript is a high-level, dynamic programming language primarily "
                    "used for web development. It runs in browsers and enables interactive "
                    "web pages. With Node.js, JavaScript can also run on servers. It supports "
                    "event-driven, functional, and object-oriented programming styles."
                ),
                "keywords": ["javascript", "js", "web", "browser", "node", "frontend"],
                "category": "programming_language"
            },
            "rust": {
                "title": "Rust Programming Language",
                "content": (
                    "Rust is a systems programming language focused on safety, speed, and "
                    "concurrency. It prevents memory errors at compile time through its "
                    "ownership system. Rust is used for system programming, WebAssembly, "
                    "CLI tools, and performance-critical applications."
                ),
                "keywords": ["rust", "systems", "memory", "safety", "performance"],
                "category": "programming_language"
            },
            "machine_learning": {
                "title": "Machine Learning",
                "content": (
                    "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn and improve from experience without being explicitly "
                    "programmed. It uses algorithms to analyze data, learn patterns, and "
                    "make predictions. Common types include supervised learning, "
                    "unsupervised learning, and reinforcement learning."
                ),
                "keywords": ["machine learning", "ml", "ai", "artificial intelligence", "algorithms"],
                "category": "technology"
            },
            "api": {
                "title": "Application Programming Interface (API)",
                "content": (
                    "An API (Application Programming Interface) is a set of protocols and "
                    "tools for building software applications. APIs define how software "
                    "components should interact. REST APIs use HTTP methods (GET, POST, PUT, "
                    "DELETE) to perform operations on resources. GraphQL is an alternative "
                    "that allows clients to request specific data."
                ),
                "keywords": ["api", "rest", "graphql", "http", "interface", "web service"],
                "category": "technology"
            },
            "docker": {
                "title": "Docker Containerization",
                "content": (
                    "Docker is a platform for developing, shipping, and running applications "
                    "in containers. Containers package an application with all its dependencies, "
                    "ensuring consistent behavior across environments. Docker uses images "
                    "(blueprints) to create containers (running instances)."
                ),
                "keywords": ["docker", "container", "containerization", "devops", "deployment"],
                "category": "technology"
            }
        }

    def add_entry(
        self,
        key: str,
        title: str,
        content: str,
        keywords: List[str] = None,
        category: str = "general"
    ):
        """Add a knowledge entry."""
        self._entries[key] = {
            "title": title,
            "content": content,
            "keywords": keywords or [],
            "category": category
        }

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of matching entries with relevance scores
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        results = []

        for key, entry in self._entries.items():
            score = 0.0

            # Check key match
            if key in query_lower:
                score += 0.5

            # Check keyword matches
            for keyword in entry.get("keywords", []):
                if keyword.lower() in query_lower:
                    score += 0.3

            # Check title match
            title_lower = entry.get("title", "").lower()
            for word in query_words:
                if word in title_lower:
                    score += 0.2

            # Check content match
            content_lower = entry.get("content", "").lower()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 0.1

            if score > 0:
                results.append({
                    "key": key,
                    "title": entry["title"],
                    "content": entry["content"],
                    "category": entry.get("category", "general"),
                    "score": min(score, 1.0)
                })

        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]


# Global knowledge base instance
_knowledge_base = KnowledgeBase()


async def search_knowledge(query: str, max_results: int = 3) -> str:
    """
    Search the knowledge base for information.

    Args:
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        Formatted search results or "no results" message
    """
    results = _knowledge_base.search(query, max_results)

    if not results:
        return f"No information found for: {query}"

    # Format results
    output = f"Knowledge base results for '{query}':\n"
    for i, result in enumerate(results, 1):
        output += f"\n{i}. {result['title']} (relevance: {result['score']:.0%})\n"
        output += f"   {result['content'][:200]}...\n"

    return output


async def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information.

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        Formatted search results
    """
    try:
        from ..web_tools import search_web as do_search
        results = await do_search(query, num_results=num_results)

        if not results:
            return f"No web results found for: {query}"

        output = f"Web search results for '{query}':\n"
        for i, r in enumerate(results, 1):
            output += f"\n{i}. {r.title}\n"
            output += f"   URL: {r.url}\n"
            snippet = r.snippet[:150] if r.snippet else "No description"
            output += f"   {snippet}...\n"

        return output

    except ImportError:
        return "Web search not available: web_tools module not found"
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search error: {str(e)}"


def create_knowledge_search_tool() -> BaseTool:
    """Create the knowledge search tool."""
    return BaseTool(
        name="search",
        description=(
            "Search the knowledge base for information about programming languages, "
            "technologies, and concepts."
        ),
        parameters=[
            ToolParameter.string(
                name="query",
                description="Search query",
                required=True
            ),
            ToolParameter.integer(
                name="max_results",
                description="Maximum number of results (default: 3)",
                required=False,
                default=3
            )
        ],
        function=search_knowledge,
        category=ToolCategory.SEARCH,
        timeout=10.0,
        tags=["search", "knowledge", "information"]
    )


def create_web_search_tool() -> BaseTool:
    """Create the web search tool."""
    return BaseTool(
        name="web_search",
        description=(
            "Search the web for current information. Use this for recent events, "
            "news, or topics not in the knowledge base."
        ),
        parameters=[
            ToolParameter.string(
                name="query",
                description="Search query",
                required=True
            ),
            ToolParameter.integer(
                name="num_results",
                description="Number of results (default: 5)",
                required=False,
                default=5
            )
        ],
        function=web_search,
        category=ToolCategory.SEARCH,
        timeout=30.0,
        tags=["search", "web", "internet"]
    )


def register_search_tools(registry: "ToolRegistry") -> None:
    """Register all search tools in the registry."""
    registry.register(create_knowledge_search_tool())
    registry.register(create_web_search_tool())
    logger.debug("Registered search tools")


# Export knowledge base for external use
def get_knowledge_base() -> KnowledgeBase:
    """Get the global knowledge base instance."""
    return _knowledge_base
