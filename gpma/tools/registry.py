"""
Central Tool Registry

The ToolRegistry is the single source of truth for all tools in GPMA.
It provides:
- Tool registration and discovery
- Category-based organization
- Tag-based filtering
- Tool sets for different use cases
- Global statistics

Usage:
    from gpma.tools import registry, BaseTool

    # Register a tool
    registry.register(my_tool)

    # Get tools
    tool = registry.get("my_tool")
    web_tools = registry.get_by_category(ToolCategory.WEB)
    all_tools = registry.get_all()

    # Create tool sets for agents
    research_tools = registry.create_toolset(["web_search", "fetch_url", "read_file"])
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Set, Union

from .base import BaseTool, ToolCategory, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Centralized registry for managing all tools.

    This is implemented as a singleton to ensure there's only one
    registry across the entire application. All tools should be
    registered here for discovery and management.

    Features:
    - Singleton pattern for global access
    - Category-based organization
    - Tag-based filtering
    - Tool sets for specific use cases
    - Usage statistics aggregation
    - Lazy loading of built-in tools
    """

    _instance: Optional[ToolRegistry] = None
    _initialized: bool = False

    def __new__(cls) -> ToolRegistry:
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry (only runs once due to singleton)."""
        if not ToolRegistry._initialized:
            self._tools: Dict[str, BaseTool] = {}
            self._categories: Dict[ToolCategory, Set[str]] = {
                cat: set() for cat in ToolCategory
            }
            self._tags: Dict[str, Set[str]] = {}
            self._toolsets: Dict[str, List[str]] = {}
            self._builtins_loaded = False

            ToolRegistry._initialized = True
            logger.debug("ToolRegistry initialized")

    def register(self, tool: BaseTool, replace: bool = False) -> None:
        """
        Register a tool in the registry.

        Args:
            tool: The tool to register
            replace: If True, replace existing tool with same name

        Raises:
            ValueError: If tool with same name exists and replace=False
        """
        if tool.name in self._tools and not replace:
            raise ValueError(
                f"Tool '{tool.name}' already registered. "
                "Use replace=True to override."
            )

        self._tools[tool.name] = tool
        self._categories[tool.category].add(tool.name)

        # Index by tags
        for tag in tool.tags:
            if tag not in self._tags:
                self._tags[tag] = set()
            self._tags[tag].add(tool.name)

        logger.debug(f"Registered tool: {tool.name} [{tool.category.value}]")

    def unregister(self, name: str) -> bool:
        """
        Remove a tool from the registry.

        Args:
            name: Name of the tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name not in self._tools:
            return False

        tool = self._tools[name]

        # Remove from category index
        self._categories[tool.category].discard(name)

        # Remove from tag indices
        for tag in tool.tags:
            if tag in self._tags:
                self._tags[tag].discard(name)

        # Remove from main storage
        del self._tools[name]

        logger.debug(f"Unregistered tool: {name}")
        return True

    def get(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            The tool or None if not found
        """
        self._ensure_builtins_loaded()
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        self._ensure_builtins_loaded()
        return list(self._tools.values())

    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """
        Get all tools in a category.

        Args:
            category: The category to filter by

        Returns:
            List of tools in the category
        """
        self._ensure_builtins_loaded()
        names = self._categories.get(category, set())
        return [self._tools[name] for name in names if name in self._tools]

    def get_by_tag(self, tag: str) -> List[BaseTool]:
        """
        Get all tools with a specific tag.

        Args:
            tag: The tag to filter by

        Returns:
            List of tools with the tag
        """
        self._ensure_builtins_loaded()
        names = self._tags.get(tag, set())
        return [self._tools[name] for name in names if name in self._tools]

    def get_by_tags(self, tags: List[str], match_all: bool = False) -> List[BaseTool]:
        """
        Get tools matching multiple tags.

        Args:
            tags: List of tags to match
            match_all: If True, tool must have all tags. If False, any tag.

        Returns:
            List of matching tools
        """
        self._ensure_builtins_loaded()

        if match_all:
            # Intersection of all tag sets
            name_sets = [self._tags.get(tag, set()) for tag in tags]
            if not name_sets:
                return []
            names = set.intersection(*name_sets)
        else:
            # Union of all tag sets
            names = set()
            for tag in tags:
                names.update(self._tags.get(tag, set()))

        return [self._tools[name] for name in names if name in self._tools]

    def search(self, query: str) -> List[BaseTool]:
        """
        Search for tools by name or description.

        Args:
            query: Search query

        Returns:
            List of matching tools, sorted by relevance
        """
        self._ensure_builtins_loaded()
        query_lower = query.lower()
        results = []

        for tool in self._tools.values():
            score = 0.0

            # Name match
            if query_lower in tool.name.lower():
                score += 0.5
            if tool.name.lower().startswith(query_lower):
                score += 0.3

            # Description match
            if query_lower in tool.description.lower():
                score += 0.2

            # Tag match
            for tag in tool.tags:
                if query_lower in tag.lower():
                    score += 0.1

            if score > 0:
                results.append((tool, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in results]

    def list_tools(self) -> List[Dict[str, str]]:
        """
        Get a summary list of all tools.

        Returns:
            List of dicts with name, description, category
        """
        self._ensure_builtins_loaded()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "tags": tool.tags
            }
            for tool in self._tools.values()
        ]

    def list_categories(self) -> Dict[str, int]:
        """
        Get counts of tools per category.

        Returns:
            Dict mapping category name to tool count
        """
        self._ensure_builtins_loaded()
        return {
            cat.value: len(names)
            for cat, names in self._categories.items()
        }

    def list_tags(self) -> Dict[str, int]:
        """
        Get counts of tools per tag.

        Returns:
            Dict mapping tag to tool count
        """
        self._ensure_builtins_loaded()
        return {tag: len(names) for tag, names in self._tags.items()}

    # =========================================================================
    # Tool Sets
    # =========================================================================

    def define_toolset(self, name: str, tool_names: List[str]) -> None:
        """
        Define a named set of tools for specific use cases.

        Args:
            name: Name for the tool set (e.g., "research", "file_ops")
            tool_names: List of tool names to include

        Example:
            registry.define_toolset("research", ["web_search", "fetch_url", "read_file"])
        """
        self._toolsets[name] = tool_names
        logger.debug(f"Defined toolset: {name} with {len(tool_names)} tools")

    def get_toolset(self, name: str) -> List[BaseTool]:
        """
        Get tools in a named tool set.

        Args:
            name: Tool set name

        Returns:
            List of tools in the set

        Raises:
            ValueError: If tool set not found
        """
        self._ensure_builtins_loaded()

        if name not in self._toolsets:
            raise ValueError(f"Tool set not found: {name}")

        return [
            self._tools[tool_name]
            for tool_name in self._toolsets[name]
            if tool_name in self._tools
        ]

    def create_toolset(self, tool_names: List[str]) -> List[BaseTool]:
        """
        Create an ad-hoc tool set from a list of names.

        Args:
            tool_names: List of tool names

        Returns:
            List of tools (missing tools are skipped with warning)
        """
        self._ensure_builtins_loaded()
        tools = []

        for name in tool_names:
            tool = self._tools.get(name)
            if tool:
                tools.append(tool)
            else:
                logger.warning(f"Tool not found for toolset: {name}")

        return tools

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics for all tools.

        Returns:
            Dict with total stats and per-tool stats
        """
        self._ensure_builtins_loaded()

        total_calls = 0
        total_errors = 0
        total_time = 0.0
        tool_stats = []

        for tool in self._tools.values():
            stats = tool.get_stats()
            total_calls += stats["call_count"]
            total_errors += stats["error_count"]
            total_time += stats["total_time"]
            tool_stats.append(stats)

        return {
            "total_tools": len(self._tools),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "total_time": total_time,
            "error_rate": total_errors / max(total_calls, 1),
            "tools": tool_stats
        }

    def reset_all_stats(self) -> None:
        """Reset statistics for all tools."""
        for tool in self._tools.values():
            tool.reset_stats()
        logger.debug("Reset stats for all tools")

    # =========================================================================
    # Built-in Tools
    # =========================================================================

    def _ensure_builtins_loaded(self) -> None:
        """Lazy load built-in tools on first access."""
        if self._builtins_loaded:
            return

        try:
            from .builtin import register_all_builtins
            register_all_builtins(self)
            self._builtins_loaded = True
            logger.debug(f"Loaded {len(self._tools)} built-in tools")
        except ImportError as e:
            logger.warning(f"Could not load built-in tools: {e}")
            self._builtins_loaded = True  # Don't retry

    def load_builtins(self) -> None:
        """Explicitly load built-in tools."""
        self._builtins_loaded = False
        self._ensure_builtins_loaded()

    # =========================================================================
    # Async Execution Helper
    # =========================================================================

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            **kwargs: Parameters to pass to the tool

        Returns:
            ToolResult from the execution

        Raises:
            ValueError: If tool not found
        """
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        return await tool.execute(**kwargs)


# ============================================================================
# Global registry instance
# ============================================================================

# The global registry instance - use this for all tool operations
registry = ToolRegistry()


# Convenience functions that delegate to the global registry
def register(tool: BaseTool, replace: bool = False) -> None:
    """Register a tool in the global registry."""
    registry.register(tool, replace=replace)


def get_tool(name: str) -> Optional[BaseTool]:
    """Get a tool from the global registry."""
    return registry.get(name)


def get_all_tools() -> List[BaseTool]:
    """Get all tools from the global registry."""
    return registry.get_all()


def get_tools_by_category(category: ToolCategory) -> List[BaseTool]:
    """Get tools by category from the global registry."""
    return registry.get_by_category(category)


def get_default_tools() -> List[BaseTool]:
    """Get the default set of tools for agentic loops."""
    return registry.get_all()


async def execute_tool(name: str, **kwargs) -> ToolResult:
    """Execute a tool by name."""
    return await registry.execute(name, **kwargs)
