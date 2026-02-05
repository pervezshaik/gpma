"""
Built-in Tools

Pre-built tools that come with GPMA, organized by category.

Categories:
- web: Web fetching, searching, parsing
- file: File operations (read, write, list)
- math: Calculator and math functions
- search: Knowledge base and search

Usage:
    # Tools are auto-loaded when accessing the registry
    from gpma.tools import registry

    # Or explicitly load them
    from gpma.tools.builtin import register_all_builtins
    register_all_builtins(registry)

    # Or import specific tools
    from gpma.tools.builtin.web import web_search_tool, fetch_url_tool
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..registry import ToolRegistry


def register_all_builtins(registry: "ToolRegistry") -> None:
    """
    Register all built-in tools in the registry.

    This is called automatically when accessing the registry.

    Args:
        registry: The ToolRegistry to register tools in
    """
    from .math import register_math_tools
    from .search import register_search_tools
    from .web import register_web_tools
    from .file import register_file_tools

    # Register in order of dependency (math has no deps, others may)
    register_math_tools(registry)
    register_search_tools(registry)
    register_file_tools(registry)
    register_web_tools(registry)


__all__ = ["register_all_builtins"]
