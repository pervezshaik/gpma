"""
GPMA Tools Module

Centralized tool system for agents. All tools are defined using a unified
BaseTool class and managed through a central ToolRegistry.

Quick Start:
    from gpma.tools import registry, tool, ToolCategory

    # Get all available tools
    tools = registry.get_all()

    # Get tools by category
    web_tools = registry.get_by_category(ToolCategory.WEB)

    # Create a custom tool
    @tool(description="My custom tool", category=ToolCategory.CUSTOM)
    async def my_tool(param: str) -> str:
        '''Do something.

        Args:
            param: A parameter
        '''
        return f"Result: {param}"

Architecture:
    - BaseTool: Unified tool class with validation, timeout, stats
    - ToolRegistry: Central singleton for tool management
    - @tool decorator: Easy tool creation from functions
    - Adapters: Convert to OpenAI/Anthropic formats

Categories:
    - WEB: Web fetching, searching, parsing
    - FILE: File operations (read, write, list)
    - MATH: Calculator and math functions
    - SEARCH: Knowledge base and web search
    - SYSTEM: System operations
    - DATA: Data processing
    - CUSTOM: User-defined tools
"""

# Core classes
from .base import (
    BaseTool,
    ToolParameter,
    ToolResult,
    ToolResultStatus,
    ToolCategory,
)

# Registry
from .registry import (
    ToolRegistry,
    registry,
    register,
    get_tool,
    get_all_tools,
    get_tools_by_category,
    get_default_tools,
    execute_tool,
)

# Decorators
from .decorators import (
    tool,
    simple_tool,
    ToolBuilder,
)

# Adapters
from .adapters import (
    to_openai_format,
    to_anthropic_format,
    OpenAIToolAdapter,
    AnthropicToolAdapter,
)

# Legacy imports for backward compatibility
# These import from the old modules that still exist
try:
    from .web_tools import (
        fetch_url,
        parse_html,
        extract_links,
        extract_text,
        search_web,
        WebFetcher,
    )
except ImportError:
    pass

try:
    from .file_tools import (
        read_file,
        write_file,
        list_directory,
        file_exists,
        FileManager,
    )
except ImportError:
    pass

# Legacy agentic tools exports
try:
    from .agentic_tools import (
        AgenticTool,
        SafeCalculator,
        KnowledgeBase,
        create_demo_tools,
        safe_calculate,
        search_knowledge,
        web_search,
        fetch_webpage,
        read_file_content,
        list_files,
        auto_tool,
    )
    # Also re-export as ToolRegistry for backward compat
    from .agentic_tools import ToolRegistry as LegacyToolRegistry
except ImportError:
    pass


__all__ = [
    # Core
    'BaseTool',
    'ToolParameter',
    'ToolResult',
    'ToolResultStatus',
    'ToolCategory',

    # Registry
    'ToolRegistry',
    'registry',
    'register',
    'get_tool',
    'get_all_tools',
    'get_tools_by_category',
    'get_default_tools',
    'execute_tool',

    # Decorators
    'tool',
    'simple_tool',
    'ToolBuilder',

    # Adapters
    'to_openai_format',
    'to_anthropic_format',
    'OpenAIToolAdapter',
    'AnthropicToolAdapter',

    # Legacy web tools
    'fetch_url',
    'parse_html',
    'extract_links',
    'extract_text',
    'search_web',
    'WebFetcher',

    # Legacy file tools
    'read_file',
    'write_file',
    'list_directory',
    'file_exists',
    'FileManager',

    # Legacy agentic tools
    'AgenticTool',
    'SafeCalculator',
    'KnowledgeBase',
    'create_demo_tools',
    'safe_calculate',
    'search_knowledge',
    'web_search',
    'fetch_webpage',
    'read_file_content',
    'list_files',
    'auto_tool',
]
