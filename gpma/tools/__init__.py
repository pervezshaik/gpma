"""
GPMA Tools Module

Tools are reusable functions that agents can invoke to perform actions.
They are the "hands" of the agents - enabling them to interact with the world.

Categories:
- Web Tools: Fetching URLs, parsing HTML, web searches
- File Tools: Reading, writing, managing files
- Agentic Tools: Production-grade tools for agentic loops
- System Tools: Running commands, system information
"""

from .web_tools import (
    fetch_url,
    parse_html,
    extract_links,
    extract_text,
    search_web,
    WebFetcher,
)

from .file_tools import (
    read_file,
    write_file,
    list_directory,
    file_exists,
    FileManager,
)

from .agentic_tools import (
    AgenticTool,
    ToolRegistry,
    SafeCalculator,
    KnowledgeBase,
    get_default_tools,
    get_tool,
    get_tools_by_category,
    create_demo_tools,
    safe_calculate,
    search_knowledge,
    web_search,
    fetch_webpage,
    read_file_content,
    list_files,
    auto_tool,
)

__all__ = [
    # Web tools
    'fetch_url',
    'parse_html',
    'extract_links',
    'extract_text',
    'search_web',
    'WebFetcher',
    # File tools
    'read_file',
    'write_file',
    'list_directory',
    'file_exists',
    'FileManager',
    # Agentic tools
    'AgenticTool',
    'ToolRegistry',
    'SafeCalculator',
    'KnowledgeBase',
    'get_default_tools',
    'get_tool',
    'get_tools_by_category',
    'create_demo_tools',
    'safe_calculate',
    'search_knowledge',
    'web_search',
    'fetch_webpage',
    'read_file_content',
    'list_files',
    'auto_tool',
]
