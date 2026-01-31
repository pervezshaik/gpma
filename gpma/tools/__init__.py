"""
GPMA Tools Module

Tools are reusable functions that agents can invoke to perform actions.
They are the "hands" of the agents - enabling them to interact with the world.

Categories:
- Web Tools: Fetching URLs, parsing HTML, web searches
- File Tools: Reading, writing, managing files
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
]
