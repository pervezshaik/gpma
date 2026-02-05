"""
File Tools

File system operations with safety checks.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, List

from ..base import BaseTool, ToolCategory, ToolParameter, ToolResult

if TYPE_CHECKING:
    from ..registry import ToolRegistry

logger = logging.getLogger(__name__)


async def read_file(filepath: str, max_lines: int = 100) -> str:
    """
    Read content from a file (with safety checks).

    Args:
        filepath: Path to the file
        max_lines: Maximum lines to read

    Returns:
        File content or error message
    """
    # Security: Prevent path traversal
    filepath = os.path.normpath(filepath)
    if ".." in filepath:
        return "Error: Path traversal not allowed"

    try:
        # Try to import the existing file tools
        try:
            from ..file_tools import read_file as read_file_impl
            content = await read_file_impl(filepath)
        except ImportError:
            # Fallback to direct file read
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        # Limit output
        lines = content.split('\n')
        if len(lines) > max_lines:
            content = '\n'.join(lines[:max_lines])
            content += f"\n... (truncated, {len(lines) - max_lines} more lines)"

        return f"Content of {filepath}:\n\n{content}"

    except FileNotFoundError:
        return f"Error: File not found: {filepath}"
    except PermissionError:
        return f"Error: Permission denied: {filepath}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


async def write_file(filepath: str, content: str) -> str:
    """
    Write content to a file.

    Args:
        filepath: Path to the file
        content: Content to write

    Returns:
        Success message or error
    """
    # Security: Prevent path traversal
    filepath = os.path.normpath(filepath)
    if ".." in filepath:
        return "Error: Path traversal not allowed"

    try:
        # Try to import the existing file tools
        try:
            from ..file_tools import write_file as write_file_impl
            await write_file_impl(filepath, content)
        except ImportError:
            # Fallback to direct file write
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        return f"Successfully wrote {len(content)} characters to {filepath}"

    except PermissionError:
        return f"Error: Permission denied: {filepath}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


async def list_directory(directory: str = ".") -> str:
    """
    List files in a directory.

    Args:
        directory: Directory path

    Returns:
        List of files or error message
    """
    # Security: Prevent path traversal
    directory = os.path.normpath(directory)
    if ".." in directory:
        return "Error: Path traversal not allowed"

    try:
        # Try to import the existing file tools
        try:
            from ..file_tools import list_directory as list_dir_impl
            files = await list_dir_impl(directory)
        except ImportError:
            # Fallback to direct directory listing
            files = os.listdir(directory)

        output = f"Files in {directory}:\n"
        for f in files[:50]:  # Limit to 50 files
            output += f"  - {f}\n"

        if len(files) > 50:
            output += f"  ... and {len(files) - 50} more files"

        return output

    except FileNotFoundError:
        return f"Error: Directory not found: {directory}"
    except PermissionError:
        return f"Error: Permission denied: {directory}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


async def file_exists(filepath: str) -> str:
    """
    Check if a file exists.

    Args:
        filepath: Path to check

    Returns:
        "true" or "false"
    """
    filepath = os.path.normpath(filepath)
    exists = os.path.exists(filepath)
    return f"File {'exists' if exists else 'does not exist'}: {filepath}"


def create_read_file_tool() -> BaseTool:
    """Create the read file tool."""
    return BaseTool(
        name="read_file",
        description="Read content from a file. Use for examining code, configs, or documents.",
        parameters=[
            ToolParameter.string(
                name="filepath",
                description="Path to the file",
                required=True
            ),
            ToolParameter.integer(
                name="max_lines",
                description="Maximum lines to read (default: 100)",
                required=False,
                default=100
            )
        ],
        function=read_file,
        category=ToolCategory.FILE,
        timeout=10.0,
        tags=["file", "read", "content"]
    )


def create_write_file_tool() -> BaseTool:
    """Create the write file tool."""
    return BaseTool(
        name="write_file",
        description="Write content to a file. Creates parent directories if needed.",
        parameters=[
            ToolParameter.string(
                name="filepath",
                description="Path to the file",
                required=True
            ),
            ToolParameter.string(
                name="content",
                description="Content to write",
                required=True
            )
        ],
        function=write_file,
        category=ToolCategory.FILE,
        timeout=10.0,
        requires_confirmation=True,  # Writing files needs confirmation
        tags=["file", "write", "create"]
    )


def create_list_files_tool() -> BaseTool:
    """Create the list files tool."""
    return BaseTool(
        name="list_files",
        description="List files in a directory.",
        parameters=[
            ToolParameter.string(
                name="directory",
                description="Directory path (default: current directory)",
                required=False,
                default="."
            )
        ],
        function=list_directory,
        category=ToolCategory.FILE,
        timeout=10.0,
        tags=["file", "list", "directory"]
    )


def create_file_exists_tool() -> BaseTool:
    """Create the file exists tool."""
    return BaseTool(
        name="file_exists",
        description="Check if a file or directory exists.",
        parameters=[
            ToolParameter.string(
                name="filepath",
                description="Path to check",
                required=True
            )
        ],
        function=file_exists,
        category=ToolCategory.FILE,
        timeout=5.0,
        tags=["file", "check", "exists"]
    )


def register_file_tools(registry: "ToolRegistry") -> None:
    """Register all file tools in the registry."""
    registry.register(create_read_file_tool())
    registry.register(create_write_file_tool())
    registry.register(create_list_files_tool())
    registry.register(create_file_exists_tool())
    logger.debug("Registered file tools")
