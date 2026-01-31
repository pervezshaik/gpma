"""
File Tools Module

Tools for file system operations.

CAPABILITIES:
- Read/write files
- Directory operations
- File search and filtering
- Safe file operations with error handling

LEARNING POINTS:
- File operations should be atomic where possible
- Always handle encoding properly
- Use async for better performance with multiple files
- Validate paths to prevent security issues
"""

import os
import asyncio
import aiofiles
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import json
import fnmatch

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """
    Information about a file.
    """
    path: str
    name: str
    size: int
    is_directory: bool
    modified_time: datetime
    extension: str
    permissions: str


@dataclass
class FileContent:
    """
    Result of reading a file.
    """
    path: str
    content: str
    encoding: str
    size: int
    success: bool
    error: Optional[str] = None


class FileManager:
    """
    Comprehensive file management with safety features.

    FEATURES:
    - Async file operations
    - Path validation
    - Backup before modification
    - Transaction-like operations

    USAGE:
        fm = FileManager(base_path="./data")
        content = await fm.read("config.json")
        await fm.write("output.txt", "Hello World")
    """

    def __init__(
        self,
        base_path: str = ".",
        create_backup: bool = True,
        allowed_extensions: List[str] = None
    ):
        """
        Initialize file manager.

        Args:
            base_path: Root directory for operations
            create_backup: Create backups before modifying files
            allowed_extensions: Whitelist of file extensions (None = all allowed)
        """
        self.base_path = Path(base_path).resolve()
        self.create_backup = create_backup
        self.allowed_extensions = allowed_extensions

    def _validate_path(self, path: str) -> Path:
        """
        Validate and resolve a path.

        Prevents directory traversal attacks.
        """
        # Resolve the full path
        full_path = (self.base_path / path).resolve()

        # Ensure it's within base_path
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(f"Path '{path}' is outside allowed directory")

        # Check extension if whitelist is set
        if self.allowed_extensions:
            ext = full_path.suffix.lower()
            if ext and ext not in self.allowed_extensions:
                raise ValueError(f"Extension '{ext}' is not allowed")

        return full_path

    async def read(self, path: str, encoding: str = "utf-8") -> FileContent:
        """
        Read a file's content.

        Args:
            path: Relative path to the file
            encoding: Text encoding

        Returns:
            FileContent object with content or error
        """
        try:
            full_path = self._validate_path(path)

            if not full_path.exists():
                return FileContent(
                    path=str(full_path),
                    content="",
                    encoding=encoding,
                    size=0,
                    success=False,
                    error="File not found"
                )

            async with aiofiles.open(full_path, "r", encoding=encoding) as f:
                content = await f.read()

            return FileContent(
                path=str(full_path),
                content=content,
                encoding=encoding,
                size=len(content),
                success=True
            )

        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return FileContent(
                path=path,
                content="",
                encoding=encoding,
                size=0,
                success=False,
                error=str(e)
            )

    async def write(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        append: bool = False
    ) -> bool:
        """
        Write content to a file.

        Args:
            path: Relative path to the file
            content: Content to write
            encoding: Text encoding
            append: Append to file instead of overwriting

        Returns:
            True if successful
        """
        try:
            full_path = self._validate_path(path)

            # Create backup if file exists and backup is enabled
            if self.create_backup and full_path.exists() and not append:
                backup_path = full_path.with_suffix(full_path.suffix + ".bak")
                async with aiofiles.open(full_path, "r", encoding=encoding) as src:
                    original = await src.read()
                async with aiofiles.open(backup_path, "w", encoding=encoding) as dst:
                    await dst.write(original)

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            mode = "a" if append else "w"
            async with aiofiles.open(full_path, mode, encoding=encoding) as f:
                await f.write(content)

            logger.debug(f"File written: {full_path}")
            return True

        except Exception as e:
            logger.error(f"Error writing file {path}: {e}")
            return False

    async def delete(self, path: str) -> bool:
        """Delete a file."""
        try:
            full_path = self._validate_path(path)

            if not full_path.exists():
                return False

            # Create backup before deleting
            if self.create_backup:
                backup_path = full_path.with_suffix(full_path.suffix + ".deleted")
                full_path.rename(backup_path)
            else:
                full_path.unlink()

            return True

        except Exception as e:
            logger.error(f"Error deleting file {path}: {e}")
            return False

    def list_directory(
        self,
        path: str = ".",
        pattern: str = "*",
        recursive: bool = False
    ) -> List[FileInfo]:
        """
        List files in a directory.

        Args:
            path: Directory path
            pattern: Glob pattern to match
            recursive: Include subdirectories

        Returns:
            List of FileInfo objects
        """
        try:
            full_path = self._validate_path(path)

            if not full_path.is_dir():
                return []

            results = []
            if recursive:
                files = full_path.rglob(pattern)
            else:
                files = full_path.glob(pattern)

            for file_path in files:
                try:
                    stat = file_path.stat()
                    results.append(FileInfo(
                        path=str(file_path),
                        name=file_path.name,
                        size=stat.st_size,
                        is_directory=file_path.is_dir(),
                        modified_time=datetime.fromtimestamp(stat.st_mtime),
                        extension=file_path.suffix,
                        permissions=oct(stat.st_mode)[-3:]
                    ))
                except Exception:
                    continue

            return results

        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return []

    def exists(self, path: str) -> bool:
        """Check if a file or directory exists."""
        try:
            full_path = self._validate_path(path)
            return full_path.exists()
        except ValueError:
            return False

    async def copy(self, source: str, destination: str) -> bool:
        """Copy a file."""
        try:
            source_path = self._validate_path(source)
            dest_path = self._validate_path(destination)

            async with aiofiles.open(source_path, "rb") as src:
                content = await src.read()

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(dest_path, "wb") as dst:
                await dst.write(content)

            return True

        except Exception as e:
            logger.error(f"Error copying {source} to {destination}: {e}")
            return False

    async def move(self, source: str, destination: str) -> bool:
        """Move a file."""
        if await self.copy(source, destination):
            return await self.delete(source)
        return False

    def get_info(self, path: str) -> Optional[FileInfo]:
        """Get information about a file."""
        try:
            full_path = self._validate_path(path)

            if not full_path.exists():
                return None

            stat = full_path.stat()
            return FileInfo(
                path=str(full_path),
                name=full_path.name,
                size=stat.st_size,
                is_directory=full_path.is_dir(),
                modified_time=datetime.fromtimestamp(stat.st_mtime),
                extension=full_path.suffix,
                permissions=oct(stat.st_mode)[-3:]
            )

        except Exception as e:
            logger.error(f"Error getting file info {path}: {e}")
            return None

    async def search(
        self,
        pattern: str,
        content_pattern: str = None,
        path: str = ".",
        max_results: int = 100
    ) -> List[str]:
        """
        Search for files, optionally filtering by content.

        Args:
            pattern: Filename glob pattern
            content_pattern: Text to search for in files
            path: Directory to search in
            max_results: Maximum number of results

        Returns:
            List of matching file paths
        """
        matches = []
        files = self.list_directory(path, pattern, recursive=True)

        for file_info in files[:max_results * 2]:  # Get more than needed for content filter
            if file_info.is_directory:
                continue

            if content_pattern:
                try:
                    result = await self.read(file_info.path)
                    if result.success and content_pattern in result.content:
                        matches.append(file_info.path)
                except Exception:
                    continue
            else:
                matches.append(file_info.path)

            if len(matches) >= max_results:
                break

        return matches


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

async def read_file(path: str, encoding: str = "utf-8") -> str:
    """
    Simple async file read.

    Args:
        path: Path to the file
        encoding: Text encoding

    Returns:
        File content as string, or empty string on error
    """
    try:
        async with aiofiles.open(path, "r", encoding=encoding) as f:
            return await f.read()
    except Exception as e:
        logger.error(f"Error reading {path}: {e}")
        return ""


async def write_file(path: str, content: str, encoding: str = "utf-8") -> bool:
    """
    Simple async file write.

    Args:
        path: Path to the file
        content: Content to write
        encoding: Text encoding

    Returns:
        True if successful
    """
    try:
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        async with aiofiles.open(path, "w", encoding=encoding) as f:
            await f.write(content)
        return True
    except Exception as e:
        logger.error(f"Error writing {path}: {e}")
        return False


def list_directory(path: str, pattern: str = "*") -> List[str]:
    """
    List files in a directory matching a pattern.

    Args:
        path: Directory path
        pattern: Glob pattern

    Returns:
        List of file paths
    """
    try:
        p = Path(path)
        return [str(f) for f in p.glob(pattern)]
    except Exception as e:
        logger.error(f"Error listing {path}: {e}")
        return []


def file_exists(path: str) -> bool:
    """Check if a file exists."""
    return Path(path).exists()


async def read_json(path: str) -> Optional[Dict[str, Any]]:
    """Read and parse a JSON file."""
    content = await read_file(path)
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {path}: {e}")
    return None


async def write_json(path: str, data: Any, indent: int = 2) -> bool:
    """Write data as JSON to a file."""
    try:
        content = json.dumps(data, indent=indent, default=str)
        return await write_file(path, content)
    except Exception as e:
        logger.error(f"Error writing JSON to {path}: {e}")
        return False
