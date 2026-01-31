"""
Task Executor Agent

An agent specialized in executing system commands and file operations.

CAPABILITIES:
- Run shell commands
- File operations (read, write, manage)
- System information gathering
- Automation scripts

SECURITY CONSIDERATIONS:
- Commands are validated before execution
- Dangerous operations require confirmation
- File operations are sandboxed to allowed paths

LEARNING POINTS:
- Some agents interact with the system
- Security is critical for these agents
- Proper error handling prevents issues
"""

import os
import asyncio
import subprocess
import platform
from typing import Any, Dict, List, Optional

from ..core.base_agent import BaseAgent, AgentCapability, TaskResult, Tool
from ..tools.file_tools import FileManager, read_file, write_file

import logging

logger = logging.getLogger(__name__)


class TaskExecutorAgent(BaseAgent):
    """
    Agent for executing tasks and system operations.

    SECURITY:
    - Command whitelist/blacklist
    - Path restrictions
    - Confirmation for dangerous operations

    SUPPORTED ACTIONS:
    - run_command: Execute a shell command
    - read_file: Read file content
    - write_file: Write to a file
    - list_files: List directory contents
    - system_info: Get system information

    EXAMPLE:
        agent = TaskExecutorAgent(
            allowed_paths=["./data", "./output"],
            command_whitelist=["ls", "cat", "echo"]
        )
        result = await agent.run_task({
            "action": "read_file",
            "input": "./data/config.json"
        })
    """

    # Default dangerous commands to block
    DANGEROUS_COMMANDS = [
        "rm -rf", "del /f", "format", "mkfs",
        "dd if=", ":(){:|:&};:", "chmod -R 777",
        "> /dev/sda", "mv /* ", "wget | sh",
        "curl | bash", "sudo rm"
    ]

    def __init__(
        self,
        name: str = None,
        allowed_paths: List[str] = None,
        command_whitelist: List[str] = None,
        allow_dangerous: bool = False
    ):
        super().__init__(name or "TaskExecutor")

        self.allowed_paths = allowed_paths or ["."]
        self.command_whitelist = command_whitelist
        self.allow_dangerous = allow_dangerous

        # File manager for safe file operations
        self._file_manager = FileManager(
            base_path=self.allowed_paths[0] if allowed_paths else ".",
            create_backup=True
        )

        # Register tools
        self.register_tool(Tool(
            name="run_command",
            description="Execute a shell command",
            function=self._run_command,
            parameters={"command": "Shell command to run"}
        ))

        self.register_tool(Tool(
            name="read_file",
            description="Read a file's contents",
            function=self._read_file,
            parameters={"path": "Path to the file"}
        ))

        self.register_tool(Tool(
            name="write_file",
            description="Write content to a file",
            function=self._write_file,
            parameters={"path": "File path", "content": "Content to write"}
        ))

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="command_execution",
                description="Execute shell commands",
                keywords=["run", "execute", "command", "shell", "terminal", "cmd"],
                priority=2
            ),
            AgentCapability(
                name="file_operations",
                description="Read, write, and manage files",
                keywords=["file", "read", "write", "save", "load", "create", "delete"],
                priority=2
            ),
            AgentCapability(
                name="system_info",
                description="Get system information",
                keywords=["system", "info", "status", "disk", "memory", "cpu"],
                priority=1
            ),
            AgentCapability(
                name="automation",
                description="Automate tasks and workflows",
                keywords=["automate", "script", "batch", "schedule"],
                priority=1
            )
        ]

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a task execution request.

        Actions:
        - run_command: Execute a shell command
        - read_file: Read file content
        - write_file: Write to file
        - list_files: List directory
        - system_info: Get system info
        """
        action = task.get("action", "")
        input_data = task.get("input", "")
        options = task.get("options", {})

        try:
            if action == "run_command" or action == "execute":
                return await self._handle_command(input_data, options)
            elif action == "read_file" or action == "read":
                return await self._handle_read_file(input_data)
            elif action == "write_file" or action == "write":
                content = options.get("content", "")
                return await self._handle_write_file(input_data, content)
            elif action == "list_files" or action == "list":
                return await self._handle_list_files(input_data, options)
            elif action == "system_info":
                return await self._handle_system_info()
            elif action == "delete_file" or action == "delete":
                return await self._handle_delete_file(input_data)
            else:
                # Try to infer action
                if os.path.isfile(input_data):
                    return await self._handle_read_file(input_data)
                elif os.path.isdir(input_data):
                    return await self._handle_list_files(input_data, options)
                else:
                    return await self._handle_command(input_data, options)

        except Exception as e:
            logger.error(f"TaskExecutorAgent error: {e}")
            return TaskResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def _handle_command(self, command: str, options: Dict) -> TaskResult:
        """Execute a shell command."""
        # Security checks
        if not self._is_command_allowed(command):
            return TaskResult(
                success=False,
                data=None,
                error=f"Command not allowed: {command}"
            )

        if self._is_dangerous_command(command) and not self.allow_dangerous:
            return TaskResult(
                success=False,
                data=None,
                error=f"Dangerous command blocked: {command}"
            )

        # Execute the command
        timeout = options.get("timeout", 30)
        cwd = options.get("cwd", None)

        try:
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            return TaskResult(
                success=process.returncode == 0,
                data={
                    "command": command,
                    "stdout": stdout.decode("utf-8", errors="replace"),
                    "stderr": stderr.decode("utf-8", errors="replace"),
                    "return_code": process.returncode
                },
                error=stderr.decode() if process.returncode != 0 else None
            )

        except asyncio.TimeoutError:
            return TaskResult(
                success=False,
                data=None,
                error=f"Command timed out after {timeout} seconds"
            )

    async def _handle_read_file(self, path: str) -> TaskResult:
        """Read a file's contents."""
        # Security check
        if not self._is_path_allowed(path):
            return TaskResult(
                success=False,
                data=None,
                error=f"Path not allowed: {path}"
            )

        result = await self._file_manager.read(path)

        if result.success:
            return TaskResult(
                success=True,
                data={
                    "path": path,
                    "content": result.content,
                    "size": result.size
                }
            )
        else:
            return TaskResult(
                success=False,
                data=None,
                error=result.error
            )

    async def _handle_write_file(self, path: str, content: str) -> TaskResult:
        """Write content to a file."""
        if not self._is_path_allowed(path):
            return TaskResult(
                success=False,
                data=None,
                error=f"Path not allowed: {path}"
            )

        success = await self._file_manager.write(path, content)

        return TaskResult(
            success=success,
            data={"path": path, "bytes_written": len(content)} if success else None,
            error="Failed to write file" if not success else None
        )

    async def _handle_list_files(self, path: str, options: Dict) -> TaskResult:
        """List files in a directory."""
        if not self._is_path_allowed(path):
            return TaskResult(
                success=False,
                data=None,
                error=f"Path not allowed: {path}"
            )

        pattern = options.get("pattern", "*")
        recursive = options.get("recursive", False)

        files = self._file_manager.list_directory(path, pattern, recursive)

        return TaskResult(
            success=True,
            data={
                "path": path,
                "count": len(files),
                "files": [
                    {
                        "name": f.name,
                        "path": f.path,
                        "size": f.size,
                        "is_directory": f.is_directory,
                        "modified": f.modified_time.isoformat()
                    }
                    for f in files
                ]
            }
        )

    async def _handle_delete_file(self, path: str) -> TaskResult:
        """Delete a file (moves to backup)."""
        if not self._is_path_allowed(path):
            return TaskResult(
                success=False,
                data=None,
                error=f"Path not allowed: {path}"
            )

        success = await self._file_manager.delete(path)

        return TaskResult(
            success=success,
            data={"path": path, "deleted": success},
            error="Failed to delete file" if not success else None
        )

    async def _handle_system_info(self) -> TaskResult:
        """Get system information."""
        import shutil

        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        }

        # Disk usage (for current directory)
        try:
            disk = shutil.disk_usage(".")
            info["disk"] = {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 1)
            }
        except Exception:
            pass

        # Current working directory
        info["cwd"] = os.getcwd()

        # Environment (safe subset)
        safe_env_vars = ["PATH", "HOME", "USER", "SHELL", "TERM"]
        info["environment"] = {
            k: v for k, v in os.environ.items()
            if k in safe_env_vars
        }

        return TaskResult(
            success=True,
            data=info
        )

    # Security methods
    def _is_command_allowed(self, command: str) -> bool:
        """Check if a command is in the whitelist (if configured)."""
        if not self.command_whitelist:
            return True

        # Check if any whitelisted command is the start of this command
        cmd_base = command.split()[0] if command else ""
        return cmd_base in self.command_whitelist

    def _is_dangerous_command(self, command: str) -> bool:
        """Check if command matches known dangerous patterns."""
        command_lower = command.lower()
        return any(danger in command_lower for danger in self.DANGEROUS_COMMANDS)

    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories."""
        if not self.allowed_paths:
            return True

        abs_path = os.path.abspath(path)
        for allowed in self.allowed_paths:
            allowed_abs = os.path.abspath(allowed)
            if abs_path.startswith(allowed_abs):
                return True

        return False

    # Tool implementations
    async def _run_command(self, command: str) -> Dict[str, Any]:
        """Tool wrapper for running commands."""
        result = await self._handle_command(command, {})
        return result.data if result.success else {"error": result.error}

    async def _read_file(self, path: str) -> str:
        """Tool wrapper for reading files."""
        result = await self._handle_read_file(path)
        return result.data.get("content", "") if result.success else ""

    async def _write_file(self, path: str, content: str) -> bool:
        """Tool wrapper for writing files."""
        result = await self._handle_write_file(path, content)
        return result.success
