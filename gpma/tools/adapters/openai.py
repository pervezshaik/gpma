"""
OpenAI Tool Format Adapter

Converts BaseTool to OpenAI function calling format.
This format is also compatible with:
- Azure OpenAI
- Groq
- Together AI
- Ollama (with function calling support)
- Any OpenAI-compatible API

OpenAI Format:
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "Tool description",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "..."},
                "param2": {"type": "integer", "description": "..."}
            },
            "required": ["param1"]
        }
    }
}
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from ..base import BaseTool, ToolResult


class OpenAIToolAdapter:
    """
    Adapter for converting tools to OpenAI format.

    Handles both conversion to OpenAI format and parsing of
    OpenAI tool call responses.

    Usage:
        adapter = OpenAIToolAdapter(tools)

        # Convert for API call
        openai_tools = adapter.to_format()

        # Parse tool call from response
        tool_name, args = adapter.parse_tool_call(response.tool_calls[0])

        # Execute the tool
        result = await adapter.execute(tool_name, args)
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initialize adapter with tools.

        Args:
            tools: List of BaseTools to adapt
        """
        self._tools = {tool.name: tool for tool in tools}

    def to_format(self, strict: bool = False) -> List[Dict[str, Any]]:
        """
        Convert all tools to OpenAI format.

        Args:
            strict: Enable OpenAI strict mode (requires all params defined)

        Returns:
            List of tool definitions in OpenAI format
        """
        return [
            self._tool_to_openai(tool, strict)
            for tool in self._tools.values()
        ]

    def _tool_to_openai(self, tool: BaseTool, strict: bool = False) -> Dict[str, Any]:
        """Convert a single tool to OpenAI format."""
        result = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.get_parameters_schema()
            }
        }

        if strict:
            result["function"]["strict"] = True

        return result

    def parse_tool_call(
        self,
        tool_call: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Parse an OpenAI tool call object.

        Args:
            tool_call: OpenAI tool call object with function.name and function.arguments

        Returns:
            Tuple of (tool_name, arguments_dict)
        """
        # Handle both object and dict formats
        if hasattr(tool_call, "function"):
            name = tool_call.function.name
            args_str = tool_call.function.arguments
        else:
            name = tool_call.get("function", {}).get("name", "")
            args_str = tool_call.get("function", {}).get("arguments", "{}")

        # Parse arguments
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        return name, args

    async def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            ToolResult from execution

        Raises:
            ValueError: If tool not found
        """
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")

        return await tool.execute(**args)

    def format_result_for_api(self, tool_call_id: str, result: ToolResult) -> Dict[str, Any]:
        """
        Format a tool result for the OpenAI messages array.

        Args:
            tool_call_id: The ID from the original tool call
            result: The ToolResult from execution

        Returns:
            Dict suitable for messages array with role="tool"
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result.to_string()
        }


def to_openai_format(
    tools: Union[List[BaseTool], BaseTool],
    strict: bool = False
) -> List[Dict[str, Any]]:
    """
    Convert tools to OpenAI function calling format.

    This is a convenience function for quick conversion.

    Args:
        tools: Single tool or list of tools
        strict: Enable OpenAI strict mode

    Returns:
        List of tool definitions in OpenAI format

    Example:
        from gpma.tools import registry
        from gpma.tools.adapters import to_openai_format

        tools = registry.get_all()
        openai_tools = to_openai_format(tools)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=openai_tools
        )
    """
    if isinstance(tools, BaseTool):
        tools = [tools]

    adapter = OpenAIToolAdapter(tools)
    return adapter.to_format(strict=strict)


def tool_to_openai(tool: BaseTool, strict: bool = False) -> Dict[str, Any]:
    """
    Convert a single tool to OpenAI format.

    Args:
        tool: The tool to convert
        strict: Enable OpenAI strict mode

    Returns:
        Tool definition in OpenAI format
    """
    return OpenAIToolAdapter([tool])._tool_to_openai(tool, strict)
