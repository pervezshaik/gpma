"""
Anthropic Tool Format Adapter

Converts BaseTool to Anthropic (Claude) tool use format.

Anthropic Format:
{
    "name": "tool_name",
    "description": "Tool description",
    "input_schema": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "integer", "description": "..."}
        },
        "required": ["param1"]
    }
}

Note: Anthropic uses "input_schema" instead of "parameters" and
doesn't wrap in a "function" object like OpenAI.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Union

from ..base import BaseTool, ToolResult


class AnthropicToolAdapter:
    """
    Adapter for converting tools to Anthropic format.

    Handles both conversion to Anthropic format and parsing of
    Anthropic tool use responses.

    Usage:
        adapter = AnthropicToolAdapter(tools)

        # Convert for API call
        anthropic_tools = adapter.to_format()

        # Parse tool use from response
        tool_name, args = adapter.parse_tool_use(content_block)

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

    def to_format(self) -> List[Dict[str, Any]]:
        """
        Convert all tools to Anthropic format.

        Returns:
            List of tool definitions in Anthropic format
        """
        return [
            self._tool_to_anthropic(tool)
            for tool in self._tools.values()
        ]

    def _tool_to_anthropic(self, tool: BaseTool) -> Dict[str, Any]:
        """Convert a single tool to Anthropic format."""
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.get_parameters_schema()
        }

    def parse_tool_use(
        self,
        content_block: Any
    ) -> tuple[str, Dict[str, Any]]:
        """
        Parse an Anthropic tool use content block.

        Args:
            content_block: Anthropic tool_use content block

        Returns:
            Tuple of (tool_name, arguments_dict)
        """
        # Handle both object and dict formats
        if hasattr(content_block, "name"):
            name = content_block.name
            args = content_block.input
        else:
            name = content_block.get("name", "")
            args = content_block.get("input", {})

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

    def format_result_for_api(
        self,
        tool_use_id: str,
        result: ToolResult,
        is_error: bool = None
    ) -> Dict[str, Any]:
        """
        Format a tool result for the Anthropic messages array.

        Args:
            tool_use_id: The ID from the original tool_use block
            result: The ToolResult from execution
            is_error: Whether this is an error result (auto-detected if None)

        Returns:
            Dict suitable for messages array content
        """
        if is_error is None:
            is_error = result.is_error

        content = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result.to_string()
        }

        if is_error:
            content["is_error"] = True

        return content


def to_anthropic_format(
    tools: Union[List[BaseTool], BaseTool]
) -> List[Dict[str, Any]]:
    """
    Convert tools to Anthropic tool use format.

    This is a convenience function for quick conversion.

    Args:
        tools: Single tool or list of tools

    Returns:
        List of tool definitions in Anthropic format

    Example:
        from gpma.tools import registry
        from gpma.tools.adapters import to_anthropic_format

        tools = registry.get_all()
        anthropic_tools = to_anthropic_format(tools)

        response = client.messages.create(
            model="claude-3-opus-20240229",
            messages=messages,
            tools=anthropic_tools
        )
    """
    if isinstance(tools, BaseTool):
        tools = [tools]

    adapter = AnthropicToolAdapter(tools)
    return adapter.to_format()


def tool_to_anthropic(tool: BaseTool) -> Dict[str, Any]:
    """
    Convert a single tool to Anthropic format.

    Args:
        tool: The tool to convert

    Returns:
        Tool definition in Anthropic format
    """
    return AnthropicToolAdapter([tool])._tool_to_anthropic(tool)
