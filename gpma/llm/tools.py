"""
LLM Tools Module

Utilities for creating tools that LLMs can call via function calling.

This module provides:
1. Easy tool creation from Python functions
2. Automatic JSON schema generation
3. Tool execution helpers

USAGE:
    from gpma.llm.tools import create_llm_tool, LLMTool

    # From a function
    @create_llm_tool
    def get_weather(city: str, units: str = "celsius") -> str:
        '''Get the current weather for a city.

        Args:
            city: The city name
            units: Temperature units (celsius or fahrenheit)
        '''
        return f"Weather in {city}: 72°F"

    # Manual creation
    tool = LLMTool(
        name="search",
        description="Search the web",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        },
        function=my_search_func
    )
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_type_hints
import inspect
import json
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMTool:
    """
    A tool that can be called by an LLM.

    Follows the OpenAI function calling format, which is
    widely supported by other providers.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    strict: bool = False  # OpenAI strict mode

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict
            }
        }

    async def execute(self, **kwargs) -> Any:
        """Execute the tool function."""
        try:
            if asyncio.iscoroutinefunction(self.function):
                return await self.function(**kwargs)
            else:
                return self.function(**kwargs)
        except Exception as e:
            logger.error(f"Tool execution error ({self.name}): {e}")
            return {"error": str(e)}


def create_llm_tool(func: Callable = None, *, name: str = None, description: str = None) -> LLMTool:
    """
    Decorator to create an LLM tool from a Python function.

    Automatically generates JSON schema from type hints and docstring.

    USAGE:
        @create_llm_tool
        def my_function(arg1: str, arg2: int = 10) -> str:
            '''Description of what the function does.

            Args:
                arg1: Description of arg1
                arg2: Description of arg2
            '''
            return "result"

        # Or with custom name/description
        @create_llm_tool(name="custom_name", description="Custom description")
        def my_function(...):
            ...
    """
    def decorator(fn: Callable) -> LLMTool:
        tool_name = name or fn.__name__
        tool_description = description or _extract_description(fn)
        parameters = _generate_schema(fn)

        return LLMTool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=fn
        )

    if func is not None:
        # Called without arguments: @create_llm_tool
        return decorator(func)
    else:
        # Called with arguments: @create_llm_tool(name="...")
        return decorator


def _extract_description(func: Callable) -> str:
    """Extract description from function docstring."""
    doc = func.__doc__
    if not doc:
        return f"Function {func.__name__}"

    # Get first paragraph (before Args:)
    lines = doc.strip().split("\n")
    description_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(("args:", "arguments:", "parameters:", "returns:", "raises:")):
            break
        description_lines.append(stripped)

    return " ".join(description_lines).strip() or f"Function {func.__name__}"


def _generate_schema(func: Callable) -> Dict[str, Any]:
    """Generate JSON schema from function signature and type hints."""
    sig = inspect.signature(func)
    hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}

    properties = {}
    required = []

    # Parse docstring for argument descriptions
    arg_descriptions = _parse_docstring_args(func.__doc__ or "")

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        # Get type
        param_type = hints.get(param_name, Any)
        json_type = _python_type_to_json(param_type)

        # Get description from docstring
        description = arg_descriptions.get(param_name, f"Parameter {param_name}")

        properties[param_name] = {
            "type": json_type,
            "description": description
        }

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema = {
        "type": "object",
        "properties": properties
    }

    if required:
        schema["required"] = required

    return schema


def _parse_docstring_args(docstring: str) -> Dict[str, str]:
    """Parse argument descriptions from docstring."""
    descriptions = {}

    if not docstring:
        return descriptions

    lines = docstring.split("\n")
    in_args = False
    current_arg = None
    current_desc = []

    for line in lines:
        stripped = line.strip()

        if stripped.lower().startswith(("args:", "arguments:", "parameters:")):
            in_args = True
            continue

        if stripped.lower().startswith(("returns:", "raises:", "yields:", "examples:")):
            # Save current arg if any
            if current_arg and current_desc:
                descriptions[current_arg] = " ".join(current_desc).strip()
            in_args = False
            continue

        if in_args:
            # Check for new argument (name: description)
            if ":" in stripped and not stripped.startswith(" "):
                # Save previous arg
                if current_arg and current_desc:
                    descriptions[current_arg] = " ".join(current_desc).strip()

                parts = stripped.split(":", 1)
                current_arg = parts[0].strip()
                current_desc = [parts[1].strip()] if len(parts) > 1 else []
            elif current_arg and stripped:
                # Continuation of current arg description
                current_desc.append(stripped)

    # Don't forget the last argument
    if current_arg and current_desc:
        descriptions[current_arg] = " ".join(current_desc).strip()

    return descriptions


def _python_type_to_json(python_type) -> str:
    """Convert Python type hint to JSON schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }

    # Handle basic types
    if python_type in type_map:
        return type_map[python_type]

    # Handle typing module types
    type_str = str(python_type)

    if "List" in type_str or "list" in type_str:
        return "array"
    if "Dict" in type_str or "dict" in type_str:
        return "object"
    if "Optional" in type_str:
        return "string"  # Default for optional
    if "Union" in type_str:
        return "string"  # Default for union
    if "Any" in type_str:
        return "string"  # Default for any

    return "string"  # Default


class ToolRegistry:
    """
    Registry for managing LLM tools.

    Useful when you have many tools and want to:
    - Organize them by category
    - Enable/disable groups of tools
    - Share tools across agents
    """

    def __init__(self):
        self._tools: Dict[str, LLMTool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: LLMTool, category: str = "general") -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

        if category not in self._categories:
            self._categories[category] = []
        self._categories[category].append(tool.name)

    def get(self, name: str) -> Optional[LLMTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_category(self, category: str) -> List[LLMTool]:
        """Get all tools in a category."""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_all(self) -> List[LLMTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def to_openai_format(self, categories: List[str] = None) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format, optionally filtered by category."""
        if categories:
            tools = []
            for cat in categories:
                tools.extend(self.get_category(cat))
        else:
            tools = self.get_all()

        return [tool.to_openai_format() for tool in tools]

    async def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        return await tool.execute(**kwargs)


# ============================================================================
# COMMON TOOL DEFINITIONS
# ============================================================================

def create_web_search_tool(search_func: Callable) -> LLMTool:
    """Create a web search tool."""
    return LLMTool(
        name="web_search",
        description="Search the web for information. Use this when you need current information or facts you don't know.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)"
                }
            },
            "required": ["query"]
        },
        function=search_func
    )


def create_fetch_url_tool(fetch_func: Callable) -> LLMTool:
    """Create a URL fetch tool."""
    return LLMTool(
        name="fetch_url",
        description="Fetch the content of a webpage. Use this to read articles or get specific information from URLs.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        },
        function=fetch_func
    )


def create_file_read_tool(read_func: Callable) -> LLMTool:
    """Create a file read tool."""
    return LLMTool(
        name="read_file",
        description="Read the contents of a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["path"]
        },
        function=read_func
    )


def create_file_write_tool(write_func: Callable) -> LLMTool:
    """Create a file write tool."""
    return LLMTool(
        name="write_file",
        description="Write content to a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        },
        function=write_func
    )


def create_calculator_tool() -> LLMTool:
    """Create a simple calculator tool."""
    def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        # Very basic safe eval for math
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        try:
            result = eval(expression)  # Safe because we validated chars
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    return LLMTool(
        name="calculator",
        description="Perform mathematical calculations. Supports +, -, *, /, parentheses.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., '2 + 2 * 3'"
                }
            },
            "required": ["expression"]
        },
        function=calculate
    )
