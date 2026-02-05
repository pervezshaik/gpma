"""
Tool Decorators

Easy-to-use decorators for creating tools from Python functions.
These decorators automatically:
- Extract parameter information from type hints
- Parse descriptions from docstrings
- Create proper JSON schemas
- Handle async/sync functions

Usage:
    from gpma.tools import tool, ToolCategory

    @tool(description="Search the web for information", category=ToolCategory.WEB)
    async def web_search(query: str, num_results: int = 5) -> str:
        '''Perform a web search.

        Args:
            query: The search query
            num_results: Number of results to return
        '''
        # Implementation
        return results

    # The function is now a BaseTool and can be used directly
    result = await web_search.execute(query="python tutorials")
"""

from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints

from .base import BaseTool, ToolCategory, ToolParameter, ToolResult

logger = logging.getLogger(__name__)


def tool(
    description: str = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    timeout: float = 30.0,
    retry_count: int = 0,
    rate_limit: int = 0,
    requires_confirmation: bool = False,
    tags: List[str] = None,
    auto_register: bool = True
) -> Callable[[Callable], BaseTool]:
    """
    Decorator to create a BaseTool from a function.

    Automatically extracts:
    - Tool name from function name
    - Description from docstring (or parameter)
    - Parameters from type hints and docstring Args section

    Args:
        description: Tool description (uses docstring if not provided)
        category: Tool category for organization
        timeout: Maximum execution time in seconds
        retry_count: Number of retries on failure
        rate_limit: Max calls per minute (0 = unlimited)
        requires_confirmation: Whether to ask user before executing
        tags: Additional tags for filtering
        auto_register: Whether to automatically register in global registry

    Returns:
        A decorator that transforms a function into a BaseTool

    Example:
        @tool(category=ToolCategory.WEB, timeout=60.0)
        async def fetch_url(url: str) -> str:
            '''Fetch content from a URL.

            Args:
                url: The URL to fetch
            '''
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    return await response.text()
    """
    def decorator(func: Callable) -> BaseTool:
        # Get function name
        tool_name = func.__name__

        # Get description
        tool_description = description
        if not tool_description:
            tool_description = _extract_description(func)

        # Get parameters from type hints and docstring
        parameters = _extract_parameters(func)

        # Create the tool
        base_tool = BaseTool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=func,
            category=category,
            timeout=timeout,
            retry_count=retry_count,
            rate_limit=rate_limit,
            requires_confirmation=requires_confirmation,
            tags=tags or []
        )

        # Auto-register if requested
        if auto_register:
            try:
                from .registry import registry
                registry.register(base_tool, replace=True)
            except ImportError:
                pass  # Registry not available

        logger.debug(f"Created tool from function: {tool_name}")
        return base_tool

    return decorator


def simple_tool(func: Callable) -> BaseTool:
    """
    Simple decorator for creating a tool with minimal configuration.

    Uses all defaults - just provide a function with type hints and docstring.

    Example:
        @simple_tool
        def calculate(expression: str) -> str:
            '''Evaluate a math expression.'''
            return str(eval(expression))
    """
    return tool()(func)


def _extract_description(func: Callable) -> str:
    """Extract tool description from function docstring."""
    doc = func.__doc__
    if not doc:
        return f"Execute {func.__name__}"

    # Get first paragraph (before Args:, Returns:, etc.)
    lines = doc.strip().split("\n")
    description_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith(("args:", "arguments:", "parameters:",
                                        "returns:", "raises:", "yields:",
                                        "examples:", "example:")):
            break
        description_lines.append(stripped)

    description = " ".join(description_lines).strip()
    return description or f"Execute {func.__name__}"


def _extract_parameters(func: Callable) -> List[ToolParameter]:
    """Extract tool parameters from function signature and docstring."""
    sig = inspect.signature(func)

    # Get type hints
    try:
        hints = get_type_hints(func)
    except Exception:
        hints = getattr(func, "__annotations__", {})

    # Parse docstring for argument descriptions
    arg_descriptions = _parse_docstring_args(func.__doc__ or "")

    parameters = []

    for param_name, param in sig.parameters.items():
        # Skip self, cls, and *args/**kwargs
        if param_name in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        # Get type
        python_type = hints.get(param_name, str)
        json_type = _python_type_to_json(python_type)

        # Get description from docstring
        param_description = arg_descriptions.get(param_name, f"Parameter: {param_name}")

        # Check if required (no default value)
        is_required = param.default is inspect.Parameter.empty

        # Get default value
        default_value = None if param.default is inspect.Parameter.empty else param.default

        # Check for enum types (Literal, Enum)
        enum_values = _extract_enum_values(python_type)

        tool_param = ToolParameter(
            name=param_name,
            type=json_type,
            description=param_description,
            required=is_required,
            default=default_value,
            enum=enum_values
        )

        parameters.append(tool_param)

    return parameters


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
            current_arg = None
            current_desc = []
            continue

        if in_args:
            # Check for new argument (name: description or name (type): description)
            if ":" in stripped and not stripped.startswith(" "):
                # Save previous arg
                if current_arg and current_desc:
                    descriptions[current_arg] = " ".join(current_desc).strip()

                # Parse new arg
                parts = stripped.split(":", 1)
                arg_part = parts[0].strip()

                # Handle "name (type)" format
                if "(" in arg_part:
                    arg_part = arg_part.split("(")[0].strip()

                current_arg = arg_part
                current_desc = [parts[1].strip()] if len(parts) > 1 else []

            elif current_arg and stripped:
                # Continuation of current arg description
                current_desc.append(stripped)

    # Don't forget the last argument
    if current_arg and current_desc:
        descriptions[current_arg] = " ".join(current_desc).strip()

    return descriptions


def _python_type_to_json(python_type: Any) -> str:
    """Convert Python type hint to JSON Schema type."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
        bytes: "string",
    }

    # Handle basic types
    if python_type in type_map:
        return type_map[python_type]

    # Get origin for generic types
    origin = getattr(python_type, "__origin__", None)

    if origin is not None:
        # Handle List[X]
        if origin is list:
            return "array"
        # Handle Dict[X, Y]
        if origin is dict:
            return "object"
        # Handle Optional[X] (Union[X, None])
        if origin is Union:
            args = getattr(python_type, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return _python_type_to_json(non_none[0])

    # Handle string representation for complex types
    type_str = str(python_type)
    if "List" in type_str or "list" in type_str:
        return "array"
    if "Dict" in type_str or "dict" in type_str:
        return "object"
    if "int" in type_str.lower():
        return "integer"
    if "float" in type_str.lower():
        return "number"
    if "bool" in type_str.lower():
        return "boolean"

    return "string"  # Default


def _extract_enum_values(python_type: Any) -> Optional[List[Any]]:
    """Extract enum values from Literal or Enum types."""
    # Check for Literal type
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        # Python 3.8+ Literal
        type_str = str(origin)
        if "Literal" in type_str:
            return list(getattr(python_type, "__args__", ()))

    # Check for Enum class
    if inspect.isclass(python_type):
        from enum import Enum
        if issubclass(python_type, Enum):
            return [e.value for e in python_type]

    return None


# ============================================================================
# Tool Builder for More Complex Configurations
# ============================================================================

class ToolBuilder:
    """
    Fluent builder for creating tools with complex configurations.

    Use this when the decorator doesn't provide enough flexibility.

    Example:
        tool = (ToolBuilder("fetch_url")
            .description("Fetch content from a URL")
            .category(ToolCategory.WEB)
            .parameter("url", "string", "The URL to fetch", required=True)
            .parameter("timeout", "integer", "Request timeout", default=30)
            .timeout(60.0)
            .retry(3)
            .function(fetch_url_impl)
            .build())
    """

    def __init__(self, name: str):
        self._name = name
        self._description = f"Execute {name}"
        self._category = ToolCategory.CUSTOM
        self._parameters: List[ToolParameter] = []
        self._function: Optional[Callable] = None
        self._timeout = 30.0
        self._retry_count = 0
        self._rate_limit = 0
        self._requires_confirmation = False
        self._tags: List[str] = []

    def description(self, desc: str) -> ToolBuilder:
        """Set tool description."""
        self._description = desc
        return self

    def category(self, cat: ToolCategory) -> ToolBuilder:
        """Set tool category."""
        self._category = cat
        return self

    def parameter(
        self,
        name: str,
        type: str,
        description: str,
        required: bool = True,
        default: Any = None,
        enum: List[Any] = None
    ) -> ToolBuilder:
        """Add a parameter."""
        self._parameters.append(ToolParameter(
            name=name,
            type=type,
            description=description,
            required=required,
            default=default,
            enum=enum
        ))
        return self

    def timeout(self, seconds: float) -> ToolBuilder:
        """Set timeout in seconds."""
        self._timeout = seconds
        return self

    def retry(self, count: int) -> ToolBuilder:
        """Set retry count."""
        self._retry_count = count
        return self

    def rate_limit(self, calls_per_minute: int) -> ToolBuilder:
        """Set rate limit."""
        self._rate_limit = calls_per_minute
        return self

    def requires_confirmation(self, required: bool = True) -> ToolBuilder:
        """Set confirmation requirement."""
        self._requires_confirmation = required
        return self

    def tag(self, *tags: str) -> ToolBuilder:
        """Add tags."""
        self._tags.extend(tags)
        return self

    def function(self, func: Callable) -> ToolBuilder:
        """Set the implementation function."""
        self._function = func
        return self

    def build(self) -> BaseTool:
        """Build the tool."""
        if not self._function:
            raise ValueError("Tool function is required")

        return BaseTool(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
            function=self._function,
            category=self._category,
            timeout=self._timeout,
            retry_count=self._retry_count,
            rate_limit=self._rate_limit,
            requires_confirmation=self._requires_confirmation,
            tags=self._tags
        )

    def register(self) -> BaseTool:
        """Build and register the tool."""
        tool = self.build()
        from .registry import registry
        registry.register(tool)
        return tool
