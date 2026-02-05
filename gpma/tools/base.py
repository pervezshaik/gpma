"""
Unified Tool Base Classes

This module provides the core tool abstractions that all tools inherit from.
It unifies the previously fragmented tool definitions (AgenticTool, LLMTool, Tool)
into a single, consistent system.

Key Classes:
- ToolResult: Standard response type for all tool executions
- ToolParameter: Type-safe parameter definition
- BaseTool: The unified tool class with all features

Features:
- Parameter validation with JSON Schema
- Timeout protection
- Rate limiting
- Usage statistics
- Retry logic
- Async/sync support
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ToolCategory(str, Enum):
    """Standard tool categories for organization."""
    WEB = "web"
    FILE = "file"
    MATH = "math"
    SEARCH = "search"
    SYSTEM = "system"
    DATA = "data"
    CUSTOM = "custom"


class ToolResultStatus(str, Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ToolResult:
    """
    Standard result type for all tool executions.

    This provides a consistent interface for tool outputs, making it easy
    for agents to handle results uniformly regardless of which tool was called.

    Attributes:
        status: The execution status (success, error, timeout, etc.)
        data: The actual result data (can be any type)
        error: Error message if status is not SUCCESS
        execution_time: Time taken to execute in seconds
        metadata: Additional context about the execution

    Usage:
        # Success case
        result = ToolResult.success(data={"weather": "sunny"})

        # Error case
        result = ToolResult.error("Connection failed", code="CONN_ERROR")

        # Check result
        if result.is_success:
            print(result.data)
        else:
            print(f"Error: {result.error}")
    """
    status: ToolResultStatus
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        """Check if the execution was successful."""
        return self.status == ToolResultStatus.SUCCESS

    @property
    def is_error(self) -> bool:
        """Check if the execution failed."""
        return self.status != ToolResultStatus.SUCCESS

    @classmethod
    def success(cls, data: Any = None, metadata: Dict[str, Any] = None) -> ToolResult:
        """Create a successful result."""
        return cls(
            status=ToolResultStatus.SUCCESS,
            data=data,
            metadata=metadata or {}
        )

    @classmethod
    def error(
        cls,
        message: str,
        code: str = None,
        data: Any = None,
        metadata: Dict[str, Any] = None
    ) -> ToolResult:
        """Create an error result."""
        return cls(
            status=ToolResultStatus.ERROR,
            error=message,
            error_code=code,
            data=data,
            metadata=metadata or {}
        )

    @classmethod
    def timeout(cls, timeout_seconds: float) -> ToolResult:
        """Create a timeout result."""
        return cls(
            status=ToolResultStatus.TIMEOUT,
            error=f"Tool execution timed out after {timeout_seconds} seconds",
            error_code="TIMEOUT"
        )

    @classmethod
    def rate_limited(cls, retry_after: float = None) -> ToolResult:
        """Create a rate limited result."""
        msg = "Rate limit exceeded"
        if retry_after:
            msg += f", retry after {retry_after:.1f} seconds"
        return cls(
            status=ToolResultStatus.RATE_LIMITED,
            error=msg,
            error_code="RATE_LIMITED",
            metadata={"retry_after": retry_after} if retry_after else {}
        )

    @classmethod
    def validation_error(cls, message: str, parameter: str = None) -> ToolResult:
        """Create a validation error result."""
        return cls(
            status=ToolResultStatus.VALIDATION_ERROR,
            error=message,
            error_code="VALIDATION_ERROR",
            metadata={"parameter": parameter} if parameter else {}
        )

    def to_string(self) -> str:
        """Convert result to a string representation for LLM consumption."""
        if self.is_success:
            if isinstance(self.data, str):
                return self.data
            elif self.data is None:
                return "Success (no data returned)"
            else:
                import json
                try:
                    return json.dumps(self.data, indent=2, default=str)
                except:
                    return str(self.data)
        else:
            return f"Error: {self.error}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


@dataclass
class ToolParameter:
    """
    Definition of a tool parameter with full type information.

    Attributes:
        name: Parameter name
        type: JSON Schema type (string, integer, number, boolean, array, object)
        description: Human-readable description for LLM
        required: Whether this parameter is required
        default: Default value if not provided
        enum: List of allowed values (for string types)
        items: Schema for array items (for array types)
        properties: Schema for object properties (for object types)
    """
    name: str
    type: str  # JSON Schema type
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    items: Optional[Dict[str, Any]] = None  # For array types
    properties: Optional[Dict[str, Any]] = None  # For object types

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema = {
            "type": self.type,
            "description": self.description
        }

        if self.default is not None:
            schema["default"] = self.default
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.items is not None:
            schema["items"] = self.items
        if self.properties is not None:
            schema["properties"] = self.properties

        return schema

    @classmethod
    def string(cls, name: str, description: str, required: bool = True,
               default: str = None, enum: List[str] = None) -> ToolParameter:
        """Create a string parameter."""
        return cls(name=name, type="string", description=description,
                   required=required, default=default, enum=enum)

    @classmethod
    def integer(cls, name: str, description: str, required: bool = True,
                default: int = None) -> ToolParameter:
        """Create an integer parameter."""
        return cls(name=name, type="integer", description=description,
                   required=required, default=default)

    @classmethod
    def number(cls, name: str, description: str, required: bool = True,
               default: float = None) -> ToolParameter:
        """Create a number (float) parameter."""
        return cls(name=name, type="number", description=description,
                   required=required, default=default)

    @classmethod
    def boolean(cls, name: str, description: str, required: bool = True,
                default: bool = None) -> ToolParameter:
        """Create a boolean parameter."""
        return cls(name=name, type="boolean", description=description,
                   required=required, default=default)

    @classmethod
    def array(cls, name: str, description: str, items_type: str = "string",
              required: bool = True) -> ToolParameter:
        """Create an array parameter."""
        return cls(name=name, type="array", description=description,
                   required=required, items={"type": items_type})


@dataclass
class BaseTool:
    """
    Unified tool class that combines all features from previous implementations.

    This is the single tool definition that all tools should use. It provides:
    - Parameter validation
    - Timeout protection
    - Rate limiting
    - Usage statistics
    - Retry logic
    - Async/sync execution support

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description (shown to LLM)
        parameters: List of ToolParameter definitions
        function: The implementation function (async or sync)
        category: Tool category for organization
        timeout: Maximum execution time in seconds
        retry_count: Number of retries on transient failures
        rate_limit: Maximum calls per minute (0 = unlimited)
        requires_confirmation: Whether to ask user before executing
        tags: Additional tags for filtering/discovery

    Usage:
        # Create a tool
        tool = BaseTool(
            name="get_weather",
            description="Get current weather for a city",
            parameters=[
                ToolParameter.string("city", "City name", required=True),
                ToolParameter.string("units", "Temperature units", required=False,
                                     default="celsius", enum=["celsius", "fahrenheit"])
            ],
            function=get_weather_impl,
            category=ToolCategory.WEB
        )

        # Execute
        result = await tool.execute(city="London")
        if result.is_success:
            print(result.data)
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable
    category: ToolCategory = ToolCategory.CUSTOM
    timeout: float = 30.0
    retry_count: int = 0
    rate_limit: int = 0  # calls per minute, 0 = unlimited
    requires_confirmation: bool = False
    tags: List[str] = field(default_factory=list)

    # Runtime statistics (not part of tool definition)
    _call_count: int = field(default=0, repr=False)
    _error_count: int = field(default=0, repr=False)
    _total_time: float = field(default=0.0, repr=False)
    _last_call: Optional[datetime] = field(default=None, repr=False)
    _last_error: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate tool configuration after initialization."""
        if not self.name:
            raise ValueError("Tool name is required")
        if not self.description:
            raise ValueError("Tool description is required")
        if not self.function:
            raise ValueError("Tool function is required")

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.

        This method handles:
        1. Parameter validation
        2. Rate limiting
        3. Timeout protection
        4. Retry logic
        5. Statistics tracking

        Args:
            **kwargs: Parameters to pass to the tool function

        Returns:
            ToolResult with the execution outcome
        """
        start_time = time.time()

        try:
            # 1. Validate parameters
            validation_error = self._validate_parameters(kwargs)
            if validation_error:
                return validation_error

            # 2. Apply defaults
            kwargs = self._apply_defaults(kwargs)

            # 3. Check rate limit
            if self.rate_limit > 0:
                rate_result = await self._check_rate_limit()
                if rate_result:
                    return rate_result

            # 4. Execute with retry logic
            last_error = None
            for attempt in range(max(1, self.retry_count + 1)):
                try:
                    result = await self._execute_with_timeout(kwargs)

                    # Track success
                    self._call_count += 1
                    self._last_call = datetime.now()
                    self._total_time += time.time() - start_time

                    # Wrap raw return values in ToolResult
                    if isinstance(result, ToolResult):
                        result.execution_time = time.time() - start_time
                        return result
                    else:
                        return ToolResult.success(
                            data=result,
                            metadata={"execution_time": time.time() - start_time}
                        )

                except asyncio.TimeoutError:
                    self._error_count += 1
                    self._last_error = f"Timeout after {self.timeout}s"
                    return ToolResult.timeout(self.timeout)

                except Exception as e:
                    last_error = e
                    if attempt < self.retry_count:
                        logger.warning(f"Tool {self.name} attempt {attempt + 1} failed: {e}")
                        await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                    continue

            # All retries exhausted
            self._error_count += 1
            self._last_error = str(last_error)
            logger.error(f"Tool {self.name} failed after {self.retry_count + 1} attempts: {last_error}")
            return ToolResult.error(str(last_error))

        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            logger.error(f"Tool {self.name} unexpected error: {e}")
            return ToolResult.error(str(e))

    def _validate_parameters(self, kwargs: Dict[str, Any]) -> Optional[ToolResult]:
        """Validate parameters against the schema."""
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                if param.default is None:
                    return ToolResult.validation_error(
                        f"Missing required parameter: {param.name}",
                        parameter=param.name
                    )

            if param.name in kwargs:
                value = kwargs[param.name]

                # Type validation
                if not self._validate_type(value, param.type):
                    return ToolResult.validation_error(
                        f"Invalid type for {param.name}: expected {param.type}, got {type(value).__name__}",
                        parameter=param.name
                    )

                # Enum validation
                if param.enum is not None and value not in param.enum:
                    return ToolResult.validation_error(
                        f"Invalid value for {param.name}: must be one of {param.enum}",
                        parameter=param.name
                    )

        return None

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": (list, tuple),
            "object": dict,
            "null": type(None)
        }

        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, allow

        return isinstance(value, expected)

    def _apply_defaults(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default values for missing parameters."""
        result = dict(kwargs)
        for param in self.parameters:
            if param.name not in result and param.default is not None:
                result[param.name] = param.default
        return result

    async def _check_rate_limit(self) -> Optional[ToolResult]:
        """Check and enforce rate limiting."""
        if self._last_call:
            elapsed = (datetime.now() - self._last_call).total_seconds()
            min_interval = 60.0 / self.rate_limit

            if elapsed < min_interval:
                wait_time = min_interval - elapsed
                # For small waits, just sleep. For larger waits, return rate limited.
                if wait_time > 5.0:
                    return ToolResult.rate_limited(retry_after=wait_time)
                await asyncio.sleep(wait_time)

        return None

    async def _execute_with_timeout(self, kwargs: Dict[str, Any]) -> Any:
        """Execute the function with timeout protection."""
        if asyncio.iscoroutinefunction(self.function):
            return await asyncio.wait_for(
                self.function(**kwargs),
                timeout=self.timeout
            )
        else:
            # Run sync function in executor to not block event loop
            loop = asyncio.get_event_loop()
            return await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self.function(**kwargs)),
                timeout=self.timeout
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this tool."""
        return {
            "name": self.name,
            "category": self.category.value,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._call_count, 1),
            "total_time": self._total_time,
            "avg_time": self._total_time / max(self._call_count, 1),
            "last_call": self._last_call.isoformat() if self._last_call else None,
            "last_error": self._last_error
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._call_count = 0
        self._error_count = 0
        self._total_time = 0.0
        self._last_call = None
        self._last_error = None

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters as JSON Schema (for LLM function calling)."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

        schema = {
            "type": "object",
            "properties": properties
        }

        if required:
            schema["required"] = required

        return schema

    def __repr__(self) -> str:
        return f"<BaseTool name={self.name} category={self.category.value}>"
