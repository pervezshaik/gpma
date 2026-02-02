"""
Agentic Tools Module - Production-Grade Tools for Agentic Loops

This module provides production-ready tools that can be used with the AgenticLoop.
Tools are designed with:
- Safety: Input validation, sandboxed execution
- Reliability: Error handling, retries, timeouts
- Observability: Logging, metrics
- Reusability: Centralized registry, consistent interface

USAGE:
    from gpma.tools.agentic_tools import ToolRegistry, get_default_tools

    # Get all default tools
    tools = get_default_tools()

    # Or get specific tools
    registry = ToolRegistry()
    search_tool = registry.get("web_search")
    calc_tool = registry.get("calculator")

    # Use with AgenticLoop
    result = await loop.run(goal="...", tools=tools)
"""

import ast
import asyncio
import logging
import operator
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Base Classes
# =============================================================================

@dataclass
class AgenticTool:
    """
    Production-grade tool definition for agentic loops.
    
    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description (used by LLM)
        parameters: Parameter schema for validation
        function: The async function to execute
        category: Tool category for organization
        requires_confirmation: Whether to ask user before executing
        timeout: Maximum execution time in seconds
        retry_count: Number of retries on failure
        rate_limit: Max calls per minute (0 = unlimited)
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    category: str = "general"
    requires_confirmation: bool = False
    timeout: float = 30.0
    retry_count: int = 2
    rate_limit: int = 0
    
    # Runtime tracking
    _call_count: int = field(default=0, repr=False)
    _last_call: Optional[datetime] = field(default=None, repr=False)
    _total_time: float = field(default=0.0, repr=False)
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with validation and error handling."""
        import time
        start = time.time()
        
        try:
            # Validate parameters
            self._validate_params(kwargs)
            
            # Check rate limit
            if self.rate_limit > 0:
                await self._check_rate_limit()
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(self.function):
                result = await asyncio.wait_for(
                    self.function(**kwargs),
                    timeout=self.timeout
                )
            else:
                result = self.function(**kwargs)
            
            self._call_count += 1
            self._last_call = datetime.now()
            self._total_time += time.time() - start
            
            logger.debug(f"Tool '{self.name}' executed successfully")
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Tool '{self.name}' timed out after {self.timeout}s")
            return f"Error: Tool execution timed out after {self.timeout} seconds"
        except Exception as e:
            logger.error(f"Tool '{self.name}' failed: {e}")
            return f"Error: {str(e)}"
    
    def _validate_params(self, kwargs: Dict[str, Any]) -> None:
        """Validate parameters against schema."""
        for param_name, param_spec in self.parameters.items():
            if param_spec.get("required", False) and param_name not in kwargs:
                raise ValueError(f"Missing required parameter: {param_name}")
    
    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        if self._last_call:
            elapsed = (datetime.now() - self._last_call).total_seconds()
            min_interval = 60.0 / self.rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / max(self._call_count, 1),
            "last_call": self._last_call.isoformat() if self._last_call else None
        }


# =============================================================================
# Safe Calculator Tool
# =============================================================================

class SafeCalculator:
    """
    Safe mathematical expression evaluator.
    
    Uses AST parsing instead of eval() to prevent code injection.
    Supports: +, -, *, /, **, %, //, abs, round, min, max, sqrt
    """
    
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    ALLOWED_FUNCTIONS = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sqrt': lambda x: x ** 0.5,
        'pow': pow,
        'sum': sum,
        'int': int,
        'float': float,
    }
    
    MAX_VALUE = 10 ** 100  # Prevent overflow attacks
    
    def evaluate(self, expression: str) -> Union[int, float, str]:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression string
            
        Returns:
            Numeric result or error message
        """
        try:
            # Clean the expression
            expression = expression.strip()
            
            # Parse to AST
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate safely
            result = self._eval_node(tree.body)
            
            # Check for overflow
            if isinstance(result, (int, float)) and abs(result) > self.MAX_VALUE:
                return "Error: Result too large"
            
            return result
            
        except SyntaxError:
            return f"Error: Invalid expression syntax"
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _eval_node(self, node: ast.AST) -> Union[int, float]:
        """Recursively evaluate an AST node."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
            return op(left, right)
        
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.ALLOWED_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            return op(operand)
        
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self.ALLOWED_FUNCTIONS:
                raise ValueError(f"Function not allowed: {func_name}")
            
            args = [self._eval_node(arg) for arg in node.args]
            return self.ALLOWED_FUNCTIONS[func_name](*args)
        
        elif isinstance(node, ast.List):
            return [self._eval_node(elem) for elem in node.elts]
        
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_node(elem) for elem in node.elts)
        
        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


async def safe_calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Math expression (e.g., "25 * 4", "sqrt(16)", "max(1, 2, 3)")
        
    Returns:
        Result string or error message
    """
    calculator = SafeCalculator()
    result = calculator.evaluate(expression)
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    return f"Result: {result}"


# =============================================================================
# Knowledge Search Tool
# =============================================================================

class KnowledgeBase:
    """
    Extensible knowledge base for search operations.
    
    Supports:
    - In-memory knowledge entries
    - Web search fallback
    - Vector similarity search (when available)
    """
    
    def __init__(self):
        self._entries: Dict[str, Dict[str, Any]] = {}
        self._load_default_knowledge()
    
    def _load_default_knowledge(self):
        """Load default programming knowledge."""
        self._entries = {
            "python": {
                "title": "Python Programming Language",
                "content": "Python is a high-level, interpreted programming language known for its clear syntax and readability. Created by Guido van Rossum in 1991, it emphasizes code readability and supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used for web development, data science, AI/ML, automation, and scripting.",
                "keywords": ["python", "programming", "language", "guido", "interpreted"],
                "category": "programming_language"
            },
            "javascript": {
                "title": "JavaScript Programming Language",
                "content": "JavaScript is a high-level, dynamic programming language primarily used for web development. It runs in browsers and enables interactive web pages. With Node.js, JavaScript can also run on servers. It supports event-driven, functional, and object-oriented programming styles.",
                "keywords": ["javascript", "js", "web", "browser", "node", "frontend"],
                "category": "programming_language"
            },
            "rust": {
                "title": "Rust Programming Language",
                "content": "Rust is a systems programming language focused on safety, speed, and concurrency. It prevents memory errors at compile time through its ownership system. Rust is used for system programming, WebAssembly, CLI tools, and performance-critical applications.",
                "keywords": ["rust", "systems", "memory", "safety", "performance"],
                "category": "programming_language"
            },
            "machine_learning": {
                "title": "Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, learn patterns, and make predictions. Common types include supervised learning, unsupervised learning, and reinforcement learning.",
                "keywords": ["machine learning", "ml", "ai", "artificial intelligence", "algorithms", "data"],
                "category": "technology"
            },
            "api": {
                "title": "Application Programming Interface (API)",
                "content": "An API (Application Programming Interface) is a set of protocols and tools for building software applications. APIs define how software components should interact. REST APIs use HTTP methods (GET, POST, PUT, DELETE) to perform operations on resources. GraphQL is an alternative that allows clients to request specific data.",
                "keywords": ["api", "rest", "graphql", "http", "interface", "web service"],
                "category": "technology"
            },
            "docker": {
                "title": "Docker Containerization",
                "content": "Docker is a platform for developing, shipping, and running applications in containers. Containers package an application with all its dependencies, ensuring consistent behavior across environments. Docker uses images (blueprints) to create containers (running instances).",
                "keywords": ["docker", "container", "containerization", "devops", "deployment"],
                "category": "technology"
            }
        }
    
    def add_entry(self, key: str, title: str, content: str, 
                  keywords: List[str] = None, category: str = "general"):
        """Add a knowledge entry."""
        self._entries[key] = {
            "title": title,
            "content": content,
            "keywords": keywords or [],
            "category": category
        }
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of matching entries with relevance scores
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        results = []
        
        for key, entry in self._entries.items():
            score = 0.0
            
            # Check key match
            if key in query_lower:
                score += 0.5
            
            # Check keyword matches
            for keyword in entry.get("keywords", []):
                if keyword.lower() in query_lower:
                    score += 0.3
            
            # Check title match
            title_lower = entry.get("title", "").lower()
            for word in query_words:
                if word in title_lower:
                    score += 0.2
            
            # Check content match
            content_lower = entry.get("content", "").lower()
            for word in query_words:
                if len(word) > 3 and word in content_lower:
                    score += 0.1
            
            if score > 0:
                results.append({
                    "key": key,
                    "title": entry["title"],
                    "content": entry["content"],
                    "category": entry.get("category", "general"),
                    "score": min(score, 1.0)
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]


# Global knowledge base instance
_knowledge_base = KnowledgeBase()


async def search_knowledge(query: str, max_results: int = 3) -> str:
    """
    Search the knowledge base for information.
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        Formatted search results or "no results" message
    """
    results = _knowledge_base.search(query, max_results)
    
    if not results:
        # Try web search as fallback
        try:
            from .web_tools import search_web
            web_results = await search_web(query, num_results=3)
            if web_results:
                output = f"Web search results for '{query}':\n"
                for i, r in enumerate(web_results[:3], 1):
                    output += f"\n{i}. {r.title}\n   {r.snippet[:150]}...\n"
                return output
        except Exception as e:
            logger.warning(f"Web search fallback failed: {e}")
        
        return f"No information found for: {query}"
    
    # Format results
    output = f"Knowledge base results for '{query}':\n"
    for i, result in enumerate(results, 1):
        output += f"\n{i}. {result['title']} (relevance: {result['score']:.0%})\n"
        output += f"   {result['content'][:200]}...\n"
    
    return output


async def web_search(query: str, num_results: int = 5) -> str:
    """
    Search the web for information.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results
    """
    try:
        from .web_tools import search_web as do_search
        results = await do_search(query, num_results=num_results)
        
        if not results:
            return f"No web results found for: {query}"
        
        output = f"Web search results for '{query}':\n"
        for i, r in enumerate(results, 1):
            output += f"\n{i}. {r.title}\n"
            output += f"   URL: {r.url}\n"
            output += f"   {r.snippet[:150]}...\n"
        
        return output
        
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Web search error: {str(e)}"


async def fetch_webpage(url: str) -> str:
    """
    Fetch and extract text from a webpage.
    
    Args:
        url: URL to fetch
        
    Returns:
        Extracted text content or error message
    """
    try:
        from .web_tools import fetch_url
        page = await fetch_url(url)
        
        if page.status_code == 0:
            return f"Error fetching URL: {page.metadata.get('error', 'Unknown error')}"
        
        text = page.text or page.content[:2000]
        return f"Content from {url}:\n\n{text[:3000]}..."
        
    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return f"Error fetching URL: {str(e)}"


# =============================================================================
# File Operations Tool
# =============================================================================

async def read_file_content(filepath: str, max_lines: int = 100) -> str:
    """
    Read content from a file (with safety checks).
    
    Args:
        filepath: Path to the file
        max_lines: Maximum lines to read
        
    Returns:
        File content or error message
    """
    import os
    
    # Security: Prevent path traversal
    filepath = os.path.normpath(filepath)
    if ".." in filepath:
        return "Error: Path traversal not allowed"
    
    try:
        from .file_tools import read_file
        content = await read_file(filepath)
        
        # Limit output
        lines = content.split('\n')
        if len(lines) > max_lines:
            content = '\n'.join(lines[:max_lines])
            content += f"\n... (truncated, {len(lines) - max_lines} more lines)"
        
        return f"Content of {filepath}:\n\n{content}"
        
    except FileNotFoundError:
        return f"Error: File not found: {filepath}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


async def list_files(directory: str = ".") -> str:
    """
    List files in a directory.
    
    Args:
        directory: Directory path
        
    Returns:
        List of files or error message
    """
    import os
    
    # Security: Prevent path traversal
    directory = os.path.normpath(directory)
    if ".." in directory:
        return "Error: Path traversal not allowed"
    
    try:
        from .file_tools import list_directory
        files = await list_directory(directory)
        
        output = f"Files in {directory}:\n"
        for f in files[:50]:  # Limit to 50 files
            output += f"  - {f}\n"
        
        if len(files) > 50:
            output += f"  ... and {len(files) - 50} more files"
        
        return output
        
    except Exception as e:
        return f"Error listing directory: {str(e)}"


# =============================================================================
# Tool Registry
# =============================================================================

class ToolRegistry:
    """
    Centralized registry for managing agentic tools.
    
    Features:
    - Tool registration and lookup
    - Category-based organization
    - Usage statistics
    - Tool discovery
    """
    
    _instance: Optional['ToolRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools: Dict[str, AgenticTool] = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._register_default_tools()
            self._initialized = True
    
    def _register_default_tools(self):
        """Register all default tools."""
        # Calculator
        self.register(AgenticTool(
            name="calculator",
            description="Safely evaluate mathematical expressions. Supports: +, -, *, /, **, %, sqrt(), abs(), round(), min(), max(), pow(), sum(). Example: 'sqrt(16) + 5 * 2'",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate",
                    "required": True
                }
            },
            function=safe_calculate,
            category="math",
            timeout=5.0
        ))
        
        # Knowledge search
        self.register(AgenticTool(
            name="search",
            description="Search the knowledge base for information about programming languages, technologies, and concepts. Falls back to web search if no local results found.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": True
                }
            },
            function=search_knowledge,
            category="search",
            timeout=10.0
        ))
        
        # Web search
        self.register(AgenticTool(
            name="web_search",
            description="Search the web for current information. Use this for recent events, news, or topics not in the knowledge base.",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query",
                    "required": True
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5)",
                    "required": False
                }
            },
            function=web_search,
            category="search",
            timeout=30.0
        ))
        
        # Fetch webpage
        self.register(AgenticTool(
            name="fetch_url",
            description="Fetch and extract text content from a webpage URL.",
            parameters={
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                    "required": True
                }
            },
            function=fetch_webpage,
            category="web",
            timeout=30.0
        ))
        
        # Read file
        self.register(AgenticTool(
            name="read_file",
            description="Read content from a file. Use for examining code, configs, or documents.",
            parameters={
                "filepath": {
                    "type": "string",
                    "description": "Path to the file",
                    "required": True
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum lines to read (default: 100)",
                    "required": False
                }
            },
            function=read_file_content,
            category="file",
            timeout=10.0
        ))
        
        # List files
        self.register(AgenticTool(
            name="list_files",
            description="List files in a directory.",
            parameters={
                "directory": {
                    "type": "string",
                    "description": "Directory path (default: current directory)",
                    "required": False
                }
            },
            function=list_files,
            category="file",
            timeout=10.0
        ))
    
    def register(self, tool: AgenticTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
    
    def get(self, name: str) -> Optional[AgenticTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_all(self) -> List[AgenticTool]:
        """Get all registered tools."""
        return list(self._tools.values())
    
    def get_by_category(self, category: str) -> List[AgenticTool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if t.category == category]
    
    def list_tools(self) -> List[Dict[str, str]]:
        """List all tools with descriptions."""
        return [
            {"name": t.name, "description": t.description, "category": t.category}
            for t in self._tools.values()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics for all tools."""
        return {
            "total_tools": len(self._tools),
            "tools": [t.get_stats() for t in self._tools.values()]
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_default_tools() -> List[AgenticTool]:
    """
    Get the default set of production tools for agentic loops.
    
    Returns:
        List of AgenticTool instances
    """
    registry = ToolRegistry()
    return registry.get_all()


def get_tool(name: str) -> Optional[AgenticTool]:
    """
    Get a specific tool by name.
    
    Args:
        name: Tool name
        
    Returns:
        AgenticTool or None
    """
    registry = ToolRegistry()
    return registry.get(name)


def get_tools_by_category(category: str) -> List[AgenticTool]:
    """
    Get tools by category.
    
    Args:
        category: Category name (math, search, web, file)
        
    Returns:
        List of matching tools
    """
    registry = ToolRegistry()
    return registry.get_by_category(category)


# For backward compatibility with demo
def create_demo_tools() -> List[AgenticTool]:
    """
    Create tools compatible with the agentic demo.
    
    This provides the same interface as the inline demo tools
    but with production-grade implementations.
    """
    registry = ToolRegistry()
    
    # Return search and calculator (same as demo)
    return [
        registry.get("search"),
        registry.get("calculator")
    ]


# =============================================================================
# AUTO-TOOL DECORATOR
# =============================================================================

def auto_tool(
    description: str = None,
    category: str = "general",
    timeout: float = 30.0,
    requires_confirmation: bool = False
):
    """
    Decorator to automatically create an AgenticTool from a function.
    
    Infers parameter schema from type hints and docstring.
    
    USAGE:
        @auto_tool("Search the web for information")
        async def search(query: str, max_results: int = 5) -> str:
            '''Search the web and return results.
            
            Args:
                query: The search query
                max_results: Maximum number of results to return
            '''
            # Implementation
            return results
        
        # 'search' is now an AgenticTool with proper schema
        tools = [search]
        result = await loop.run(goal="...", tools=tools)
    
    Args:
        description: Tool description (uses docstring if not provided)
        category: Tool category for organization
        timeout: Maximum execution time
        requires_confirmation: Whether to ask user before executing
    
    Returns:
        Decorated function wrapped as AgenticTool
    """
    def decorator(func: Callable) -> AgenticTool:
        import inspect
        
        # Get function name
        tool_name = func.__name__
        
        # Get description from docstring if not provided
        tool_description = description
        if not tool_description:
            tool_description = func.__doc__.split('\n')[0] if func.__doc__ else f"Execute {tool_name}"
        
        # Infer parameters from type hints
        sig = inspect.signature(func)
        hints = func.__annotations__ if hasattr(func, '__annotations__') else {}
        
        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'cls'):
                continue
            
            # Get type
            param_type = hints.get(param_name, str)
            type_name = _python_type_to_json_type(param_type)
            
            # Get description from docstring
            param_desc = _extract_param_description(func.__doc__, param_name) if func.__doc__ else ""
            
            # Check if required (no default value)
            is_required = param.default == inspect.Parameter.empty
            
            parameters[param_name] = {
                "type": type_name,
                "description": param_desc or f"Parameter: {param_name}",
                "required": is_required
            }
            
            # Add default if present
            if param.default != inspect.Parameter.empty:
                parameters[param_name]["default"] = param.default
        
        # Create the AgenticTool
        tool = AgenticTool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=func,
            category=category,
            timeout=timeout,
            requires_confirmation=requires_confirmation
        )
        
        return tool
    
    return decorator


def _python_type_to_json_type(python_type) -> str:
    """Convert Python type hint to JSON schema type."""
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null"
    }
    
    # Handle Optional, Union, etc.
    origin = getattr(python_type, '__origin__', None)
    if origin is not None:
        # Handle List[X], Dict[X, Y], etc.
        if origin is list:
            return "array"
        elif origin is dict:
            return "object"
        # Handle Optional[X] (Union[X, None])
        elif origin is Union:
            args = getattr(python_type, '__args__', ())
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return _python_type_to_json_type(non_none[0])
    
    return type_mapping.get(python_type, "string")


def _extract_param_description(docstring: str, param_name: str) -> str:
    """Extract parameter description from docstring."""
    if not docstring:
        return ""
    
    # Look for Args: section
    lines = docstring.split('\n')
    in_args = False
    
    for line in lines:
        stripped = line.strip()
        
        if stripped.lower().startswith('args:'):
            in_args = True
            continue
        
        if in_args:
            # Check if this line describes our parameter
            if stripped.startswith(f'{param_name}:'):
                return stripped.split(':', 1)[1].strip()
            elif stripped.startswith(f'{param_name} '):
                # Handle "param (type): description" format
                if ':' in stripped:
                    return stripped.split(':', 1)[1].strip()
            
            # Check if we've moved to a new section
            if stripped and not stripped.startswith(' ') and ':' in stripped and not stripped.startswith(param_name):
                if stripped.lower().startswith(('returns:', 'raises:', 'yields:', 'examples:')):
                    break
    
    return ""


# Import Union for type checking
from typing import Union
