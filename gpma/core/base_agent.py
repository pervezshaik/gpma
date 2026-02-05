"""
Base Agent Module

This is the foundation of the multi-agent system. Every agent inherits from BaseAgent.

KEY CONCEPTS:
1. Agent State - Tracks what the agent is doing (idle, working, waiting, error)
2. Capabilities - Declares what the agent can do (used by orchestrator for routing)
3. Tools - Functions the agent can execute
4. Memory - Stores context and results
5. Lifecycle - Initialize -> Process -> Respond pattern

LEARNING POINTS:
- Agents are autonomous units that receive tasks and produce results
- They communicate through messages, not direct method calls
- Each agent has a specific purpose (Single Responsibility Principle)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
import uuid
import asyncio
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """
    Represents the current state of an agent.

    State machine:
    IDLE -> PROCESSING -> IDLE (success)
                      -> ERROR (failure)
                      -> WAITING (needs input)
    """
    IDLE = auto()        # Ready to accept tasks
    PROCESSING = auto()  # Currently working on a task
    WAITING = auto()     # Waiting for external input or another agent
    ERROR = auto()       # Encountered an error
    TERMINATED = auto()  # Agent has been shut down


@dataclass
class AgentCapability:
    """
    Describes what an agent can do.

    The orchestrator uses these to route tasks to appropriate agents.

    Example:
        capability = AgentCapability(
            name="web_fetch",
            description="Fetch and parse web pages",
            keywords=["url", "webpage", "fetch", "browse"]
        )
    """
    name: str
    description: str
    keywords: List[str] = field(default_factory=list)
    priority: int = 1  # Higher = more preferred for matching tasks

    def matches(self, query: str) -> float:
        """
        Returns a score (0-1) indicating how well this capability matches a query.
        Used by the orchestrator to find the best agent for a task.
        """
        query_lower = query.lower()
        score = 0.0

        # Check keyword matches
        for keyword in self.keywords:
            if keyword.lower() in query_lower:
                score += 0.2

        # Check if name matches
        if self.name.lower() in query_lower:
            score += 0.3

        # Cap at 1.0
        return min(score, 1.0)


@dataclass
class Tool:
    """
    A function that an agent can execute.

    Tools are the "hands" of an agent - they perform actual work.

    NOTE: This is the legacy tool class. For new code, use BaseTool from
    gpma.tools instead:

        from gpma.tools import BaseTool, ToolParameter, registry

    Example (legacy):
        tool = Tool(
            name="fetch_url",
            description="Fetches content from a URL",
            function=fetch_url_impl,
            parameters={"url": "The URL to fetch"}
        )

    Example (new - recommended):
        from gpma.tools import tool, ToolCategory

        @tool(category=ToolCategory.WEB)
        async def fetch_url(url: str) -> str:
            '''Fetch content from a URL.'''
            return await do_fetch(url)
    """
    name: str
    description: str
    function: Callable
    parameters: Dict[str, str] = field(default_factory=dict)

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        return self.function(**kwargs)

    @classmethod
    def from_base_tool(cls, base_tool: "BaseTool") -> "Tool":
        """
        Create a legacy Tool from a new BaseTool.

        This allows using the new centralized tool system with
        agents that still expect the legacy Tool class.

        Usage:
            from gpma.tools import registry
            from gpma.core.base_agent import Tool

            base_tools = registry.get_all()
            legacy_tools = [Tool.from_base_tool(t) for t in base_tools]
        """
        # Convert parameters
        params = {}
        for param in base_tool.parameters:
            params[param.name] = param.description

        return cls(
            name=base_tool.name,
            description=base_tool.description,
            function=base_tool.function,
            parameters=params
        )


# Type alias for BaseTool to avoid circular import
try:
    from ..tools.base import BaseTool
except ImportError:
    BaseTool = None  # Not available


@dataclass
class TaskResult:
    """
    The result of processing a task.

    Contains both the output and metadata about execution.
    """
    success: bool
    data: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system.

    INHERITANCE PATTERN:
    BaseAgent (abstract)
        ├── WebBrowserAgent
        ├── ResearchAgent
        ├── TaskExecutorAgent
        └── ... your custom agents

    LIFECYCLE:
    1. __init__: Set up the agent
    2. initialize(): Async setup (connections, resources)
    3. process(): Handle incoming tasks (called repeatedly)
    4. shutdown(): Clean up resources

    EXAMPLE IMPLEMENTATION:

    class MyAgent(BaseAgent):
        @property
        def capabilities(self) -> List[AgentCapability]:
            return [AgentCapability("my_task", "Does something cool", ["cool", "task"])]

        async def process(self, task: Dict[str, Any]) -> TaskResult:
            # Do the work
            result = self._do_something(task["input"])
            return TaskResult(success=True, data=result)
    """

    def __init__(self, name: str = None, tools: List["Tool"] = None):
        """
        Initialize the agent.

        Args:
            name: Unique identifier for this agent instance
            tools: Optional list of tools to inject. Can be:
                   - List of legacy Tool objects
                   - List of new BaseTool objects (will be converted)
                   - None to use no tools initially

        Example with tool injection:
            from gpma.tools import registry

            # Get tools from registry
            tools = registry.get_all()

            # Create agent with injected tools
            agent = MyAgent(tools=tools)
        """
        self.id = str(uuid.uuid4())[:8]
        self.name = name or f"{self.__class__.__name__}_{self.id}"
        self.state = AgentState.IDLE
        self._tools: Dict[str, Tool] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._created_at = datetime.now()
        self._task_count = 0
        self._error_count = 0

        # Memory will be injected by the orchestrator
        self.memory = None

        # Message bus reference (set by orchestrator)
        self.message_bus = None

        # Register injected tools
        if tools:
            self.inject_tools(tools)

        logger.info(f"Agent created: {self.name}")

    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """
        Declare what this agent can do.

        This is used by the orchestrator to route tasks.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a task and return the result.

        This is the main work method. Subclasses implement their
        specific logic here.

        Args:
            task: Dictionary containing task details
                  Typically has: {"action": "...", "input": "...", "context": {...}}

        Returns:
            TaskResult with the outcome
        """
        pass

    async def initialize(self) -> None:
        """
        Async initialization hook.

        Override this for setup that requires async operations
        (e.g., connecting to databases, loading models).
        """
        logger.info(f"Agent initialized: {self.name}")

    async def shutdown(self) -> None:
        """
        Clean up resources before the agent is destroyed.

        Override this to close connections, save state, etc.
        """
        self.state = AgentState.TERMINATED
        logger.info(f"Agent shut down: {self.name}")

    def register_tool(self, tool: Tool) -> None:
        """
        Register a tool that this agent can use.

        Tools are registered during initialization and can be
        invoked during task processing.
        """
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name} on {self.name}")

    def inject_tools(self, tools: List[Any]) -> None:
        """
        Inject multiple tools into this agent.

        Supports both legacy Tool objects and new BaseTool objects.
        BaseTool objects are automatically converted to legacy Tool format.

        Args:
            tools: List of Tool or BaseTool objects

        Example:
            from gpma.tools import registry

            # Inject all tools from registry
            agent.inject_tools(registry.get_all())

            # Or inject specific tools
            agent.inject_tools([
                registry.get("calculator"),
                registry.get("web_search")
            ])
        """
        for t in tools:
            # Check if it's a new BaseTool (has 'parameters' as list)
            if hasattr(t, 'parameters') and isinstance(t.parameters, list):
                # Convert BaseTool to legacy Tool
                legacy_tool = Tool.from_base_tool(t)
                self.register_tool(legacy_tool)
            else:
                # It's already a legacy Tool
                self.register_tool(t)

    def inject_tools_from_registry(
        self,
        tool_names: List[str] = None,
        category: str = None
    ) -> None:
        """
        Inject tools directly from the global registry.

        Args:
            tool_names: Optional list of specific tool names to inject
            category: Optional category to filter tools

        Example:
            # Inject specific tools
            agent.inject_tools_from_registry(["calculator", "web_search"])

            # Inject all tools in a category
            agent.inject_tools_from_registry(category="web")

            # Inject all tools
            agent.inject_tools_from_registry()
        """
        try:
            from ..tools import registry, ToolCategory

            if tool_names:
                tools = [registry.get(name) for name in tool_names]
                tools = [t for t in tools if t is not None]
            elif category:
                cat = ToolCategory(category) if isinstance(category, str) else category
                tools = registry.get_by_category(cat)
            else:
                tools = registry.get_all()

            self.inject_tools(tools)
            logger.info(f"Injected {len(tools)} tools into {self.name}")

        except ImportError as e:
            logger.warning(f"Could not import tools registry: {e}")

    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """
        Execute a registered tool.

        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool

        Returns:
            The tool's output

        Raises:
            ValueError: If tool not found
        """
        if tool_name not in self._tools:
            raise ValueError(f"Tool not found: {tool_name}")

        tool = self._tools[tool_name]
        return await tool.execute(**kwargs)

    async def run_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Main entry point for task execution.

        This wraps the process() method with state management,
        error handling, and metrics collection.

        DO NOT OVERRIDE THIS - override process() instead.
        """
        import time
        start_time = time.time()

        self.state = AgentState.PROCESSING
        self._task_count += 1

        try:
            result = await self.process(task)
            result.execution_time = time.time() - start_time
            self.state = AgentState.IDLE

            # Store result in memory if available
            if self.memory:
                self.memory.store(f"task_{self._task_count}", {
                    "task": task,
                    "result": result.data,
                    "timestamp": datetime.now().isoformat()
                })

            return result

        except Exception as e:
            self._error_count += 1
            self.state = AgentState.ERROR
            logger.error(f"Agent {self.name} error: {str(e)}")

            return TaskResult(
                success=False,
                data=None,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def send_message(self, to_agent: str, content: Any, msg_type: str = "request") -> None:
        """
        Send a message to another agent via the message bus.

        Args:
            to_agent: Name of the target agent
            content: Message payload
            msg_type: Type of message (request, response, broadcast)
        """
        if self.message_bus:
            from .message_bus import Message, MessageType

            msg = Message(
                sender=self.name,
                receiver=to_agent,
                content=content,
                msg_type=MessageType[msg_type.upper()]
            )
            await self.message_bus.publish(msg)

    async def receive_message(self, timeout: float = None) -> Optional[Any]:
        """
        Receive a message from the queue.

        Args:
            timeout: How long to wait (None = wait forever)

        Returns:
            The message content or None if timeout
        """
        try:
            if timeout:
                return await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=timeout
                )
            return await self._message_queue.get()
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics for monitoring.
        """
        return {
            "name": self.name,
            "state": self.state.name,
            "tasks_processed": self._task_count,
            "errors": self._error_count,
            "uptime_seconds": (datetime.now() - self._created_at).total_seconds(),
            "tools": list(self._tools.keys()),
            "capabilities": [c.name for c in self.capabilities]
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} state={self.state.name}>"


# ============================================================================
# EXAMPLE: Simple Echo Agent (for learning)
# ============================================================================

class EchoAgent(BaseAgent):
    """
    A simple agent that echoes back its input.

    This is a minimal example to understand the agent pattern.
    """

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="echo",
                description="Echoes back the input",
                keywords=["echo", "repeat", "test"]
            )
        ]

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        input_data = task.get("input", "")
        return TaskResult(
            success=True,
            data=f"Echo: {input_data}",
            metadata={"original_input": input_data}
        )
