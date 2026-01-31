"""
Orchestrator Module

The central coordinator for the multi-agent system. The orchestrator:
1. Receives user requests
2. Analyzes and breaks down complex tasks
3. Routes tasks to appropriate agents
4. Manages agent lifecycle
5. Aggregates results

KEY CONCEPTS:
1. Task Decomposition - Breaking complex tasks into subtasks
2. Agent Selection - Matching tasks to capable agents
3. Execution Strategies - Parallel vs sequential execution
4. Result Aggregation - Combining outputs from multiple agents

LEARNING POINTS:
- The orchestrator is like a project manager for agents
- It doesn't do work itself, it delegates
- It understands agent capabilities and matches them to tasks
- It handles failures and retries
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type
import asyncio
import logging
import re

from .base_agent import BaseAgent, AgentCapability, TaskResult, AgentState
from .message_bus import MessageBus, Message, MessageType
from .memory import CompositeMemory

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """
    How to execute multiple subtasks.

    SEQUENTIAL - One at a time, in order
    PARALLEL - All at once, combine results
    PIPELINE - Output of one feeds into next
    """
    SEQUENTIAL = auto()
    PARALLEL = auto()
    PIPELINE = auto()


@dataclass
class Task:
    """
    A unit of work to be executed by an agent.

    Tasks can be:
    - Simple: Single action by one agent
    - Complex: Multiple subtasks, potentially by different agents
    """
    id: str
    action: str
    input: Any
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    subtasks: List['Task'] = field(default_factory=list)
    parent_id: Optional[str] = None
    assigned_agent: Optional[str] = None
    status: str = "pending"
    result: Optional[TaskResult] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentRegistration:
    """
    Metadata about a registered agent.
    """
    agent: BaseAgent
    capabilities: List[AgentCapability]
    priority: int = 1
    max_concurrent_tasks: int = 5
    current_tasks: int = 0


class Orchestrator:
    """
    The brain of the multi-agent system.

    RESPONSIBILITIES:
    1. Agent Registry - Keep track of available agents
    2. Task Queue - Manage pending work
    3. Routing - Match tasks to agents
    4. Execution - Run tasks and collect results
    5. Monitoring - Track system health

    USAGE:
        # Create orchestrator
        orchestrator = Orchestrator()

        # Register agents
        orchestrator.register_agent(WebBrowserAgent())
        orchestrator.register_agent(ResearchAgent())

        # Execute a complex task
        result = await orchestrator.execute(
            "Research the latest AI news and summarize it"
        )

    TASK ROUTING ALGORITHM:
    1. Parse the input to identify required capabilities
    2. Score each agent based on capability match
    3. Select the best agent (or multiple for complex tasks)
    4. Execute and aggregate results
    """

    def __init__(self, memory_path: str = None):
        """
        Initialize the orchestrator.

        Args:
            memory_path: Path for persistent memory storage
        """
        # Agent registry
        self._agents: Dict[str, AgentRegistration] = {}

        # Message bus for inter-agent communication
        self.message_bus = MessageBus()

        # Shared memory system
        self.memory = CompositeMemory(
            stm_capacity=500,
            ltm_path=memory_path
        )

        # Task tracking
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._active_tasks: Dict[str, Task] = {}
        self._completed_tasks: List[Task] = []

        # Execution lock
        self._lock = asyncio.Lock()

        # Custom task decomposer (can be overridden)
        self._decomposer: Optional[Callable] = None

        logger.info("Orchestrator initialized")

    def register_agent(self, agent: BaseAgent, priority: int = 1) -> None:
        """
        Register an agent with the orchestrator.

        After registration, the agent:
        - Can receive tasks
        - Has access to shared memory
        - Is connected to the message bus
        """
        # Set up agent's memory and message bus
        agent.memory = self.memory
        agent.message_bus = self.message_bus

        # Create registration entry
        registration = AgentRegistration(
            agent=agent,
            capabilities=agent.capabilities,
            priority=priority
        )
        self._agents[agent.name] = registration

        # Subscribe agent to message bus
        async def agent_message_handler(msg: Message):
            await agent._message_queue.put(msg)

        self.message_bus.subscribe(agent.name, agent_message_handler)

        logger.info(f"Agent registered: {agent.name} with {len(agent.capabilities)} capabilities")

    def unregister_agent(self, agent_name: str) -> None:
        """Remove an agent from the system."""
        if agent_name in self._agents:
            self.message_bus.unsubscribe(agent_name)
            del self._agents[agent_name]
            logger.info(f"Agent unregistered: {agent_name}")

    async def initialize_agents(self) -> None:
        """Initialize all registered agents."""
        init_tasks = [
            reg.agent.initialize()
            for reg in self._agents.values()
        ]
        await asyncio.gather(*init_tasks)
        logger.info("All agents initialized")

    async def shutdown(self) -> None:
        """Shutdown all agents gracefully."""
        shutdown_tasks = [
            reg.agent.shutdown()
            for reg in self._agents.values()
        ]
        await asyncio.gather(*shutdown_tasks)
        logger.info("All agents shut down")

    def find_best_agent(self, task_description: str) -> Optional[str]:
        """
        Find the best agent for a given task.

        Scoring algorithm:
        1. Check each agent's capabilities
        2. Score based on keyword matches
        3. Factor in agent priority and availability
        4. Return the highest-scoring agent
        """
        best_agent = None
        best_score = 0.0

        for name, reg in self._agents.items():
            # Skip busy agents
            if reg.agent.state != AgentState.IDLE:
                continue

            # Score based on capabilities
            for capability in reg.capabilities:
                score = capability.matches(task_description) * reg.priority

                if score > best_score:
                    best_score = score
                    best_agent = name

        return best_agent if best_score > 0 else None

    def find_capable_agents(self, task_description: str) -> List[str]:
        """
        Find all agents that can handle a task.

        Returns a list of agent names sorted by capability score.
        """
        scored_agents = []

        for name, reg in self._agents.items():
            max_score = 0.0
            for capability in reg.capabilities:
                score = capability.matches(task_description)
                max_score = max(max_score, score)

            if max_score > 0:
                scored_agents.append((name, max_score * reg.priority))

        # Sort by score descending
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored_agents]

    async def execute(
        self,
        request: str,
        context: Dict[str, Any] = None,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    ) -> TaskResult:
        """
        Execute a user request.

        This is the main entry point for the orchestrator.

        Args:
            request: Natural language description of what to do
            context: Additional context (previous results, user info, etc.)
            strategy: How to handle multi-step tasks

        Returns:
            Combined result from all involved agents
        """
        logger.info(f"Executing request: {request[:50]}...")

        # Store the request in memory
        self.memory.store("last_request", {
            "text": request,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })

        # Create the root task
        import uuid
        task = Task(
            id=str(uuid.uuid4())[:8],
            action="execute",
            input=request,
            context=context or {}
        )

        # Decompose if complex
        subtasks = await self._decompose_task(task)

        if subtasks:
            # Execute subtasks according to strategy
            return await self._execute_subtasks(subtasks, strategy)
        else:
            # Simple task - route to single agent
            return await self._execute_single_task(task)

    async def _decompose_task(self, task: Task) -> List[Task]:
        """
        Break a complex task into subtasks.

        This is a simple rule-based decomposition.
        Advanced implementations might use LLMs for this.
        """
        request = task.input.lower()
        subtasks = []

        # Pattern matching for common multi-step requests
        patterns = [
            # "do X and Y" -> two subtasks
            (r"(.+)\s+and\s+(then\s+)?(.+)", ["$1", "$3"]),
            # "first X, then Y" -> two subtasks
            (r"first\s+(.+),?\s+then\s+(.+)", ["$1", "$2"]),
            # "X, Y, and Z" -> multiple subtasks
            (r"(.+),\s*(.+),\s*and\s+(.+)", ["$1", "$2", "$3"]),
        ]

        for pattern, groups in patterns:
            match = re.match(pattern, request, re.IGNORECASE)
            if match:
                for i, group in enumerate(groups):
                    group_text = match.group(i + 1) if group.startswith("$") else group
                    subtask = Task(
                        id=f"{task.id}_{i}",
                        action="subtask",
                        input=group_text.strip(),
                        context=task.context,
                        parent_id=task.id
                    )
                    subtasks.append(subtask)
                break

        # If custom decomposer is set, use it
        if self._decomposer and not subtasks:
            subtasks = await self._decomposer(task)

        return subtasks

    async def _execute_single_task(self, task: Task) -> TaskResult:
        """Execute a task using a single agent."""
        # Find the best agent
        agent_name = self.find_best_agent(task.input)

        if not agent_name:
            return TaskResult(
                success=False,
                data=None,
                error=f"No agent found capable of: {task.input}"
            )

        # Get the agent
        reg = self._agents[agent_name]
        agent = reg.agent

        logger.info(f"Routing task to agent: {agent_name}")

        # Execute the task
        result = await agent.run_task({
            "action": task.action,
            "input": task.input,
            "context": task.context
        })

        # Store result in memory
        self.memory.store(f"result_{task.id}", {
            "task": task.input,
            "agent": agent_name,
            "result": result.data,
            "success": result.success
        })

        return result

    async def _execute_subtasks(
        self,
        subtasks: List[Task],
        strategy: ExecutionStrategy
    ) -> TaskResult:
        """Execute multiple subtasks according to strategy."""

        if strategy == ExecutionStrategy.PARALLEL:
            # Execute all at once
            results = await asyncio.gather(*[
                self._execute_single_task(task)
                for task in subtasks
            ], return_exceptions=True)

            # Aggregate results
            all_data = []
            errors = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Subtask {i}: {str(result)}")
                elif result.success:
                    all_data.append(result.data)
                else:
                    errors.append(f"Subtask {i}: {result.error}")

            return TaskResult(
                success=len(errors) == 0,
                data=all_data if len(errors) == 0 else None,
                error="; ".join(errors) if errors else None,
                metadata={"subtask_count": len(subtasks)}
            )

        elif strategy == ExecutionStrategy.PIPELINE:
            # Each task feeds into the next
            current_result = None
            for task in subtasks:
                if current_result:
                    task.context["previous_result"] = current_result.data
                current_result = await self._execute_single_task(task)
                if not current_result.success:
                    return current_result
            return current_result

        else:  # SEQUENTIAL
            # Execute one at a time, collect all results
            results = []
            for task in subtasks:
                result = await self._execute_single_task(task)
                results.append(result)
                if not result.success:
                    # Stop on first failure
                    return TaskResult(
                        success=False,
                        data=results,
                        error=f"Failed at subtask: {result.error}"
                    )

            return TaskResult(
                success=True,
                data=[r.data for r in results],
                metadata={"subtask_count": len(subtasks)}
            )

    async def broadcast(self, message: str, msg_type: MessageType = MessageType.BROADCAST) -> None:
        """Send a message to all agents."""
        msg = Message(
            sender="orchestrator",
            receiver="*",
            content=message,
            msg_type=msg_type
        )
        await self.message_bus.publish(msg)

    def set_decomposer(self, decomposer: Callable) -> None:
        """
        Set a custom task decomposition function.

        For advanced use cases, you might want to use an LLM
        to decompose complex tasks intelligently.

        Args:
            decomposer: Async function(Task) -> List[Task]
        """
        self._decomposer = decomposer

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        return {
            name: {
                "state": reg.agent.state.name,
                "capabilities": [c.name for c in reg.capabilities],
                "stats": reg.agent.get_stats()
            }
            for name, reg in self._agents.items()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "agents_registered": len(self._agents),
            "active_tasks": len(self._active_tasks),
            "completed_tasks": len(self._completed_tasks),
            "message_bus": self.message_bus.get_stats(),
            "agents": self.get_agent_status()
        }


# ============================================================================
# DYNAMIC AGENT SPAWNING
# ============================================================================

class DynamicOrchestrator(Orchestrator):
    """
    Extended orchestrator that can dynamically create agents as needed.

    This is useful when you don't know upfront what agents you'll need,
    or when you want to create specialized agents on-the-fly.
    """

    def __init__(self, memory_path: str = None):
        super().__init__(memory_path)
        self._agent_factories: Dict[str, Type[BaseAgent]] = {}

    def register_agent_factory(self, capability_name: str, agent_class: Type[BaseAgent]) -> None:
        """
        Register a factory for creating agents with specific capabilities.

        Args:
            capability_name: The capability this agent type provides
            agent_class: The class to instantiate
        """
        self._agent_factories[capability_name] = agent_class

    async def spawn_agent(self, capability_name: str) -> Optional[BaseAgent]:
        """
        Dynamically create an agent for a specific capability.

        Returns:
            The new agent, or None if no factory exists
        """
        if capability_name not in self._agent_factories:
            return None

        agent_class = self._agent_factories[capability_name]
        agent = agent_class()

        self.register_agent(agent)
        await agent.initialize()

        logger.info(f"Spawned new agent: {agent.name} for capability: {capability_name}")
        return agent

    async def execute_with_spawn(self, request: str, context: Dict[str, Any] = None) -> TaskResult:
        """
        Execute a request, spawning agents if needed.

        If no existing agent can handle the request, try to spawn one.
        """
        # First, try with existing agents
        agent_name = self.find_best_agent(request)

        if not agent_name:
            # Try to identify needed capability and spawn
            for cap_name in self._agent_factories.keys():
                if cap_name.lower() in request.lower():
                    agent = await self.spawn_agent(cap_name)
                    if agent:
                        agent_name = agent.name
                        break

        if not agent_name:
            return TaskResult(
                success=False,
                data=None,
                error="No agent available and none could be spawned"
            )

        return await super().execute(request, context)
