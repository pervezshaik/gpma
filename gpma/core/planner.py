"""
Intelligent Planning Module

This module replaces simple regex-based task decomposition with
LLM-powered intelligent planning.

THE PLANNING PROBLEM:
Given a complex goal, how do we break it down into achievable steps
while respecting dependencies and constraints?

PLANNING APPROACH:
1. Goal Analysis - Understand what the user wants to achieve
2. Decomposition - Break into subtasks
3. Dependency Analysis - Determine which tasks depend on others
4. Resource Mapping - Match tasks to available agents/tools
5. Optimization - Order tasks for efficiency
6. Fallback Planning - What to do if steps fail

PLAN EXECUTION:
- Sequential: A → B → C (dependencies)
- Parallel: A, B, C simultaneously (independent)
- Pipeline: Output of A feeds into B
- Conditional: If A fails, try B

USAGE:
    from gpma.core.planner import TaskPlanner, PlanExecutor

    planner = TaskPlanner(llm_provider)

    # Create a plan
    plan = await planner.plan(
        goal="Research AI trends, analyze findings, create report",
        available_agents=[web_agent, analysis_agent, writer_agent]
    )

    print(plan.steps)
    print(plan.dependency_graph)

    # Execute the plan
    executor = PlanExecutor()
    result = await executor.execute(plan, agents)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from enum import Enum, auto
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class TaskComplexity(Enum):
    """Complexity estimation for tasks."""
    TRIVIAL = "trivial"      # < 1 min
    SIMPLE = "simple"        # 1-5 min
    MODERATE = "moderate"    # 5-15 min
    COMPLEX = "complex"      # 15-60 min
    VERY_COMPLEX = "very_complex"  # > 60 min


class ExecutionStrategy(Enum):
    """How to execute a group of tasks."""
    SEQUENTIAL = "sequential"  # One after another
    PARALLEL = "parallel"      # All at once
    PIPELINE = "pipeline"      # Output feeds to next
    CONDITIONAL = "conditional"  # Based on conditions


@dataclass
class PlannedTask:
    """
    A single task in the execution plan.

    This represents one atomic unit of work that can be
    assigned to an agent.
    """
    id: str
    description: str
    action: str  # The action type (e.g., "search", "analyze", "write")
    dependencies: List[str] = field(default_factory=list)  # IDs of tasks that must complete first
    agent_hint: Optional[str] = None  # Suggested agent type
    priority: TaskPriority = TaskPriority.MEDIUM
    complexity: TaskComplexity = TaskComplexity.MODERATE
    estimated_time_seconds: int = 60
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    fallback_action: Optional[str] = None

    # Execution state (filled during execution)
    status: str = "pending"  # pending, running, completed, failed, skipped
    result: Any = None
    error: Optional[str] = None
    actual_time_seconds: float = 0.0
    assigned_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "action": self.action,
            "dependencies": self.dependencies,
            "agent_hint": self.agent_hint,
            "priority": self.priority.name,
            "complexity": self.complexity.value,
            "estimated_time_seconds": self.estimated_time_seconds,
            "parameters": self.parameters,
            "success_criteria": self.success_criteria,
            "fallback_action": self.fallback_action,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlannedTask":
        return cls(
            id=data["id"],
            description=data["description"],
            action=data["action"],
            dependencies=data.get("dependencies", []),
            agent_hint=data.get("agent_hint"),
            priority=TaskPriority[data.get("priority", "MEDIUM")],
            complexity=TaskComplexity(data.get("complexity", "moderate")),
            estimated_time_seconds=data.get("estimated_time_seconds", 60),
            parameters=data.get("parameters", {}),
            success_criteria=data.get("success_criteria", []),
            fallback_action=data.get("fallback_action")
        )


@dataclass
class ExecutionPlan:
    """
    A complete execution plan for achieving a goal.

    Contains:
    - Ordered list of tasks
    - Dependency graph
    - Execution strategy
    - Fallback strategies
    - Time/resource estimates
    """
    id: str
    goal: str
    tasks: List[PlannedTask]
    execution_order: List[List[str]]  # Groups of task IDs that can run together
    strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL
    estimated_total_time: int = 0  # seconds
    created_at: datetime = field(default_factory=datetime.now)

    # Fallback plans for critical failures
    fallback_strategies: Dict[str, str] = field(default_factory=dict)

    # Metadata
    reasoning: str = ""  # Why this plan was chosen
    alternatives_considered: List[str] = field(default_factory=list)
    risk_assessment: str = ""

    def get_task(self, task_id: str) -> Optional[PlannedTask]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> List[PlannedTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        ready = []

        for task in self.tasks:
            if task.status != "pending":
                continue

            # Check if all dependencies are completed
            deps_satisfied = all(dep in completed_ids for dep in task.dependencies)
            if deps_satisfied:
                ready.append(task)

        return ready

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph as adjacency list."""
        return {task.id: task.dependencies for task in self.tasks}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "tasks": [t.to_dict() for t in self.tasks],
            "execution_order": self.execution_order,
            "strategy": self.strategy.value,
            "estimated_total_time": self.estimated_total_time,
            "reasoning": self.reasoning,
            "risk_assessment": self.risk_assessment
        }

    def __str__(self) -> str:
        lines = [f"Plan: {self.goal}", f"Strategy: {self.strategy.value}", "Tasks:"]
        for task in self.tasks:
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            lines.append(f"  [{task.id}] {task.description}{deps}")
        return "\n".join(lines)


class DependencyGraph:
    """
    Manages task dependencies and execution ordering.

    Provides:
    - Topological sorting for execution order
    - Cycle detection
    - Parallel execution grouping
    """

    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Dict[str, Set[str]] = {}  # task_id -> set of dependencies

    def add_task(self, task_id: str, dependencies: List[str] = None):
        """Add a task with its dependencies."""
        self.nodes.add(task_id)
        self.edges[task_id] = set(dependencies or [])

        # Ensure all dependencies are in the graph
        for dep in (dependencies or []):
            self.nodes.add(dep)
            if dep not in self.edges:
                self.edges[dep] = set()

    def has_cycle(self) -> bool:
        """Check if the graph has any cycles (which would be invalid)."""
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in self.edges.get(node, set()):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in self.nodes:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def topological_sort(self) -> List[str]:
        """
        Return tasks in topological order (respecting dependencies).

        Tasks with no dependencies come first.
        """
        if self.has_cycle():
            raise ValueError("Cannot sort: dependency graph has cycles")

        in_degree = {node: 0 for node in self.nodes}
        for node in self.nodes:
            for dep in self.edges.get(node, set()):
                # dep must come before node, so node has an incoming edge
                pass

        # Calculate in-degrees (how many tasks depend on this one)
        reverse_edges: Dict[str, Set[str]] = {node: set() for node in self.nodes}
        for node, deps in self.edges.items():
            for dep in deps:
                reverse_edges[dep].add(node)

        in_degree = {node: len(self.edges.get(node, set())) for node in self.nodes}

        # Start with nodes that have no dependencies
        queue = [node for node in self.nodes if in_degree[node] == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree of dependent nodes
            for dependent in reverse_edges.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        return result

    def get_parallel_groups(self) -> List[List[str]]:
        """
        Group tasks that can be executed in parallel.

        Returns groups where all tasks in a group can run simultaneously.
        """
        if self.has_cycle():
            raise ValueError("Cannot group: dependency graph has cycles")

        groups = []
        remaining = set(self.nodes)
        completed = set()

        while remaining:
            # Find all tasks whose dependencies are all completed
            ready = []
            for node in remaining:
                deps = self.edges.get(node, set())
                if deps.issubset(completed):
                    ready.append(node)

            if not ready:
                # This shouldn't happen if no cycles
                break

            groups.append(ready)
            completed.update(ready)
            remaining -= set(ready)

        return groups


class TaskPlanner:
    """
    LLM-powered intelligent task planner.

    This replaces simple regex decomposition with intelligent planning
    that considers:
    - Goal semantics
    - Task dependencies
    - Available resources
    - Risk and fallbacks
    """

    def __init__(
        self,
        llm_provider,
        enable_optimization: bool = True,
        max_tasks: int = 20
    ):
        """
        Initialize the planner.

        Args:
            llm_provider: LLM for planning intelligence
            enable_optimization: Optimize task ordering
            max_tasks: Maximum tasks in a plan
        """
        self.llm = llm_provider
        self.enable_optimization = enable_optimization
        self.max_tasks = max_tasks

    async def plan(
        self,
        goal: str,
        available_agents: List[Any] = None,
        context: Dict[str, Any] = None,
        constraints: List[str] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for a goal.

        Args:
            goal: The goal to achieve
            available_agents: List of available agents
            context: Additional context for planning
            constraints: Constraints to respect

        Returns:
            ExecutionPlan ready for execution
        """
        import uuid

        # Step 1: Analyze the goal
        goal_analysis = await self._analyze_goal(goal, context)

        # Step 2: Decompose into tasks
        raw_tasks = await self._decompose_goal(goal, goal_analysis, constraints)

        # Step 3: Determine dependencies
        tasks_with_deps = await self._analyze_dependencies(raw_tasks)

        # Step 4: Map to agents (if available)
        if available_agents:
            tasks_with_deps = self._map_to_agents(tasks_with_deps, available_agents)

        # Step 5: Build dependency graph and execution order
        graph = DependencyGraph()
        for task in tasks_with_deps:
            graph.add_task(task.id, task.dependencies)

        # Check for cycles
        if graph.has_cycle():
            logger.warning("Dependency cycle detected, removing problematic edges")
            tasks_with_deps = self._remove_cycles(tasks_with_deps)
            graph = DependencyGraph()
            for task in tasks_with_deps:
                graph.add_task(task.id, task.dependencies)

        # Get parallel execution groups
        execution_order = graph.get_parallel_groups()

        # Step 6: Determine execution strategy
        strategy = self._determine_strategy(tasks_with_deps, execution_order)

        # Step 7: Estimate time
        total_time = self._estimate_total_time(tasks_with_deps, strategy, execution_order)

        # Step 8: Generate fallback strategies
        fallbacks = await self._generate_fallbacks(tasks_with_deps, goal)

        # Create the plan
        plan = ExecutionPlan(
            id=str(uuid.uuid4())[:8],
            goal=goal,
            tasks=tasks_with_deps,
            execution_order=execution_order,
            strategy=strategy,
            estimated_total_time=total_time,
            fallback_strategies=fallbacks,
            reasoning=goal_analysis.get("reasoning", ""),
            risk_assessment=goal_analysis.get("risks", "")
        )

        logger.info(f"Created plan with {len(tasks_with_deps)} tasks, strategy: {strategy.value}")
        return plan

    async def _analyze_goal(
        self,
        goal: str,
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze the goal to understand what needs to be done."""
        from ..llm.providers import Message, MessageRole, LLMConfig

        prompt = f"""Analyze this goal and provide insights:

GOAL: {goal}

{f"CONTEXT: {json.dumps(context)}" if context else ""}

Provide analysis as JSON:
{{
    "main_objective": "the core thing to achieve",
    "sub_objectives": ["list of sub-objectives"],
    "required_capabilities": ["web_search", "analysis", "writing", etc.],
    "potential_challenges": ["list of challenges"],
    "success_criteria": ["how to know if goal is achieved"],
    "reasoning": "your analysis of this goal",
    "risks": "potential risks and how to mitigate them"
}}
"""

        messages = [
            Message(MessageRole.SYSTEM, "You are a planning assistant that analyzes goals."),
            Message(MessageRole.USER, prompt)
        ]

        config = LLMConfig(temperature=0.3, max_tokens=1024)
        response = await self.llm.chat(messages, config)

        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {
            "main_objective": goal,
            "sub_objectives": [],
            "required_capabilities": [],
            "reasoning": response.content
        }

    async def _decompose_goal(
        self,
        goal: str,
        analysis: Dict[str, Any],
        constraints: Optional[List[str]]
    ) -> List[PlannedTask]:
        """Decompose goal into executable tasks."""
        from ..llm.providers import Message, MessageRole, LLMConfig

        prompt = f"""Break down this goal into specific, executable tasks:

GOAL: {goal}

ANALYSIS:
- Main objective: {analysis.get('main_objective', goal)}
- Sub-objectives: {analysis.get('sub_objectives', [])}
- Required capabilities: {analysis.get('required_capabilities', [])}

{f"CONSTRAINTS: {constraints}" if constraints else ""}

Create a list of tasks as JSON array. Each task should be atomic and executable:
[
    {{
        "id": "task_1",
        "description": "Clear description of what to do",
        "action": "search|analyze|write|fetch|compute|etc",
        "agent_hint": "web_browser|research|task_executor|llm|null",
        "priority": "CRITICAL|HIGH|MEDIUM|LOW",
        "complexity": "trivial|simple|moderate|complex|very_complex",
        "estimated_time_seconds": 60,
        "parameters": {{"key": "value"}},
        "success_criteria": ["criterion 1", "criterion 2"]
    }}
]

Rules:
1. Tasks should be specific and actionable
2. Each task should do ONE thing
3. Order tasks logically (but don't specify dependencies yet)
4. Maximum {self.max_tasks} tasks
5. Include all necessary steps to achieve the goal
"""

        messages = [
            Message(MessageRole.SYSTEM, "You are a task decomposition expert."),
            Message(MessageRole.USER, prompt)
        ]

        config = LLMConfig(temperature=0.3, max_tokens=2048)
        response = await self.llm.chat(messages, config)

        tasks = self._parse_tasks(response.content)

        # Ensure we have at least one task
        if not tasks:
            tasks = [PlannedTask(
                id="task_1",
                description=f"Complete: {goal}",
                action="execute",
                complexity=TaskComplexity.MODERATE
            )]

        return tasks[:self.max_tasks]

    def _parse_tasks(self, llm_response: str) -> List[PlannedTask]:
        """Parse LLM response into PlannedTask objects."""
        import re

        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', llm_response)
            if json_match:
                tasks_data = json.loads(json_match.group())
                return [PlannedTask.from_dict(t) for t in tasks_data]
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to parse tasks JSON: {e}")

        # Fallback: try to extract tasks from text
        tasks = []
        lines = llm_response.split('\n')
        task_num = 0

        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                task_num += 1
                # Extract task description
                desc = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if desc:
                    tasks.append(PlannedTask(
                        id=f"task_{task_num}",
                        description=desc,
                        action="execute"
                    ))

        return tasks

    async def _analyze_dependencies(
        self,
        tasks: List[PlannedTask]
    ) -> List[PlannedTask]:
        """Determine dependencies between tasks."""
        from ..llm.providers import Message, MessageRole, LLMConfig

        if len(tasks) <= 1:
            return tasks

        task_descriptions = "\n".join([
            f"- {t.id}: {t.description}"
            for t in tasks
        ])

        prompt = f"""Analyze these tasks and determine their dependencies:

TASKS:
{task_descriptions}

For each task, determine which other tasks must complete BEFORE it can start.
Only include direct dependencies (not transitive).

Respond with JSON mapping task_id to list of dependency task_ids:
{{
    "task_1": [],
    "task_2": ["task_1"],
    "task_3": ["task_1", "task_2"]
}}

Rules:
1. A task depends on another if it needs that task's output
2. Only include necessary dependencies
3. Avoid circular dependencies
4. Tasks with no dependencies have empty arrays
"""

        messages = [
            Message(MessageRole.SYSTEM, "You are a dependency analysis expert."),
            Message(MessageRole.USER, prompt)
        ]

        config = LLMConfig(temperature=0.2, max_tokens=1024)
        response = await self.llm.chat(messages, config)

        # Parse dependencies
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                deps_map = json.loads(json_match.group())

                # Update tasks with dependencies
                task_ids = {t.id for t in tasks}
                for task in tasks:
                    if task.id in deps_map:
                        # Filter to only valid task IDs
                        valid_deps = [d for d in deps_map[task.id] if d in task_ids and d != task.id]
                        task.dependencies = valid_deps

        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse dependencies: {e}")
            # Default: sequential dependencies
            for i, task in enumerate(tasks[1:], 1):
                task.dependencies = [tasks[i-1].id]

        return tasks

    def _map_to_agents(
        self,
        tasks: List[PlannedTask],
        available_agents: List[Any]
    ) -> List[PlannedTask]:
        """Map tasks to the most suitable agents."""
        # Build capability map
        agent_capabilities = {}
        for agent in available_agents:
            if hasattr(agent, 'capabilities'):
                for cap in agent.capabilities:
                    if cap.name not in agent_capabilities:
                        agent_capabilities[cap.name] = []
                    agent_capabilities[cap.name].append(agent.name)

        # Map tasks to agents based on hints and capabilities
        for task in tasks:
            if task.agent_hint:
                # Check if hint matches a capability
                for cap_name, agents in agent_capabilities.items():
                    if task.agent_hint.lower() in cap_name.lower():
                        task.assigned_agent = agents[0]
                        break

            # If no match, try to match action to capability
            if not task.assigned_agent:
                for cap_name, agents in agent_capabilities.items():
                    if task.action.lower() in cap_name.lower():
                        task.assigned_agent = agents[0]
                        break

        return tasks

    def _remove_cycles(self, tasks: List[PlannedTask]) -> List[PlannedTask]:
        """Remove dependency cycles by breaking weakest links."""
        # Simple approach: keep dependencies only to earlier tasks
        task_order = {t.id: i for i, t in enumerate(tasks)}

        for task in tasks:
            valid_deps = [
                dep for dep in task.dependencies
                if dep in task_order and task_order[dep] < task_order[task.id]
            ]
            task.dependencies = valid_deps

        return tasks

    def _determine_strategy(
        self,
        tasks: List[PlannedTask],
        execution_order: List[List[str]]
    ) -> ExecutionStrategy:
        """Determine the best execution strategy."""
        # If all groups have single tasks, use sequential
        if all(len(group) == 1 for group in execution_order):
            return ExecutionStrategy.SEQUENTIAL

        # If first group has multiple tasks, parallel is beneficial
        if execution_order and len(execution_order[0]) > 1:
            return ExecutionStrategy.PARALLEL

        # Default to sequential for safety
        return ExecutionStrategy.SEQUENTIAL

    def _estimate_total_time(
        self,
        tasks: List[PlannedTask],
        strategy: ExecutionStrategy,
        execution_order: List[List[str]]
    ) -> int:
        """Estimate total execution time."""
        task_times = {t.id: t.estimated_time_seconds for t in tasks}

        if strategy == ExecutionStrategy.PARALLEL:
            # Time is max of each parallel group
            total = 0
            for group in execution_order:
                group_time = max(task_times.get(tid, 60) for tid in group)
                total += group_time
            return total
        else:
            # Sequential: sum of all tasks
            return sum(t.estimated_time_seconds for t in tasks)

    async def _generate_fallbacks(
        self,
        tasks: List[PlannedTask],
        goal: str
    ) -> Dict[str, str]:
        """Generate fallback strategies for critical tasks."""
        fallbacks = {}

        critical_tasks = [t for t in tasks if t.priority == TaskPriority.CRITICAL]

        for task in critical_tasks:
            fallbacks[task.id] = f"If '{task.description}' fails: " \
                f"1) Retry with different parameters, " \
                f"2) Try alternative approach, " \
                f"3) Skip and note in final output"

        return fallbacks


class PlanExecutor:
    """
    Executes an ExecutionPlan.

    Handles:
    - Task execution based on strategy
    - Dependency management
    - Error handling and fallbacks
    - Progress tracking
    """

    def __init__(
        self,
        max_retries: int = 2,
        retry_delay: float = 1.0
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._stop_requested = False

    async def execute(
        self,
        plan: ExecutionPlan,
        agents: Dict[str, Any],
        on_progress: Optional[Callable[[PlannedTask, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Execute a plan.

        Args:
            plan: The plan to execute
            agents: Dict mapping agent names to agent instances
            on_progress: Callback for progress updates

        Returns:
            Dict with execution results
        """
        import time
        start_time = time.time()
        self._stop_requested = False

        results = {}
        failed_tasks = []

        logger.info(f"Executing plan: {plan.goal}")

        for group_idx, group in enumerate(plan.execution_order):
            if self._stop_requested:
                break

            logger.info(f"Executing group {group_idx + 1}/{len(plan.execution_order)}: {group}")

            if plan.strategy == ExecutionStrategy.PARALLEL and len(group) > 1:
                # Execute group in parallel
                group_results = await self._execute_parallel(
                    [plan.get_task(tid) for tid in group],
                    agents,
                    results,
                    on_progress
                )
            else:
                # Execute sequentially
                group_results = await self._execute_sequential(
                    [plan.get_task(tid) for tid in group],
                    agents,
                    results,
                    on_progress
                )

            results.update(group_results)

            # Check for failures
            for tid, result in group_results.items():
                task = plan.get_task(tid)
                if task and not result.get("success"):
                    failed_tasks.append(tid)

                    # Try fallback if available
                    if tid in plan.fallback_strategies:
                        logger.info(f"Task {tid} failed, trying fallback")
                        # Could implement fallback execution here

        total_time = time.time() - start_time

        return {
            "success": len(failed_tasks) == 0,
            "results": results,
            "failed_tasks": failed_tasks,
            "total_time": total_time,
            "tasks_completed": len(results),
            "tasks_failed": len(failed_tasks)
        }

    async def _execute_sequential(
        self,
        tasks: List[PlannedTask],
        agents: Dict[str, Any],
        previous_results: Dict[str, Any],
        on_progress: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute tasks sequentially."""
        results = {}

        for task in tasks:
            if self._stop_requested or task is None:
                break

            if on_progress:
                on_progress(task, "starting")

            result = await self._execute_task(task, agents, previous_results)
            results[task.id] = result

            if on_progress:
                on_progress(task, "completed" if result.get("success") else "failed")

            # Update previous results for next task
            previous_results[task.id] = result

        return results

    async def _execute_parallel(
        self,
        tasks: List[PlannedTask],
        agents: Dict[str, Any],
        previous_results: Dict[str, Any],
        on_progress: Optional[Callable]
    ) -> Dict[str, Any]:
        """Execute tasks in parallel."""
        async def execute_one(task: PlannedTask) -> Tuple[str, Dict[str, Any]]:
            if on_progress:
                on_progress(task, "starting")

            result = await self._execute_task(task, agents, previous_results)

            if on_progress:
                on_progress(task, "completed" if result.get("success") else "failed")

            return task.id, result

        # Execute all in parallel
        coroutines = [execute_one(t) for t in tasks if t is not None]
        completed = await asyncio.gather(*coroutines, return_exceptions=True)

        results = {}
        for item in completed:
            if isinstance(item, Exception):
                logger.error(f"Parallel execution error: {item}")
            else:
                task_id, result = item
                results[task_id] = result

        return results

    async def _execute_task(
        self,
        task: PlannedTask,
        agents: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single task."""
        import time
        start_time = time.time()

        task.status = "running"

        # Find the appropriate agent
        agent = None
        if task.assigned_agent and task.assigned_agent in agents:
            agent = agents[task.assigned_agent]
        elif task.agent_hint:
            # Try to find agent matching hint
            for name, ag in agents.items():
                if task.agent_hint.lower() in name.lower():
                    agent = ag
                    break
        else:
            # Use first available agent
            if agents:
                agent = list(agents.values())[0]

        if not agent:
            task.status = "failed"
            task.error = "No suitable agent found"
            return {
                "success": False,
                "error": "No suitable agent found",
                "task_id": task.id
            }

        # Execute with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                # Build task input
                task_input = {
                    "action": task.action,
                    "input": task.description,
                    "parameters": task.parameters,
                    "context": context
                }

                # Run the task
                result = await agent.run_task(task_input)

                if result.success:
                    task.status = "completed"
                    task.result = result.data
                    task.actual_time_seconds = time.time() - start_time

                    return {
                        "success": True,
                        "data": result.data,
                        "task_id": task.id,
                        "execution_time": task.actual_time_seconds
                    }
                else:
                    last_error = result.error

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Task {task.id} attempt {attempt + 1} failed: {e}")

            # Retry delay
            if attempt < self.max_retries:
                await asyncio.sleep(self.retry_delay)

        # All retries failed
        task.status = "failed"
        task.error = last_error
        task.actual_time_seconds = time.time() - start_time

        return {
            "success": False,
            "error": last_error,
            "task_id": task.id,
            "execution_time": task.actual_time_seconds
        }

    def stop(self):
        """Request execution to stop."""
        self._stop_requested = True


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_plan(
    goal: str,
    llm_provider,
    available_agents: List[Any] = None
) -> ExecutionPlan:
    """
    Convenience function to create a plan.

    Usage:
        plan = await create_plan(
            "Research and summarize AI trends",
            llm_provider,
            [web_agent, writer_agent]
        )
    """
    planner = TaskPlanner(llm_provider)
    return await planner.plan(goal, available_agents)


async def execute_plan(
    plan: ExecutionPlan,
    agents: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to execute a plan.

    Usage:
        result = await execute_plan(plan, {"web": web_agent, "writer": writer_agent})
    """
    executor = PlanExecutor()
    return await executor.execute(plan, agents)
