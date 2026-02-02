"""
Enhanced Agentic Base Agent Module

This module provides an upgraded BaseAgent that incorporates true agentic
capabilities including:
- Autonomous goal pursuit
- ReAct (Reasoning + Acting) loop
- Self-reflection and correction
- Planning and replanning
- Tool reasoning

THE DIFFERENCE:
Original BaseAgent: Receives task → Executes → Returns result
AgenticAgent: Receives goal → Plans → Executes → Reflects → Adapts → Achieves

KEY CAPABILITIES:
1. Goal-Oriented: Pursues goals, not just tasks
2. Reasoning: Uses LLM to reason about what to do
3. Self-Correcting: Evaluates outputs and improves them
4. Adaptive: Replans when obstacles arise
5. Explainable: Provides reasoning traces

USAGE:
    from gpma.core.agentic_agent import AgenticAgent, AgentConfig

    class MySmartAgent(AgenticAgent):
        @property
        def capabilities(self):
            return [AgentCapability("research", "Research topics", ["search", "find"])]

        async def execute_action(self, action, context):
            # Implement action execution
            return {"success": True, "data": "result"}

    agent = MySmartAgent(
        name="ResearchAgent",
        llm_provider=llm_provider,
        config=AgentConfig(enable_reflection=True)
    )

    result = await agent.pursue_goal("Find and summarize AI trends")
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum, auto
from datetime import datetime
import asyncio
import json
import logging

from .base_agent import BaseAgent, AgentCapability, TaskResult, Tool, AgentState
from .agentic_loop import AgenticLoop, AgenticTool, LoopResult, LoopStatus
from .reflection import ReflectionEngine, SuccessCriteria, ReflectionResult
from .goal_manager import GoalManager, Goal, GoalResult, GoalStatus, GoalPriority
from .planner import TaskPlanner, ExecutionPlan, PlanExecutor

logger = logging.getLogger(__name__)


class AgentMode(Enum):
    """Operating mode for the agent."""
    REACTIVE = auto()     # Traditional: respond to tasks
    PROACTIVE = auto()    # Pursue goals autonomously
    COLLABORATIVE = auto() # Work with other agents


@dataclass
class AgentConfig:
    """Configuration for AgenticAgent."""
    # Core settings
    mode: AgentMode = AgentMode.PROACTIVE

    # Agentic features
    enable_planning: bool = True
    enable_reflection: bool = True
    enable_goal_decomposition: bool = True
    enable_self_correction: bool = True

    # Limits
    max_iterations: int = 20
    max_retries: int = 3
    max_reflection_rounds: int = 3
    max_planning_depth: int = 5

    # Quality thresholds
    quality_threshold: float = 0.7
    confidence_threshold: float = 0.6

    # Behavior
    verbose: bool = False
    explain_reasoning: bool = True
    store_reasoning_trace: bool = True


@dataclass
class AgentThought:
    """Represents the agent's reasoning at a point in time."""
    content: str
    action_chosen: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgenticResult:
    """
    Enhanced result from agentic processing.

    Includes not just the output but the reasoning trace,
    reflection, and any corrections made.
    """
    success: bool
    data: Any
    goal_achieved: bool
    reasoning_trace: List[AgentThought]
    reflections: List[ReflectionResult]
    corrections_made: int
    iterations: int
    total_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "goal_achieved": self.goal_achieved,
            "reasoning_steps": len(self.reasoning_trace),
            "reflections": len(self.reflections),
            "corrections_made": self.corrections_made,
            "iterations": self.iterations,
            "total_time": self.total_time,
            "error": self.error
        }


class AgenticAgent(BaseAgent):
    """
    Enhanced base agent with true agentic capabilities.

    This extends BaseAgent with:
    - ReAct loop for autonomous reasoning
    - Goal management and decomposition
    - Self-reflection and correction
    - Planning and adaptive execution

    SUBCLASS REQUIREMENTS:
    Subclasses must implement:
    1. capabilities (property) - What the agent can do
    2. execute_action(action, context) - How to execute actions

    EXAMPLE:
        class WebResearchAgent(AgenticAgent):
            @property
            def capabilities(self):
                return [
                    AgentCapability("web_search", "Search the web", ["search", "find"]),
                    AgentCapability("summarize", "Summarize content", ["summarize"])
                ]

            async def execute_action(self, action, context):
                if action == "search":
                    return await self._do_search(context.get("query"))
                elif action == "summarize":
                    return await self._do_summarize(context.get("text"))

            async def _do_search(self, query):
                # Implementation
                return {"success": True, "data": results}
    """

    def __init__(
        self,
        name: str = None,
        llm_provider=None,
        config: AgentConfig = None
    ):
        """
        Initialize the agentic agent.

        Args:
            name: Agent name/identifier
            llm_provider: LLM for reasoning
            config: Agent configuration
        """
        super().__init__(name)

        self.llm = llm_provider
        self.config = config or AgentConfig()

        # Agentic components
        self._agentic_loop: Optional[AgenticLoop] = None
        self._reflection_engine: Optional[ReflectionEngine] = None
        self._goal_manager: Optional[GoalManager] = None
        self._planner: Optional[TaskPlanner] = None

        # State
        self._reasoning_trace: List[AgentThought] = []
        self._reflections: List[ReflectionResult] = []
        self._current_goal: Optional[Goal] = None

        # Initialize components if LLM provided
        if self.llm:
            self._init_agentic_components()

        logger.info(f"AgenticAgent initialized: {self.name} (mode: {self.config.mode.name})")

    def _init_agentic_components(self):
        """Initialize agentic components."""
        self._agentic_loop = AgenticLoop(
            llm_provider=self.llm,
            enable_reflection=self.config.enable_reflection,
            verbose=self.config.verbose
        )

        if self.config.enable_reflection:
            self._reflection_engine = ReflectionEngine(
                llm_provider=self.llm,
                max_correction_iterations=self.config.max_reflection_rounds
            )

        if self.config.enable_goal_decomposition:
            self._goal_manager = GoalManager(
                llm_provider=self.llm,
                max_depth=self.config.max_planning_depth,
                auto_decompose=True
            )

        if self.config.enable_planning:
            self._planner = TaskPlanner(
                llm_provider=self.llm
            )

    @property
    def capabilities(self) -> List[AgentCapability]:
        """
        Declare agent capabilities.

        Must be overridden by subclasses.
        """
        return [
            AgentCapability(
                name="general",
                description="General agentic capabilities",
                keywords=["help", "assist", "do"]
            )
        ]

    async def execute_action(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a specific action.

        Override this in subclasses to implement actual actions.

        Args:
            action: The action to perform
            context: Context including parameters

        Returns:
            Dict with "success" and "data" keys
        """
        # Default implementation - use tools if available
        if action in self._tools:
            tool = self._tools[action]
            result = await tool.execute(**context)
            return {"success": True, "data": result}

        return {
            "success": False,
            "data": None,
            "error": f"Unknown action: {action}"
        }

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a task (BaseAgent interface).

        This routes to either reactive or proactive processing
        based on configuration.
        """
        action = task.get("action", "")
        input_data = task.get("input", "")

        # If it looks like a goal, use proactive mode
        if self.config.mode == AgentMode.PROACTIVE and self.llm:
            is_goal = self._is_goal_like(input_data)
            if is_goal:
                result = await self.pursue_goal(input_data)
                return TaskResult(
                    success=result.success,
                    data=result.data,
                    metadata={
                        "agentic": True,
                        "iterations": result.iterations,
                        "goal_achieved": result.goal_achieved
                    }
                )

        # Reactive mode - direct execution
        context = task.get("context", {})
        context["input"] = input_data

        result = await self.execute_action(action, context)

        return TaskResult(
            success=result.get("success", False),
            data=result.get("data"),
            error=result.get("error")
        )

    def _is_goal_like(self, text: str) -> bool:
        """Determine if input looks like a goal vs a simple command."""
        text_lower = text.lower()

        # Goal indicators
        goal_words = [
            "research", "find out", "understand", "learn",
            "create", "build", "develop", "implement",
            "analyze", "investigate", "explore", "discover",
            "summarize", "compile", "gather", "collect"
        ]

        return any(word in text_lower for word in goal_words)

    async def pursue_goal(
        self,
        goal_description: str,
        success_criteria: Optional[SuccessCriteria] = None
    ) -> AgenticResult:
        """
        Pursue a goal using full agentic capabilities.

        This is the main entry point for goal-oriented behavior.
        It combines:
        - Goal decomposition
        - ReAct loop execution
        - Self-reflection and correction

        Args:
            goal_description: What to achieve
            success_criteria: Optional criteria for success

        Returns:
            AgenticResult with outcome and reasoning trace
        """
        import time
        start_time = time.time()

        self._reasoning_trace = []
        self._reflections = []

        if not self.llm:
            return AgenticResult(
                success=False,
                data=None,
                goal_achieved=False,
                reasoning_trace=[],
                reflections=[],
                corrections_made=0,
                iterations=0,
                total_time=0,
                error="No LLM provider configured"
            )

        logger.info(f"{self.name} pursuing goal: {goal_description}")

        # Build tools from capabilities
        tools = self._build_agentic_tools()

        # Run the agentic loop
        loop_result = await self._agentic_loop.run(
            goal=goal_description,
            tools=tools,
            max_iterations=self.config.max_iterations
        )

        # Extract reasoning trace from loop result
        for step in loop_result.reasoning_trace:
            self._reasoning_trace.append(AgentThought(
                content=step.get("thought", {}).get("reasoning", ""),
                action_chosen=step.get("action", {}).get("action_type", ""),
                confidence=step.get("thought", {}).get("confidence", 0.0)
            ))

        # Apply reflection if enabled and we have output
        final_output = loop_result.final_answer
        corrections_made = 0

        if self.config.enable_reflection and self._reflection_engine and final_output:
            criteria = success_criteria or SuccessCriteria(
                quality_threshold=self.config.quality_threshold
            )

            reflection_result = await self._reflection_engine.evaluate_and_correct(
                output=str(final_output),
                task_description=goal_description,
                criteria=criteria
            )

            self._reflections.append(reflection_result)

            if reflection_result.correction_applied:
                final_output = reflection_result.corrected_output
                corrections_made = reflection_result.iterations

        total_time = time.time() - start_time

        return AgenticResult(
            success=loop_result.status == LoopStatus.COMPLETED,
            data=final_output,
            goal_achieved=loop_result.achieved,
            reasoning_trace=self._reasoning_trace,
            reflections=self._reflections,
            corrections_made=corrections_made,
            iterations=loop_result.steps_taken,
            total_time=total_time,
            metadata={
                "artifacts": loop_result.artifacts,
                "status": loop_result.status.name
            }
        )

    def _build_agentic_tools(self) -> List[AgenticTool]:
        """Build AgenticTools from registered tools and capabilities."""
        agentic_tools = []

        # Convert registered tools
        for name, tool in self._tools.items():
            agentic_tool = AgenticTool(
                name=name,
                description=tool.description,
                parameters={
                    param: {"type": "string", "description": desc, "required": True}
                    for param, desc in tool.parameters.items()
                },
                function=tool.function
            )
            agentic_tools.append(agentic_tool)

        # Add execute_action as a meta-tool
        async def execute_action_wrapper(**kwargs):
            action = kwargs.pop("action", "")
            return await self.execute_action(action, kwargs)

        action_tool = AgenticTool(
            name="execute_action",
            description="Execute an agent action",
            parameters={
                "action": {
                    "type": "string",
                    "description": "The action to execute",
                    "required": True
                }
            },
            function=execute_action_wrapper
        )
        agentic_tools.append(action_tool)

        return agentic_tools

    async def plan_and_execute(
        self,
        goal: str,
        available_agents: List["AgenticAgent"] = None
    ) -> Dict[str, Any]:
        """
        Plan and execute a goal using the planning system.

        This is useful for complex goals that benefit from
        explicit planning before execution.

        Args:
            goal: The goal to plan and execute
            available_agents: Other agents that can help

        Returns:
            Execution results
        """
        if not self._planner:
            return {
                "success": False,
                "error": "Planning not enabled"
            }

        # Create plan
        agents_list = available_agents or [self]
        plan = await self._planner.plan(goal, agents_list)

        logger.info(f"Created plan with {len(plan.tasks)} tasks")

        # Execute plan
        executor = PlanExecutor()
        agents_dict = {a.name: a for a in agents_list}

        result = await executor.execute(plan, agents_dict)

        return result

    async def reflect_and_improve(
        self,
        output: str,
        task_description: str,
        criteria: Optional[SuccessCriteria] = None
    ) -> ReflectionResult:
        """
        Reflect on an output and potentially improve it.

        Args:
            output: The output to reflect on
            task_description: What the task was
            criteria: Success criteria

        Returns:
            ReflectionResult with evaluation and any corrections
        """
        if not self._reflection_engine:
            # Return a pass-through result if reflection not enabled
            return ReflectionResult(
                passed=True,
                original_output=output,
                evaluation_result=None,
                criteria_checks=[],
                quality_assessment=None,
                issues_found=[],
                correction_applied=False,
                corrected_output=None,
                correction_strategy=None,
                improvement_description="Reflection not enabled",
                iterations=0,
                total_time=0
            )

        return await self._reflection_engine.evaluate_and_correct(
            output=output,
            task_description=task_description,
            criteria=criteria or SuccessCriteria()
        )

    async def reason_about(self, question: str) -> Dict[str, Any]:
        """
        Use the agent's reasoning capabilities to think about a question.

        This doesn't take action, just reasons.

        Args:
            question: The question to reason about

        Returns:
            Dict with reasoning and conclusions
        """
        if not self.llm:
            return {
                "success": False,
                "error": "No LLM for reasoning"
            }

        try:
            from ..llm.providers import Message, MessageRole, LLMConfig

            prompt = f"""Think through this question step by step:

QUESTION: {question}

CAPABILITIES YOU HAVE:
{[c.name + ': ' + c.description for c in self.capabilities]}

Provide your reasoning as JSON:
{{
    "understanding": "what the question is asking",
    "reasoning_steps": ["step 1", "step 2", ...],
    "conclusion": "your conclusion",
    "confidence": 0.0-1.0,
    "caveats": ["any limitations or uncertainties"]
}}
"""

            messages = [
                Message(MessageRole.SYSTEM, f"You are {self.name}, an intelligent agent. Think carefully."),
                Message(MessageRole.USER, prompt)
            ]

            config = LLMConfig(temperature=0.4, max_tokens=1500)
            response = await self.llm.chat(messages, config)

            import re
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return {
                    "success": True,
                    "reasoning": json.loads(json_match.group())
                }

            return {
                "success": True,
                "reasoning": {"raw_response": response.content}
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_reasoning_trace(self) -> List[Dict[str, Any]]:
        """Get the reasoning trace from the last goal pursuit."""
        return [
            {
                "content": t.content,
                "action": t.action_chosen,
                "confidence": t.confidence,
                "timestamp": t.timestamp.isoformat()
            }
            for t in self._reasoning_trace
        ]

    def get_reflections(self) -> List[Dict[str, Any]]:
        """Get reflections from the last goal pursuit."""
        return [r.to_dict() for r in self._reflections]

    def get_stats(self) -> Dict[str, Any]:
        """Get extended agent statistics."""
        base_stats = super().get_stats()

        base_stats.update({
            "mode": self.config.mode.name,
            "agentic_features": {
                "planning": self.config.enable_planning,
                "reflection": self.config.enable_reflection,
                "goal_decomposition": self.config.enable_goal_decomposition,
                "self_correction": self.config.enable_self_correction
            },
            "reasoning_steps_total": len(self._reasoning_trace),
            "reflections_total": len(self._reflections),
            "current_goal": self._current_goal.description if self._current_goal else None
        })

        return base_stats


class SimpleAgenticAgent(AgenticAgent):
    """
    A simple agentic agent that can be configured with actions.

    Use this when you don't want to create a subclass.

    USAGE:
        agent = SimpleAgenticAgent(
            name="MyAgent",
            llm_provider=provider,
            actions={
                "search": lambda ctx: search_function(ctx["query"]),
                "summarize": lambda ctx: summarize_function(ctx["text"])
            }
        )

        result = await agent.pursue_goal("Find and summarize AI news")
    """

    def __init__(
        self,
        name: str = None,
        llm_provider=None,
        config: AgentConfig = None,
        actions: Dict[str, Callable] = None,
        capability_list: List[AgentCapability] = None
    ):
        super().__init__(name, llm_provider, config)

        self._actions = actions or {}
        self._capability_list = capability_list or []

        # Register actions as tools
        for action_name, action_func in self._actions.items():
            self.register_tool(Tool(
                name=action_name,
                description=f"Execute {action_name}",
                function=action_func
            ))

    @property
    def capabilities(self) -> List[AgentCapability]:
        if self._capability_list:
            return self._capability_list

        # Generate from actions
        return [
            AgentCapability(
                name=action_name,
                description=f"Can {action_name}",
                keywords=[action_name]
            )
            for action_name in self._actions.keys()
        ]

    async def execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        if action in self._actions:
            try:
                func = self._actions[action]
                if asyncio.iscoroutinefunction(func):
                    result = await func(context)
                else:
                    result = func(context)

                return {"success": True, "data": result}

            except Exception as e:
                return {"success": False, "error": str(e)}

        return await super().execute_action(action, context)

    def add_action(self, name: str, func: Callable, description: str = None):
        """Add an action at runtime."""
        self._actions[name] = func
        self.register_tool(Tool(
            name=name,
            description=description or f"Execute {name}",
            function=func
        ))


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_agentic_agent(
    name: str,
    llm_provider,
    actions: Dict[str, Callable] = None,
    capabilities: List[AgentCapability] = None,
    enable_reflection: bool = True,
    enable_planning: bool = True
) -> SimpleAgenticAgent:
    """
    Convenience function to create an agentic agent.

    Usage:
        agent = create_agentic_agent(
            name="ResearchAgent",
            llm_provider=provider,
            actions={
                "search": search_func,
                "analyze": analyze_func
            },
            enable_reflection=True
        )
    """
    config = AgentConfig(
        enable_reflection=enable_reflection,
        enable_planning=enable_planning
    )

    return SimpleAgenticAgent(
        name=name,
        llm_provider=llm_provider,
        config=config,
        actions=actions,
        capability_list=capabilities
    )


async def run_agentic(
    goal: str,
    llm_provider,
    actions: Dict[str, Callable] = None,
    agent_name: str = "Agent"
) -> AgenticResult:
    """
    Convenience function to run an agentic goal pursuit.

    Usage:
        result = await run_agentic(
            "Research and summarize Python best practices",
            llm_provider,
            actions={"search": search_func, "summarize": summarize_func}
        )
    """
    agent = create_agentic_agent(
        name=agent_name,
        llm_provider=llm_provider,
        actions=actions or {}
    )

    return await agent.pursue_goal(goal)


# ============================================================================
# AGENT BUILDER - FLUENT API FOR AGENT CREATION
# ============================================================================

class AgentBuilder:
    """
    Fluent builder for creating agentic agents with minimal boilerplate.
    
    USAGE:
        agent = (AgentBuilder("ResearchAgent")
            .with_llm(provider)
            .with_mode(AgentMode.PROACTIVE)
            .add_tool("search", search_func, "Search the web")
            .add_tool("analyze", analyze_func, "Analyze content")
            .enable_reflection()
            .enable_planning()
            .build())
        
        result = await agent.pursue_goal("Research AI trends")
    
    QUICK START:
        # Minimal agent
        agent = AgentBuilder("MyAgent").with_llm(provider).build()
        
        # With tools
        agent = (AgentBuilder("ToolAgent")
            .with_llm(provider)
            .add_tool("calc", lambda x: eval(x), "Calculate expression")
            .build())
    """
    
    def __init__(self, name: str = "Agent"):
        """Initialize the builder with an agent name."""
        self._name = name
        self._llm_provider = None
        self._mode = AgentMode.PROACTIVE
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._capabilities: List[AgentCapability] = []
        
        # Feature flags
        self._enable_reflection = True
        self._enable_planning = True
        self._enable_goal_decomposition = True
        self._enable_self_correction = True
        
        # Limits
        self._max_iterations = 20
        self._max_retries = 3
        self._quality_threshold = 0.7
        
        # Behavior
        self._verbose = False
        
        # Event handlers (for observability)
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def with_llm(self, provider) -> 'AgentBuilder':
        """Set the LLM provider for reasoning."""
        self._llm_provider = provider
        return self
    
    def with_mode(self, mode: AgentMode) -> 'AgentBuilder':
        """Set the agent operating mode."""
        self._mode = mode
        return self
    
    def add_tool(
        self, 
        name: str, 
        func: Callable, 
        description: str = None,
        parameters: Dict[str, Dict] = None
    ) -> 'AgentBuilder':
        """
        Add a tool/action the agent can use.
        
        Args:
            name: Tool name (used in LLM reasoning)
            func: The function to execute (sync or async)
            description: Human-readable description for LLM
            parameters: Optional parameter schema {param_name: {type, description, required}}
        """
        self._tools[name] = {
            "function": func,
            "description": description or f"Execute {name}",
            "parameters": parameters or {}
        }
        return self
    
    def add_capability(
        self,
        name: str,
        description: str,
        keywords: List[str] = None
    ) -> 'AgentBuilder':
        """Add a capability declaration."""
        self._capabilities.append(AgentCapability(
            name=name,
            description=description,
            keywords=keywords or [name]
        ))
        return self
    
    def enable_reflection(self, enabled: bool = True) -> 'AgentBuilder':
        """Enable/disable self-reflection."""
        self._enable_reflection = enabled
        return self
    
    def enable_planning(self, enabled: bool = True) -> 'AgentBuilder':
        """Enable/disable planning."""
        self._enable_planning = enabled
        return self
    
    def enable_goal_decomposition(self, enabled: bool = True) -> 'AgentBuilder':
        """Enable/disable goal decomposition."""
        self._enable_goal_decomposition = enabled
        return self
    
    def with_max_iterations(self, max_iter: int) -> 'AgentBuilder':
        """Set maximum iterations for goal pursuit."""
        self._max_iterations = max_iter
        return self
    
    def with_quality_threshold(self, threshold: float) -> 'AgentBuilder':
        """Set quality threshold (0.0 - 1.0)."""
        self._quality_threshold = threshold
        return self
    
    def verbose(self, enabled: bool = True) -> 'AgentBuilder':
        """Enable verbose output."""
        self._verbose = enabled
        return self
    
    def on_event(self, event_type: str, handler: Callable) -> 'AgentBuilder':
        """
        Register an event handler for observability.
        
        Event types: 'thinking', 'action_start', 'action_complete', 
                     'reflection', 'goal_progress', 'error'
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        return self
    
    def build(self) -> SimpleAgenticAgent:
        """Build and return the configured agent."""
        if not self._llm_provider:
            raise ValueError("LLM provider is required. Use .with_llm(provider)")
        
        # Build config
        config = AgentConfig(
            mode=self._mode,
            enable_planning=self._enable_planning,
            enable_reflection=self._enable_reflection,
            enable_goal_decomposition=self._enable_goal_decomposition,
            enable_self_correction=self._enable_self_correction,
            max_iterations=self._max_iterations,
            max_retries=self._max_retries,
            quality_threshold=self._quality_threshold,
            verbose=self._verbose
        )
        
        # Build actions dict - wrap functions to be async-compatible
        actions = {}
        for tool_name, tool_info in self._tools.items():
            func = tool_info["function"]
            
            if asyncio.iscoroutinefunction(func):
                # Already async, just wrap to ensure **kwargs compatibility
                def make_async_action(f=func):
                    async def action(**kwargs):
                        return await f(**kwargs)
                    return action
                actions[tool_name] = make_async_action()
            else:
                # Sync function, wrap in async
                def make_sync_action(f=func):
                    async def action(**kwargs):
                        return f(**kwargs)
                    return action
                actions[tool_name] = make_sync_action()
        
        # Build capabilities
        capabilities = self._capabilities.copy()
        if not capabilities:
            # Auto-generate from tools
            for tool_name, tool_info in self._tools.items():
                capabilities.append(AgentCapability(
                    name=tool_name,
                    description=tool_info["description"],
                    keywords=[tool_name]
                ))
        
        # Create agent
        agent = SimpleAgenticAgent(
            name=self._name,
            llm_provider=self._llm_provider,
            config=config,
            actions=actions,
            capability_list=capabilities
        )
        
        # Attach event handlers
        agent._event_handlers = self._event_handlers
        
        return agent
    
    def build_and_run(self, goal: str) -> 'asyncio.Task':
        """Build agent and start pursuing a goal (returns awaitable)."""
        agent = self.build()
        return agent.pursue_goal(goal)
