"""
GPMA Core Module

This module contains the foundational components for the multi-agent system:

ORIGINAL COMPONENTS:
- BaseAgent: Abstract base class for all agents
- Orchestrator: Coordinates multiple agents
- MessageBus: Inter-agent communication
- Memory: Agent memory systems

AGENTIC CAPABILITIES (Phase 1):
- AgenticLoop: ReAct (Reasoning + Acting) loop for autonomous behavior
- TaskPlanner: Intelligent LLM-powered task planning and decomposition
- ReflectionEngine: Self-evaluation and correction of outputs
- GoalManager: Hierarchical goal-oriented behavior
- AgenticAgent: Enhanced base agent with all agentic capabilities
"""

# Original components
from .base_agent import BaseAgent, AgentCapability, AgentState, Tool, TaskResult
from .message_bus import MessageBus, Message, MessageType
from .memory import Memory, ShortTermMemory, LongTermMemory
from .orchestrator import Orchestrator

# Agentic capabilities (Phase 1)
from .agentic_loop import (
    AgenticLoop,
    AgenticTool,
    AgenticContext,
    LoopResult,
    LoopStatus,
    Observation,
    Thought,
    Action,
    Reflection,
    Step,
    run_agentic_task,
    create_tool,
)

from .planner import (
    TaskPlanner,
    PlanExecutor,
    ExecutionPlan,
    PlannedTask,
    DependencyGraph,
    TaskPriority,
    TaskComplexity,
    ExecutionStrategy,
    create_plan,
    execute_plan,
)

from .reflection import (
    ReflectionEngine,
    ReflectionResult,
    SuccessCriteria,
    QualityAssessment,
    CriterionCheck,
    EvaluationResult,
    CorrectionStrategy,
    evaluate_output,
    create_criteria,
)

from .goal_manager import (
    GoalManager,
    Goal,
    GoalTree,
    GoalResult,
    GoalStatus,
    GoalPriority,
    GoalType,
    pursue_goal,
    create_goal,
)

from .agentic_agent import (
    AgenticAgent,
    SimpleAgenticAgent,
    AgentConfig,
    AgentMode,
    AgenticResult,
    AgentThought,
    AgentBuilder,
    create_agentic_agent,
    run_agentic,
)

from .observability import (
    AgentObserver,
    EventType,
    AgentEvent,
    ThinkingEvent,
    ActionEvent,
    GoalEvent,
    ReflectionEvent,
    IterationEvent,
    EventFormatter,
    JSONFormatter,
    SimpleFormatter,
    ConsoleFormatter,
    ProgressTracker,
    create_console_observer,
    create_logging_observer,
)

from .console_ui import (
    AgentConsole,
    ProgressBar,
    Colors,
    Symbols,
    colorize,
)

__all__ = [
    # Original components
    'BaseAgent',
    'AgentCapability',
    'AgentState',
    'Tool',
    'TaskResult',
    'MessageBus',
    'Message',
    'MessageType',
    'Memory',
    'ShortTermMemory',
    'LongTermMemory',
    'Orchestrator',

    # Agentic Loop
    'AgenticLoop',
    'AgenticTool',
    'AgenticContext',
    'LoopResult',
    'LoopStatus',
    'Observation',
    'Thought',
    'Action',
    'Reflection',
    'Step',
    'run_agentic_task',
    'create_tool',

    # Planning
    'TaskPlanner',
    'PlanExecutor',
    'ExecutionPlan',
    'PlannedTask',
    'DependencyGraph',
    'TaskPriority',
    'TaskComplexity',
    'ExecutionStrategy',
    'create_plan',
    'execute_plan',

    # Reflection
    'ReflectionEngine',
    'ReflectionResult',
    'SuccessCriteria',
    'QualityAssessment',
    'CriterionCheck',
    'EvaluationResult',
    'CorrectionStrategy',
    'evaluate_output',
    'create_criteria',

    # Goal Management
    'GoalManager',
    'Goal',
    'GoalTree',
    'GoalResult',
    'GoalStatus',
    'GoalPriority',
    'GoalType',
    'pursue_goal',
    'create_goal',

    # Agentic Agent
    'AgenticAgent',
    'SimpleAgenticAgent',
    'AgentConfig',
    'AgentMode',
    'AgenticResult',
    'AgentThought',
    'AgentBuilder',
    'create_agentic_agent',
    'run_agentic',
    
    # Observability
    'AgentObserver',
    'EventType',
    'AgentEvent',
    'ThinkingEvent',
    'ActionEvent',
    'GoalEvent',
    'ReflectionEvent',
    'IterationEvent',
    'EventFormatter',
    'JSONFormatter',
    'SimpleFormatter',
    'ConsoleFormatter',
    'ProgressTracker',
    'create_console_observer',
    'create_logging_observer',
    
    # Console UI
    'AgentConsole',
    'ProgressBar',
    'Colors',
    'Symbols',
    'colorize',
]
