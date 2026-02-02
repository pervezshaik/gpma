"""
GPMA - General Purpose Multi-Agent System

A modular, extensible framework for building multi-agent AI systems.
Now with TRUE AGENTIC CAPABILITIES.

QUICK START (without LLM):
    from gpma import PersonalAssistant

    async def main():
        assistant = PersonalAssistant()
        await assistant.initialize()

        result = await assistant.ask("What is the capital of France?")
        print(result)

        await assistant.shutdown()

QUICK START (with LLM):
    from gpma import PersonalAssistant
    from gpma.llm import OpenAIProvider, OllamaProvider

    # With OpenAI
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
    assistant = PersonalAssistant(llm_provider=provider)

    # With Ollama (local)
    provider = OllamaProvider(model="llama2")
    assistant = PersonalAssistant(llm_provider=provider)

    async with assistant:
        response = await assistant.chat("Hello!")
        print(response)

AGENTIC CAPABILITIES (NEW):
    from gpma import AgenticAgent, AgenticLoop, GoalManager
    from gpma.llm import OllamaProvider

    provider = OllamaProvider(model="llama3.1")

    # Use the agentic loop directly
    loop = AgenticLoop(provider)
    result = await loop.run("Research and summarize AI trends", tools=[...])

    # Or use an agentic agent
    agent = AgenticAgent(name="ResearchAgent", llm_provider=provider)
    result = await agent.pursue_goal("Find Python best practices")

COMPONENTS:
- Core: BaseAgent, Orchestrator, MessageBus, Memory
- Agentic: AgenticLoop, TaskPlanner, ReflectionEngine, GoalManager, AgenticAgent
- Agents: WebBrowserAgent, ResearchAgent, TaskExecutorAgent
- LLM: OpenAIProvider, OllamaProvider, LLMAgent
- Tools: web_tools, file_tools

See PROFESSIONAL_UPGRADE_PLAN.md for the complete upgrade roadmap.
"""

# Original core components
from .core import (
    # Base components
    BaseAgent,
    AgentCapability,
    AgentState,
    Tool,
    TaskResult,
    MessageBus,
    Message,
    MessageType,
    Memory,
    ShortTermMemory,
    LongTermMemory,
    Orchestrator,

    # Agentic Loop (ReAct)
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

    # Planning
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

    # Reflection
    ReflectionEngine,
    ReflectionResult,
    SuccessCriteria,
    QualityAssessment,
    CriterionCheck,
    EvaluationResult,
    CorrectionStrategy,
    evaluate_output,
    create_criteria,

    # Goal Management
    GoalManager,
    Goal,
    GoalTree,
    GoalResult,
    GoalStatus,
    GoalPriority,
    GoalType,
    pursue_goal,
    create_goal,

    # Agentic Agent
    AgenticAgent,
    SimpleAgenticAgent,
    AgentConfig,
    AgentMode,
    AgenticResult,
    AgentThought,
    create_agentic_agent,
    run_agentic,
)

from .agents import (
    WebBrowserAgent,
    ResearchAgent,
    TaskExecutorAgent,
)

from .personal_assistant import (
    PersonalAssistant,
    create_openai_assistant,
    create_ollama_assistant,
)

__version__ = "0.3.0"  # Bumped for agentic capabilities
__author__ = "GPMA Team"

__all__ = [
    # Original Core
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

    # Agentic Loop (ReAct)
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
    'create_agentic_agent',
    'run_agentic',

    # Agents
    'WebBrowserAgent',
    'ResearchAgent',
    'TaskExecutorAgent',

    # Main
    'PersonalAssistant',
    'create_openai_assistant',
    'create_ollama_assistant',
]

# LLM module is optional - import separately if needed
# from gpma.llm import OpenAIProvider, OllamaProvider, LLMAgent
