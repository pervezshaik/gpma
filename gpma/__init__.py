"""
GPMA - General Purpose Multi-Agent System

A modular, extensible framework for building multi-agent AI systems.

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

COMPONENTS:
- Core: BaseAgent, Orchestrator, MessageBus, Memory
- Agents: WebBrowserAgent, ResearchAgent, TaskExecutorAgent
- LLM: OpenAIProvider, OllamaProvider, LLMAgent
- Tools: web_tools, file_tools
"""

from .core import (
    BaseAgent,
    AgentCapability,
    AgentState,
    MessageBus,
    Message,
    MessageType,
    Memory,
    ShortTermMemory,
    LongTermMemory,
    Orchestrator,
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

__version__ = "0.2.0"
__author__ = "GPMA Team"

__all__ = [
    # Core
    'BaseAgent',
    'AgentCapability',
    'AgentState',
    'MessageBus',
    'Message',
    'MessageType',
    'Memory',
    'ShortTermMemory',
    'LongTermMemory',
    'Orchestrator',
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
