"""
GPMA LLM Module

Provides LLM integration for intelligent agents.

Supported Providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5-turbo)
- Ollama (Local models: Llama, Mistral, CodeLlama, etc.)
- Azure OpenAI

Usage:
    from gpma.llm import OpenAIProvider, OllamaProvider, LLMAgent

    # OpenAI
    llm = OpenAIProvider(api_key="sk-...", model="gpt-4")

    # Local (Ollama)
    llm = OllamaProvider(model="llama2")

    # Use in agent
    agent = LLMAgent(llm_provider=llm)
"""

from .providers import (
    LLMProvider,
    OpenAIProvider,
    OllamaProvider,
    AzureOpenAIProvider,
    LLMResponse,
    LLMConfig,
)

from .agent import LLMAgent
from .decomposer import LLMTaskDecomposer
from .tools import create_llm_tool, LLMTool

__all__ = [
    # Providers
    'LLMProvider',
    'OpenAIProvider',
    'OllamaProvider',
    'AzureOpenAIProvider',
    'LLMResponse',
    'LLMConfig',
    # Agent
    'LLMAgent',
    # Decomposer
    'LLMTaskDecomposer',
    # Tools
    'create_llm_tool',
    'LLMTool',
]
