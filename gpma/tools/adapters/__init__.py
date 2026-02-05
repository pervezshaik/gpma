"""
Tool Format Adapters

Convert BaseTool to provider-specific formats for function calling.

Supported providers:
- OpenAI (and compatible APIs like Azure OpenAI, Groq, Together)
- Anthropic (Claude)
- Ollama (local models)

Usage:
    from gpma.tools import registry
    from gpma.tools.adapters import to_openai_format, to_anthropic_format

    tools = registry.get_all()

    # For OpenAI API
    openai_tools = to_openai_format(tools)

    # For Anthropic API
    anthropic_tools = to_anthropic_format(tools)
"""

from .openai import to_openai_format, OpenAIToolAdapter
from .anthropic import to_anthropic_format, AnthropicToolAdapter

__all__ = [
    "to_openai_format",
    "to_anthropic_format",
    "OpenAIToolAdapter",
    "AnthropicToolAdapter",
]
