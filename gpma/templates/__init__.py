"""
Agent Templates Module - Pre-built Agent Configurations

This module provides ready-to-use agent templates for common use cases.
Templates reduce boilerplate and provide best-practice configurations.

AVAILABLE TEMPLATES:
- ResearchAgentTemplate: Web search, analysis, summarization
- CodingAgentTemplate: Code generation, review, testing
- DataAnalystTemplate: Data processing, visualization
- WriterAgentTemplate: Content creation, editing

USAGE:
    from gpma.templates import ResearchAgentTemplate
    
    # Quick creation
    agent = ResearchAgentTemplate.create(provider)
    result = await agent.pursue_goal("Research AI trends")
    
    # Customized
    agent = (ResearchAgentTemplate.builder(provider)
        .with_name("MyResearcher")
        .add_tool("custom_search", my_func)
        .build())
"""

from .research import ResearchAgentTemplate
from .coding import CodingAgentTemplate
from .data import DataAnalystTemplate
from .writer import WriterAgentTemplate

__all__ = [
    "ResearchAgentTemplate",
    "CodingAgentTemplate",
    "DataAnalystTemplate",
    "WriterAgentTemplate",
]
