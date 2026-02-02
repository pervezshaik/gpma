"""
Research Agent Template - Pre-configured for Research Tasks

This template provides a ready-to-use research agent with:
- Web search capabilities
- Content analysis
- Summarization
- Source citation

USAGE:
    from gpma.templates import ResearchAgentTemplate
    
    # Quick creation
    agent = ResearchAgentTemplate.create(provider)
    result = await agent.pursue_goal("Research Python async patterns")
    
    # With customization
    agent = (ResearchAgentTemplate.builder(provider)
        .with_name("CustomResearcher")
        .with_max_iterations(15)
        .build())
"""

from typing import Any, Callable, Dict, List, Optional
from ..core.agentic_agent import AgentBuilder, AgentMode, SimpleAgenticAgent
from ..core.base_agent import AgentCapability


class ResearchAgentTemplate:
    """
    Template for creating research-focused agents.
    
    Pre-configured with:
    - search: Search for information
    - analyze: Analyze content for key themes
    - summarize: Create summaries
    - cite: Generate citations
    """
    
    # Default tools for research
    @staticmethod
    async def _default_search(query: str, **kwargs) -> str:
        """Default search implementation (simulated)."""
        return f"Search results for '{query}': Found relevant information about the topic including key concepts, best practices, and recent developments."
    
    @staticmethod
    async def _default_analyze(text: str, **kwargs) -> str:
        """Default analysis implementation."""
        return f"Analysis complete: Identified key themes, patterns, and important points in the provided content."
    
    @staticmethod
    async def _default_summarize(text: str, max_length: int = 500, **kwargs) -> str:
        """Default summarization implementation."""
        if len(text) > max_length:
            return text[:max_length] + "... (summarized)"
        return f"Summary: {text}"
    
    @staticmethod
    async def _default_cite(source: str, **kwargs) -> str:
        """Default citation implementation."""
        return f"[Source: {source}]"
    
    @classmethod
    def create(
        cls,
        llm_provider,
        name: str = "ResearchAgent",
        custom_tools: Dict[str, Callable] = None,
        verbose: bool = False
    ) -> SimpleAgenticAgent:
        """
        Create a research agent with default configuration.
        
        Args:
            llm_provider: LLM provider for reasoning
            name: Agent name
            custom_tools: Optional dict of custom tools to add/override
            verbose: Enable verbose output
        
        Returns:
            Configured SimpleAgenticAgent
        """
        builder = cls.builder(llm_provider, name)
        
        if verbose:
            builder.verbose(True)
        
        if custom_tools:
            for tool_name, tool_func in custom_tools.items():
                builder.add_tool(tool_name, tool_func)
        
        return builder.build()
    
    @classmethod
    def builder(cls, llm_provider, name: str = "ResearchAgent") -> AgentBuilder:
        """
        Get a pre-configured builder for customization.
        
        Args:
            llm_provider: LLM provider for reasoning
            name: Agent name
        
        Returns:
            AgentBuilder with research tools pre-configured
        """
        return (AgentBuilder(name)
            .with_llm(llm_provider)
            .with_mode(AgentMode.PROACTIVE)
            .add_tool("search", cls._default_search, "Search for information on a topic")
            .add_tool("analyze", cls._default_analyze, "Analyze content for key themes and patterns")
            .add_tool("summarize", cls._default_summarize, "Summarize content into key points")
            .add_tool("cite", cls._default_cite, "Generate a citation for a source")
            .add_capability("research", "Research topics using web search", ["search", "find", "research", "look up"])
            .add_capability("analyze", "Analyze and examine content", ["analyze", "examine", "review"])
            .add_capability("summarize", "Summarize and condense information", ["summarize", "brief", "condense"])
            .enable_reflection(True)
            .enable_planning(True)
            .with_max_iterations(10)
            .with_quality_threshold(0.7))
    
    @classmethod
    def with_web_search(cls, llm_provider, web_search_func: Callable, name: str = "WebResearchAgent") -> SimpleAgenticAgent:
        """
        Create a research agent with real web search capability.
        
        Args:
            llm_provider: LLM provider
            web_search_func: Function that performs actual web search
            name: Agent name
        
        Returns:
            Research agent with web search
        """
        return (cls.builder(llm_provider, name)
            .add_tool("search", web_search_func, "Search the web for information")
            .build())
