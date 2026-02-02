"""
Writer Agent Template - Pre-configured for Content Creation

This template provides a ready-to-use writing agent with:
- Content drafting
- Editing and revision
- Style adjustment
- Grammar checking

USAGE:
    from gpma.templates import WriterAgentTemplate
    
    agent = WriterAgentTemplate.create(provider)
    result = await agent.pursue_goal("Write a blog post about AI trends")
"""

from typing import Any, Callable, Dict, List, Optional
from ..core.agentic_agent import AgentBuilder, AgentMode, SimpleAgenticAgent
from ..core.base_agent import AgentCapability


class WriterAgentTemplate:
    """
    Template for creating content writing agents.
    
    Pre-configured with:
    - draft: Create initial content
    - edit: Edit and improve content
    - adjust_style: Modify tone and style
    - check_grammar: Check for grammar issues
    """
    
    @staticmethod
    async def _default_draft(topic: str, style: str = "professional", length: str = "medium", **kwargs) -> str:
        """Default content drafting (simulated)."""
        return f"Draft content about '{topic}' in {style} style. Length: {length}. Content covers key points and provides valuable insights."
    
    @staticmethod
    async def _default_edit(content: str, focus: str = "clarity", **kwargs) -> str:
        """Default editing."""
        return f"Edited content with focus on {focus}. Improved flow, removed redundancy, enhanced readability."
    
    @staticmethod
    async def _default_adjust_style(content: str, target_style: str = "casual", **kwargs) -> str:
        """Default style adjustment."""
        return f"Adjusted content to {target_style} style. Tone and vocabulary modified appropriately."
    
    @staticmethod
    async def _default_check_grammar(content: str, **kwargs) -> str:
        """Default grammar checking."""
        return "Grammar check complete. No major issues found. Minor suggestions: consider active voice in paragraph 2."
    
    @classmethod
    def create(
        cls,
        llm_provider,
        name: str = "WriterAgent",
        custom_tools: Dict[str, Callable] = None,
        verbose: bool = False
    ) -> SimpleAgenticAgent:
        """Create a writer agent with default configuration."""
        builder = cls.builder(llm_provider, name)
        
        if verbose:
            builder.verbose(True)
        
        if custom_tools:
            for tool_name, tool_func in custom_tools.items():
                builder.add_tool(tool_name, tool_func)
        
        return builder.build()
    
    @classmethod
    def builder(cls, llm_provider, name: str = "WriterAgent") -> AgentBuilder:
        """Get a pre-configured builder for customization."""
        return (AgentBuilder(name)
            .with_llm(llm_provider)
            .with_mode(AgentMode.PROACTIVE)
            .add_tool("draft", cls._default_draft, "Create initial content draft")
            .add_tool("edit", cls._default_edit, "Edit and improve content")
            .add_tool("adjust_style", cls._default_adjust_style, "Adjust tone and writing style")
            .add_tool("check_grammar", cls._default_check_grammar, "Check for grammar and spelling issues")
            .add_capability("write", "Write and create content", ["write", "draft", "create", "compose"])
            .add_capability("edit", "Edit and revise content", ["edit", "revise", "improve", "refine"])
            .add_capability("style", "Adjust writing style", ["style", "tone", "voice", "format"])
            .enable_reflection(True)
            .enable_planning(True)
            .with_max_iterations(10)
            .with_quality_threshold(0.8))
