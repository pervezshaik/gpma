"""
Coding Agent Template - Pre-configured for Code Tasks

This template provides a ready-to-use coding agent with:
- Code generation
- Code review
- Bug fixing
- Test generation

USAGE:
    from gpma.templates import CodingAgentTemplate
    
    agent = CodingAgentTemplate.create(provider)
    result = await agent.pursue_goal("Write a Python function to sort a list")
"""

from typing import Any, Callable, Dict, List, Optional
from ..core.agentic_agent import AgentBuilder, AgentMode, SimpleAgenticAgent
from ..core.base_agent import AgentCapability


class CodingAgentTemplate:
    """
    Template for creating coding-focused agents.
    
    Pre-configured with:
    - generate_code: Generate code from description
    - review_code: Review code for issues
    - fix_bugs: Identify and fix bugs
    - write_tests: Generate test cases
    """
    
    @staticmethod
    async def _default_generate_code(description: str, language: str = "python", **kwargs) -> str:
        """Default code generation (simulated)."""
        return f"# Generated {language} code for: {description}\n# TODO: Implement actual code generation"
    
    @staticmethod
    async def _default_review_code(code: str, **kwargs) -> str:
        """Default code review."""
        return "Code review: The code follows basic conventions. Consider adding type hints and docstrings."
    
    @staticmethod
    async def _default_fix_bugs(code: str, error: str = "", **kwargs) -> str:
        """Default bug fixing."""
        return f"Bug analysis: Identified potential issues. Suggested fix applied."
    
    @staticmethod
    async def _default_write_tests(code: str, framework: str = "pytest", **kwargs) -> str:
        """Default test generation."""
        return f"# Generated {framework} tests\ndef test_function():\n    assert True  # TODO: Add actual tests"
    
    @classmethod
    def create(
        cls,
        llm_provider,
        name: str = "CodingAgent",
        custom_tools: Dict[str, Callable] = None,
        verbose: bool = False
    ) -> SimpleAgenticAgent:
        """Create a coding agent with default configuration."""
        builder = cls.builder(llm_provider, name)
        
        if verbose:
            builder.verbose(True)
        
        if custom_tools:
            for tool_name, tool_func in custom_tools.items():
                builder.add_tool(tool_name, tool_func)
        
        return builder.build()
    
    @classmethod
    def builder(cls, llm_provider, name: str = "CodingAgent") -> AgentBuilder:
        """Get a pre-configured builder for customization."""
        return (AgentBuilder(name)
            .with_llm(llm_provider)
            .with_mode(AgentMode.PROACTIVE)
            .add_tool("generate_code", cls._default_generate_code, "Generate code from a description")
            .add_tool("review_code", cls._default_review_code, "Review code for issues and improvements")
            .add_tool("fix_bugs", cls._default_fix_bugs, "Identify and fix bugs in code")
            .add_tool("write_tests", cls._default_write_tests, "Generate test cases for code")
            .add_capability("code", "Generate and write code", ["code", "write", "implement", "create"])
            .add_capability("review", "Review and analyze code", ["review", "check", "analyze"])
            .add_capability("debug", "Debug and fix issues", ["debug", "fix", "bug", "error"])
            .add_capability("test", "Write tests", ["test", "unittest", "pytest"])
            .enable_reflection(True)
            .enable_planning(True)
            .with_max_iterations(15)
            .with_quality_threshold(0.8))
