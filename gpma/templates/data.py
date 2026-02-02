"""
Data Analyst Agent Template - Pre-configured for Data Tasks

This template provides a ready-to-use data analysis agent with:
- Data loading and parsing
- Statistical analysis
- Pattern detection
- Visualization suggestions

USAGE:
    from gpma.templates import DataAnalystTemplate
    
    agent = DataAnalystTemplate.create(provider)
    result = await agent.pursue_goal("Analyze sales data and find trends")
"""

from typing import Any, Callable, Dict, List, Optional
from ..core.agentic_agent import AgentBuilder, AgentMode, SimpleAgenticAgent
from ..core.base_agent import AgentCapability


class DataAnalystTemplate:
    """
    Template for creating data analysis agents.
    
    Pre-configured with:
    - load_data: Load and parse data
    - analyze_stats: Compute statistics
    - find_patterns: Detect patterns and trends
    - suggest_viz: Suggest visualizations
    """
    
    @staticmethod
    async def _default_load_data(source: str, format: str = "csv", **kwargs) -> str:
        """Default data loading (simulated)."""
        return f"Loaded data from {source} ({format} format). Found columns and rows ready for analysis."
    
    @staticmethod
    async def _default_analyze_stats(data: str, metrics: List[str] = None, **kwargs) -> str:
        """Default statistical analysis."""
        return "Statistical analysis: Mean, median, std computed. Distribution appears normal with some outliers."
    
    @staticmethod
    async def _default_find_patterns(data: str, **kwargs) -> str:
        """Default pattern detection."""
        return "Pattern analysis: Identified seasonal trends, correlations between variables, and potential anomalies."
    
    @staticmethod
    async def _default_suggest_viz(data: str, insight: str = "", **kwargs) -> str:
        """Default visualization suggestions."""
        return "Visualization suggestions: Line chart for trends, scatter plot for correlations, histogram for distribution."
    
    @classmethod
    def create(
        cls,
        llm_provider,
        name: str = "DataAnalyst",
        custom_tools: Dict[str, Callable] = None,
        verbose: bool = False
    ) -> SimpleAgenticAgent:
        """Create a data analyst agent with default configuration."""
        builder = cls.builder(llm_provider, name)
        
        if verbose:
            builder.verbose(True)
        
        if custom_tools:
            for tool_name, tool_func in custom_tools.items():
                builder.add_tool(tool_name, tool_func)
        
        return builder.build()
    
    @classmethod
    def builder(cls, llm_provider, name: str = "DataAnalyst") -> AgentBuilder:
        """Get a pre-configured builder for customization."""
        return (AgentBuilder(name)
            .with_llm(llm_provider)
            .with_mode(AgentMode.PROACTIVE)
            .add_tool("load_data", cls._default_load_data, "Load and parse data from a source")
            .add_tool("analyze_stats", cls._default_analyze_stats, "Compute statistical metrics")
            .add_tool("find_patterns", cls._default_find_patterns, "Detect patterns and trends in data")
            .add_tool("suggest_viz", cls._default_suggest_viz, "Suggest appropriate visualizations")
            .add_capability("data", "Load and process data", ["data", "load", "parse", "read"])
            .add_capability("analyze", "Analyze data statistically", ["analyze", "statistics", "metrics"])
            .add_capability("patterns", "Find patterns and trends", ["pattern", "trend", "correlation"])
            .add_capability("visualize", "Create visualizations", ["visualize", "chart", "plot", "graph"])
            .enable_reflection(True)
            .enable_planning(True)
            .with_max_iterations(12)
            .with_quality_threshold(0.75))
