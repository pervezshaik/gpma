"""
GPMA Agents Module

Specialized agents for different types of tasks:
- WebBrowserAgent: Fetches and parses web content
- ResearchAgent: Searches and analyzes information
- TaskExecutorAgent: Executes commands and file operations
"""

from .web_browser import WebBrowserAgent
from .research import ResearchAgent
from .task_executor import TaskExecutorAgent

__all__ = [
    'WebBrowserAgent',
    'ResearchAgent',
    'TaskExecutorAgent',
]
