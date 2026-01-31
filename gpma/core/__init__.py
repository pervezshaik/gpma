"""
GPMA Core Module

This module contains the foundational components for the multi-agent system:
- BaseAgent: Abstract base class for all agents
- Orchestrator: Coordinates multiple agents
- MessageBus: Inter-agent communication
- Memory: Agent memory systems
"""

from .base_agent import BaseAgent, AgentCapability, AgentState
from .message_bus import MessageBus, Message, MessageType
from .memory import Memory, ShortTermMemory, LongTermMemory
from .orchestrator import Orchestrator

__all__ = [
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
]
