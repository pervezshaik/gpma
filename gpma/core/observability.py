"""
Observability Module - Real-time Agent Monitoring

This module provides event-based observability for agentic systems,
allowing developers to see what agents are doing in real-time.

FEATURES:
- Event-based architecture for agent lifecycle events
- Pluggable formatters (console, JSON, custom)
- Progress tracking with metrics
- Event history and replay

USAGE:
    from gpma.core.observability import AgentObserver, EventType
    
    # Create observer
    observer = AgentObserver()
    
    # Subscribe to events
    observer.on(EventType.THINKING, lambda e: print(f"🤔 {e.content}"))
    observer.on(EventType.ACTION_START, lambda e: print(f"⚡ {e.tool_name}"))
    
    # Attach to agent
    agent = AgentBuilder("MyAgent").with_llm(provider).build()
    observer.attach(agent)
    
    # Or use the console formatter for pretty output
    from gpma.core.observability import ConsoleFormatter
    observer.add_formatter(ConsoleFormatter())
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from enum import Enum, auto
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# EVENT TYPES
# ============================================================================

class EventType(Enum):
    """Types of events emitted during agent execution."""
    # Lifecycle events
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    
    # Reasoning events
    THINKING = "thinking"
    THOUGHT_COMPLETE = "thought_complete"
    
    # Action events
    ACTION_START = "action_start"
    ACTION_COMPLETE = "action_complete"
    ACTION_ERROR = "action_error"
    
    # Reflection events
    REFLECTION_START = "reflection_start"
    REFLECTION_COMPLETE = "reflection_complete"
    
    # Goal events
    GOAL_SET = "goal_set"
    GOAL_PROGRESS = "goal_progress"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_FAILED = "goal_failed"
    
    # Planning events
    PLAN_CREATED = "plan_created"
    PLAN_STEP_START = "plan_step_start"
    PLAN_STEP_COMPLETE = "plan_step_complete"
    
    # Tool events
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    
    # Iteration events
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"


# ============================================================================
# EVENT DATA CLASSES
# ============================================================================

@dataclass
class AgentEvent:
    """Base event emitted by agents."""
    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.now)
    agent_name: str = ""
    iteration: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "iteration": self.iteration,
            "metadata": self.metadata
        }


@dataclass
class ThinkingEvent(AgentEvent):
    """Event when agent is reasoning."""
    content: str = ""
    confidence: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.THINKING


@dataclass
class ActionEvent(AgentEvent):
    """Event for action execution."""
    tool_name: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: str = ""
    execution_time: float = 0.0
    
    def __post_init__(self):
        if self.result is not None:
            self.event_type = EventType.ACTION_COMPLETE
        else:
            self.event_type = EventType.ACTION_START


@dataclass
class GoalEvent(AgentEvent):
    """Event for goal-related updates."""
    goal: str = ""
    progress: float = 0.0
    subgoals: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.progress >= 1.0:
            self.event_type = EventType.GOAL_ACHIEVED
        elif self.progress > 0:
            self.event_type = EventType.GOAL_PROGRESS
        else:
            self.event_type = EventType.GOAL_SET


@dataclass
class ReflectionEvent(AgentEvent):
    """Event for reflection updates."""
    assessment: str = ""
    quality_score: float = 0.0
    needs_correction: bool = False
    suggestions: List[str] = field(default_factory=list)


@dataclass
class IterationEvent(AgentEvent):
    """Event for iteration tracking."""
    current: int = 0
    max_iterations: int = 0
    elapsed_time: float = 0.0


# ============================================================================
# EVENT HANDLERS AND FORMATTERS
# ============================================================================

class EventFormatter:
    """Base class for event formatters."""
    
    def format(self, event: AgentEvent) -> str:
        """Format an event for output."""
        raise NotImplementedError


class JSONFormatter(EventFormatter):
    """Format events as JSON."""
    
    def format(self, event: AgentEvent) -> str:
        return json.dumps(event.to_dict(), indent=2)


class SimpleFormatter(EventFormatter):
    """Simple text formatter."""
    
    def format(self, event: AgentEvent) -> str:
        timestamp = event.timestamp.strftime("%H:%M:%S")
        return f"[{timestamp}] {event.event_type.value}: {event.agent_name}"


class ConsoleFormatter(EventFormatter):
    """
    Rich console formatter with colors and symbols.
    
    Provides beautiful, readable output for terminal display.
    """
    
    # Event type to emoji/symbol mapping
    SYMBOLS = {
        EventType.AGENT_START: "🚀",
        EventType.AGENT_COMPLETE: "✅",
        EventType.AGENT_ERROR: "❌",
        EventType.THINKING: "🤔",
        EventType.THOUGHT_COMPLETE: "💭",
        EventType.ACTION_START: "⚡",
        EventType.ACTION_COMPLETE: "✓",
        EventType.ACTION_ERROR: "⚠️",
        EventType.REFLECTION_START: "🔍",
        EventType.REFLECTION_COMPLETE: "📝",
        EventType.GOAL_SET: "🎯",
        EventType.GOAL_PROGRESS: "📊",
        EventType.GOAL_ACHIEVED: "🏆",
        EventType.GOAL_FAILED: "💔",
        EventType.PLAN_CREATED: "📋",
        EventType.PLAN_STEP_START: "▶️",
        EventType.PLAN_STEP_COMPLETE: "✔️",
        EventType.TOOL_CALL: "🔧",
        EventType.TOOL_RESULT: "📤",
        EventType.ITERATION_START: "🔄",
        EventType.ITERATION_COMPLETE: "⏱️",
    }
    
    # ANSI color codes
    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "red": "\033[31m",
        "white": "\033[37m",
    }
    
    def __init__(self, use_colors: bool = True, show_timestamp: bool = True):
        self.use_colors = use_colors
        self.show_timestamp = show_timestamp
    
    def _color(self, text: str, color: str) -> str:
        if not self.use_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['reset']}"
    
    def format(self, event: AgentEvent) -> str:
        symbol = self.SYMBOLS.get(event.event_type, "•")
        
        # Build timestamp
        timestamp = ""
        if self.show_timestamp:
            ts = event.timestamp.strftime("%H:%M:%S")
            timestamp = self._color(f"[{ts}] ", "dim")
        
        # Format based on event type
        if isinstance(event, ThinkingEvent):
            content = event.content[:100] + "..." if len(event.content) > 100 else event.content
            return f"{timestamp}{symbol} {self._color('Thinking:', 'cyan')} {content}"
        
        elif isinstance(event, ActionEvent):
            if event.event_type == EventType.ACTION_START:
                params = ", ".join(f"{k}={v}" for k, v in list(event.parameters.items())[:2])
                return f"{timestamp}{symbol} {self._color('Action:', 'yellow')} {event.tool_name}({params})"
            else:
                status = self._color("✓", "green") if event.success else self._color("✗", "red")
                time_str = f" ({event.execution_time:.2f}s)" if event.execution_time > 0 else ""
                return f"{timestamp}{status} {event.tool_name} complete{time_str}"
        
        elif isinstance(event, GoalEvent):
            progress_bar = self._progress_bar(event.progress)
            return f"{timestamp}{symbol} {self._color('Goal:', 'magenta')} {event.goal[:50]}... {progress_bar}"
        
        elif isinstance(event, ReflectionEvent):
            quality = self._color(f"{event.quality_score:.0%}", "green" if event.quality_score > 0.7 else "yellow")
            return f"{timestamp}{symbol} {self._color('Reflection:', 'blue')} Quality: {quality}"
        
        elif isinstance(event, IterationEvent):
            progress = event.current / event.max_iterations if event.max_iterations > 0 else 0
            bar = self._progress_bar(progress)
            return f"{timestamp}{symbol} Iteration {event.current}/{event.max_iterations} {bar}"
        
        else:
            return f"{timestamp}{symbol} {event.event_type.value}"
    
    def _progress_bar(self, progress: float, width: int = 20) -> str:
        filled = int(width * progress)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        pct = f"{progress:.0%}"
        if self.use_colors:
            color = "green" if progress >= 0.7 else "yellow" if progress >= 0.3 else "red"
            return f"{self._color(bar, color)} {pct}"
        return f"[{bar}] {pct}"


# ============================================================================
# AGENT OBSERVER
# ============================================================================

class AgentObserver:
    """
    Central observer for agent events.
    
    Collects events from agents and dispatches to handlers/formatters.
    
    USAGE:
        observer = AgentObserver()
        
        # Add handlers
        observer.on(EventType.THINKING, my_handler)
        observer.on(EventType.ACTION_COMPLETE, log_action)
        
        # Add formatter for console output
        observer.add_formatter(ConsoleFormatter())
        
        # Attach to agent
        observer.attach(agent)
    """
    
    def __init__(self):
        self._handlers: Dict[EventType, List[Callable]] = {}
        self._formatters: List[EventFormatter] = []
        self._history: List[AgentEvent] = []
        self._max_history = 1000
        self._attached_agents: List[str] = []
    
    def on(self, event_type: EventType, handler: Callable[[AgentEvent], None]) -> 'AgentObserver':
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        return self
    
    def on_all(self, handler: Callable[[AgentEvent], None]) -> 'AgentObserver':
        """Register a handler for all event types."""
        for event_type in EventType:
            self.on(event_type, handler)
        return self
    
    def add_formatter(self, formatter: EventFormatter) -> 'AgentObserver':
        """Add a formatter for event output."""
        self._formatters.append(formatter)
        return self
    
    def emit(self, event: AgentEvent) -> None:
        """Emit an event to all handlers and formatters."""
        # Store in history
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        
        # Call handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        
        # Output via formatters
        for formatter in self._formatters:
            try:
                output = formatter.format(event)
                print(output)
            except Exception as e:
                logger.error(f"Formatter error: {e}")
    
    def attach(self, agent) -> 'AgentObserver':
        """
        Attach observer to an agent.
        
        This injects the observer's emit method into the agent's
        event emission system.
        """
        agent_name = getattr(agent, 'name', str(agent))
        self._attached_agents.append(agent_name)
        
        # Inject observer reference
        agent._observer = self
        
        return self
    
    def get_history(
        self, 
        event_type: EventType = None,
        agent_name: str = None,
        limit: int = None
    ) -> List[AgentEvent]:
        """Get event history with optional filtering."""
        events = self._history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if agent_name:
            events = [e for e in events if e.agent_name == agent_name]
        
        if limit:
            events = events[-limit:]
        
        return events
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics from event history."""
        if not self._history:
            return {}
        
        # Count events by type
        event_counts = {}
        for event in self._history:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Calculate timing metrics
        action_events = [e for e in self._history if isinstance(e, ActionEvent) and e.execution_time > 0]
        avg_action_time = sum(e.execution_time for e in action_events) / len(action_events) if action_events else 0
        
        # Success rate
        completed_actions = [e for e in self._history if e.event_type == EventType.ACTION_COMPLETE]
        success_rate = sum(1 for e in completed_actions if isinstance(e, ActionEvent) and e.success) / len(completed_actions) if completed_actions else 0
        
        return {
            "total_events": len(self._history),
            "event_counts": event_counts,
            "avg_action_time": avg_action_time,
            "action_success_rate": success_rate,
            "attached_agents": self._attached_agents
        }
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._history = []


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_console_observer(use_colors: bool = True) -> AgentObserver:
    """Create an observer with console output."""
    observer = AgentObserver()
    observer.add_formatter(ConsoleFormatter(use_colors=use_colors))
    return observer


def create_logging_observer(logger_name: str = "gpma.agent") -> AgentObserver:
    """Create an observer that logs events."""
    obs_logger = logging.getLogger(logger_name)
    observer = AgentObserver()
    
    def log_event(event: AgentEvent):
        obs_logger.info(f"{event.event_type.value}: {event.agent_name} - {event.metadata}")
    
    observer.on_all(log_event)
    return observer


# ============================================================================
# PROGRESS TRACKER
# ============================================================================

class ProgressTracker:
    """
    Track and display progress of agent execution.
    
    USAGE:
        tracker = ProgressTracker(total_steps=10)
        tracker.start("Research AI trends")
        
        for i in range(10):
            tracker.update(i + 1, f"Step {i + 1}")
        
        tracker.complete("Research complete!")
    """
    
    def __init__(self, total_steps: int = 10, show_bar: bool = True):
        self.total_steps = total_steps
        self.current_step = 0
        self.show_bar = show_bar
        self.start_time: Optional[datetime] = None
        self.task_name = ""
        self._formatter = ConsoleFormatter()
    
    def start(self, task_name: str) -> None:
        """Start tracking a task."""
        self.task_name = task_name
        self.current_step = 0
        self.start_time = datetime.now()
        print(f"\n🚀 Starting: {task_name}")
        print("=" * 60)
    
    def update(self, step: int, message: str = "") -> None:
        """Update progress."""
        self.current_step = step
        progress = step / self.total_steps
        
        if self.show_bar:
            bar = self._formatter._progress_bar(progress)
            print(f"\r{bar} {message}", end="", flush=True)
        else:
            print(f"  Step {step}/{self.total_steps}: {message}")
    
    def complete(self, message: str = "Complete!") -> None:
        """Mark task as complete."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        print(f"\n{'=' * 60}")
        print(f"✅ {message} (took {elapsed:.2f}s)")
    
    def fail(self, error: str) -> None:
        """Mark task as failed."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        print(f"\n{'=' * 60}")
        print(f"❌ Failed: {error} (after {elapsed:.2f}s)")
