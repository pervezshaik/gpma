"""
Console UI Module - Rich Terminal Output for Agents

This module provides beautiful, informative console output for
monitoring agent execution in real-time.

FEATURES:
- Progress bars with ETA
- Collapsible sections
- Color-coded output
- Live updating displays
- Structured output panels

USAGE:
    from gpma.core.console_ui import AgentConsole
    
    console = AgentConsole()
    console.start_agent("ResearchAgent", "Research AI trends")
    
    console.show_thinking("Analyzing the goal...")
    console.show_action("search", {"query": "AI trends 2024"})
    console.show_result("Found 5 relevant articles")
    
    console.complete("Research complete!")
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime
import sys
import os


# ============================================================================
# ANSI ESCAPE CODES
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


class Symbols:
    """Unicode symbols for visual output."""
    # Status
    CHECK = "✓"
    CROSS = "✗"
    WARNING = "⚠"
    INFO = "ℹ"
    
    # Progress
    SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    PROGRESS_FULL = "█"
    PROGRESS_EMPTY = "░"
    PROGRESS_PARTIAL = ["▏", "▎", "▍", "▌", "▋", "▊", "▉"]
    
    # Actions
    THINKING = "🤔"
    ACTION = "⚡"
    TOOL = "🔧"
    SEARCH = "🔍"
    ANALYZE = "📊"
    WRITE = "✏️"
    SUCCESS = "✅"
    FAILURE = "❌"
    GOAL = "🎯"
    PLAN = "📋"
    REFLECT = "💭"
    ROCKET = "🚀"
    CLOCK = "⏱️"
    
    # Boxes
    BOX_TL = "┌"
    BOX_TR = "┐"
    BOX_BL = "└"
    BOX_BR = "┘"
    BOX_H = "─"
    BOX_V = "│"
    BOX_T = "┬"
    BOX_B = "┴"
    BOX_L = "├"
    BOX_R = "┤"
    BOX_X = "┼"
    
    # Arrows
    ARROW_RIGHT = "→"
    ARROW_LEFT = "←"
    ARROW_UP = "↑"
    ARROW_DOWN = "↓"
    ARROW_DOUBLE = "⇒"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def supports_color() -> bool:
    """Check if terminal supports color output."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


def colorize(text: str, *styles) -> str:
    """Apply color/style codes to text."""
    if not supports_color():
        return text
    codes = "".join(styles)
    return f"{codes}{text}{Colors.RESET}"


def truncate(text: str, max_length: int = 80, suffix: str = "...") -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# ============================================================================
# PROGRESS BAR
# ============================================================================

class ProgressBar:
    """
    Customizable progress bar for terminal display.
    
    USAGE:
        bar = ProgressBar(total=100, width=40)
        for i in range(100):
            bar.update(i + 1)
        bar.finish()
    """
    
    def __init__(
        self,
        total: int = 100,
        width: int = 40,
        fill_char: str = Symbols.PROGRESS_FULL,
        empty_char: str = Symbols.PROGRESS_EMPTY,
        show_percentage: bool = True,
        show_count: bool = True,
        show_eta: bool = True,
        prefix: str = "",
        suffix: str = ""
    ):
        self.total = total
        self.width = width
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.show_percentage = show_percentage
        self.show_count = show_count
        self.show_eta = show_eta
        self.prefix = prefix
        self.suffix = suffix
        
        self.current = 0
        self.start_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start the progress bar."""
        self.start_time = datetime.now()
        self.current = 0
        self._render()
    
    def update(self, current: int, message: str = "") -> None:
        """Update progress."""
        self.current = current
        self.suffix = message
        self._render()
    
    def increment(self, amount: int = 1) -> None:
        """Increment progress by amount."""
        self.update(self.current + amount)
    
    def finish(self, message: str = "Complete!") -> None:
        """Finish and show completion message."""
        self.current = self.total
        self.suffix = message
        self._render()
        print()  # New line after completion
    
    def _render(self) -> None:
        """Render the progress bar."""
        progress = self.current / self.total if self.total > 0 else 0
        filled = int(self.width * progress)
        empty = self.width - filled
        
        # Build bar
        bar = self.fill_char * filled + self.empty_char * empty
        
        # Color the bar
        if progress >= 0.7:
            bar = colorize(bar, Colors.GREEN)
        elif progress >= 0.3:
            bar = colorize(bar, Colors.YELLOW)
        else:
            bar = colorize(bar, Colors.RED)
        
        # Build info string
        info_parts = []
        
        if self.show_percentage:
            pct = f"{progress:.0%}"
            info_parts.append(pct)
        
        if self.show_count:
            count = f"{self.current}/{self.total}"
            info_parts.append(count)
        
        if self.show_eta and self.start_time and self.current > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            eta = (elapsed / self.current) * (self.total - self.current)
            if eta < 60:
                eta_str = f"ETA: {eta:.0f}s"
            else:
                eta_str = f"ETA: {eta/60:.1f}m"
            info_parts.append(eta_str)
        
        info = " | ".join(info_parts)
        
        # Build full line
        line = f"\r{self.prefix}[{bar}] {info}"
        if self.suffix:
            line += f" {self.suffix}"
        
        # Clear to end of line and print
        print(f"{line}\033[K", end="", flush=True)


# ============================================================================
# AGENT CONSOLE
# ============================================================================

class AgentConsole:
    """
    Rich console interface for agent monitoring.
    
    Provides structured, beautiful output for tracking agent execution.
    
    USAGE:
        console = AgentConsole()
        
        # Start agent session
        console.start_agent("ResearchAgent", "Research Python best practices")
        
        # Show thinking
        console.show_thinking("Analyzing the research goal...")
        
        # Show action
        console.show_action("search", {"query": "Python best practices"})
        
        # Show result
        console.show_result("Found 10 relevant articles")
        
        # Show reflection
        console.show_reflection("Good coverage of topics", quality=0.8)
        
        # Complete
        console.complete("Research complete with 3 key findings")
    """
    
    def __init__(self, use_colors: bool = True, width: int = 80):
        self.use_colors = use_colors and supports_color()
        self.width = width
        self.indent_level = 0
        self.start_time: Optional[datetime] = None
        self.iteration = 0
        self.agent_name = ""
        self.goal = ""
    
    def _print(self, text: str, indent: bool = True) -> None:
        """Print with optional indentation."""
        prefix = "  " * self.indent_level if indent else ""
        print(f"{prefix}{text}")
    
    def _header(self, title: str, char: str = "=") -> None:
        """Print a header line."""
        line = char * self.width
        self._print(line, indent=False)
        centered = title.center(self.width)
        if self.use_colors:
            centered = colorize(centered, Colors.BOLD, Colors.CYAN)
        self._print(centered, indent=False)
        self._print(line, indent=False)
    
    def _subheader(self, title: str) -> None:
        """Print a subheader."""
        line = "-" * (self.width - 4)
        self._print(f"  {title}")
        self._print(f"  {line}")
    
    def _box(self, content: List[str], title: str = "") -> None:
        """Print content in a box."""
        inner_width = self.width - 4
        
        # Top border
        if title:
            title_part = f" {title} "
            padding = inner_width - len(title_part)
            top = f"{Symbols.BOX_TL}{Symbols.BOX_H}{title_part}{Symbols.BOX_H * padding}{Symbols.BOX_TR}"
        else:
            top = f"{Symbols.BOX_TL}{Symbols.BOX_H * (inner_width + 2)}{Symbols.BOX_TR}"
        
        self._print(top, indent=False)
        
        # Content
        for line in content:
            padded = line.ljust(inner_width)[:inner_width]
            self._print(f"{Symbols.BOX_V} {padded} {Symbols.BOX_V}", indent=False)
        
        # Bottom border
        bottom = f"{Symbols.BOX_BL}{Symbols.BOX_H * (inner_width + 2)}{Symbols.BOX_BR}"
        self._print(bottom, indent=False)
    
    def start_agent(self, agent_name: str, goal: str) -> None:
        """Start an agent session."""
        self.agent_name = agent_name
        self.goal = goal
        self.start_time = datetime.now()
        self.iteration = 0
        
        print()
        self._header(f"{Symbols.ROCKET} {agent_name}")
        
        goal_display = truncate(goal, self.width - 10)
        self._print(f"\n{Symbols.GOAL} Goal: {colorize(goal_display, Colors.BOLD) if self.use_colors else goal_display}")
        self._print(f"{Symbols.CLOCK} Started: {self.start_time.strftime('%H:%M:%S')}\n")
    
    def start_iteration(self, iteration: int, max_iterations: int = 0) -> None:
        """Start a new iteration."""
        self.iteration = iteration
        
        if max_iterations > 0:
            progress = iteration / max_iterations
            bar = ProgressBar(total=max_iterations, width=30, show_eta=False)
            bar.current = iteration
            bar._render()
            print()
        
        iter_text = f"Iteration {iteration}"
        if max_iterations > 0:
            iter_text += f"/{max_iterations}"
        
        self._subheader(f"{Symbols.ARROW_RIGHT} {iter_text}")
    
    def show_thinking(self, thought: str, confidence: float = 0.0) -> None:
        """Display agent thinking."""
        thought_display = truncate(thought, self.width - 20)
        
        if self.use_colors:
            label = colorize("Thinking:", Colors.CYAN)
        else:
            label = "Thinking:"
        
        self._print(f"  {Symbols.THINKING} {label} {thought_display}")
        
        if confidence > 0:
            conf_color = Colors.GREEN if confidence > 0.7 else Colors.YELLOW
            conf_text = f"(confidence: {confidence:.0%})"
            if self.use_colors:
                conf_text = colorize(conf_text, conf_color)
            self._print(f"     {conf_text}")
    
    def show_action(self, tool_name: str, parameters: Dict[str, Any] = None) -> None:
        """Display action being taken."""
        params_str = ""
        if parameters:
            params_list = [f"{k}={repr(v)[:30]}" for k, v in list(parameters.items())[:3]]
            params_str = f"({', '.join(params_list)})"
        
        if self.use_colors:
            label = colorize("Action:", Colors.YELLOW)
            tool = colorize(tool_name, Colors.BOLD)
        else:
            label = "Action:"
            tool = tool_name
        
        self._print(f"  {Symbols.ACTION} {label} {tool}{params_str}")
    
    def show_result(self, result: str, success: bool = True) -> None:
        """Display action result."""
        result_display = truncate(str(result), self.width - 20)
        
        if success:
            symbol = Symbols.CHECK
            color = Colors.GREEN
        else:
            symbol = Symbols.CROSS
            color = Colors.RED
        
        if self.use_colors:
            result_display = colorize(result_display, color)
        
        self._print(f"  {symbol} Result: {result_display}")
    
    def show_reflection(self, assessment: str, quality: float = 0.0, needs_correction: bool = False) -> None:
        """Display reflection result."""
        if self.use_colors:
            label = colorize("Reflection:", Colors.BLUE)
        else:
            label = "Reflection:"
        
        self._print(f"  {Symbols.REFLECT} {label} {truncate(assessment, 60)}")
        
        if quality > 0:
            quality_bar = self._quality_indicator(quality)
            self._print(f"     Quality: {quality_bar}")
        
        if needs_correction:
            warning = "Needs correction"
            if self.use_colors:
                warning = colorize(warning, Colors.YELLOW)
            self._print(f"     {Symbols.WARNING} {warning}")
    
    def show_plan(self, steps: List[str]) -> None:
        """Display execution plan."""
        self._print(f"\n  {Symbols.PLAN} Plan:")
        for i, step in enumerate(steps, 1):
            step_display = truncate(step, self.width - 10)
            self._print(f"     {i}. {step_display}")
        print()
    
    def show_goal_progress(self, progress: float, message: str = "") -> None:
        """Display goal progress."""
        bar = self._quality_indicator(progress)
        msg = f" - {message}" if message else ""
        self._print(f"  {Symbols.GOAL} Progress: {bar}{msg}")
    
    def complete(self, message: str = "Complete!", success: bool = True) -> None:
        """Complete the agent session."""
        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        print()
        if success:
            symbol = Symbols.SUCCESS
            color = Colors.GREEN
            status = "SUCCESS"
        else:
            symbol = Symbols.FAILURE
            color = Colors.RED
            status = "FAILED"
        
        self._header(f"{symbol} {status}")
        
        if self.use_colors:
            message = colorize(message, color, Colors.BOLD)
        
        self._print(f"\n{message}")
        self._print(f"{Symbols.CLOCK} Duration: {elapsed:.2f}s")
        self._print(f"Iterations: {self.iteration}\n")
    
    def error(self, error_message: str) -> None:
        """Display an error."""
        if self.use_colors:
            error_message = colorize(error_message, Colors.RED)
        self._print(f"  {Symbols.FAILURE} Error: {error_message}")
    
    def info(self, message: str) -> None:
        """Display info message."""
        if self.use_colors:
            message = colorize(message, Colors.DIM)
        self._print(f"  {Symbols.INFO} {message}")
    
    def _quality_indicator(self, value: float) -> str:
        """Create a quality indicator bar."""
        width = 10
        filled = int(width * value)
        empty = width - filled
        
        bar = Symbols.PROGRESS_FULL * filled + Symbols.PROGRESS_EMPTY * empty
        pct = f"{value:.0%}"
        
        if self.use_colors:
            if value >= 0.7:
                bar = colorize(bar, Colors.GREEN)
            elif value >= 0.4:
                bar = colorize(bar, Colors.YELLOW)
            else:
                bar = colorize(bar, Colors.RED)
        
        return f"[{bar}] {pct}"


# ============================================================================
# LIVE DISPLAY (for async updates)
# ============================================================================

class LiveDisplay:
    """
    Live-updating display for real-time agent monitoring.
    
    USAGE:
        display = LiveDisplay()
        display.start()
        
        display.update_status("Thinking...")
        display.update_progress(0.5)
        display.add_log("Found 3 results")
        
        display.stop()
    """
    
    def __init__(self, refresh_rate: float = 0.1):
        self.refresh_rate = refresh_rate
        self.status = ""
        self.progress = 0.0
        self.logs: List[str] = []
        self.max_logs = 5
        self._running = False
    
    def start(self) -> None:
        """Start the live display."""
        self._running = True
        print("\033[?25l")  # Hide cursor
    
    def stop(self) -> None:
        """Stop the live display."""
        self._running = False
        print("\033[?25h")  # Show cursor
        print()
    
    def update_status(self, status: str) -> None:
        """Update status line."""
        self.status = status
        self._render()
    
    def update_progress(self, progress: float) -> None:
        """Update progress (0.0 - 1.0)."""
        self.progress = progress
        self._render()
    
    def add_log(self, message: str) -> None:
        """Add a log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        self._render()
    
    def _render(self) -> None:
        """Render the display."""
        if not self._running:
            return
        
        # Move cursor up and clear
        lines_to_clear = self.max_logs + 3
        print(f"\033[{lines_to_clear}A\033[J", end="")
        
        # Status line
        print(f"Status: {self.status}")
        
        # Progress bar
        bar = ProgressBar(total=100, width=50)
        bar.current = int(self.progress * 100)
        bar._render()
        print()
        
        # Recent logs
        print("Recent activity:")
        for log in self.logs:
            print(f"  {log}")
        
        # Pad remaining lines
        for _ in range(self.max_logs - len(self.logs)):
            print()
