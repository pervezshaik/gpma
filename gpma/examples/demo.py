"""
GPMA Demo Script

This script demonstrates the key features of the General Purpose Multi-Agent System.

RUN THIS DEMO:
    python -m gpma.examples.demo

WHAT YOU'LL LEARN:
1. How to create and use the PersonalAssistant
2. How agents collaborate
3. How to add custom agents
4. How memory works
"""

import asyncio
import logging
from typing import Any, Dict, List

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import from GPMA
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from gpma import PersonalAssistant, BaseAgent, AgentCapability, WebBrowserAgent
from gpma.core.base_agent import TaskResult


# ============================================================================
# DEMO 1: Basic Usage
# ============================================================================

async def demo_basic_usage():
    """
    Demonstrates basic PersonalAssistant usage.
    """
    print("\n" + "="*60)
    print("DEMO 1: Basic Usage")
    print("="*60)

    # Create and initialize the assistant
    assistant = PersonalAssistant(name="MyAssistant")
    await assistant.initialize()

    try:
        # Check status
        status = assistant.get_status()
        print(f"\nAssistant Status:")
        print(f"  - Name: {status['name']}")
        print(f"  - Agents: {status['agents']}")
        print(f"  - Initialized: {status['initialized']}")

        # List available agents
        agents = assistant.get_agents()
        print(f"\nAvailable Agents: {agents}")

        # Ask a simple question (will route to appropriate agent)
        print("\n--- Asking a question ---")
        # Note: This will try to search the web
        result = await assistant.ask("What is Python programming language?")
        print(f"Answer: {result[:200]}...")

        # Use memory
        print("\n--- Using Memory ---")
        assistant.remember("user_preference", "dark_mode", permanent=True)
        assistant.remember("last_topic", "Python")

        recalled = assistant.recall("user_preference")
        print(f"Recalled preference: {recalled}")

        print("\nDemo 1 completed!")

    finally:
        await assistant.shutdown()


# ============================================================================
# DEMO 2: Web Browsing
# ============================================================================

async def demo_web_browsing():
    """
    Demonstrates web browsing capabilities.
    """
    print("\n" + "="*60)
    print("DEMO 2: Web Browsing")
    print("="*60)

    async with PersonalAssistant() as assistant:
        # Browse a webpage
        print("\n--- Fetching a webpage ---")
        page = await assistant.browse("https://httpbin.org/html")

        if "error" not in page:
            print(f"Title: {page.get('title', 'No title')}")
            print(f"Text length: {page.get('text_length', 0)} characters")
            print(f"Links found: {page.get('links_count', 0)}")
        else:
            print(f"Error: {page.get('error')}")

        # Search the web
        print("\n--- Searching the web ---")
        results = await assistant.search("Python programming", num_results=3)

        if results:
            print("Search Results:")
            for i, r in enumerate(results, 1):
                print(f"  {i}. {r.get('title', 'No title')}")
                print(f"     URL: {r.get('url', '')[:50]}...")
        else:
            print("No results found (search may be rate limited)")

        print("\nDemo 2 completed!")


# ============================================================================
# DEMO 3: Agent Collaboration
# ============================================================================

async def demo_agent_collaboration():
    """
    Demonstrates how agents work together.
    """
    print("\n" + "="*60)
    print("DEMO 3: Agent Collaboration")
    print("="*60)

    async with PersonalAssistant() as assistant:
        # Research uses WebBrowserAgent internally
        print("\n--- Research (agents collaborating) ---")
        print("Note: Research agent uses Web agent to fetch content")

        # This demonstrates the collaboration:
        # 1. Research agent generates search queries
        # 2. Research agent asks Web agent to search
        # 3. Research agent asks Web agent to fetch top results
        # 4. Research agent synthesizes findings

        # For demo, we'll just show the orchestrator status
        status = assistant.get_status()
        print("\nAgent Status:")
        for name, agent_status in status.get("agent_status", {}).items():
            print(f"  - {name}: {agent_status.get('state', 'unknown')}")
            print(f"    Capabilities: {agent_status.get('capabilities', [])}")

        print("\nDemo 3 completed!")


# ============================================================================
# DEMO 4: Creating a Custom Agent
# ============================================================================

class CalculatorAgent(BaseAgent):
    """
    A custom agent that performs calculations.

    This demonstrates how to create your own agent.
    """

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="calculate",
                description="Perform mathematical calculations",
                keywords=["calculate", "math", "compute", "add", "subtract", "multiply", "divide", "sum"],
                priority=5
            )
        ]

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a calculation request.

        Supported operations:
        - Basic arithmetic: 2 + 3, 10 - 5, 4 * 6, 20 / 4
        - Multiple numbers: sum 1 2 3 4 5
        """
        import re

        input_text = task.get("input", "")

        try:
            # Try to parse arithmetic expression
            # Extract numbers and operators
            numbers = [float(x) for x in re.findall(r'-?\d+\.?\d*', input_text)]

            if not numbers:
                return TaskResult(
                    success=False,
                    data=None,
                    error="No numbers found in input"
                )

            # Check for operation keywords
            input_lower = input_text.lower()

            if "sum" in input_lower or "add" in input_lower or "+" in input_text:
                result = sum(numbers)
                operation = "sum"
            elif "subtract" in input_lower or "-" in input_text:
                result = numbers[0] - sum(numbers[1:]) if len(numbers) > 1 else numbers[0]
                operation = "subtract"
            elif "multiply" in input_lower or "product" in input_lower or "*" in input_text:
                result = 1
                for n in numbers:
                    result *= n
                operation = "multiply"
            elif "divide" in input_lower or "/" in input_text:
                if len(numbers) >= 2 and numbers[1] != 0:
                    result = numbers[0] / numbers[1]
                    operation = "divide"
                else:
                    return TaskResult(
                        success=False,
                        data=None,
                        error="Division requires two numbers and non-zero divisor"
                    )
            elif "average" in input_lower or "mean" in input_lower:
                result = sum(numbers) / len(numbers)
                operation = "average"
            else:
                # Default to sum
                result = sum(numbers)
                operation = "sum"

            return TaskResult(
                success=True,
                data={
                    "operation": operation,
                    "numbers": numbers,
                    "result": result,
                    "formatted": f"{operation}({', '.join(map(str, numbers))}) = {result}"
                }
            )

        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error=str(e)
            )


async def demo_custom_agent():
    """
    Demonstrates adding a custom agent.
    """
    print("\n" + "="*60)
    print("DEMO 4: Custom Agent")
    print("="*60)

    async with PersonalAssistant() as assistant:
        # Create and add custom agent
        calc_agent = CalculatorAgent(name="Calculator")
        await calc_agent.initialize()
        assistant.add_agent(calc_agent, priority=5)

        print("\nCustom CalculatorAgent added!")
        print(f"Agents now: {assistant.get_agents()}")

        # Use the calculator agent
        print("\n--- Calculator Operations ---")

        # Direct agent usage
        result = await calc_agent.run_task({
            "action": "calculate",
            "input": "sum 10 20 30 40"
        })

        if result.success:
            print(f"Calculation: {result.data.get('formatted')}")

        # Another calculation
        result = await calc_agent.run_task({
            "action": "calculate",
            "input": "multiply 5 6 7"
        })

        if result.success:
            print(f"Calculation: {result.data.get('formatted')}")

        print("\nDemo 4 completed!")


# ============================================================================
# DEMO 5: Task Execution
# ============================================================================

async def demo_task_execution():
    """
    Demonstrates task execution capabilities.
    """
    print("\n" + "="*60)
    print("DEMO 5: Task Execution")
    print("="*60)

    from gpma import TaskExecutorAgent

    # Create task executor with restrictions
    executor = TaskExecutorAgent(
        allowed_paths=["."],
        command_whitelist=["echo", "dir", "ls", "pwd", "cd"]
    )
    await executor.initialize()

    try:
        # Get system info
        print("\n--- System Information ---")
        result = await executor.run_task({
            "action": "system_info"
        })

        if result.success:
            data = result.data
            print(f"Platform: {data.get('platform')}")
            print(f"Python: {data.get('python_version')}")
            print(f"CWD: {data.get('cwd')}")

            disk = data.get('disk', {})
            if disk:
                print(f"Disk: {disk.get('free_gb')}GB free of {disk.get('total_gb')}GB")

        # Run a safe command
        print("\n--- Running a command ---")
        result = await executor.run_task({
            "action": "run_command",
            "input": "echo Hello from GPMA!"
        })

        if result.success:
            print(f"Output: {result.data.get('stdout', '').strip()}")

        print("\nDemo 5 completed!")

    finally:
        await executor.shutdown()


# ============================================================================
# DEMO 6: Memory System
# ============================================================================

async def demo_memory_system():
    """
    Demonstrates the memory system.
    """
    print("\n" + "="*60)
    print("DEMO 6: Memory System")
    print("="*60)

    from gpma.core.memory import ShortTermMemory, LongTermMemory, CompositeMemory

    # Short-term memory
    print("\n--- Short-Term Memory ---")
    stm = ShortTermMemory(capacity=5)

    # Store some values
    stm.store("task1", "Completed web search")
    stm.store("task2", "Analyzed 3 websites")
    stm.store("task3", "Generated summary")

    print(f"Stored {len(stm)} items")
    print(f"Retrieving 'task2': {stm.retrieve('task2')}")

    # Search memory
    results = stm.search("web")
    print(f"Search 'web' found: {len(results)} items")

    # Long-term memory (in-memory, no file)
    print("\n--- Long-Term Memory ---")
    ltm = LongTermMemory()

    ltm.store("user_name", "Alice", tags=["user", "profile"])
    ltm.store("preference_theme", "dark", tags=["user", "settings"])

    print(f"User name: {ltm.retrieve('user_name')}")

    # Search by tags
    user_items = ltm.search(tags=["user"])
    print(f"Items with 'user' tag: {len(user_items)}")

    # Composite memory
    print("\n--- Composite Memory ---")
    memory = CompositeMemory(stm_capacity=10)

    memory.store("recent_query", "What is AI?")  # Short-term
    memory.store("api_key", "sk-xxx", long_term=True)  # Long-term

    # Retrieve from either
    print(f"Recent query: {memory.retrieve('recent_query')}")
    print(f"API key: {memory.retrieve('api_key')}")

    # Get recent context
    context = memory.get_context(limit=5)
    print(f"Recent context: {len(context)} items")

    print("\nDemo 6 completed!")


# ============================================================================
# DEMO 7: Message Bus
# ============================================================================

async def demo_message_bus():
    """
    Demonstrates inter-agent communication.
    """
    print("\n" + "="*60)
    print("DEMO 7: Message Bus")
    print("="*60)

    from gpma.core.message_bus import MessageBus, Message, MessageType

    bus = MessageBus()

    # Track received messages
    received = []

    # Create message handlers
    async def agent_a_handler(msg: Message):
        print(f"  Agent A received: '{msg.content}' from {msg.sender}")
        received.append(("A", msg))

    async def agent_b_handler(msg: Message):
        print(f"  Agent B received: '{msg.content}' from {msg.sender}")
        received.append(("B", msg))

    # Subscribe agents
    bus.subscribe("agent_a", agent_a_handler)
    bus.subscribe("agent_b", agent_b_handler)

    print("\n--- Direct Message ---")
    await bus.publish(Message(
        sender="agent_b",
        receiver="agent_a",
        content="Hello Agent A!",
        msg_type=MessageType.REQUEST
    ))

    print("\n--- Broadcast Message ---")
    await bus.publish(Message(
        sender="system",
        receiver="*",
        content="System announcement to all agents",
        msg_type=MessageType.BROADCAST
    ))

    # Stats
    print("\n--- Message Bus Stats ---")
    stats = bus.get_stats()
    print(f"Subscribers: {stats['subscribers']}")
    print(f"Total messages: {stats['total_messages']}")

    print("\nDemo 7 completed!")


# ============================================================================
# MAIN
# ============================================================================

async def run_all_demos():
    """Run all demos in sequence."""
    print("\n" + "="*60)
    print("GPMA - General Purpose Multi-Agent System Demos")
    print("="*60)

    demos = [
        ("Basic Usage", demo_basic_usage),
        ("Web Browsing", demo_web_browsing),
        ("Agent Collaboration", demo_agent_collaboration),
        ("Custom Agent", demo_custom_agent),
        ("Task Execution", demo_task_execution),
        ("Memory System", demo_memory_system),
        ("Message Bus", demo_message_bus),
    ]

    for name, demo_func in demos:
        try:
            await demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All demos completed!")
    print("="*60)


async def run_single_demo(demo_number: int):
    """Run a specific demo."""
    demos = {
        1: demo_basic_usage,
        2: demo_web_browsing,
        3: demo_agent_collaboration,
        4: demo_custom_agent,
        5: demo_task_execution,
        6: demo_memory_system,
        7: demo_message_bus,
    }

    if demo_number in demos:
        await demos[demo_number]()
    else:
        print(f"Demo {demo_number} not found. Available: 1-7")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        try:
            demo_num = int(sys.argv[1])
            asyncio.run(run_single_demo(demo_num))
        except ValueError:
            print("Usage: python demo.py [demo_number]")
            print("Demo numbers: 1-7")
    else:
        asyncio.run(run_all_demos())
