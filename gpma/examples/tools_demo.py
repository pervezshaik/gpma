"""
Centralized Tools Demo

This example demonstrates the new centralized tool system in GPMA.

Key Features:
1. Unified BaseTool class for all tools
2. Central ToolRegistry for tool management
3. @tool decorator for easy tool creation
4. Format adapters for OpenAI/Anthropic
5. Tool injection into agents

Run with: python -m gpma.examples.tools_demo
"""

import asyncio
from typing import List


async def main():
    """Demonstrate the centralized tool system."""

    print("=" * 60)
    print("GPMA Centralized Tool System Demo")
    print("=" * 60)

    # =========================================================================
    # 1. Import the new tool system
    # =========================================================================
    print("\n1. Importing the centralized tool system...")

    from gpma.tools import (
        BaseTool,
        ToolParameter,
        ToolResult,
        ToolCategory,
        registry,
        tool,
        ToolBuilder,
        to_openai_format,
        to_anthropic_format,
    )

    print("   [OK] Imported successfully")

    # =========================================================================
    # 2. View available tools from registry
    # =========================================================================
    print("\n2. Viewing tools from the central registry...")

    all_tools = registry.get_all()
    print(f"   Total tools available: {len(all_tools)}")

    # List by category
    categories = registry.list_categories()
    print(f"   Categories: {categories}")

    # List tool names
    tool_list = registry.list_tools()
    print("\n   Available tools:")
    for t in tool_list:
        print(f"     - {t['name']} [{t['category']}]: {t['description'][:50]}...")

    # =========================================================================
    # 3. Use tools directly from registry
    # =========================================================================
    print("\n3. Using tools directly from registry...")

    # Get the calculator tool
    calc = registry.get("calculator")
    if calc:
        result = await calc.execute(expression="sqrt(16) + 5 * 2")
        print(f"   Calculator: sqrt(16) + 5 * 2 = {result.data if result.is_success else result.error}")

    # Get the search tool
    search = registry.get("search")
    if search:
        result = await search.execute(query="python", max_results=1)
        print(f"   Search for 'python': {result.data[:100] if result.is_success else result.error}...")

    # =========================================================================
    # 4. Create custom tools with @tool decorator
    # =========================================================================
    print("\n4. Creating custom tools with @tool decorator...")

    @tool(description="Get current weather for a city", category=ToolCategory.WEB)
    async def get_weather(city: str, units: str = "celsius") -> str:
        """Fetch current weather.

        Args:
            city: City name
            units: Temperature units (celsius or fahrenheit)
        """
        # Mock implementation
        temps = {"new york": 22, "london": 15, "tokyo": 28}
        temp = temps.get(city.lower(), 20)

        if units == "fahrenheit":
            temp = temp * 9 / 5 + 32

        return f"Weather in {city}: {temp}°{'F' if units == 'fahrenheit' else 'C'}"

    # The decorated function is now a BaseTool
    print(f"   Created tool: {get_weather.name}")
    print(f"   Category: {get_weather.category}")
    print(f"   Parameters: {[p.name for p in get_weather.parameters]}")

    # Use it
    result = await get_weather.execute(city="Tokyo")
    print(f"   Result: {result.data if result.is_success else result.error}")

    # =========================================================================
    # 5. Create tools with ToolBuilder (for complex configs)
    # =========================================================================
    print("\n5. Creating tools with ToolBuilder...")

    async def translate_text(text: str, target_lang: str) -> str:
        # Mock translation
        return f"[Translated to {target_lang}]: {text}"

    translator = (
        ToolBuilder("translate")
        .description("Translate text to another language")
        .category(ToolCategory.DATA)
        .parameter("text", "string", "Text to translate", required=True)
        .parameter("target_lang", "string", "Target language code",
                   required=True, enum=["en", "es", "fr", "de", "ja"])
        .timeout(30.0)
        .retry(2)
        .tag("translation", "language", "text")
        .function(translate_text)
        .register()  # Automatically registers in global registry
    )

    print(f"   Created and registered: {translator.name}")
    print(f"   Tags: {translator.tags}")

    result = await translator.execute(text="Hello world", target_lang="es")
    print(f"   Result: {result.data}")

    # =========================================================================
    # 6. Convert tools for LLM providers
    # =========================================================================
    print("\n6. Converting tools to LLM provider formats...")

    # Get some tools
    selected_tools = registry.create_toolset(["calculator", "search", "get_weather"])

    # Convert to OpenAI format
    openai_tools = to_openai_format(selected_tools)
    print(f"   OpenAI format ({len(openai_tools)} tools):")
    for ot in openai_tools:
        print(f"     - {ot['function']['name']}")

    # Convert to Anthropic format
    anthropic_tools = to_anthropic_format(selected_tools)
    print(f"   Anthropic format ({len(anthropic_tools)} tools):")
    for at in anthropic_tools:
        print(f"     - {at['name']}")

    # =========================================================================
    # 7. Inject tools into agents
    # =========================================================================
    print("\n7. Injecting tools into agents...")

    from gpma.core.base_agent import BaseAgent, AgentCapability, TaskResult as AgentTaskResult

    class DemoAgent(BaseAgent):
        @property
        def capabilities(self) -> List[AgentCapability]:
            return [AgentCapability("demo", "Demo agent", ["demo", "test"])]

        async def process(self, task):
            # Use injected tools
            if task.get("action") == "calculate":
                result = await self.use_tool("calculator", expression=task["input"])
                return AgentTaskResult(success=True, data=result)

            return AgentTaskResult(success=True, data="OK")

    # Create agent with tools from registry
    agent = DemoAgent(name="demo-agent")
    agent.inject_tools_from_registry(["calculator", "search"])

    print(f"   Agent '{agent.name}' has tools: {list(agent._tools.keys())}")

    # Or inject all tools in a category
    agent2 = DemoAgent(name="web-agent")
    agent2.inject_tools_from_registry(category="web")
    print(f"   Agent '{agent2.name}' has tools: {list(agent2._tools.keys())}")

    # =========================================================================
    # 8. View tool statistics
    # =========================================================================
    print("\n8. Viewing tool statistics...")

    stats = registry.get_stats()
    print(f"   Total tools: {stats['total_tools']}")
    print(f"   Total calls: {stats['total_calls']}")
    print(f"   Total time: {stats['total_time']:.3f}s")

    # Individual tool stats
    print("\n   Per-tool stats:")
    for ts in stats['tools'][:5]:  # Show first 5
        if ts['call_count'] > 0:
            print(f"     - {ts['name']}: {ts['call_count']} calls, "
                  f"avg {ts['avg_time']*1000:.1f}ms")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("""
Key takeaways:
1. Use 'registry' to access all tools centrally
2. Use '@tool' decorator to create new tools easily
3. Use 'ToolBuilder' for complex tool configurations
4. Use 'to_openai_format()' / 'to_anthropic_format()' for LLM APIs
5. Use 'agent.inject_tools_from_registry()' to add tools to agents
6. All tools share the same BaseTool interface with validation,
   timeout, and statistics tracking
""")


if __name__ == "__main__":
    asyncio.run(main())
