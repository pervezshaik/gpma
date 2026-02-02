# GPMA Complete Tutorial: Building Intelligent Multi-Agent Systems

This comprehensive tutorial teaches you how to build sophisticated AI agents and multi-agent systems using GPMA (General Purpose Multi-Agent) framework.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Core Concepts](#core-concepts)
4. [Building Your First Agent](#building-your-first-agent)
5. [Agentic Capabilities](#agentic-capabilities)
6. [The New Developer Experience](#the-new-developer-experience)
7. [Multi-Agent Orchestration](#multi-agent-orchestration)
8. [Tools and Capabilities](#tools-and-capabilities)
9. [Real-World Examples](#real-world-examples)
10. [Advanced Patterns](#advanced-patterns)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Introduction

GPMA is a Python framework for building intelligent multi-agent systems with autonomous capabilities. Unlike traditional LLM chains, GPMA agents can:

- **Reason** about complex goals
- **Plan** multi-step approaches
- **Reflect** on their outputs
- **Adapt** to obstacles
- **Collaborate** with other agents

### What Makes GPMA Special?

```
Traditional LLM App:     Prompt → LLM → Response
GPMA Agent:              Goal → Plan → Act → Reflect → Adapt
Multi-Agent System:      Goal → Orchestrate → Collaborate → Synthesize
```

---

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/gpma.git
cd gpma

# Install dependencies
pip install -r requirements.txt

# Optional: Install for web capabilities
pip install playwright
playwright install chromium
```

### Setup LLM Provider

GPMA supports various LLM providers. Here's how to set up Ollama:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1
# or
ollama pull glm-4.7-flash
```

---

## Core Concepts

### 1. Agents

An agent is an autonomous entity with:
- **Capabilities**: What it can do
- **Tools**: How it does things
- **Memory**: What it remembers
- **State**: Current status

```python
from gpma.core import BaseAgent, AgentCapability

class MyAgent(BaseAgent):
    @property
    def capabilities(self):
        return [
            AgentCapability("research", "Research topics", ["search", "find"])
        ]
    
    async def process(self, task):
        # Agent logic here
        return TaskResult(success=True, data="Result")
```

### 2. The Agentic Loop

The core of GPMA's intelligence - the ReAct pattern:

```
1. OBSERVE - What's the current state?
2. THINK - What should I do next? (LLM reasoning)
3. ACT - Execute the chosen action
4. REFLECT - Did it work? Should I continue?
5. REPEAT - Until goal is achieved
```

### 3. Orchestrator

Coordinates multiple agents:

```
User Request → Orchestrator → Route to Best Agent → Execute → Synthesize
```

---

## Building Your First Agent

### The Old Way (Verbose)

```python
from gpma.core import SimpleAgenticAgent, AgentConfig, AgentMode
from gpma.llm.providers import OllamaProvider

# Define actions
async def search(query: str) -> str:
    return f"Search results for: {query}"

async def analyze(text: str) -> str:
    return f"Analysis of: {text}"

# Create config
config = AgentConfig(
    mode=AgentMode.PROACTIVE,
    enable_planning=True,
    enable_reflection=True,
    max_iterations=10
)

# Create agent
agent = SimpleAgenticAgent(
    name="ResearchAgent",
    llm_provider=OllamaProvider(model="llama3.1"),
    config=config,
    actions={
        "search": search,
        "analyze": analyze
    },
    capability_list=[
        AgentCapability("research", "Research topics", ["search", "find"])
    ]
)

# Use agent
result = await agent.pursue_goal("Research Python async patterns")
```

### The New Way (AgentBuilder)

```python
from gpma.core import AgentBuilder, AgentMode
from gpma.llm.providers import OllamaProvider

provider = OllamaProvider(model="llama3.1")

agent = (AgentBuilder("ResearchAgent")
    .with_llm(provider)
    .with_mode(AgentMode.PROACTIVE)
    .add_tool("search", search, "Search for information")
    .add_tool("analyze", analyze, "Analyze content")
    .enable_reflection()
    .enable_planning()
    .with_max_iterations(10)
    .build())

result = await agent.pursue_goal("Research Python async patterns")
```

**80% less code!**

---

## Agentic Capabilities

### 1. Autonomous Goal Pursuit

Agents pursue goals intelligently, not just execute tasks:

```python
# Task-oriented (old)
await agent.process("Summarize this text")

# Goal-oriented (new)
await agent.pursue_goal("Create a comprehensive summary that highlights key insights")
```

### 2. Planning

Agents break down complex goals:

```python
goal = "Research and compare AI frameworks"

# Agent automatically creates plan:
# 1. Search for AI frameworks
# 2. Analyze each framework
# 3. Compare features
# 4. Synthesize comparison
```

### 3. Self-Reflection

Agents evaluate and improve their work:

```python
# Agent reflects:
# "Is this summary comprehensive enough?"
# "Did I miss any important frameworks?"
# "Should I add more details?"
```

### 4. Goal Decomposition

Complex goals are broken into subgoals:

```python
goal = "Build a weather prediction system"

# Decomposes to:
# - Research weather data sources
# - Design prediction model
# - Implement data pipeline
# - Test accuracy
# - Deploy system
```

---

## The New Developer Experience

### 1. @auto_tool Decorator

Automatically create tools from functions:

```python
from gpma.tools import auto_tool

@auto_tool("Search the web for information")
async def search_web(query: str, max_results: int = 5) -> str:
    """Search and return relevant information.
    
    Args:
        query: The search query
        max_results: Maximum results to return
    """
    # Implementation
    return results

# The decorator creates an AgenticTool with:
# - Automatic parameter schema from type hints
# - Description from docstring
# - Validation and error handling
```

### 2. Agent Templates

Pre-built agents for common use cases:

```python
from gpma.templates import ResearchAgentTemplate, CodingAgentTemplate

# Quick research agent
research_agent = ResearchAgentTemplate.create(provider)

# Custom research agent
agent = (ResearchAgentTemplate.builder(provider)
    .with_name("CustomResearcher")
    .add_tool("custom_search", my_search_func)
    .build())

# Available templates:
# - ResearchAgentTemplate: Search, analyze, summarize
# - CodingAgentTemplate: Generate, review, test code
# - DataAnalystTemplate: Load, analyze, visualize data
# - WriterAgentTemplate: Draft, edit, style content
```

### 3. Rich Console UI

Beautiful terminal output for monitoring:

```python
from gpma.core import AgentConsole

console = AgentConsole()
console.start_agent("ResearchAgent", "Research AI trends")

console.start_iteration(1, 3)
console.show_thinking("Analyzing the goal...")
console.show_action("search", {"query": "AI trends 2024"})
console.show_result("Found 10 articles", success=True)

console.complete("Research complete!")
```

Output:
```
================================================================================
                               🚀 ResearchAgent
================================================================================

🎯 Goal: Research AI trends
⏱️ Started: 10:30:00

[██████████░░░░░░░░░░░░░░░░░░░░] 33% | 1/3
  → Iteration 1/3
  ----------------------------------------------------------------------------
  🤔 Thinking: Analyzing the goal...
  ⚡ Action: search(query='AI trends 2024')
  ✓ Result: Found 10 articles

================================================================================
                                   ✅ SUCCESS
================================================================================

Research complete!
⏱️ Duration: 2.5s
Iterations: 3
```

### 4. Observability

Real-time event tracking:

```python
from gpma.core import AgentObserver, EventType, ConsoleFormatter

# Create observer
observer = AgentObserver()
observer.add_formatter(ConsoleFormatter())

# Subscribe to events
observer.on(EventType.THINKING, lambda e: print(f"🤔 {e.content}"))
observer.on(EventType.ACTION_START, lambda e: print(f"⚡ {e.tool_name}"))

# Attach to agent
observer.attach(agent)

# Now see what agent is doing in real-time!
```

---

## Multi-Agent Orchestration

### 1. The Orchestrator Pattern

```python
from gpma.core import Orchestrator

# Create specialized agents
researcher = ResearchAgentTemplate.create(provider, name="Researcher")
analyzer = DataAnalystTemplate.create(provider, name="Analyst")
writer = WriterAgentTemplate.create(provider, name="Writer")

# Create orchestrator
orchestrator = Orchestrator()

# Register agents
orchestrator.register(researcher)
orchestrator.register(analyzer)
orchestrator.register(writer)

# Execute complex task
result = await orchestrator.execute(
    "Research AI trends, analyze the data, and write a report"
)

# Orchestrator:
# 1. Breaks down task
# 2. Routes to appropriate agents
# 3. Coordinates execution
# 4. Synthesizes results
```

### 2. Dynamic Agent Creation

```python
from gpma.core import DynamicOrchestrator

# Create orchestrator with agent factories
orchestrator = DynamicOrchestrator()

# Register agent factory
orchestrator.register_factory(
    "research",
    lambda: ResearchAgentTemplate.create(provider)
)

# Create agents on demand
agent = await orchestrator.create_agent("research", "TaskResearcher")
```

### 3. Agent Communication

```python
# Agents can communicate through the message bus
from gpma.core import MessageBus, Message, MessageType

bus = MessageBus()

# Agent sends message
await bus.send(Message(
    sender="Researcher",
    receiver="Analyst",
    content="Research data attached",
    type=MessageType.DATA,
    payload={"data": research_results}
))

# Agent receives message
message = await bus.receive("Analyst")
```

---

## Tools and Capabilities

### 1. Built-in Tools

```python
from gpma.tools import get_default_tools

# Get all production-grade tools
tools = get_default_tools()

# Available tools:
# - calculator: Safe math evaluation
# - search: Web search with multiple engines
# - knowledge_base: Search knowledge base with web fallback
# - file_operations: Read, write, list files
```

### 2. Custom Tools

```python
from gpma.tools import AgenticTool

# Create custom tool
weather_tool = AgenticTool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "city": {
            "type": "string",
            "description": "City name",
            "required": True
        }
    },
    function=get_weather_data,
    category="weather",
    timeout=10.0
)

# Register with agent
agent.register_tool(weather_tool)
```

### 3. Tool Composition

```python
# Tools can use other tools
@auto_tool("Comprehensive research")
async def deep_research(topic: str) -> str:
    """Perform comprehensive research on a topic."""
    
    # Use search tool
    search_results = await search(topic, max_results=10)
    
    # Use analyze tool
    analysis = await analyze(search_results)
    
    # Use summarize tool
    summary = await summarize(analysis)
    
    return summary
```

---

## Real-World Examples

### Example 1: Research Assistant

```python
from gpma.core import AgentBuilder
from gpma.templates import ResearchAgentTemplate
from gpma.tools import auto_tool

# Custom search tool
@auto_tool("Search academic papers")
async def search_papers(query: str, field: str = "computer_science") -> str:
    """Search academic papers for research."""
    # Implementation using arXiv API or similar
    return f"Found 5 papers on {query} in {field}"

# Create research assistant
research_assistant = (AgentBuilder("ResearchAssistant")
    .with_llm(provider)
    .add_tool("search_papers", search_papers)
    .add_capability("academic_research", "Research academic papers")
    .enable_planning()
    .enable_reflection()
    .build())

# Use it
result = await research_assistant.pursue_goal(
    "Find recent papers on transformer architectures and summarize key innovations"
)
```

### Example 2: Code Review Agent

```python
from gpma.templates import CodingAgentTemplate
from gpma.tools import auto_tool

@auto_tool("Analyze code quality")
async def analyze_code(code: str, language: str = "python") -> str:
    """Analyze code for quality issues."""
    # Implementation using static analysis
    return f"Code analysis: 3 issues found, 2 suggestions"

@auto_tool("Run tests")
async def run_tests(code: str, test_type: str = "unit") -> str:
    """Run tests on code."""
    # Implementation
    return f"Tests passed: 8/10"

# Create code reviewer
reviewer = (CodingAgentTemplate.builder(provider)
    .with_name("CodeReviewer")
    .add_tool("analyze_code", analyze_code)
    .add_tool("run_tests", run_tests)
    .build())

# Review code
result = await reviewer.pursue_goal(
    "Review this Python code for quality, security, and test coverage"
)
```

### Example 3: Multi-Agent Data Pipeline

```python
from gpma.core import Orchestrator
from gpma.templates import DataAnalystTemplate, ResearchAgentTemplate, WriterAgentTemplate

# Create specialized agents
collector = ResearchAgentTemplate.create(provider, name="DataCollector")
analyzer = DataAnalystTemplate.create(provider, name="DataAnalyzer")
reporter = WriterAgentTemplate.create(provider, name="DataReporter")

# Create orchestrator
orchestrator = Orchestrator()
orchestrator.register(collector, analyzer, reporter)

# Execute data pipeline
result = await orchestrator.execute(
    "Collect sales data for Q1, analyze trends, and generate executive report"
)

# Result includes:
# - Data collection summary
# - Analysis insights
# - Final report
# - Agent collaboration logs
```

---

## Advanced Patterns

### 1. Hierarchical Agents

```python
# Manager agent with team
manager = (AgentBuilder("ProjectManager")
    .with_llm(provider)
    .add_tool("delegate", delegate_to_specialist)
    .add_capability("management", "Coordinate team efforts")
    .build())

# Specialist agents
specialists = {
    "research": ResearchAgentTemplate.create(provider),
    "coding": CodingAgentTemplate.create(provider),
    "analysis": DataAnalystTemplate.create(provider)
}

async def delegate_to_specialist(task_type: str, task: str) -> str:
    specialist = specialists[task_type]
    return await specialist.pursue_goal(task)
```

### 2. Memory-Augmented Agents

```python
from gpma.core import LongTermMemory, ShortTermMemory

# Create agent with memory
agent = (AgentBuilder("MemoryAgent")
    .with_llm(provider)
    .with_memory(ShortTermMemory(max_items=100))
    .with_memory(LongTermMemory(storage_path="./agent_memory"))
    .build())

# Agent remembers across sessions
await agent.pursue_goal("Remember my preferences")
# Later...
await agent.pursue_goal("Use my preferences to customize response")
```

### 3. Tool-Using Agents

```python
# Agent that can create and use tools
@auto_tool("Create custom tool")
async def create_tool(description: str, code: str) -> str:
    """Create a new tool from code."""
    # Dynamically create and register tool
    new_tool = eval(code)  # In production, use safe execution
    agent.register_tool(new_tool)
    return f"Tool created: {description}"

tool_creator = (AgentBuilder("ToolCreator")
    .with_llm(provider)
    .add_tool("create_tool", create_tool)
    .build())
```

---

## Best Practices

### 1. Agent Design

- **Single Responsibility**: Each agent should have one primary purpose
- **Clear Capabilities**: Define what the agent can do explicitly
- **Error Handling**: Always handle errors gracefully
- **Resource Limits**: Set timeouts and retry limits

### 2. Tool Design

- **Descriptive Names**: Use clear, action-oriented names
- **Type Hints**: Always provide type hints for auto-tool
- **Documentation**: Write clear docstrings
- **Validation**: Validate inputs and handle edge cases

### 3. Prompt Engineering

- **Clear Goals**: Be specific about what you want
- **Context**: Provide relevant context
- **Constraints**: Set clear boundaries and limits
- **Examples**: Give examples of desired output

### 4. Multi-Agent Systems

- **Clear Roles**: Define each agent's role clearly
- **Communication**: Establish communication protocols
- **Coordination**: Use orchestrator for complex tasks
- **Monitoring**: Track agent interactions and performance

---

## Troubleshooting

### Common Issues

1. **Agent Not Responding**
   - Check LLM provider connection
   - Verify model is pulled (for Ollama)
   - Check API keys and permissions

2. **Tools Not Working**
   - Verify tool registration
   - Check parameter names match
   - Ensure async/await used correctly

3. **Memory Issues**
   - Clear old memories
   - Use persistent storage for long-term memory
   - Monitor memory usage

4. **Performance Issues**
   - Reduce max_iterations
   - Optimize tool execution
   - Use caching for repeated operations

### Debug Mode

```python
# Enable verbose output
agent = (AgentBuilder("DebugAgent")
    .with_llm(provider)
    .verbose(True)
    .build())

# Use console UI for debugging
console = AgentConsole()
console.start_agent("DebugAgent", "Debug goal")
```

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# GPMA will log detailed information
logger = logging.getLogger("gpma")
```

---

## Next Steps

1. **Run the Demos**: `python -m gpma.examples.agentic_demo`
2. **Study Examples**: Check the `examples/` directory
3. **Build Your Own**: Start with a simple agent
4. **Join Community**: Contribute and get help

### Resources

- [API Documentation](./docs/api.md)
- [Examples](./examples/)
- [Contributing Guide](./CONTRIBUTING.md)
- [Discord Community](https://discord.gg/gpma)

---

## Conclusion

GPMA provides a powerful framework for building intelligent multi-agent systems. With its agentic capabilities, developer-friendly APIs, and rich ecosystem, you can create sophisticated AI applications that reason, plan, and collaborate.

Remember:
- Start simple and build complexity gradually
- Use the new developer experience features (AgentBuilder, templates, @auto_tool)
- Monitor your agents with observability tools
- Leverage multi-agent orchestration for complex tasks

Happy building! 🚀
