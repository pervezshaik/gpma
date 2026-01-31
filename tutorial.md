# Building Multi-Agent Systems: A Complete Tutorial

This tutorial will teach you how to build multi-agent AI systems using the GPMA framework. By the end, you'll understand the core concepts and be able to create your own agents.

## Table of Contents

1. [What is a Multi-Agent System?](#what-is-a-multi-agent-system)
2. [Core Concepts](#core-concepts)
3. [Building Your First Agent](#building-your-first-agent)
4. [Agent Communication](#agent-communication)
5. [The Orchestrator Pattern](#the-orchestrator-pattern)
6. [Memory Systems](#memory-systems)
7. [LLM Integration](#llm-integration)
8. [Creating a Personal Assistant](#creating-a-personal-assistant)
9. [Advanced Topics](#advanced-topics)

---

## What is a Multi-Agent System?

A multi-agent system (MAS) is a collection of autonomous agents that work together to solve complex problems. Think of it like a team where each member has specialized skills:

```
┌─────────────────────────────────────────────────────┐
│                     USER REQUEST                    │
│         "Research AI trends and summarize"          │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                     │
│   Breaks down the task, assigns to specialists      │
└─────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  Web Browser  │ │   Research    │ │   Summary     │
│    Agent      │ │    Agent      │ │    Agent      │
│               │ │               │ │               │
│ Fetches URLs  │ │Analyzes data  │ │ Synthesizes   │
└───────────────┘ └───────────────┘ └───────────────┘
```

### Why Multi-Agent?

1. **Specialization**: Each agent is an expert at one thing
2. **Scalability**: Add more agents as needed
3. **Reliability**: If one fails, others can continue
4. **Flexibility**: Mix and match agents for different tasks

---

## Core Concepts

### 1. Agents

An agent is an autonomous unit that:
- Has specific **capabilities** (what it can do)
- Uses **tools** to perform actions
- Has **memory** to store context
- Communicates via **messages**

```python
from gpma import BaseAgent, AgentCapability, TaskResult

class MyAgent(BaseAgent):
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="my_task",
                description="Does something useful",
                keywords=["do", "make", "create"]
            )
        ]

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        # Do the work here
        result = self._do_something(task["input"])
        return TaskResult(success=True, data=result)
```

### 2. Tools

Tools are functions that agents use to interact with the world:

```python
from gpma.core.base_agent import Tool

# Define a tool
fetch_tool = Tool(
    name="fetch_url",
    description="Fetches content from a URL",
    function=fetch_url_impl,
    parameters={"url": "The URL to fetch"}
)

# Register with agent
agent.register_tool(fetch_tool)

# Use in processing
result = await agent.use_tool("fetch_url", url="https://example.com")
```

### 3. Message Bus

Agents communicate through a message bus (pub/sub pattern):

```python
from gpma.core.message_bus import MessageBus, Message, MessageType

bus = MessageBus()

# Subscribe to messages
async def handler(msg):
    print(f"Received: {msg.content}")

bus.subscribe("my_agent", handler)

# Send a message
await bus.publish(Message(
    sender="other_agent",
    receiver="my_agent",
    content={"data": "hello"},
    msg_type=MessageType.REQUEST
))
```

### 4. Memory

Agents have both short-term and long-term memory:

```python
from gpma.core.memory import CompositeMemory

memory = CompositeMemory(stm_capacity=100)

# Store temporarily
memory.store("current_task", "analyzing data")

# Store permanently
memory.store("user_preference", "dark_mode", long_term=True)

# Retrieve
value = memory.retrieve("user_preference")
```

---

## Building Your First Agent

Let's create a simple agent step by step:

### Step 1: Define the Agent Class

```python
from typing import Any, Dict, List
from gpma import BaseAgent, AgentCapability
from gpma.core.base_agent import TaskResult, Tool

class WeatherAgent(BaseAgent):
    """An agent that provides weather information."""

    def __init__(self, name: str = None):
        super().__init__(name or "Weather")
        # Register any tools
        self.register_tool(Tool(
            name="get_weather",
            description="Get current weather",
            function=self._get_weather,
            parameters={"city": "City name"}
        ))
```

### Step 2: Define Capabilities

Capabilities tell the orchestrator what this agent can do:

```python
    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="weather",
                description="Get weather information",
                keywords=["weather", "temperature", "forecast", "rain", "sunny"],
                priority=2  # Higher = more preferred
            )
        ]
```

### Step 3: Implement Processing Logic

```python
    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """Handle weather requests."""
        action = task.get("action", "current")
        city = task.get("input", "New York")

        try:
            if action == "current":
                weather = await self.use_tool("get_weather", city=city)
                return TaskResult(
                    success=True,
                    data={
                        "city": city,
                        "weather": weather,
                        "formatted": f"Weather in {city}: {weather}"
                    }
                )
            else:
                return TaskResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            return TaskResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def _get_weather(self, city: str) -> Dict[str, Any]:
        """Tool implementation."""
        # In reality, this would call a weather API
        return {
            "temperature": 72,
            "condition": "sunny",
            "humidity": 45
        }
```

### Step 4: Use the Agent

```python
async def main():
    agent = WeatherAgent()
    await agent.initialize()

    result = await agent.run_task({
        "action": "current",
        "input": "San Francisco"
    })

    if result.success:
        print(result.data["formatted"])
    else:
        print(f"Error: {result.error}")

    await agent.shutdown()

asyncio.run(main())
```

---

## Agent Communication

Agents can communicate directly or through broadcasts:

### Direct Communication

```python
# In an agent's process method:
async def process(self, task):
    # Ask another agent for help
    await self.send_message(
        to_agent="web_browser",
        content={"action": "fetch", "url": "https://example.com"},
        msg_type="request"
    )

    # Wait for response
    response = await self.receive_message(timeout=30)
    if response:
        # Use the response
        pass
```

### Request-Response Pattern

```python
# Using the message bus directly
response = await bus.request(
    Message(
        sender="research_agent",
        receiver="web_agent",
        content={"url": "https://example.com"},
        msg_type=MessageType.REQUEST
    ),
    timeout=30.0
)

if response:
    print(f"Got response: {response.content}")
```

### Broadcasting

```python
# Send to all agents
await bus.publish(Message(
    sender="system",
    receiver="*",  # Broadcast
    content="System is shutting down",
    msg_type=MessageType.BROADCAST
))
```

---

## The Orchestrator Pattern

The orchestrator is the brain that coordinates agents:

```python
from gpma import Orchestrator, WebBrowserAgent, ResearchAgent

# Create orchestrator
orchestrator = Orchestrator(memory_path="./memory.json")

# Register agents
orchestrator.register_agent(WebBrowserAgent(), priority=2)
orchestrator.register_agent(ResearchAgent(), priority=3)

# Initialize all agents
await orchestrator.initialize_agents()

# Execute a request - orchestrator routes to best agent
result = await orchestrator.execute(
    "Research quantum computing",
    context={}
)

print(result.data)
```

### How Routing Works

1. User sends request: "Research quantum computing"
2. Orchestrator scores each agent's capabilities
3. Best match (ResearchAgent with keywords ["research"]) is selected
4. Task is sent to that agent
5. Result is returned

```python
# You can also find agents manually
agent_name = orchestrator.find_best_agent("fetch https://example.com")
print(f"Best agent: {agent_name}")  # "WebBrowser"

# Or get all capable agents
agents = orchestrator.find_capable_agents("research AI trends")
print(f"Capable agents: {agents}")  # ["Researcher", "WebBrowser"]
```

### Execution Strategies

For complex tasks with multiple subtasks:

```python
from gpma.core.orchestrator import ExecutionStrategy

# Sequential: One at a time
result = await orchestrator.execute(
    "First search for X, then analyze Y",
    strategy=ExecutionStrategy.SEQUENTIAL
)

# Parallel: All at once
result = await orchestrator.execute(
    "Get weather for NYC, LA, and Chicago",
    strategy=ExecutionStrategy.PARALLEL
)

# Pipeline: Output feeds into next
result = await orchestrator.execute(
    "Fetch page then summarize",
    strategy=ExecutionStrategy.PIPELINE
)
```

---

## Memory Systems

GPMA provides three types of memory:

### Short-Term Memory (STM)

- Fast access, limited capacity
- Uses LRU eviction
- Items can expire (TTL)

```python
from gpma.core.memory import ShortTermMemory
from datetime import timedelta

stm = ShortTermMemory(capacity=100)

# Store with expiration
stm.store(
    "current_task",
    "analyzing data",
    ttl=timedelta(minutes=30)
)

# Store with tags for searching
stm.store(
    "search_results",
    [...],
    tags=["search", "web"]
)

# Search by tags
results = stm.search(tags=["web"])
```

### Long-Term Memory (LTM)

- Persistent storage
- No expiration
- Survives restarts

```python
from gpma.core.memory import LongTermMemory

ltm = LongTermMemory(storage_path="./memory.json")

# Store permanently
ltm.store(
    "user_preferences",
    {"theme": "dark", "language": "en"},
    tags=["user", "settings"]
)

# Still accessible after restart
prefs = ltm.retrieve("user_preferences")
```

### Composite Memory

Combines both for typical agent usage:

```python
from gpma.core.memory import CompositeMemory

memory = CompositeMemory(
    stm_capacity=500,
    ltm_path="./long_term.json"
)

# Default: short-term
memory.store("temp_data", "...")

# Explicit long-term
memory.store("important", "...", long_term=True)

# Retrieval checks both (STM first)
value = memory.retrieve("some_key")

# Promote important STM items to LTM
memory.promote_to_long_term("temp_data")
```

---

## LLM Integration

GPMA supports integration with Large Language Models (LLMs) to power intelligent agents. This enables natural language understanding, complex reasoning, and AI-powered task decomposition.

### Supported Providers

GPMA supports three LLM providers:

| Provider | Models | Use Case |
|----------|--------|----------|
| **OpenAI** | GPT-4, GPT-4o, GPT-3.5-turbo | Cloud-based, most capable |
| **Ollama** | Llama 2/3, Mistral, CodeLlama | Local, privacy-focused |
| **Azure OpenAI** | GPT-4, GPT-3.5-turbo | Enterprise, compliance |

### Quick Start with OpenAI

```python
from gpma.llm import OpenAIProvider, LLMAgent

# Create provider (uses OPENAI_API_KEY env var, or pass api_key)
provider = OpenAIProvider(model="gpt-4")

# Simple generation
response = await provider.generate(
    "What is Python?",
    system_prompt="You are a concise assistant."
)
print(response.content)
print(f"Tokens used: {response.total_tokens}")

# Don't forget to close when done
await provider.close()
```

### Quick Start with Ollama (Local)

```python
from gpma.llm import OllamaProvider

# Requires Ollama installed and running (ollama serve)
# Pull a model first: ollama pull llama2
provider = OllamaProvider(
    model="llama2",
    base_url="http://localhost:11434"
)

# Same API as OpenAI
response = await provider.generate("Explain recursion briefly")
print(response.content)

# Check available models
models = await provider.list_models()
print([m.get('name') for m in models])
```

### LLM Agent

The `LLMAgent` is a full-featured agent powered by an LLM:

```python
from gpma.llm import LLMAgent, OpenAIProvider

provider = OpenAIProvider(api_key="sk-...", model="gpt-4")

agent = LLMAgent(
    llm_provider=provider,
    system_prompt="You are a helpful coding assistant.",
    max_history=20  # Conversation memory
)

await agent.initialize()

# Chat mode (maintains conversation history)
result = await agent.run_task({
    "input": "What's the difference between a list and a tuple?",
    "action": "chat"
})
print(result.data)

# Ask a follow-up (context is maintained)
result = await agent.run_task({
    "input": "Which one should I use for constants?",
    "action": "chat"
})
print(result.data)

# Single completion (no history)
result = await agent.run_task({
    "input": "Write a haiku about Python",
    "action": "complete"
})

# Analysis mode
result = await agent.run_task({
    "input": "def foo(x): return x*2 if x > 0 else -x",
    "action": "analyze"
})

await agent.shutdown()
```

### Tool/Function Calling

LLM agents can call functions to interact with the world:

```python
from gpma.llm import LLMAgent, OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
agent = LLMAgent(llm_provider=provider, enable_tools=True)

# Define a tool
def get_weather(city: str) -> dict:
    """Get weather for a city."""
    # In reality, call a weather API
    return {"city": city, "temp": 72, "condition": "sunny"}

# Register the tool
agent.register_llm_tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            }
        },
        "required": ["city"]
    },
    function=get_weather
)

await agent.initialize()

# The LLM will automatically call the tool when needed
result = await agent.run_task({
    "input": "What's the weather in San Francisco?",
    "action": "chat"
})
print(result.data)  # "The weather in San Francisco is 72°F and sunny."
```

### Specialized LLM Agents

GPMA includes pre-built specialized agents:

```python
from gpma.llm import OpenAIProvider
from gpma.llm.agent import (
    CodeAssistantAgent,
    ResearchAssistantAgent,
    WritingAssistantAgent
)

provider = OpenAIProvider(api_key="sk-...")

# Code assistant - optimized for programming tasks
code_agent = CodeAssistantAgent(llm_provider=provider)
result = await code_agent.run_task({
    "input": "Write a Python function to merge two sorted lists"
})

# Research assistant - analysis and synthesis
research_agent = ResearchAssistantAgent(llm_provider=provider)
result = await research_agent.run_task({
    "input": "Compare REST and GraphQL APIs",
    "action": "analyze"
})

# Writing assistant - content creation
writing_agent = WritingAssistantAgent(llm_provider=provider)
result = await writing_agent.run_task({
    "input": "Write a professional email declining a meeting"
})
```

### LLM-Powered Task Decomposition

Use an LLM to intelligently break down complex tasks:

```python
from gpma.llm import LLMTaskDecomposer, OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
decomposer = LLMTaskDecomposer(provider)

# Decompose a complex task
result = await decomposer.decompose(
    "Research AI trends, summarize the top 3 findings, and create a report"
)

print(f"Main Goal: {result['main_goal']}")
print(f"Strategy: {result['execution_strategy']}")

for subtask in result['subtasks']:
    print(f"\n{subtask.id}. {subtask.description}")
    print(f"   Action: {subtask.action}")
    print(f"   Agent: {subtask.agent_hint}")
    print(f"   Depends on: {subtask.dependencies}")
    print(f"   Complexity: {subtask.complexity}")
```

Example output:
```
Main Goal: Research AI trends and create a summary report
Strategy: pipeline

1. Search for recent AI trends articles
   Action: search
   Agent: web_browser
   Depends on: []
   Complexity: low

2. Analyze and extract key findings
   Action: analyze
   Agent: research
   Depends on: ['1']
   Complexity: medium

3. Summarize top 3 findings
   Action: write
   Agent: llm
   Depends on: ['2']
   Complexity: medium

4. Create formatted report
   Action: write
   Agent: llm
   Depends on: ['3']
   Complexity: low
```

### Streaming Responses

For real-time output, use streaming:

```python
from gpma.llm import OllamaProvider

provider = OllamaProvider(model="llama2")

# Stream tokens as they're generated
async for token in provider.generate_stream(
    "Write a short story about a robot",
    system_prompt="You are a creative writer."
):
    print(token, end="", flush=True)

print()  # Newline at end
```

### Multi-turn Conversations

For explicit control over conversation history:

```python
from gpma.llm import OpenAIProvider
from gpma.llm.providers import Message, MessageRole

provider = OpenAIProvider(api_key="sk-...")

messages = [
    Message(MessageRole.SYSTEM, "You are a helpful math tutor."),
    Message(MessageRole.USER, "What is 15% of 80?"),
]

response = await provider.chat(messages)
print(f"Assistant: {response.content}")

# Continue the conversation
messages.append(Message(MessageRole.ASSISTANT, response.content))
messages.append(Message(MessageRole.USER, "How did you calculate that?"))

response = await provider.chat(messages)
print(f"Assistant: {response.content}")
```

### Configuration Options

Fine-tune LLM behavior:

```python
from gpma.llm import OpenAIProvider, LLMConfig

config = LLMConfig(
    temperature=0.7,      # Creativity (0.0-2.0)
    max_tokens=2048,      # Maximum response length
    top_p=1.0,            # Nucleus sampling
    frequency_penalty=0,  # Reduce repetition
    presence_penalty=0,   # Encourage new topics
    stop=["\n\n"]         # Stop sequences
)

provider = OpenAIProvider(
    api_key="sk-...",
    model="gpt-4",
    config=config
)

# Or override per-request
response = await provider.generate(
    "Write a creative story",
    config=LLMConfig(temperature=1.2)  # More creative
)
```

### Factory Function

Use the factory for dynamic provider creation:

```python
from gpma.llm.providers import create_provider

# Create based on configuration
provider = create_provider("openai", model="gpt-4", api_key="sk-...")
provider = create_provider("ollama", model="llama2")
provider = create_provider("azure",
    endpoint="https://your-resource.openai.azure.com",
    deployment_name="gpt-4",
    api_key="..."
)
```

---

## Creating a Personal Assistant

Now let's put it all together. The `PersonalAssistant` combines multiple agents with optional LLM integration:

### Basic Usage (Without LLM)

```python
from gpma import PersonalAssistant

async def main():
    # Create assistant
    assistant = PersonalAssistant(
        name="Jarvis",
        memory_path="./jarvis_memory.json"
    )

    # Initialize (sets up all agents)
    await assistant.initialize()

    # Do research
    research = await assistant.research("AI trends 2024", depth="deep")
    print(research["synthesis"])

    # Browse web
    page = await assistant.browse("https://example.com")
    print(page["title"])

    # Search
    results = await assistant.search("Python tutorials")
    for r in results:
        print(f"- {r['title']}")

    # Memory
    assistant.remember("user_name", "Alice", permanent=True)
    name = assistant.recall("user_name")

    # Clean up
    await assistant.shutdown()

asyncio.run(main())
```

### With LLM Integration (Recommended)

```python
from gpma import PersonalAssistant
from gpma.llm import OpenAIProvider

async def main():
    # Create LLM provider
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4")

    # Create assistant with LLM
    assistant = PersonalAssistant(
        name="Jarvis",
        llm_provider=provider,  # Enable LLM capabilities
        memory_path="./jarvis_memory.json"
    )

    async with assistant:  # Auto initialize/shutdown
        # Natural language chat
        response = await assistant.chat("Tell me a joke about programming")
        print(response)

        # Ask questions (routes to best agent)
        answer = await assistant.ask("What is machine learning?")
        print(answer)

        # Check status
        status = assistant.get_status()
        print(f"Active agents: {status['agents']}")
        print(f"LLM enabled: {'llm' in assistant.get_agents()}")

asyncio.run(main())
```

### Using Factory Functions

For quick setup, use the factory functions:

```python
from gpma.personal_assistant import create_openai_assistant, create_ollama_assistant

# OpenAI-powered assistant
assistant = create_openai_assistant(
    api_key="sk-...",
    model="gpt-4"
)

# Local Ollama-powered assistant
assistant = create_ollama_assistant(
    model="llama2"  # or mistral, codellama, etc.
)

async with assistant:
    response = await assistant.chat("Hello!")
    print(response)
```

### Adding Custom Agents

```python
class EmailAgent(BaseAgent):
    @property
    def capabilities(self):
        return [AgentCapability(
            name="email",
            description="Send and read emails",
            keywords=["email", "mail", "send", "inbox"]
        )]

    async def process(self, task):
        # Email handling logic
        pass

# Add to assistant
async with PersonalAssistant() as assistant:
    email_agent = EmailAgent()
    await email_agent.initialize()
    assistant.add_agent(email_agent, priority=5)

    # Now it can handle email requests
    result = await assistant.ask("Send email to bob@example.com")
```

---

## Advanced Topics

### Dynamic Agent Spawning

Create agents on-demand:

```python
from gpma.core.orchestrator import DynamicOrchestrator

orchestrator = DynamicOrchestrator()

# Register factories
orchestrator.register_agent_factory("email", EmailAgent)
orchestrator.register_agent_factory("calendar", CalendarAgent)

# Agents are created when needed
result = await orchestrator.execute_with_spawn(
    "Send an email to the team"
)
# EmailAgent is automatically spawned
```

### Custom Task Decomposition

For intelligent task decomposition, use the built-in `LLMTaskDecomposer` (see [LLM Integration](#llm-integration) section) or create custom logic:

```python
from gpma.llm import LLMTaskDecomposer, SmartDecomposer, OpenAIProvider

# Basic LLM decomposer
provider = OpenAIProvider(api_key="sk-...")
decomposer = LLMTaskDecomposer(provider)

# Smart decomposer (aware of available agents)
smart_decomposer = SmartDecomposer(
    llm_provider=provider,
    available_agents=["web_browser", "research", "llm"]
)

# Decompose with agent awareness
result = await smart_decomposer.decompose(
    "Research Python best practices and create a guide",
    agent_capabilities={
        "web_browser": ["fetch", "search"],
        "research": ["analyze", "summarize"],
        "llm": ["write", "explain"]
    }
)

# Optimize execution plan
plan = await smart_decomposer.optimize_execution_plan(result["subtasks"])
print(f"Parallel groups: {plan['parallel_groups']}")
print(f"Can parallelize: {plan['can_parallelize']}")
```

### Error Handling and Retries

```python
class RobustAgent(BaseAgent):
    async def process(self, task):
        max_retries = 3

        for attempt in range(max_retries):
            try:
                result = await self._do_work(task)
                return TaskResult(success=True, data=result)
            except TransientError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                raise
            except PermanentError as e:
                return TaskResult(success=False, error=str(e))
```

### Monitoring and Logging

```python
# Get system stats
stats = orchestrator.get_stats()
print(f"Agents: {stats['agents_registered']}")
print(f"Active tasks: {stats['active_tasks']}")
print(f"Message bus: {stats['message_bus']}")

# Per-agent stats
for name, reg in orchestrator._agents.items():
    agent_stats = reg.agent.get_stats()
    print(f"{name}: {agent_stats['tasks_processed']} tasks, {agent_stats['errors']} errors")
```

---

## Next Steps

1. **Run the basic demos**: `python -m gpma.examples.demo`
2. **Run the LLM demos**:
   ```bash
   # All LLM demos
   python -m gpma.examples.llm_demo

   # OpenAI demos only
   python -m gpma.examples.llm_demo openai

   # Ollama (local) demos only
   python -m gpma.examples.llm_demo ollama

   # Quick start examples
   python -m gpma.examples.llm_demo quick
   ```
3. **Create your own agent**: Start with a simple capability
4. **Add LLM integration**: Use `OpenAIProvider` or `OllamaProvider` to power your agents
5. **Add persistence**: Use a real database for memory
6. **Scale up**: Deploy multiple orchestrators

### Prerequisites for LLM Integration

**For OpenAI:**
```bash
export OPENAI_API_KEY='sk-your-key-here'
```

**For Ollama (local models):**
```bash
# Install Ollama from https://ollama.ai
ollama pull llama2   # or mistral, codellama, phi, etc.
ollama serve         # Start the server (usually auto-starts)
```

Happy building!
