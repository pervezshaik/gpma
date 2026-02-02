# GPMA - General Purpose Multi-Agent System

A professional-grade, modular multi-agent framework for building AI-powered applications with LLM integration, autonomous agents, and advanced agentic capabilities.

## Features

- **Multi-Agent Architecture** - Orchestrated agents with specialized capabilities
- **LLM Integration** - OpenAI, Ollama (local), Azure OpenAI support
- **Agentic Capabilities** - Goal-driven behavior, planning, reflection, learning
- **Web Scraping** - Anti-bot evasion, JavaScript rendering via Playwright
- **Memory Systems** - Short-term, long-term, and composite memory
- **Observability** - Metrics, tracing, and monitoring built-in

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (Brain)                        │
│  - Routes tasks to appropriate agents                           │
│  - LLM-powered task decomposition                               │
│  - Manages agent lifecycle                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Web Browser  │    │   Research    │    │     LLM       │
│    Agent      │    │    Agent      │    │    Agent      │
│               │    │               │    │               │
│ - Fetch URLs  │    │ - Search web  │    │ - Chat        │
│ - Parse HTML  │    │ - Summarize   │    │ - Reason      │
│ - JavaScript  │    │ - Analyze     │    │ - Tool use    │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC CAPABILITIES                         │
│  - Goal Manager: Hierarchical goal tracking                     │
│  - Planner: Multi-step planning with dependencies               │
│  - Reflection: Self-evaluation and learning                     │
│  - Observability: Metrics, tracing, monitoring                  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/pervezshaik/gpma.git
cd gpma

# Install dependencies
pip install -r requirements.txt

# Optional: Install Playwright for JavaScript rendering
pip install playwright
playwright install chromium

# Optional: Install Ollama for local LLM
# Download from https://ollama.ai
ollama pull llama3.1
```

## Quick Start

### Basic Usage (No LLM)

```python
import asyncio
from gpma import PersonalAssistant

async def main():
    assistant = PersonalAssistant(name="MyAssistant")
    await assistant.initialize()

    # Web search
    results = await assistant.search("Python tutorials")
    print(results)

    # Browse a webpage
    page = await assistant.browse("https://example.com")
    print(page["title"])

    await assistant.shutdown()

asyncio.run(main())
```

### With Ollama (Local LLM)

```python
import asyncio
from gpma import PersonalAssistant
from gpma.llm import OllamaProvider

async def main():
    # Connect to local Ollama
    provider = OllamaProvider(model="llama3.1")

    assistant = PersonalAssistant(
        name="MyAssistant",
        llm_provider=provider
    )

    async with assistant:
        # Chat with LLM
        response = await assistant.chat("Explain Python in 2 sentences")
        print(response)

        # Ask questions (routes to best agent)
        answer = await assistant.ask("What is machine learning?")
        print(answer)

asyncio.run(main())
```

### With OpenAI

```python
from gpma import PersonalAssistant
from gpma.llm import OpenAIProvider

provider = OpenAIProvider(
    api_key="sk-...",  # or set OPENAI_API_KEY env var
    model="gpt-4"
)

assistant = PersonalAssistant(llm_provider=provider)
```

## Project Structure

```
gpma/
├── core/
│   ├── base_agent.py       # Base agent class
│   ├── orchestrator.py     # Agent coordinator
│   ├── message_bus.py      # Inter-agent communication
│   ├── memory.py           # Memory systems (STM, LTM)
│   ├── agentic_agent.py    # Advanced autonomous agent
│   ├── agentic_loop.py     # Observe-Think-Act-Reflect loop
│   ├── goal_manager.py     # Hierarchical goal tracking
│   ├── planner.py          # Multi-step planning
│   ├── reflection.py       # Self-evaluation & learning
│   ├── observability.py    # Metrics & tracing
│   └── console_ui.py       # Rich console interface
├── agents/
│   ├── web_browser.py      # Web browsing agent
│   ├── research.py         # Research & analysis agent
│   └── task_executor.py    # Task execution agent
├── llm/
│   ├── providers.py        # LLM providers (OpenAI, Ollama, Azure)
│   ├── agent.py            # LLM-powered agent
│   ├── decomposer.py       # LLM task decomposition
│   └── tools.py            # LLM tool definitions
├── tools/
│   ├── web_tools.py        # Web fetching, Playwright, search
│   ├── file_tools.py       # File operations
│   └── agentic_tools.py    # Advanced tool implementations
├── templates/
│   ├── coding.py           # Code generation templates
│   ├── research.py         # Research templates
│   ├── data.py             # Data processing templates
│   └── writer.py           # Content writing templates
├── personal_assistant.py   # Main assistant facade
└── examples/
    ├── demo.py             # Basic demos (7 examples)
    ├── llm_demo.py         # LLM integration demos
    └── agentic_demo.py     # Advanced agentic demos
```

## LLM Providers

| Provider | Models | Setup |
|----------|--------|-------|
| **Ollama** | llama3.1, gemma2, mistral, codellama | Local, free |
| **OpenAI** | gpt-4, gpt-4o, gpt-3.5-turbo | API key required |
| **Azure OpenAI** | gpt-4, gpt-3.5-turbo | Enterprise |

## Running Demos

```bash
# Run all basic demos
python -m gpma.examples.demo

# Run specific demo (1-7)
python -m gpma.examples.demo 1

# Run LLM demos
python -m gpma.examples.llm_demo

# Run with specific provider
python -m gpma.examples.llm_demo ollama
python -m gpma.examples.llm_demo openai
```

## Key Components

### Agents
Autonomous units with specialized capabilities:
- **WebBrowserAgent** - Fetch URLs, parse HTML, JavaScript rendering
- **ResearchAgent** - Search, analyze, synthesize information
- **TaskExecutorAgent** - File operations, command execution
- **LLMAgent** - Natural language understanding, reasoning

### Agentic Features
Advanced autonomous capabilities:
- **Goal Manager** - Track and decompose hierarchical goals
- **Planner** - Create multi-step execution plans
- **Reflection** - Learn from successes and failures
- **Agentic Loop** - Observe → Think → Act → Reflect cycle

### Memory Systems
- **Short-Term Memory (STM)** - LRU cache with TTL
- **Long-Term Memory (LTM)** - Persistent JSON storage
- **Composite Memory** - Unified interface for both

### Web Scraping
- User agent rotation
- Anti-bot evasion headers
- JavaScript rendering (Playwright)
- Multiple search engines (DuckDuckGo, Brave, Bing, Google)
- Proxy support

## Documentation

- [Tutorial](tutorial.md) - Complete guide to building agents
- [Design Document](DESIGN_DOC.md) - Architecture deep-dive
- [Complete Tutorial](GPMA_COMPLETE_TUTORIAL.md) - Advanced usage
- [Upgrade Plan](PROFESSIONAL_UPGRADE_PLAN.md) - Production roadmap

## Requirements

- Python 3.10+
- aiohttp, aiofiles
- beautifulsoup4 (recommended)
- playwright (optional, for JavaScript)
- Ollama or OpenAI API key (for LLM features)

## Contributing

Contributions welcome! See the [Professional Upgrade Plan](PROFESSIONAL_UPGRADE_PLAN.md) for areas that need work.

## License

MIT License
