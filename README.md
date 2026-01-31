# General Purpose Multi-Agent System (GPMA)

A modular, extensible multi-agent system designed for personal assistant tasks including web browsing, research, and task automation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR (Brain)                        │
│  - Routes tasks to appropriate agents                           │
│  - Manages agent lifecycle                                      │
│  - Handles inter-agent communication                            │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Web Browser  │    │   Research    │    │     Task      │
│    Agent      │    │    Agent      │    │    Agent      │
│               │    │               │    │               │
│ - Fetch URLs  │    │ - Search web  │    │ - File ops    │
│ - Parse HTML  │    │ - Summarize   │    │ - Commands    │
│ - Extract     │    │ - Analyze     │    │ - Automation  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌───────────────┐
                    │  Message Bus  │
                    │  (Pub/Sub)    │
                    └───────────────┘
```

## Key Concepts

### 1. Agents
Autonomous units that perform specific tasks. Each agent has:
- **Capabilities**: What it can do
- **Memory**: Short-term and long-term storage
- **Tools**: Functions it can execute

### 2. Orchestrator
The central coordinator that:
- Receives user requests
- Breaks down complex tasks
- Assigns sub-tasks to agents
- Aggregates results

### 3. Message Bus
Enables async communication between agents using pub/sub pattern.

### 4. Tools
Reusable functions that agents can invoke (web fetch, file read, etc.)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from gpma import PersonalAssistant

# Create your personal agent
assistant = PersonalAssistant()

# Ask it to do something
result = await assistant.run("Research the latest news about AI and summarize it")
print(result)
```

## Project Structure

```
gpma/
├── core/
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class
│   ├── orchestrator.py    # Agent coordinator
│   ├── message_bus.py     # Inter-agent communication
│   └── memory.py          # Agent memory systems
├── agents/
│   ├── __init__.py
│   ├── web_browser.py     # Web browsing agent
│   ├── research.py        # Research & analysis agent
│   └── task_executor.py   # Task execution agent
├── tools/
│   ├── __init__.py
│   ├── web_tools.py       # Web fetching, parsing
│   └── file_tools.py      # File operations
├── personal_assistant.py  # Main personal assistant
└── examples/
    └── demo.py            # Usage examples
```

## Learning Path

1. **Start with `core/base_agent.py`** - Understand the agent abstraction
2. **Study `core/message_bus.py`** - Learn inter-agent communication
3. **Explore `core/orchestrator.py`** - See how agents are coordinated
4. **Look at specific agents** - See implementations in action
5. **Run examples** - Try the system hands-on
