# GPMA Design Document
## General Purpose Multi-Agent System

**Version:** 1.0
**Date:** January 2025
**Status:** Active Development

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction to Agentic Design](#2-introduction-to-agentic-design)
3. [Agentic Design Patterns](#3-agentic-design-patterns)
4. [GPMA Architecture](#4-gpma-architecture)
5. [Component Design](#5-component-design)
6. [Data Flow](#6-data-flow)
7. [API Design](#7-api-design)
8. [Security Considerations](#8-security-considerations)
9. [Extensibility](#9-extensibility)
10. [Future Roadmap](#10-future-roadmap)

---

## 1. Executive Summary

### 1.1 Purpose

GPMA (General Purpose Multi-Agent System) is a modular framework for building AI-powered personal assistants using multiple specialized agents. It demonstrates best practices in agentic design while providing practical functionality for web browsing, research, and task automation.

### 1.2 Goals

- **Educational**: Teach agentic design principles through working code
- **Practical**: Provide a functional personal assistant framework
- **Extensible**: Enable easy addition of custom agents and capabilities
- **Production-Ready**: Include patterns for reliability, security, and scalability

### 1.3 Non-Goals

- Replace specialized AI frameworks (LangChain, AutoGen)
- Provide pre-trained AI models
- Handle real-time streaming (v1.0)

---

## 2. Introduction to Agentic Design

### 2.1 What is an Agent?

An **agent** is an autonomous software entity that:

1. **Perceives** its environment (receives inputs)
2. **Reasons** about what to do (processes information)
3. **Acts** to achieve goals (produces outputs)
4. **Learns** from outcomes (improves over time)

```
┌─────────────────────────────────────────────────────────────┐
│                         AGENT                               │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ PERCEIVE │───▶│  REASON  │───▶│   ACT    │             │
│  │          │    │          │    │          │             │
│  │ • Inputs │    │ • Goals  │    │ • Tools  │             │
│  │ • Events │    │ • Plans  │    │ • APIs   │             │
│  │ • State  │    │ • Memory │    │ • Output │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│        ▲                                │                  │
│        │         ┌──────────┐           │                  │
│        └─────────│  LEARN   │◀──────────┘                  │
│                  │          │                              │
│                  │ • Feedback│                              │
│                  │ • Memory │                              │
│                  └──────────┘                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Single Agent vs Multi-Agent Systems

| Aspect | Single Agent | Multi-Agent System |
|--------|--------------|-------------------|
| **Complexity** | Handles everything | Divides responsibilities |
| **Scalability** | Limited | Horizontal scaling |
| **Reliability** | Single point of failure | Fault tolerant |
| **Specialization** | Jack of all trades | Expert agents |
| **Development** | Simpler | More complex coordination |

### 2.3 When to Use Multi-Agent Systems

Use MAS when:
- Tasks require **diverse expertise** (web, files, APIs)
- **Parallel processing** improves performance
- **Reliability** is critical (redundancy)
- System needs to **scale** with demand
- **Modularity** aids development and testing

### 2.4 Core Principles of Agentic Design

#### Principle 1: Single Responsibility
Each agent should do ONE thing well.

```
❌ BAD: MonolithicAgent (does everything)
✅ GOOD: WebAgent + ResearchAgent + FileAgent
```

#### Principle 2: Loose Coupling
Agents communicate through messages, not direct calls.

```python
# ❌ BAD: Direct coupling
class ResearchAgent:
    def __init__(self, web_agent):
        self.web = web_agent

    def research(self, topic):
        return self.web.fetch(url)  # Direct dependency

# ✅ GOOD: Message-based
class ResearchAgent:
    async def research(self, topic):
        await self.send_message("web_agent", {"action": "fetch", "url": url})
        response = await self.receive_message()
```

#### Principle 3: Declarative Capabilities
Agents declare what they can do, not how.

```python
@property
def capabilities(self):
    return [
        AgentCapability(
            name="web_search",
            description="Search the web for information",
            keywords=["search", "find", "lookup", "google"]
        )
    ]
```

#### Principle 4: Graceful Degradation
System continues working even when agents fail.

```python
async def execute(self, task):
    primary_agent = self.find_best_agent(task)

    if primary_agent is None:
        # Fallback to alternatives
        alternatives = self.find_capable_agents(task)
        if alternatives:
            primary_agent = alternatives[0]
        else:
            return TaskResult(success=False, error="No capable agent")
```

#### Principle 5: Observable State
Agents expose their state for monitoring and debugging.

```python
def get_stats(self):
    return {
        "state": self.state.name,
        "tasks_processed": self._task_count,
        "errors": self._error_count,
        "uptime": self._uptime()
    }
```

---

## 3. Agentic Design Patterns

### 3.1 The Orchestrator Pattern

**Problem**: How do we coordinate multiple agents without tight coupling?

**Solution**: A central orchestrator receives requests and delegates to appropriate agents.

```
                    ┌─────────────────┐
                    │   User Request  │
                    └────────┬────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │        ORCHESTRATOR          │
              │                              │
              │  1. Parse request            │
              │  2. Match capabilities       │
              │  3. Select best agent(s)     │
              │  4. Execute & aggregate      │
              └──────────────────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
      ┌─────────┐       ┌─────────┐       ┌─────────┐
      │ Agent A │       │ Agent B │       │ Agent C │
      └─────────┘       └─────────┘       └─────────┘
```

**GPMA Implementation**: `Orchestrator` class in `core/orchestrator.py`

### 3.2 The Tool Pattern

**Problem**: How do agents perform actions in the real world?

**Solution**: Agents use "tools" - encapsulated functions for specific actions.

```
┌─────────────────────────────────────────┐
│                 AGENT                   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │           TOOL BELT             │   │
│  │                                 │   │
│  │  ┌───────┐ ┌───────┐ ┌───────┐│   │
│  │  │fetch  │ │parse  │ │search ││   │
│  │  │_url   │ │_html  │ │_web   ││   │
│  │  └───────┘ └───────┘ └───────┘│   │
│  └─────────────────────────────────┘   │
│                                         │
│  process(task):                         │
│      data = use_tool("fetch_url", url) │
│      parsed = use_tool("parse_html")   │
│      return parsed                      │
└─────────────────────────────────────────┘
```

**Benefits**:
- Tools are reusable across agents
- Easy to test in isolation
- Clear separation of concerns

**GPMA Implementation**: `Tool` class in `core/base_agent.py`

### 3.3 The Message Bus Pattern (Pub/Sub)

**Problem**: How do agents communicate without knowing about each other?

**Solution**: Agents publish and subscribe to a central message bus.

```
┌─────────┐         ┌─────────────────┐         ┌─────────┐
│ Agent A │────────▶│                 │────────▶│ Agent B │
└─────────┘ publish │   MESSAGE BUS   │subscribe└─────────┘
                    │                 │
┌─────────┐         │  • Routes msgs  │         ┌─────────┐
│ Agent C │◀────────│  • Broadcasts   │────────▶│ Agent D │
└─────────┘subscribe│  • Queues       │ publish └─────────┘
                    └─────────────────┘
```

**Message Types**:
- `REQUEST`: Ask another agent to do something
- `RESPONSE`: Reply to a request
- `BROADCAST`: Notify all agents
- `EVENT`: Something happened (informational)

**GPMA Implementation**: `MessageBus` class in `core/message_bus.py`

### 3.4 The Memory Pattern

**Problem**: How do agents maintain context and learn from history?

**Solution**: Layered memory system with different retention policies.

```
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                          │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              SHORT-TERM MEMORY (STM)                │   │
│  │                                                     │   │
│  │  • Current conversation context                     │   │
│  │  • Recent task results                              │   │
│  │  • Working variables                                │   │
│  │  • TTL: Minutes to hours                            │   │
│  │  • Capacity: Limited (LRU eviction)                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                    Promote if important                     │
│                           ▼                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              LONG-TERM MEMORY (LTM)                 │   │
│  │                                                     │   │
│  │  • User preferences                                 │   │
│  │  • Learned patterns                                 │   │
│  │  • Historical data                                  │   │
│  │  • TTL: Permanent                                   │   │
│  │  • Capacity: Unlimited (persistent storage)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            SEMANTIC MEMORY (Future)                 │   │
│  │                                                     │   │
│  │  • Vector embeddings for similarity search          │   │
│  │  • Knowledge graphs                                 │   │
│  │  • Conceptual relationships                         │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**GPMA Implementation**: `Memory` classes in `core/memory.py`

### 3.5 The Capability Matching Pattern

**Problem**: How does the orchestrator know which agent to use?

**Solution**: Agents declare capabilities; orchestrator matches against tasks.

```python
# Agent declares capabilities
capabilities = [
    AgentCapability(
        name="web_search",
        description="Search the web",
        keywords=["search", "find", "google", "lookup"],
        priority=2
    )
]

# Orchestrator matches
def find_best_agent(task: str) -> str:
    scores = {}
    for agent_name, registration in agents.items():
        for capability in registration.capabilities:
            score = capability.matches(task)  # Keyword matching
            scores[agent_name] = max(scores.get(agent_name, 0), score)

    return max(scores, key=scores.get)
```

**Matching Algorithm**:
1. Tokenize the task description
2. For each agent's capabilities:
   - Check keyword overlap
   - Check name match
   - Apply priority weighting
3. Return highest-scoring agent

### 3.6 The Task Decomposition Pattern

**Problem**: How do we handle complex multi-step tasks?

**Solution**: Break tasks into subtasks that can be routed to different agents.

```
User: "Research AI trends and create a summary report"

                    ┌─────────────────────┐
                    │   DECOMPOSITION     │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│   Subtask 1   │    │   Subtask 2   │    │   Subtask 3   │
│               │    │               │    │               │
│ "Search for   │    │ "Analyze top  │    │ "Generate     │
│  AI trends"   │    │  5 results"   │    │  summary"     │
│               │    │               │    │               │
│ → WebAgent    │    │ → Research    │    │ → Research    │
│               │    │    Agent      │    │    Agent      │
└───────────────┘    └───────────────┘    └───────────────┘
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │    AGGREGATION      │
                    │                     │
                    │  Combine results    │
                    │  into final output  │
                    └─────────────────────┘
```

**Execution Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **Sequential** | One at a time, in order | Dependencies between tasks |
| **Parallel** | All at once | Independent tasks |
| **Pipeline** | Output → Next input | Data transformation chains |

### 3.7 The Facade Pattern

**Problem**: The system is complex; users need a simple interface.

**Solution**: Provide a high-level API that hides internal complexity.

```python
# Complex internal structure
orchestrator = Orchestrator()
orchestrator.register_agent(WebBrowserAgent())
orchestrator.register_agent(ResearchAgent())
orchestrator.register_agent(TaskExecutorAgent())
await orchestrator.initialize_agents()
result = await orchestrator.execute(task, context, strategy)

# Simple facade
assistant = PersonalAssistant()
await assistant.initialize()
result = await assistant.ask("What is AI?")
```

**GPMA Implementation**: `PersonalAssistant` class in `personal_assistant.py`

---

## 4. GPMA Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              USER APPLICATION                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                           PERSONAL ASSISTANT                                │
│                              (Facade Layer)                                 │
│                                                                             │
│   • ask()      • research()      • browse()      • search()      • do()    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              ORCHESTRATOR                                   │
│                          (Coordination Layer)                               │
│                                                                             │
│   • Task routing           • Agent lifecycle          • Result aggregation │
│   • Capability matching    • Error handling           • Task decomposition │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────┐
            │                         │                         │
            ▼                         ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│                     │   │                     │   │                     │
│   WEB BROWSER       │   │   RESEARCH          │   │   TASK EXECUTOR     │
│   AGENT             │   │   AGENT             │   │   AGENT             │
│                     │   │                     │   │                     │
│ • Fetch URLs        │   │ • Search & analyze  │   │ • Run commands      │
│ • Parse HTML        │   │ • Summarize         │   │ • File operations   │
│ • Extract data      │   │ • Synthesize        │   │ • System info       │
│                     │   │                     │   │                     │
└─────────────────────┘   └─────────────────────┘   └─────────────────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                              SHARED SERVICES                                │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │   MESSAGE BUS   │    │     MEMORY      │    │     TOOLS       │        │
│  │                 │    │                 │    │                 │        │
│  │ • Pub/Sub       │    │ • Short-term    │    │ • Web tools     │        │
│  │ • Request/Reply │    │ • Long-term     │    │ • File tools    │        │
│  │ • Broadcast     │    │ • Search        │    │ • System tools  │        │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 gpma/                                       │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                              core/                                    │  │
│  │                                                                       │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │  base_agent.py │  │ orchestrator.py│  │ message_bus.py │         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ • BaseAgent    │  │ • Orchestrator │  │ • MessageBus   │         │  │
│  │  │ • AgentState   │  │ • Dynamic      │  │ • Message      │         │  │
│  │  │ • Capability   │  │   Orchestrator │  │ • MessageType  │         │  │
│  │  │ • Tool         │  │ • Task         │  │                │         │  │
│  │  │ • TaskResult   │  │ • Strategy     │  │                │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  │                                                                       │  │
│  │  ┌────────────────┐                                                  │  │
│  │  │   memory.py    │                                                  │  │
│  │  │                │                                                  │  │
│  │  │ • Memory       │                                                  │  │
│  │  │ • ShortTerm    │                                                  │  │
│  │  │ • LongTerm     │                                                  │  │
│  │  │ • Composite    │                                                  │  │
│  │  └────────────────┘                                                  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                             agents/                                   │  │
│  │                                                                       │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │  │
│  │  │ web_browser.py │  │  research.py   │  │task_executor.py│         │  │
│  │  │                │  │                │  │                │         │  │
│  │  │ • WebBrowser   │  │ • Research     │  │ • TaskExecutor │         │  │
│  │  │   Agent        │  │   Agent        │  │   Agent        │         │  │
│  │  └────────────────┘  └────────────────┘  └────────────────┘         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                             tools/                                    │  │
│  │                                                                       │  │
│  │  ┌────────────────┐  ┌────────────────┐                              │  │
│  │  │  web_tools.py  │  │ file_tools.py  │                              │  │
│  │  │                │  │                │                              │  │
│  │  │ • WebFetcher   │  │ • FileManager  │                              │  │
│  │  │ • fetch_url    │  │ • read_file    │                              │  │
│  │  │ • search_web   │  │ • write_file   │                              │  │
│  │  └────────────────┘  └────────────────┘                              │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      personal_assistant.py                            │  │
│  │                                                                       │  │
│  │  • PersonalAssistant (Main user interface)                           │  │
│  │  • quick_ask, quick_research, quick_browse (Convenience functions)   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Class Hierarchy

```
                                BaseAgent (Abstract)
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            WebBrowserAgent    ResearchAgent    TaskExecutorAgent
                    │
                    ▼
              Custom Agents...


                                Memory (Abstract)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
            ShortTermMemory                     LongTermMemory
                    │                                   │
                    └───────────────┬───────────────────┘
                                    │
                                    ▼
                            CompositeMemory


                                Orchestrator
                                      │
                                      ▼
                            DynamicOrchestrator
```

---

## 5. Component Design

### 5.1 BaseAgent

**Purpose**: Abstract foundation for all agents

**Responsibilities**:
- Define capability interface
- Manage agent lifecycle (init, process, shutdown)
- Tool registration and execution
- State management
- Statistics collection

**Key Methods**:

| Method | Description |
|--------|-------------|
| `capabilities` | Property declaring what agent can do |
| `process(task)` | Main work method (abstract) |
| `initialize()` | Async setup hook |
| `shutdown()` | Cleanup hook |
| `register_tool(tool)` | Add a tool to the agent |
| `use_tool(name, **kwargs)` | Execute a registered tool |
| `run_task(task)` | Entry point with error handling |

**State Machine**:

```
                    ┌───────────┐
                    │   IDLE    │◀─────────────────┐
                    └─────┬─────┘                  │
                          │                        │
                     run_task()                    │
                          │                        │
                          ▼                        │
                    ┌───────────┐           success│
                    │PROCESSING │──────────────────┘
                    └─────┬─────┘
                          │
                     error│waiting
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
      ┌───────────┐ ┌───────────┐ ┌───────────┐
      │   ERROR   │ │  WAITING  │ │TERMINATED │
      └───────────┘ └───────────┘ └───────────┘
```

### 5.2 Orchestrator

**Purpose**: Coordinate multiple agents

**Responsibilities**:
- Agent registration and lifecycle
- Task routing based on capabilities
- Task decomposition
- Execution strategy management
- Result aggregation

**Key Methods**:

| Method | Description |
|--------|-------------|
| `register_agent(agent)` | Add agent to the system |
| `find_best_agent(task)` | Match task to best agent |
| `execute(request, context, strategy)` | Main execution entry |
| `_decompose_task(task)` | Break complex tasks |
| `initialize_agents()` | Init all registered agents |
| `shutdown()` | Shutdown all agents |

### 5.3 MessageBus

**Purpose**: Enable agent communication

**Responsibilities**:
- Message routing (direct, broadcast, topic)
- Request-response pattern
- Message history
- Subscriber management

**Message Flow**:

```
┌─────────────┐                      ┌─────────────┐
│   Sender    │                      │  Receiver   │
└──────┬──────┘                      └──────▲──────┘
       │                                    │
       │ publish(message)                   │ handler(message)
       │                                    │
       ▼                                    │
┌─────────────────────────────────────────────────────────────┐
│                        MESSAGE BUS                          │
│                                                             │
│  1. Validate message                                        │
│  2. Store in history                                        │
│  3. Route based on receiver:                                │
│     • "*" → broadcast to all                                │
│     • "#topic" → send to topic subscribers                  │
│     • "agent_name" → send to specific agent                 │
│  4. Invoke handler                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.4 Memory

**Purpose**: Store and retrieve agent context

**Design Decisions**:

| Decision | Rationale |
|----------|-----------|
| LRU for STM | Automatic eviction of old items |
| JSON for LTM | Simple, human-readable persistence |
| Tag-based search | Flexible categorization |
| Composite pattern | Best of both worlds |

**Memory Operations**:

```
store(key, value)
    │
    ├── Check capacity (STM)
    ├── Evict if needed (LRU)
    ├── Create MemoryEntry
    ├── Store in appropriate tier
    └── Persist if LTM

retrieve(key)
    │
    ├── Check STM first (faster)
    ├── Check LTM if not found
    ├── Update access count
    └── Return value or None

search(query, tags)
    │
    ├── Filter by tags
    ├── Match query against keys/values
    └── Return matching entries
```

---

## 6. Data Flow

### 6.1 Request Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  User: "Research quantum computing and summarize"                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PersonalAssistant.ask()                                             │
│                                                                             │
│ • Store in conversation history                                             │
│ • Forward to orchestrator                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Orchestrator.execute()                                              │
│                                                                             │
│ • Store request in memory                                                   │
│ • Create root Task                                                          │
│ • Attempt decomposition                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Task Decomposition                                                  │
│                                                                             │
│ Pattern: "(.+) and (.+)" matched                                            │
│                                                                             │
│ Subtask 1: "Research quantum computing"                                     │
│ Subtask 2: "summarize"                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Capability Matching                                                 │
│                                                                             │
│ Subtask 1: "research" → ResearchAgent (score: 0.5)                         │
│ Subtask 2: "summarize" → ResearchAgent (score: 0.4)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: Sequential Execution                                                │
│                                                                             │
│ 5a. ResearchAgent.process(subtask_1)                                        │
│     • Generate search queries                                               │
│     • Use WebBrowserAgent for searches                                      │
│     • Analyze sources                                                       │
│     • Return synthesis                                                      │
│                                                                             │
│ 5b. ResearchAgent.process(subtask_2)                                        │
│     • Use previous result as context                                        │
│     • Generate summary                                                      │
│     • Return final output                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Result Aggregation                                                  │
│                                                                             │
│ • Combine subtask results                                                   │
│ • Store in memory                                                           │
│ • Return TaskResult                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: Response Formatting                                                 │
│                                                                             │
│ • Extract key data from TaskResult                                          │
│ • Format as human-readable string                                           │
│ • Add to conversation history                                               │
│ • Return to user                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Agent Collaboration Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESEARCH AGENT                                     │
│                                                                             │
│  _handle_research("quantum computing")                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ 1. Generate queries
        ▼
┌───────────────────────┐
│ ["quantum computing", │
│  "latest quantum      │
│   computing",         │
│  "quantum computing   │
│   explained"]         │
└───────────────────────┘
        │
        │ 2. For each query, call WebBrowserAgent
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WEB BROWSER AGENT                                  │
│                                                                             │
│  run_task({action: "search", input: "quantum computing"})                   │
│                                                                             │
│  → Returns: [{title, url, snippet}, ...]                                    │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ 3. For each top result, fetch page
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          WEB BROWSER AGENT                                  │
│                                                                             │
│  run_task({action: "fetch", input: "https://..."})                          │
│                                                                             │
│  → Returns: {title, text, links, metadata}                                  │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ 4. Analyze each source
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESEARCH AGENT                                     │
│                                                                             │
│  _analyze_source(url)                                                       │
│  → Extract key points                                                       │
│  → Generate summary                                                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        │ 5. Synthesize findings
        ▼
┌───────────────────────┐
│ Final synthesis       │
│ combining all sources │
└───────────────────────┘
```

---

## 7. API Design

### 7.1 PersonalAssistant API

```python
class PersonalAssistant:
    """High-level user interface."""

    # Lifecycle
    async def initialize() -> None
    async def shutdown() -> None

    # Core functionality
    async def ask(question: str, context: Dict = None) -> str
    async def research(topic: str, depth: str = "standard") -> Dict
    async def browse(url: str) -> Dict
    async def search(query: str, num_results: int = 10) -> List[Dict]
    async def do(task: str, options: Dict = None) -> TaskResult

    # Memory
    def remember(key: str, value: Any, permanent: bool = False) -> None
    def recall(key: str) -> Optional[Any]

    # Management
    def add_agent(agent: BaseAgent, priority: int = 1) -> None
    def remove_agent(agent_name: str) -> None
    def get_agents() -> List[str]
    def get_status() -> Dict

    # History
    def get_history(limit: int = 20) -> List[Dict]
    def clear_history() -> None
```

### 7.2 Agent API

```python
class BaseAgent(ABC):
    """Base class for all agents."""

    # Required (abstract)
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]: ...

    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> TaskResult: ...

    # Optional hooks
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...

    # Tool management
    def register_tool(self, tool: Tool) -> None: ...
    async def use_tool(self, tool_name: str, **kwargs) -> Any: ...

    # Communication
    async def send_message(self, to_agent: str, content: Any, msg_type: str) -> None: ...
    async def receive_message(self, timeout: float = None) -> Optional[Any]: ...

    # Execution (do not override)
    async def run_task(self, task: Dict[str, Any]) -> TaskResult: ...

    # Monitoring
    def get_stats(self) -> Dict[str, Any]: ...
```

### 7.3 Data Structures

```python
@dataclass
class AgentCapability:
    name: str                    # Unique identifier
    description: str             # Human-readable description
    keywords: List[str]          # Matching keywords
    priority: int = 1            # Higher = more preferred

@dataclass
class TaskResult:
    success: bool                # Whether task succeeded
    data: Any                    # Result data
    error: Optional[str] = None  # Error message if failed
    execution_time: float = 0.0  # Time taken
    metadata: Dict = field(...)  # Additional info

@dataclass
class Message:
    sender: str                  # Source agent
    receiver: str                # Target agent or "*"
    content: Any                 # Message payload
    msg_type: MessageType        # REQUEST, RESPONSE, etc.
    id: str                      # Unique message ID
    timestamp: datetime          # When sent
    correlation_id: str = None   # Links request/response
    topic: str = "default"       # For topic-based routing

@dataclass
class Tool:
    name: str                    # Tool identifier
    description: str             # What it does
    function: Callable           # The actual function
    parameters: Dict[str, str]   # Parameter descriptions
```

---

## 8. Security Considerations

### 8.1 Threat Model

| Threat | Risk | Mitigation |
|--------|------|------------|
| Command injection | HIGH | Whitelist commands, validate input |
| Path traversal | HIGH | Validate paths, use sandboxing |
| Sensitive data exposure | MEDIUM | Memory encryption, access control |
| Resource exhaustion | MEDIUM | Rate limiting, timeouts |
| Malicious web content | MEDIUM | Input sanitization, sandboxing |

### 8.2 Security Measures

**Command Execution**:
```python
class TaskExecutorAgent:
    DANGEROUS_COMMANDS = [
        "rm -rf", "del /f", "format", "mkfs",
        "dd if=", "chmod -R 777", "> /dev/sda"
    ]

    def _is_dangerous_command(self, command: str) -> bool:
        return any(danger in command.lower()
                   for danger in self.DANGEROUS_COMMANDS)

    def _is_command_allowed(self, command: str) -> bool:
        if not self.command_whitelist:
            return True
        return command.split()[0] in self.command_whitelist
```

**Path Validation**:
```python
class FileManager:
    def _validate_path(self, path: str) -> Path:
        full_path = (self.base_path / path).resolve()

        # Prevent directory traversal
        try:
            full_path.relative_to(self.base_path)
        except ValueError:
            raise ValueError(f"Path outside allowed directory: {path}")

        return full_path
```

**Input Sanitization**:
```python
def sanitize_url(url: str) -> str:
    """Ensure URL is safe to fetch."""
    parsed = urlparse(url)

    # Only allow http/https
    if parsed.scheme not in ["http", "https"]:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

    # Block internal IPs
    if is_internal_ip(parsed.hostname):
        raise ValueError("Internal IPs not allowed")

    return url
```

### 8.3 Best Practices

1. **Principle of Least Privilege**: Agents only have permissions they need
2. **Input Validation**: Always validate user and external input
3. **Output Encoding**: Sanitize data before displaying
4. **Secure Defaults**: Dangerous features opt-in, not opt-out
5. **Audit Logging**: Log security-relevant events

---

## 9. Extensibility

### 9.1 Adding a Custom Agent

```python
from gpma import BaseAgent, AgentCapability
from gpma.core.base_agent import TaskResult, Tool

class MyCustomAgent(BaseAgent):
    """Example custom agent."""

    def __init__(self, name: str = None, config: Dict = None):
        super().__init__(name or "MyAgent")
        self.config = config or {}

        # Register tools
        self.register_tool(Tool(
            name="my_tool",
            description="Does something useful",
            function=self._my_tool_impl,
            parameters={"param": "Description"}
        ))

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="my_capability",
                description="What this agent does",
                keywords=["keyword1", "keyword2"],
                priority=2
            )
        ]

    async def initialize(self) -> None:
        await super().initialize()
        # Custom setup (connections, resources, etc.)

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        action = task.get("action", "default")
        input_data = task.get("input", "")

        try:
            if action == "my_action":
                result = await self._handle_my_action(input_data)
                return TaskResult(success=True, data=result)
            else:
                return TaskResult(
                    success=False,
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            return TaskResult(success=False, error=str(e))

    async def shutdown(self) -> None:
        # Custom cleanup
        await super().shutdown()

    async def _my_tool_impl(self, param: str) -> Any:
        """Tool implementation."""
        return f"Processed: {param}"

    async def _handle_my_action(self, input_data: str) -> Any:
        """Handle the custom action."""
        result = await self.use_tool("my_tool", param=input_data)
        return result
```

### 9.2 Adding Custom Tools

```python
from gpma.core.base_agent import Tool

# Define standalone tool
async def translate_text(text: str, target_lang: str) -> str:
    """Translate text to target language."""
    # Implementation here
    return translated_text

# Create Tool object
translate_tool = Tool(
    name="translate",
    description="Translate text between languages",
    function=translate_text,
    parameters={
        "text": "Text to translate",
        "target_lang": "Target language code (en, es, fr, etc.)"
    }
)

# Register with an agent
agent.register_tool(translate_tool)

# Or create a tool factory
def create_api_tool(api_name: str, endpoint: str) -> Tool:
    async def call_api(**kwargs):
        # Generic API calling logic
        pass

    return Tool(
        name=f"{api_name}_api",
        description=f"Call the {api_name} API",
        function=call_api,
        parameters={"data": "Request data"}
    )
```

### 9.3 Custom Memory Backend

```python
from gpma.core.memory import Memory, MemoryEntry
import redis

class RedisMemory(Memory):
    """Redis-backed memory implementation."""

    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def store(self, key: str, value: Any, **kwargs) -> None:
        self.redis.set(key, json.dumps(value))
        if kwargs.get("ttl"):
            self.redis.expire(key, kwargs["ttl"].total_seconds())

    def retrieve(self, key: str) -> Optional[Any]:
        data = self.redis.get(key)
        return json.loads(data) if data else None

    def search(self, query: str = None, tags: List[str] = None) -> List[MemoryEntry]:
        # Implement search with Redis SCAN
        pass

    def forget(self, key: str) -> bool:
        return self.redis.delete(key) > 0

    def clear(self) -> None:
        self.redis.flushdb()
```

### 9.4 LLM Integration

```python
from gpma import BaseAgent, AgentCapability
from gpma.core.base_agent import TaskResult

class LLMAgent(BaseAgent):
    """Agent powered by an LLM."""

    def __init__(self, model: str = "gpt-4", api_key: str = None):
        super().__init__("LLM")
        self.model = model
        self.api_key = api_key
        self._client = None

    async def initialize(self) -> None:
        await super().initialize()
        # Initialize LLM client
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(api_key=self.api_key)

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="general_intelligence",
                description="Answer any question using LLM",
                keywords=["explain", "write", "create", "help", "what", "how", "why"],
                priority=1  # Low priority - fallback agent
            )
        ]

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        prompt = task.get("input", "")
        context = task.get("context", {})

        # Build messages with context
        messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]

        if context.get("history"):
            messages.extend(context["history"])

        messages.append({"role": "user", "content": prompt})

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            return TaskResult(
                success=True,
                data=response.choices[0].message.content
            )
        except Exception as e:
            return TaskResult(success=False, error=str(e))
```

---

## 10. Future Roadmap

### 10.1 Version 1.1 - Enhanced Intelligence

- [ ] LLM-based task decomposition
- [ ] Semantic memory with vector embeddings
- [ ] Learning from user feedback
- [ ] Improved capability matching with embeddings

### 10.2 Version 1.2 - Reliability & Scale

- [ ] Persistent task queue (Redis/RabbitMQ)
- [ ] Agent health checks and auto-restart
- [ ] Distributed orchestrator
- [ ] Metrics and monitoring (Prometheus)

### 10.3 Version 1.3 - Advanced Features

- [ ] Real-time streaming responses
- [ ] Multi-modal agents (vision, audio)
- [ ] Agent-to-agent learning
- [ ] Plugin marketplace

### 10.4 Version 2.0 - Enterprise

- [ ] Multi-tenant support
- [ ] Role-based access control
- [ ] Audit logging
- [ ] Compliance features (GDPR, SOC2)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Agent** | Autonomous software entity that perceives, reasons, and acts |
| **Capability** | Declaration of what an agent can do |
| **Orchestrator** | Central coordinator that routes tasks to agents |
| **Tool** | Function an agent uses to perform actions |
| **Message Bus** | Communication system for inter-agent messaging |
| **STM** | Short-Term Memory - fast, limited, temporary storage |
| **LTM** | Long-Term Memory - persistent, unlimited storage |
| **Task** | Unit of work to be executed by an agent |
| **TaskResult** | Outcome of task execution |
| **Pub/Sub** | Publish-Subscribe messaging pattern |

---

## Appendix B: References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach*
2. Wooldridge, M. (2009). *An Introduction to MultiAgent Systems*
3. Gamma, E., et al. (1994). *Design Patterns: Elements of Reusable OO Software*
4. Martin, R. C. (2017). *Clean Architecture*
5. Anthropic. (2024). *Claude Agent SDK Documentation*

---

*Document maintained by the GPMA team. Last updated: January 2025*
