"""
Personal Assistant Module

The main interface for interacting with the multi-agent system.
This is what users interact with - it provides a simple API that
hides the complexity of the underlying agent orchestration.

LEARNING POINTS:
- Facade pattern: Simple interface to complex system
- This is the entry point for all user interactions
- It manages agent lifecycle and coordination

LLM INTEGRATION:
- Supports OpenAI, Ollama (local), and Azure OpenAI
- Intelligent task decomposition with LLM
- Natural language understanding
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import logging

from .core.orchestrator import Orchestrator, ExecutionStrategy, DynamicOrchestrator
from .core.base_agent import BaseAgent, TaskResult
from .core.memory import CompositeMemory
from .agents.web_browser import WebBrowserAgent
from .agents.research import ResearchAgent
from .agents.task_executor import TaskExecutorAgent

logger = logging.getLogger(__name__)


# LLM Provider type alias for type hints
LLMProviderType = Any  # Will be LLMProvider when llm module is imported


class PersonalAssistant:
    """
    Your personal AI assistant powered by multiple specialized agents.

    The PersonalAssistant is the main interface to the GPMA system.
    It provides a simple API for:
    - Asking questions
    - Requesting research
    - Executing tasks
    - Managing agents

    ARCHITECTURE:
    ┌─────────────────────────────────────────────────────┐
    │              PersonalAssistant                      │
    │                 (You are here)                      │
    └─────────────────────────────────────────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────────────────┐
    │                 Orchestrator                        │
    │            (Manages all agents)                     │
    └─────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌─────────┐     ┌─────────┐     ┌─────────┐
    │   Web   │     │Research │     │  Task   │
    │ Browser │     │  Agent  │     │Executor │
    └─────────┘     └─────────┘     └─────────┘

    USAGE:
        # Basic usage
        assistant = PersonalAssistant()
        await assistant.initialize()

        # Ask anything
        result = await assistant.ask("What's the weather in New York?")
        print(result)

        # Research a topic
        research = await assistant.research("Machine learning trends 2024")
        print(research)

        # Execute a task
        await assistant.do("Create a file called hello.txt with 'Hello World'")

        # Clean up
        await assistant.shutdown()
    """

    def __init__(
        self,
        name: str = "Assistant",
        memory_path: str = None,
        enable_dynamic_agents: bool = False,
        llm_provider: LLMProviderType = None,
        llm_config: Dict[str, Any] = None
    ):
        """
        Create a new PersonalAssistant.

        Args:
            name: Name for this assistant instance
            memory_path: Path to persistent memory file
            enable_dynamic_agents: Allow dynamic agent spawning
            llm_provider: LLM provider for intelligent features (OpenAI, Ollama, etc.)
            llm_config: Configuration for LLM (model, temperature, etc.)

        EXAMPLE WITH LLM:
            from gpma.llm import OpenAIProvider, OllamaProvider

            # With OpenAI
            provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
            assistant = PersonalAssistant(llm_provider=provider)

            # With Ollama (local)
            provider = OllamaProvider(model="llama2")
            assistant = PersonalAssistant(llm_provider=provider)
        """
        self.name = name
        self._initialized = False

        # Choose orchestrator type
        if enable_dynamic_agents:
            self._orchestrator = DynamicOrchestrator(memory_path=memory_path)
        else:
            self._orchestrator = Orchestrator(memory_path=memory_path)

        # Pre-configured agents
        self._agents: Dict[str, BaseAgent] = {}

        # Conversation history
        self._history: List[Dict[str, Any]] = []

        # LLM integration
        self._llm_provider = llm_provider
        self._llm_config = llm_config or {}
        self._llm_agent = None
        self._task_decomposer = None

        logger.info(f"PersonalAssistant '{name}' created" + (" with LLM" if llm_provider else ""))

    async def initialize(self) -> None:
        """
        Initialize the assistant and all its agents.

        This must be called before using the assistant.
        """
        if self._initialized:
            return

        # Create default agents
        web_agent = WebBrowserAgent()
        research_agent = ResearchAgent()
        task_agent = TaskExecutorAgent()

        # Connect research agent to web agent for collaboration
        research_agent.set_web_agent(web_agent)

        # Register agents with orchestrator
        self._orchestrator.register_agent(web_agent, priority=2)
        self._orchestrator.register_agent(research_agent, priority=3)
        self._orchestrator.register_agent(task_agent, priority=2)

        # Store references
        self._agents = {
            "web": web_agent,
            "research": research_agent,
            "task": task_agent
        }

        # Initialize LLM agent if provider is configured
        if self._llm_provider:
            await self._initialize_llm()

        # Initialize all agents
        await self._orchestrator.initialize_agents()

        self._initialized = True
        logger.info(f"PersonalAssistant '{self.name}' initialized with {len(self._agents)} agents")

    async def _initialize_llm(self) -> None:
        """Initialize LLM-powered components."""
        try:
            from .llm import LLMAgent, LLMTaskDecomposer

            # Create LLM agent
            self._llm_agent = LLMAgent(
                llm_provider=self._llm_provider,
                name="LLMAgent",
                system_prompt=self._llm_config.get("system_prompt"),
                max_history=self._llm_config.get("max_history", 20)
            )

            # Register LLM agent with high priority (for general questions)
            self._orchestrator.register_agent(self._llm_agent, priority=1)
            self._agents["llm"] = self._llm_agent

            # Create task decomposer for intelligent task breakdown
            self._task_decomposer = LLMTaskDecomposer(self._llm_provider)

            # Set decomposer on orchestrator
            self._orchestrator.set_decomposer(self._llm_decompose_task)

            logger.info("LLM components initialized")

        except ImportError as e:
            logger.warning(f"LLM module not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")

    async def _llm_decompose_task(self, task) -> list:
        """Use LLM to decompose complex tasks."""
        if not self._task_decomposer:
            return []

        try:
            from .core.orchestrator import Task

            result = await self._task_decomposer.decompose(task.input)
            subtasks = []

            for st in result.get("subtasks", []):
                subtasks.append(Task(
                    id=f"{task.id}_{st.id}",
                    action=st.action,
                    input=st.description,
                    context=task.context,
                    parent_id=task.id
                ))

            return subtasks
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            return []

    async def shutdown(self) -> None:
        """
        Shutdown the assistant and clean up resources.
        """
        if not self._initialized:
            return

        await self._orchestrator.shutdown()
        self._initialized = False
        logger.info(f"PersonalAssistant '{self.name}' shut down")

    def _ensure_initialized(self) -> None:
        """Raise error if not initialized."""
        if not self._initialized:
            raise RuntimeError("PersonalAssistant not initialized. Call initialize() first.")

    async def ask(self, question: str, context: Dict[str, Any] = None, use_llm: bool = True) -> str:
        """
        Ask the assistant anything.

        This is the main interface for interacting with the assistant.
        It routes the question to the appropriate agent(s) and returns
        a human-readable response.

        Args:
            question: Your question or request
            context: Additional context (optional)
            use_llm: Whether to prefer LLM for responses (default: True)

        Returns:
            String response

        Example:
            answer = await assistant.ask("What is Python?")
            print(answer)
        """
        self._ensure_initialized()

        # Add to history
        self._history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().isoformat()
        })

        # If LLM is available and preferred, use it directly for conversational queries
        if use_llm and self._llm_agent and self._is_conversational_query(question):
            result = await self._llm_agent.run_task({
                "input": question,
                "action": "chat",
                "context": context or {}
            })
        else:
            # Execute via orchestrator (routes to best agent)
            result = await self._orchestrator.execute(question, context or {})

        # Format response
        if result.success:
            response = self._format_response(result)
        else:
            response = f"I'm sorry, I couldn't complete that request. Error: {result.error}"

        # Add response to history
        self._history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        return response

    def _is_conversational_query(self, query: str) -> bool:
        """
        Determine if a query is conversational (should use LLM)
        vs action-oriented (should use specialized agents).
        """
        # Action keywords that should use specialized agents
        action_keywords = [
            "fetch", "browse", "open", "download", "search the web",
            "run", "execute", "create file", "delete", "list files"
        ]

        query_lower = query.lower()

        # Check for action keywords
        for keyword in action_keywords:
            if keyword in query_lower:
                return False

        # Check for URLs (should use web agent)
        if "http://" in query or "https://" in query:
            return False

        # Default to conversational (use LLM)
        return True

    async def chat(self, message: str, stream: bool = False) -> Union[str, Any]:
        """
        Have a conversation with the assistant using LLM.

        This method always uses the LLM for natural conversation,
        unlike `ask()` which routes to specialized agents.

        Args:
            message: Your message
            stream: Whether to stream the response (if supported)

        Returns:
            String response (or async generator if streaming)

        Example:
            response = await assistant.chat("Tell me a joke about programming")
            print(response)
        """
        self._ensure_initialized()

        if not self._llm_agent:
            return "LLM not configured. Initialize with llm_provider to use chat."

        # Add to history
        self._history.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        result = await self._llm_agent.run_task({
            "input": message,
            "action": "chat"
        })

        response = result.data if result.success else f"Error: {result.error}"

        # Add response to history
        self._history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })

        return response

    async def research(
        self,
        topic: str,
        depth: str = "standard"
    ) -> Dict[str, Any]:
        """
        Research a topic in depth.

        Args:
            topic: What to research
            depth: "quick", "standard", or "deep"

        Returns:
            Dictionary with research results

        Example:
            results = await assistant.research("Quantum computing")
            print(results["synthesis"])
        """
        self._ensure_initialized()

        # Configure based on depth
        max_sources = {"quick": 2, "standard": 5, "deep": 10}.get(depth, 5)

        # Use research agent directly for more control
        research_agent = self._agents.get("research")
        if research_agent:
            research_agent.max_sources = max_sources

        result = await self._orchestrator.execute(
            f"Research: {topic}",
            {"depth": depth}
        )

        if result.success:
            return result.data
        else:
            return {"error": result.error, "topic": topic}

    async def browse(self, url: str) -> Dict[str, Any]:
        """
        Browse a specific URL and get its content.

        Args:
            url: URL to browse

        Returns:
            Dictionary with page content

        Example:
            page = await assistant.browse("https://example.com")
            print(page["title"])
        """
        self._ensure_initialized()

        web_agent = self._agents.get("web")
        if not web_agent:
            return {"error": "Web agent not available"}

        result = await web_agent.run_task({
            "action": "fetch",
            "input": url
        })

        return result.data if result.success else {"error": result.error}

    async def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search the web.

        Args:
            query: Search query
            num_results: Maximum number of results

        Returns:
            List of search results

        Example:
            results = await assistant.search("Python tutorials")
            for r in results:
                print(r["title"], r["url"])
        """
        self._ensure_initialized()

        web_agent = self._agents.get("web")
        if not web_agent:
            return []

        result = await web_agent.run_task({
            "action": "search",
            "input": query,
            "options": {"num_results": num_results}
        })

        if result.success and result.data:
            return result.data.get("results", [])
        return []

    async def do(self, task: str, options: Dict[str, Any] = None) -> TaskResult:
        """
        Execute a task.

        Args:
            task: Task description
            options: Task options

        Returns:
            TaskResult object

        Example:
            result = await assistant.do("List files in current directory")
        """
        self._ensure_initialized()

        return await self._orchestrator.execute(task, options or {})

    def remember(self, key: str, value: Any, permanent: bool = False) -> None:
        """
        Store something in the assistant's memory.

        Args:
            key: Memory key
            value: What to remember
            permanent: Store in long-term memory

        Example:
            assistant.remember("user_name", "Alice", permanent=True)
        """
        self._ensure_initialized()

        self._orchestrator.memory.store(key, value, long_term=permanent)

    def recall(self, key: str) -> Optional[Any]:
        """
        Recall something from memory.

        Args:
            key: Memory key

        Returns:
            The stored value, or None

        Example:
            name = assistant.recall("user_name")
        """
        self._ensure_initialized()

        return self._orchestrator.memory.retrieve(key)

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            limit: Maximum number of entries

        Returns:
            List of conversation entries
        """
        return self._history[-limit:]

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def add_agent(self, agent: BaseAgent, priority: int = 1) -> None:
        """
        Add a custom agent to the assistant.

        Args:
            agent: The agent to add
            priority: Agent priority (higher = more preferred)

        Example:
            my_agent = MyCustomAgent()
            assistant.add_agent(my_agent, priority=5)
        """
        self._orchestrator.register_agent(agent, priority)
        self._agents[agent.name] = agent
        logger.info(f"Added agent: {agent.name}")

    def remove_agent(self, agent_name: str) -> None:
        """
        Remove an agent from the assistant.

        Args:
            agent_name: Name of agent to remove
        """
        self._orchestrator.unregister_agent(agent_name)
        if agent_name in self._agents:
            del self._agents[agent_name]

    def get_agents(self) -> List[str]:
        """Get list of available agent names."""
        return list(self._agents.keys())

    def get_status(self) -> Dict[str, Any]:
        """
        Get assistant status and statistics.

        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "initialized": self._initialized,
            "agents": len(self._agents),
            "agent_status": self._orchestrator.get_agent_status() if self._initialized else {},
            "history_length": len(self._history),
            "memory_stats": {
                "short_term": len(self._orchestrator.memory.short_term) if self._initialized else 0
            }
        }

    def _format_response(self, result: TaskResult) -> str:
        """Format a TaskResult into a human-readable string."""
        data = result.data

        if data is None:
            return "Task completed successfully."

        if isinstance(data, str):
            return data

        if isinstance(data, dict):
            # Try to extract meaningful text
            if "synthesis" in data:
                return data["synthesis"]
            if "answer" in data:
                return data["answer"]
            if "summary" in data:
                return data["summary"]
            if "text" in data:
                text = data["text"]
                if len(text) > 1000:
                    return text[:1000] + "..."
                return text
            if "results" in data:
                results = data["results"]
                if isinstance(results, list):
                    lines = []
                    for i, r in enumerate(results[:5], 1):
                        if isinstance(r, dict):
                            title = r.get("title", "Result")
                            url = r.get("url", "")
                            lines.append(f"{i}. {title}\n   {url}")
                        else:
                            lines.append(f"{i}. {r}")
                    return "\n".join(lines)

            # Generic dict formatting
            return str(data)

        if isinstance(data, list):
            return "\n".join(str(item) for item in data[:10])

        return str(data)

    # Context manager support
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.shutdown()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def quick_ask(question: str) -> str:
    """
    Quick one-off question without managing lifecycle.

    Example:
        answer = await quick_ask("What is 2+2?")
    """
    async with PersonalAssistant() as assistant:
        return await assistant.ask(question)


async def quick_research(topic: str) -> Dict[str, Any]:
    """
    Quick research without managing lifecycle.

    Example:
        results = await quick_research("Python best practices")
    """
    async with PersonalAssistant() as assistant:
        return await assistant.research(topic)


async def quick_browse(url: str) -> Dict[str, Any]:
    """
    Quick URL fetch without managing lifecycle.

    Example:
        page = await quick_browse("https://example.com")
    """
    async with PersonalAssistant() as assistant:
        return await assistant.browse(url)


# ============================================================================
# LLM-POWERED ASSISTANT FACTORY
# ============================================================================

def create_openai_assistant(
    api_key: str = None,
    model: str = "gpt-4",
    name: str = "OpenAI Assistant",
    **kwargs
) -> PersonalAssistant:
    """
    Create a PersonalAssistant powered by OpenAI.

    Args:
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: Model to use (gpt-4, gpt-4-turbo, gpt-3.5-turbo)
        name: Assistant name
        **kwargs: Additional PersonalAssistant arguments

    Example:
        assistant = create_openai_assistant(api_key="sk-...")
        await assistant.initialize()
        response = await assistant.chat("Hello!")
    """
    from .llm import OpenAIProvider

    provider = OpenAIProvider(api_key=api_key, model=model)
    return PersonalAssistant(
        name=name,
        llm_provider=provider,
        **kwargs
    )


def create_ollama_assistant(
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
    name: str = "Local Assistant",
    **kwargs
) -> PersonalAssistant:
    """
    Create a PersonalAssistant powered by Ollama (local LLM).

    Requires Ollama to be installed and running.

    Args:
        model: Ollama model name (llama2, mistral, codellama, etc.)
        base_url: Ollama server URL
        name: Assistant name
        **kwargs: Additional PersonalAssistant arguments

    Example:
        # First: ollama pull llama2
        assistant = create_ollama_assistant(model="llama2")
        await assistant.initialize()
        response = await assistant.chat("Hello!")
    """
    from .llm import OllamaProvider

    provider = OllamaProvider(model=model, base_url=base_url)
    return PersonalAssistant(
        name=name,
        llm_provider=provider,
        **kwargs
    )


async def quick_chat(message: str, provider: str = "ollama", model: str = None) -> str:
    """
    Quick LLM chat without managing lifecycle.

    Args:
        message: Your message
        provider: "openai" or "ollama"
        model: Model name (optional)

    Example:
        response = await quick_chat("Tell me a joke", provider="ollama")
    """
    if provider == "openai":
        from .llm import OpenAIProvider
        llm = OpenAIProvider(model=model or "gpt-3.5-turbo")
    else:
        from .llm import OllamaProvider
        llm = OllamaProvider(model=model or "llama2")

    assistant = PersonalAssistant(llm_provider=llm)
    async with assistant:
        return await assistant.chat(message)
