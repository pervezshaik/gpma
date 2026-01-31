"""
LLM Agent Module

Provides an intelligent agent powered by Large Language Models.

This agent can:
- Understand natural language requests
- Generate intelligent responses
- Use tools with function calling
- Maintain conversation context

USAGE:
    from gpma.llm import LLMAgent, OpenAIProvider

    provider = OpenAIProvider(api_key="sk-...")
    agent = LLMAgent(llm_provider=provider)

    result = await agent.run_task({
        "input": "What is the capital of France?"
    })
    print(result.data)  # "The capital of France is Paris..."
"""

from typing import Any, Dict, List, Optional, Callable
import json
import logging

from ..core.base_agent import BaseAgent, AgentCapability, TaskResult, Tool
from .providers import LLMProvider, LLMConfig, Message, MessageRole, LLMResponse

logger = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """
    An intelligent agent powered by an LLM.

    This agent uses an LLM to:
    1. Understand user requests in natural language
    2. Generate intelligent, contextual responses
    3. Optionally call tools/functions when needed

    FEATURES:
    - Conversation memory (maintains context)
    - System prompt customization
    - Tool/function calling support
    - Streaming responses (optional)

    EXAMPLE:
        # Basic usage
        provider = OpenAIProvider(api_key="sk-...")
        agent = LLMAgent(
            llm_provider=provider,
            system_prompt="You are a helpful coding assistant."
        )

        result = await agent.run_task({
            "input": "Explain Python decorators"
        })

        # With tools
        agent = LLMAgent(llm_provider=provider)
        agent.register_llm_tool(
            name="get_weather",
            description="Get current weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            },
            function=get_weather_func
        )
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. You provide accurate,
concise, and helpful responses. When you don't know something, you say so honestly.
When asked to perform tasks, you break them down into clear steps."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        name: str = None,
        system_prompt: str = None,
        max_history: int = 20,
        enable_tools: bool = True
    ):
        """
        Initialize the LLM agent.

        Args:
            llm_provider: The LLM provider to use (OpenAI, Ollama, etc.)
            name: Agent name
            system_prompt: Custom system instructions
            max_history: Maximum conversation turns to remember
            enable_tools: Whether to enable tool/function calling
        """
        super().__init__(name or "LLMAgent")

        self.llm = llm_provider
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.max_history = max_history
        self.enable_tools = enable_tools

        # Conversation history
        self._conversation: List[Message] = []

        # LLM-callable tools (different from base agent tools)
        self._llm_tools: Dict[str, Dict[str, Any]] = {}
        self._tool_functions: Dict[str, Callable] = {}

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="general_intelligence",
                description="Answer questions and perform tasks using AI",
                keywords=[
                    "explain", "write", "create", "help", "what", "how", "why",
                    "tell", "describe", "analyze", "suggest", "recommend"
                ],
                priority=1  # Lower priority - acts as fallback
            ),
            AgentCapability(
                name="conversation",
                description="Have natural conversations",
                keywords=["chat", "talk", "discuss", "conversation"],
                priority=2
            ),
            AgentCapability(
                name="reasoning",
                description="Complex reasoning and problem solving",
                keywords=["think", "reason", "solve", "figure", "understand"],
                priority=2
            )
        ]

    def register_llm_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ) -> None:
        """
        Register a tool that the LLM can call.

        This uses OpenAI's function calling format, which is also
        supported by many other providers.

        Args:
            name: Tool name (used by LLM to call it)
            description: What the tool does (helps LLM decide when to use it)
            parameters: JSON Schema for parameters
            function: The actual function to execute

        Example:
            agent.register_llm_tool(
                name="search_web",
                description="Search the web for information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                },
                function=my_search_function
            )
        """
        self._llm_tools[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        self._tool_functions[name] = function
        logger.debug(f"Registered LLM tool: {name}")

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt."""
        self.system_prompt = prompt

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history as list of dicts."""
        return [
            {"role": m.role.value, "content": m.content}
            for m in self._conversation
        ]

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a task using the LLM.

        Task format:
        {
            "input": "User's message or question",
            "action": "chat" | "complete" | "analyze",  # optional
            "context": {...},  # optional additional context
            "config": {...}  # optional LLM config overrides
        }
        """
        input_text = task.get("input", "")
        action = task.get("action", "chat")
        context = task.get("context", {})
        config_override = task.get("config", {})

        if not input_text:
            return TaskResult(
                success=False,
                data=None,
                error="No input provided"
            )

        try:
            if action == "complete":
                # Single completion without history
                return await self._handle_completion(input_text, context, config_override)
            elif action == "analyze":
                # Analysis mode with specific prompt
                return await self._handle_analysis(input_text, context, config_override)
            else:
                # Default: conversational chat
                return await self._handle_chat(input_text, context, config_override)

        except Exception as e:
            logger.error(f"LLM Agent error: {e}")
            return TaskResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def _handle_chat(
        self,
        user_input: str,
        context: Dict[str, Any],
        config_override: Dict[str, Any]
    ) -> TaskResult:
        """Handle conversational chat with history."""

        # Add user message to history
        self._conversation.append(Message(MessageRole.USER, user_input))

        # Build messages with system prompt
        messages = [Message(MessageRole.SYSTEM, self.system_prompt)]

        # Add context if provided
        if context:
            context_str = f"Additional context: {json.dumps(context)}"
            messages.append(Message(MessageRole.SYSTEM, context_str))

        # Add conversation history (limited)
        messages.extend(self._conversation[-self.max_history:])

        # Prepare config
        config = None
        if config_override:
            config = LLMConfig(**config_override)

        # Add tools if enabled
        if self.enable_tools and self._llm_tools:
            if config is None:
                config = LLMConfig()
            config.tools = list(self._llm_tools.values())
            config.tool_choice = "auto"

        # Call LLM
        response = await self.llm.chat(messages, config)

        # Handle tool calls if present
        if response.has_tool_calls:
            response = await self._handle_tool_calls(response, messages, config)

        # Add assistant response to history
        self._conversation.append(Message(MessageRole.ASSISTANT, response.content))

        # Trim history if too long
        while len(self._conversation) > self.max_history * 2:
            self._conversation.pop(0)

        return TaskResult(
            success=True,
            data=response.content,
            metadata={
                "model": response.model,
                "usage": response.usage,
                "finish_reason": response.finish_reason
            }
        )

    async def _handle_completion(
        self,
        prompt: str,
        context: Dict[str, Any],
        config_override: Dict[str, Any]
    ) -> TaskResult:
        """Handle single completion without history."""

        # Build full prompt with context
        full_prompt = prompt
        if context:
            full_prompt = f"Context: {json.dumps(context)}\n\n{prompt}"

        config = LLMConfig(**config_override) if config_override else None

        response = await self.llm.generate(
            full_prompt,
            system_prompt=self.system_prompt,
            config=config
        )

        return TaskResult(
            success=True,
            data=response.content,
            metadata={
                "model": response.model,
                "usage": response.usage
            }
        )

    async def _handle_analysis(
        self,
        content: str,
        context: Dict[str, Any],
        config_override: Dict[str, Any]
    ) -> TaskResult:
        """Handle content analysis."""

        analysis_prompt = f"""Analyze the following content and provide:
1. A brief summary
2. Key points or findings
3. Any notable patterns or insights

Content to analyze:
{content}"""

        config = LLMConfig(**config_override) if config_override else None

        response = await self.llm.generate(
            analysis_prompt,
            system_prompt="You are an expert analyst. Provide clear, structured analysis.",
            config=config
        )

        return TaskResult(
            success=True,
            data={
                "analysis": response.content,
                "original_length": len(content)
            },
            metadata={"model": response.model}
        )

    async def _handle_tool_calls(
        self,
        response: LLMResponse,
        messages: List[Message],
        config: LLMConfig
    ) -> LLMResponse:
        """Handle tool calls from the LLM."""

        tool_calls = response.tool_calls
        if not tool_calls:
            return response

        # Add assistant's response with tool calls
        assistant_msg = Message(
            MessageRole.ASSISTANT,
            response.content or "",
            tool_calls=tool_calls
        )
        messages.append(assistant_msg)

        # Execute each tool call
        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")
            tool_args = json.loads(function.get("arguments", "{}"))
            tool_id = tool_call.get("id", "")

            if tool_name in self._tool_functions:
                try:
                    # Execute the tool
                    func = self._tool_functions[tool_name]
                    if asyncio.iscoroutinefunction(func):
                        result = await func(**tool_args)
                    else:
                        result = func(**tool_args)

                    result_str = json.dumps(result) if not isinstance(result, str) else result

                except Exception as e:
                    result_str = f"Error executing tool: {str(e)}"

                # Add tool result to messages
                tool_msg = Message(
                    MessageRole.TOOL,
                    result_str,
                    name=tool_name,
                    tool_call_id=tool_id
                )
                messages.append(tool_msg)
            else:
                # Unknown tool
                messages.append(Message(
                    MessageRole.TOOL,
                    f"Unknown tool: {tool_name}",
                    name=tool_name,
                    tool_call_id=tool_id
                ))

        # Call LLM again with tool results
        return await self.llm.chat(messages, config)

    async def shutdown(self) -> None:
        """Clean up resources."""
        await self.llm.close()
        await super().shutdown()


# Need to import asyncio for the tool call handling
import asyncio


# ============================================================================
# SPECIALIZED LLM AGENTS
# ============================================================================

class CodeAssistantAgent(LLMAgent):
    """
    An LLM agent specialized for coding tasks.
    """

    CODE_SYSTEM_PROMPT = """You are an expert software engineer and coding assistant.
You help with:
- Writing clean, efficient code
- Debugging and fixing issues
- Explaining code concepts
- Code reviews and improvements
- Architecture and design decisions

When providing code:
- Use proper formatting with language-specific syntax
- Include helpful comments
- Follow best practices and conventions
- Consider edge cases and error handling"""

    def __init__(self, llm_provider: LLMProvider, name: str = None):
        super().__init__(
            llm_provider=llm_provider,
            name=name or "CodeAssistant",
            system_prompt=self.CODE_SYSTEM_PROMPT
        )

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="coding",
                description="Write and debug code",
                keywords=[
                    "code", "program", "function", "class", "bug", "fix",
                    "implement", "debug", "error", "syntax", "algorithm"
                ],
                priority=5
            ),
            AgentCapability(
                name="code_review",
                description="Review and improve code",
                keywords=["review", "improve", "refactor", "optimize", "clean"],
                priority=4
            )
        ]


class ResearchAssistantAgent(LLMAgent):
    """
    An LLM agent specialized for research and analysis.
    """

    RESEARCH_SYSTEM_PROMPT = """You are a research assistant with expertise in:
- Gathering and synthesizing information
- Critical analysis and evaluation
- Identifying patterns and insights
- Providing balanced, well-sourced perspectives

When researching:
- Consider multiple perspectives
- Distinguish facts from opinions
- Acknowledge limitations and uncertainties
- Provide clear, structured summaries"""

    def __init__(self, llm_provider: LLMProvider, name: str = None):
        super().__init__(
            llm_provider=llm_provider,
            name=name or "ResearchAssistant",
            system_prompt=self.RESEARCH_SYSTEM_PROMPT
        )

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="research",
                description="Research and analyze topics",
                keywords=[
                    "research", "analyze", "investigate", "study",
                    "compare", "evaluate", "assess"
                ],
                priority=4
            ),
            AgentCapability(
                name="summarize",
                description="Summarize information",
                keywords=["summarize", "summary", "brief", "overview", "tldr"],
                priority=4
            )
        ]


class WritingAssistantAgent(LLMAgent):
    """
    An LLM agent specialized for writing tasks.
    """

    WRITING_SYSTEM_PROMPT = """You are a professional writer and editor.
You help with:
- Writing clear, engaging content
- Editing and proofreading
- Adapting tone and style for different audiences
- Structuring documents effectively

When writing:
- Use clear, concise language
- Maintain consistent tone and style
- Structure content logically
- Consider the target audience"""

    def __init__(self, llm_provider: LLMProvider, name: str = None):
        super().__init__(
            llm_provider=llm_provider,
            name=name or "WritingAssistant",
            system_prompt=self.WRITING_SYSTEM_PROMPT
        )

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="writing",
                description="Write content and documents",
                keywords=[
                    "write", "draft", "compose", "create", "document",
                    "article", "email", "letter", "report"
                ],
                priority=4
            ),
            AgentCapability(
                name="editing",
                description="Edit and proofread text",
                keywords=["edit", "proofread", "revise", "improve", "rewrite"],
                priority=4
            )
        ]
