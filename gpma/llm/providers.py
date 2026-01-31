"""
LLM Providers Module

Abstraction layer for different LLM providers.

SUPPORTED PROVIDERS:
1. OpenAI - GPT-4, GPT-4o, GPT-3.5-turbo
2. Ollama - Local models (Llama2, Mistral, CodeLlama, etc.)
3. Azure OpenAI - Enterprise OpenAI

DESIGN PATTERN: Strategy Pattern
- LLMProvider is the abstract interface
- Each provider implements the same interface
- Agents can swap providers without code changes

USAGE:
    # OpenAI
    provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
    response = await provider.generate("Hello, how are you?")

    # Ollama (local)
    provider = OllamaProvider(model="llama2", base_url="http://localhost:11434")
    response = await provider.generate("Hello, how are you?")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from enum import Enum
import asyncio
import aiohttp
import json
import logging

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    name: Optional[str] = None  # For function/tool messages
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API-compatible dictionary."""
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class LLMConfig:
    """Configuration for LLM generation."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: Optional[List[str]] = None
    stream: bool = False

    # Tool/Function calling
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None  # "auto", "none", or specific tool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API parameters."""
        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.frequency_penalty:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        if self.stop:
            params["stop"] = self.stop
        if self.tools:
            params["tools"] = self.tools
        if self.tool_choice:
            params["tool_choice"] = self.tool_choice
        return params


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    finish_reason: str
    usage: Dict[str, int] = field(default_factory=dict)
    tool_calls: Optional[List[Dict]] = None
    raw_response: Optional[Dict] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - generate(): Single completion
    - generate_stream(): Streaming completion
    - chat(): Multi-turn conversation
    """

    def __init__(self, model: str, config: LLMConfig = None):
        self.model = model
        self.config = config or LLMConfig()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Generate a single completion.

        Args:
            prompt: The user prompt
            system_prompt: Optional system instructions
            config: Override default config

        Returns:
            LLMResponse with the completion
        """
        pass

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """
        Multi-turn conversation.

        Args:
            messages: List of conversation messages
            config: Override default config

        Returns:
            LLMResponse with the assistant's reply
        """
        pass

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> AsyncIterator[str]:
        """
        Stream a completion token by token.

        Default implementation falls back to non-streaming.
        Override for true streaming support.
        """
        response = await self.generate(prompt, system_prompt, config)
        yield response.content

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ============================================================================
# OPENAI PROVIDER
# ============================================================================

class OpenAIProvider(LLMProvider):
    """
    OpenAI API provider.

    Supports: GPT-4, GPT-4-turbo, GPT-4o, GPT-3.5-turbo

    USAGE:
        provider = OpenAIProvider(
            api_key="sk-...",
            model="gpt-4",
            config=LLMConfig(temperature=0.7)
        )

        response = await provider.generate("What is Python?")
        print(response.content)

    ENVIRONMENT VARIABLE:
        Set OPENAI_API_KEY to avoid passing api_key explicitly
    """

    API_BASE = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-4",
        config: LLMConfig = None,
        organization: str = None
    ):
        super().__init__(model, config)

        # Get API key from env if not provided
        import os
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass api_key.")

        self.organization = organization

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        return headers

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        messages = []
        if system_prompt:
            messages.append(Message(MessageRole.SYSTEM, system_prompt))
        messages.append(Message(MessageRole.USER, prompt))

        return await self.chat(messages, config)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Multi-turn chat with OpenAI."""
        cfg = config or self.config
        session = await self._get_session()

        # Build request
        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            **cfg.to_dict()
        }

        try:
            async with session.post(
                f"{self.API_BASE}/chat/completions",
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenAI API error: {response.status} - {error_text}")

                data = await response.json()

                choice = data["choices"][0]
                message = choice["message"]

                return LLMResponse(
                    content=message.get("content", ""),
                    model=data.get("model", self.model),
                    finish_reason=choice.get("finish_reason", ""),
                    usage=data.get("usage", {}),
                    tool_calls=message.get("tool_calls"),
                    raw_response=data
                )

        except aiohttp.ClientError as e:
            logger.error(f"OpenAI request failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> AsyncIterator[str]:
        """Stream completion from OpenAI."""
        messages = []
        if system_prompt:
            messages.append(Message(MessageRole.SYSTEM, system_prompt))
        messages.append(Message(MessageRole.USER, prompt))

        cfg = config or self.config
        session = await self._get_session()

        payload = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            **cfg.to_dict()
        }

        async with session.post(
            f"{self.API_BASE}/chat/completions",
            headers=self._get_headers(),
            json=payload
        ) as response:
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


# ============================================================================
# OLLAMA PROVIDER (Local Models)
# ============================================================================

class OllamaProvider(LLMProvider):
    """
    Ollama provider for local LLM models.

    Ollama lets you run LLMs locally. Supports:
    - Llama 2, Llama 3
    - Mistral, Mixtral
    - CodeLlama
    - Phi-2
    - And many more...

    SETUP:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull llama2
        3. Start server: ollama serve (usually auto-starts)

    USAGE:
        provider = OllamaProvider(model="llama2")
        response = await provider.generate("What is Python?")

        # With custom host
        provider = OllamaProvider(
            model="mistral",
            base_url="http://192.168.1.100:11434"
        )
    """

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        model: str = "llama2",
        base_url: str = None,
        config: LLMConfig = None
    ):
        super().__init__(model, config)
        self.base_url = base_url or self.DEFAULT_BASE_URL

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate completion using Ollama."""
        cfg = config or self.config
        session = await self._get_session()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        if cfg.stop:
            payload["options"]["stop"] = cfg.stop

        try:
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)  # Longer timeout for local
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")

                data = await response.json()

                return LLMResponse(
                    content=data.get("response", ""),
                    model=data.get("model", self.model),
                    finish_reason="stop" if data.get("done") else "length",
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (
                            data.get("prompt_eval_count", 0) +
                            data.get("eval_count", 0)
                        )
                    },
                    raw_response=data
                )

        except aiohttp.ClientError as e:
            logger.error(f"Ollama request failed: {e}")
            raise Exception(f"Failed to connect to Ollama at {self.base_url}. Is it running?")

    async def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Multi-turn chat with Ollama."""
        cfg = config or self.config
        session = await self._get_session()

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "top_p": cfg.top_p,
            }
        }

        try:
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error: {response.status} - {error_text}")

                data = await response.json()
                message = data.get("message", {})

                return LLMResponse(
                    content=message.get("content", ""),
                    model=data.get("model", self.model),
                    finish_reason="stop" if data.get("done") else "length",
                    usage={
                        "prompt_tokens": data.get("prompt_eval_count", 0),
                        "completion_tokens": data.get("eval_count", 0),
                        "total_tokens": (
                            data.get("prompt_eval_count", 0) +
                            data.get("eval_count", 0)
                        )
                    },
                    raw_response=data
                )

        except aiohttp.ClientError as e:
            logger.error(f"Ollama chat failed: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> AsyncIterator[str]:
        """Stream completion from Ollama."""
        cfg = config or self.config
        session = await self._get_session()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            async for line in response.content:
                try:
                    data = json.loads(line)
                    content = data.get("response", "")
                    if content:
                        yield content
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        session = await self._get_session()

        async with session.get(f"{self.base_url}/api/tags") as response:
            if response.status != 200:
                return []
            data = await response.json()
            return data.get("models", [])

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama library."""
        session = await self._get_session()

        async with session.post(
            f"{self.base_url}/api/pull",
            json={"name": model_name}
        ) as response:
            return response.status == 200


# ============================================================================
# AZURE OPENAI PROVIDER
# ============================================================================

class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI Service provider.

    For enterprise deployments of OpenAI models.

    USAGE:
        provider = AzureOpenAIProvider(
            api_key="...",
            endpoint="https://your-resource.openai.azure.com",
            deployment_name="gpt-4",
            api_version="2024-02-01"
        )
    """

    def __init__(
        self,
        api_key: str = None,
        endpoint: str = None,
        deployment_name: str = None,
        api_version: str = "2024-02-01",
        config: LLMConfig = None
    ):
        super().__init__(deployment_name or "gpt-4", config)

        import os
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self.api_version = api_version

        if not all([self.api_key, self.endpoint, self.deployment_name]):
            raise ValueError(
                "Azure OpenAI requires: api_key, endpoint, deployment_name. "
                "Set via params or environment variables."
            )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def _get_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}"
            f"/chat/completions?api-version={self.api_version}"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append(Message(MessageRole.SYSTEM, system_prompt))
        messages.append(Message(MessageRole.USER, prompt))
        return await self.chat(messages, config)

    async def chat(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        cfg = config or self.config
        session = await self._get_session()

        payload = {
            "messages": [m.to_dict() for m in messages],
            **cfg.to_dict()
        }

        try:
            async with session.post(
                self._get_url(),
                headers=self._get_headers(),
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Azure OpenAI error: {response.status} - {error_text}")

                data = await response.json()
                choice = data["choices"][0]
                message = choice["message"]

                return LLMResponse(
                    content=message.get("content", ""),
                    model=self.deployment_name,
                    finish_reason=choice.get("finish_reason", ""),
                    usage=data.get("usage", {}),
                    tool_calls=message.get("tool_calls"),
                    raw_response=data
                )

        except aiohttp.ClientError as e:
            logger.error(f"Azure OpenAI request failed: {e}")
            raise


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_provider(
    provider_type: str,
    model: str = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_type: "openai", "ollama", or "azure"
        model: Model name
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLMProvider instance

    Example:
        provider = create_provider("openai", model="gpt-4", api_key="sk-...")
        provider = create_provider("ollama", model="llama2")
    """
    providers = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "azure": AzureOpenAIProvider,
    }

    if provider_type not in providers:
        raise ValueError(f"Unknown provider: {provider_type}. Options: {list(providers.keys())}")

    provider_class = providers[provider_type]

    if model:
        kwargs["model"] = model

    return provider_class(**kwargs)
