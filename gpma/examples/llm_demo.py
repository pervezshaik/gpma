"""
LLM Integration Demo

This script demonstrates how to use GPMA with LLM providers.

PREREQUISITES:
1. For OpenAI: Set OPENAI_API_KEY environment variable or pass api_key
2. For Ollama: Install Ollama and run `ollama pull llama2`

RUN THIS DEMO:
    python -m gpma.examples.llm_demo

DEMO OPTIONS:
    python -m gpma.examples.llm_demo openai      # OpenAI demos
    python -m gpma.examples.llm_demo ollama      # Ollama demos
    python -m gpma.examples.llm_demo all         # All demos
"""

import asyncio
import os
import sys
import logging

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# OPENAI DEMOS
# ============================================================================

async def demo_openai_basic():
    """
    Basic OpenAI integration demo.
    """
    print("\n" + "="*60)
    print("DEMO: OpenAI Basic Chat")
    print("="*60)

    from gpma.llm import OpenAIProvider, LLMAgent

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI demos.")
        print("   Set it with: export OPENAI_API_KEY='sk-...'")
        return

    try:
        # Create provider
        provider = OpenAIProvider(
            api_key=api_key,
            model="gpt-3.5-turbo"  # Use cheaper model for demo
        )

        print("\n--- Simple Generation ---")
        response = await provider.generate(
            "What is Python in one sentence?",
            system_prompt="You are a concise assistant."
        )
        print(f"Response: {response.content}")
        print(f"Tokens used: {response.total_tokens}")

        # Create an agent
        print("\n--- LLM Agent Chat ---")
        agent = LLMAgent(
            llm_provider=provider,
            system_prompt="You are a helpful coding assistant."
        )
        await agent.initialize()

        result = await agent.run_task({
            "input": "What's the difference between a list and a tuple in Python?",
            "action": "chat"
        })

        if result.success:
            print(f"Agent response: {result.data[:300]}...")

        await agent.shutdown()
        await provider.close()

        print("\n✅ OpenAI basic demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_openai_assistant():
    """
    Full PersonalAssistant with OpenAI demo.
    """
    print("\n" + "="*60)
    print("DEMO: PersonalAssistant with OpenAI")
    print("="*60)

    from gpma import PersonalAssistant
    from gpma.llm import OpenAIProvider

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping.")
        return

    try:
        provider = OpenAIProvider(api_key=api_key, model="gpt-3.5-turbo")

        assistant = PersonalAssistant(
            name="GPT Assistant",
            llm_provider=provider
        )

        async with assistant:
            print("\n--- Chat Mode ---")
            response = await assistant.chat("Tell me a short joke about programmers")
            print(f"Response: {response}")

            print("\n--- Ask Mode (routes to best agent) ---")
            response = await assistant.ask("What is machine learning?")
            print(f"Response: {response[:200]}...")

            print("\n--- Status ---")
            status = assistant.get_status()
            print(f"Agents: {status['agents']}")
            print(f"LLM enabled: {'llm' in assistant.get_agents()}")

        print("\n✅ OpenAI assistant demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_openai_tools():
    """
    Demo of LLM with tool/function calling.
    """
    print("\n" + "="*60)
    print("DEMO: OpenAI with Tool Calling")
    print("="*60)

    from gpma.llm import OpenAIProvider, LLMAgent

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set. Skipping.")
        return

    try:
        provider = OpenAIProvider(api_key=api_key, model="gpt-3.5-turbo")

        agent = LLMAgent(
            llm_provider=provider,
            enable_tools=True
        )

        # Register a custom tool
        def get_current_time(timezone: str = "UTC") -> str:
            """Get the current time in a timezone."""
            from datetime import datetime
            return f"Current time ({timezone}): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        agent.register_llm_tool(
            name="get_current_time",
            description="Get the current date and time",
            parameters={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "Timezone (e.g., UTC, EST, PST)"
                    }
                },
                "required": []
            },
            function=get_current_time
        )

        await agent.initialize()

        print("\n--- Asking about time (should trigger tool) ---")
        result = await agent.run_task({
            "input": "What time is it right now?",
            "action": "chat"
        })

        if result.success:
            print(f"Response: {result.data}")

        await agent.shutdown()
        await provider.close()

        print("\n✅ Tool calling demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# OLLAMA DEMOS (Local Models)
# ============================================================================

async def demo_ollama_basic():
    """
    Basic Ollama (local model) demo.
    """
    print("\n" + "="*60)
    print("DEMO: Ollama Basic Chat (Local)")
    print("="*60)

    from gpma.llm import OllamaProvider

    try:
        # Create provider
        provider = OllamaProvider(
            model="llama2",  # Or: mistral, codellama, phi, etc.
            base_url="http://localhost:11434"
        )

        # Check if Ollama is running
        print("Checking Ollama connection...")
        try:
            models = await provider.list_models()
            print(f"Available models: {[m.get('name') for m in models]}")
        except Exception:
            print("⚠️  Cannot connect to Ollama. Is it running?")
            print("   Start with: ollama serve")
            print("   Pull a model: ollama pull llama2")
            return

        print("\n--- Simple Generation ---")
        response = await provider.generate(
            "What is Python? Answer in one sentence.",
            system_prompt="You are a concise assistant."
        )
        print(f"Response: {response.content}")

        print("\n--- Multi-turn Chat ---")
        from gpma.llm.providers import Message, MessageRole

        messages = [
            Message(MessageRole.USER, "My name is Alice."),
        ]

        response1 = await provider.chat(messages)
        print(f"Assistant: {response1.content[:100]}...")

        messages.append(Message(MessageRole.ASSISTANT, response1.content))
        messages.append(Message(MessageRole.USER, "What's my name?"))

        response2 = await provider.chat(messages)
        print(f"Assistant: {response2.content}")

        await provider.close()
        print("\n✅ Ollama basic demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_ollama_assistant():
    """
    Full PersonalAssistant with Ollama demo.
    """
    print("\n" + "="*60)
    print("DEMO: PersonalAssistant with Ollama (Local)")
    print("="*60)

    from gpma import PersonalAssistant
    from gpma.llm import OllamaProvider

    try:
        provider = OllamaProvider(model="llama2")

        # Quick check
        try:
            await provider.list_models()
        except Exception:
            print("⚠️  Ollama not available. Skipping.")
            return

        assistant = PersonalAssistant(
            name="Local Assistant",
            llm_provider=provider
        )

        async with assistant:
            print("\n--- Chat Mode ---")
            response = await assistant.chat("What is the capital of France?")
            print(f"Response: {response[:200]}...")

            print("\n--- Memory ---")
            assistant.remember("favorite_language", "Python", permanent=True)
            recalled = assistant.recall("favorite_language")
            print(f"Remembered: {recalled}")

        print("\n✅ Ollama assistant demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


async def demo_ollama_streaming():
    """
    Demo of streaming responses from Ollama.
    """
    print("\n" + "="*60)
    print("DEMO: Ollama Streaming")
    print("="*60)

    from gpma.llm import OllamaProvider

    try:
        provider = OllamaProvider(model="llama2")

        try:
            await provider.list_models()
        except Exception:
            print("⚠️  Ollama not available. Skipping.")
            return

        print("\n--- Streaming Response ---")
        print("Response: ", end="", flush=True)

        async for token in provider.generate_stream(
            "Write a haiku about programming.",
            system_prompt="You are a poet."
        ):
            print(token, end="", flush=True)

        print("\n")
        await provider.close()
        print("\n✅ Streaming demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# TASK DECOMPOSITION DEMO
# ============================================================================

async def demo_task_decomposition():
    """
    Demo of LLM-powered task decomposition.
    """
    print("\n" + "="*60)
    print("DEMO: LLM Task Decomposition")
    print("="*60)

    from gpma.llm import OllamaProvider, LLMTaskDecomposer

    try:
        provider = OllamaProvider(model="llama2")

        try:
            await provider.list_models()
        except Exception:
            print("⚠️  Ollama not available. Trying OpenAI...")
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                from gpma.llm import OpenAIProvider
                provider = OpenAIProvider(api_key=api_key, model="gpt-3.5-turbo")
            else:
                print("No LLM available. Skipping.")
                return

        decomposer = LLMTaskDecomposer(provider)

        # Test task
        complex_task = "Research the latest AI trends, summarize the top 3 findings, and create a report"

        print(f"\n--- Decomposing Task ---")
        print(f"Task: {complex_task}")

        result = await decomposer.decompose(complex_task)

        print(f"\nMain Goal: {result.get('main_goal', 'N/A')}")
        print(f"Strategy: {result.get('execution_strategy', 'N/A')}")
        print(f"\nSubtasks:")

        for st in result.get("subtasks", []):
            print(f"  {st.id}. {st.description}")
            print(f"     Action: {st.action}, Agent: {st.agent_hint}")
            if st.dependencies:
                print(f"     Depends on: {st.dependencies}")

        await provider.close()
        print("\n✅ Task decomposition demo completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


# ============================================================================
# QUICK START EXAMPLES
# ============================================================================

async def demo_quick_start():
    """
    Quick start examples for common use cases.
    """
    print("\n" + "="*60)
    print("QUICK START EXAMPLES")
    print("="*60)

    print("""
# ============================================================
# Example 1: OpenAI Assistant
# ============================================================

from gpma import PersonalAssistant
from gpma.llm import OpenAIProvider

# Create with OpenAI
provider = OpenAIProvider(api_key="sk-...", model="gpt-4")
assistant = PersonalAssistant(llm_provider=provider)

await assistant.initialize()

# Chat naturally
response = await assistant.chat("Explain quantum computing simply")
print(response)

# Ask anything (routes to best agent)
response = await assistant.ask("Search the web for Python tutorials")
print(response)

await assistant.shutdown()


# ============================================================
# Example 2: Local Ollama Assistant
# ============================================================

from gpma import PersonalAssistant
from gpma.llm import OllamaProvider

# Create with Ollama (local)
provider = OllamaProvider(model="llama2")  # or mistral, codellama, etc.
assistant = PersonalAssistant(llm_provider=provider)

await assistant.initialize()

# Use it the same way
response = await assistant.chat("Write a Python function to reverse a string")
print(response)

await assistant.shutdown()


# ============================================================
# Example 3: Using Factory Functions
# ============================================================

from gpma.personal_assistant import create_openai_assistant, create_ollama_assistant

# Quick OpenAI setup
assistant = create_openai_assistant(api_key="sk-...", model="gpt-4")

# Quick Ollama setup
assistant = create_ollama_assistant(model="mistral")


# ============================================================
# Example 4: Custom LLM Agent
# ============================================================

from gpma.llm import LLMAgent, OpenAIProvider

provider = OpenAIProvider(api_key="sk-...")
agent = LLMAgent(
    llm_provider=provider,
    system_prompt="You are a Python expert. Always provide code examples."
)

# Register custom tools
agent.register_llm_tool(
    name="run_python",
    description="Execute Python code",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to run"}
        },
        "required": ["code"]
    },
    function=my_python_executor
)

await agent.initialize()
result = await agent.run_task({"input": "How do I read a JSON file?"})


# ============================================================
# Example 5: Streaming Responses
# ============================================================

from gpma.llm import OllamaProvider

provider = OllamaProvider(model="llama2")

async for token in provider.generate_stream("Tell me a story"):
    print(token, end="", flush=True)
""")

    print("="*60)


# ============================================================================
# MAIN
# ============================================================================

async def run_openai_demos():
    """Run all OpenAI demos."""
    await demo_openai_basic()
    await demo_openai_assistant()
    await demo_openai_tools()


async def run_ollama_demos():
    """Run all Ollama demos."""
    await demo_ollama_basic()
    await demo_ollama_assistant()
    await demo_ollama_streaming()


async def run_all_demos():
    """Run all demos."""
    await demo_quick_start()
    await run_ollama_demos()
    await run_openai_demos()
    await demo_task_decomposition()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        demo_type = sys.argv[1].lower()
        if demo_type == "openai":
            asyncio.run(run_openai_demos())
        elif demo_type == "ollama":
            asyncio.run(run_ollama_demos())
        elif demo_type == "decompose":
            asyncio.run(demo_task_decomposition())
        elif demo_type == "quick":
            asyncio.run(demo_quick_start())
        else:
            asyncio.run(run_all_demos())
    else:
        asyncio.run(run_all_demos())
