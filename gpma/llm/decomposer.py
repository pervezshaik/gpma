"""
LLM Task Decomposer Module

Uses an LLM to intelligently break down complex tasks into subtasks.

This replaces the simple regex-based decomposition with AI-powered
understanding of task requirements.

USAGE:
    from gpma.llm import LLMTaskDecomposer, OpenAIProvider

    provider = OpenAIProvider(api_key="sk-...")
    decomposer = LLMTaskDecomposer(provider)

    subtasks = await decomposer.decompose(
        "Research quantum computing, analyze the top 3 papers, and write a summary report"
    )
    # Returns list of Task objects with intelligent breakdown
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import json
import logging

from .providers import LLMProvider, LLMConfig, Message, MessageRole

logger = logging.getLogger(__name__)


@dataclass
class DecomposedTask:
    """A subtask from decomposition."""
    id: str
    description: str
    action: str
    dependencies: List[str]  # IDs of tasks this depends on
    agent_hint: Optional[str] = None  # Suggested agent type
    priority: int = 1
    estimated_complexity: str = "medium"  # low, medium, high


class LLMTaskDecomposer:
    """
    Intelligently decomposes complex tasks using an LLM.

    FEATURES:
    - Understands natural language task descriptions
    - Identifies dependencies between subtasks
    - Suggests appropriate agents for each subtask
    - Handles multi-step workflows

    DECOMPOSITION PROMPT ENGINEERING:
    The decomposer uses a carefully crafted prompt that instructs
    the LLM to:
    1. Identify the main goal
    2. Break it into logical steps
    3. Determine dependencies
    4. Assign complexity estimates

    EXAMPLE:
        Input: "Research AI trends, then create a presentation"

        Output:
        1. Task: "Search for recent AI trends articles"
           Agent: web_browser
           Dependencies: []

        2. Task: "Analyze and summarize findings"
           Agent: research
           Dependencies: [1]

        3. Task: "Create presentation outline"
           Agent: llm
           Dependencies: [2]

        4. Task: "Generate presentation content"
           Agent: llm
           Dependencies: [3]
    """

    DECOMPOSITION_PROMPT = '''You are a task decomposition expert. Break down the given task into clear, actionable subtasks.

For each subtask, provide:
1. A clear description of what needs to be done
2. The type of action (search, fetch, analyze, write, execute, etc.)
3. Dependencies on other subtasks (by number)
4. Suggested agent type: "web_browser", "research", "task_executor", "llm", or "general"
5. Complexity: "low", "medium", or "high"

Rules:
- Keep subtasks atomic and focused
- Order by logical execution sequence
- Identify true dependencies (what must complete first)
- Don't over-decompose simple tasks
- If the task is already simple, return just one subtask

Respond in JSON format:
{
    "main_goal": "Brief description of the overall goal",
    "subtasks": [
        {
            "id": "1",
            "description": "What to do",
            "action": "search|fetch|analyze|write|execute|other",
            "dependencies": [],
            "agent_hint": "web_browser|research|task_executor|llm|general",
            "complexity": "low|medium|high"
        }
    ],
    "execution_strategy": "sequential|parallel|pipeline"
}

Task to decompose:
{task}'''

    def __init__(
        self,
        llm_provider: LLMProvider,
        config: LLMConfig = None
    ):
        """
        Initialize the decomposer.

        Args:
            llm_provider: LLM provider for decomposition
            config: Optional LLM config (lower temperature recommended)
        """
        self.llm = llm_provider
        self.config = config or LLMConfig(
            temperature=0.3,  # Lower for more consistent output
            max_tokens=1024
        )

    async def decompose(
        self,
        task: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Decompose a task into subtasks.

        Args:
            task: The task description to decompose
            context: Optional context (available agents, user preferences, etc.)

        Returns:
            Dictionary with:
            - main_goal: str
            - subtasks: List[DecomposedTask]
            - execution_strategy: str
            - raw_response: str
        """
        # Build the prompt
        prompt = self.DECOMPOSITION_PROMPT.format(task=task)

        # Add context if provided
        if context:
            context_str = f"\nAvailable context:\n{json.dumps(context, indent=2)}"
            prompt += context_str

        try:
            # Get LLM response
            response = await self.llm.generate(
                prompt,
                system_prompt="You are a task planning expert. Always respond with valid JSON.",
                config=self.config
            )

            # Parse the JSON response
            result = self._parse_response(response.content)
            result["raw_response"] = response.content

            return result

        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            # Return simple single-task fallback
            return {
                "main_goal": task,
                "subtasks": [
                    DecomposedTask(
                        id="1",
                        description=task,
                        action="general",
                        dependencies=[],
                        agent_hint="general",
                        complexity="medium"
                    )
                ],
                "execution_strategy": "sequential",
                "error": str(e)
            }

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response."""
        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if "```json" in response:
                start = response.index("```json") + 7
                end = response.index("```", start)
                response = response[start:end].strip()
            elif "```" in response:
                start = response.index("```") + 3
                end = response.index("```", start)
                response = response[start:end].strip()

            data = json.loads(response)

            # Convert subtasks to DecomposedTask objects
            subtasks = []
            for st in data.get("subtasks", []):
                subtasks.append(DecomposedTask(
                    id=str(st.get("id", len(subtasks) + 1)),
                    description=st.get("description", ""),
                    action=st.get("action", "general"),
                    dependencies=st.get("dependencies", []),
                    agent_hint=st.get("agent_hint", "general"),
                    complexity=st.get("complexity", "medium")
                ))

            return {
                "main_goal": data.get("main_goal", ""),
                "subtasks": subtasks,
                "execution_strategy": data.get("execution_strategy", "sequential")
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse decomposition response: {e}")
            # Return the raw text as a single task
            return {
                "main_goal": response[:100],
                "subtasks": [
                    DecomposedTask(
                        id="1",
                        description=response,
                        action="general",
                        dependencies=[],
                        complexity="medium"
                    )
                ],
                "execution_strategy": "sequential"
            }

    async def analyze_complexity(self, task: str) -> Dict[str, Any]:
        """
        Analyze task complexity without full decomposition.

        Useful for deciding whether decomposition is needed.
        """
        prompt = f'''Analyze the complexity of this task and determine if it needs decomposition.

Task: {task}

Respond in JSON:
{{
    "complexity": "simple|moderate|complex",
    "needs_decomposition": true|false,
    "estimated_steps": 1-10,
    "required_capabilities": ["list", "of", "capabilities"],
    "reasoning": "Brief explanation"
}}'''

        try:
            response = await self.llm.generate(
                prompt,
                system_prompt="Analyze tasks briefly. Respond only with JSON.",
                config=LLMConfig(temperature=0.2, max_tokens=256)
            )

            # Parse JSON
            text = response.content
            if "```" in text:
                start = text.index("```") + 3
                if text[start:start+4] == "json":
                    start += 4
                end = text.index("```", start)
                text = text[start:end].strip()

            return json.loads(text)

        except Exception as e:
            return {
                "complexity": "moderate",
                "needs_decomposition": True,
                "estimated_steps": 3,
                "required_capabilities": ["general"],
                "error": str(e)
            }


class SmartDecomposer:
    """
    Advanced decomposer that considers available agents.

    This decomposer is aware of the agents in the system and
    creates decompositions optimized for the available capabilities.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        available_agents: List[str] = None
    ):
        self.llm = llm_provider
        self.base_decomposer = LLMTaskDecomposer(llm_provider)
        self.available_agents = available_agents or [
            "web_browser", "research", "task_executor", "llm"
        ]

    async def decompose(
        self,
        task: str,
        agent_capabilities: Dict[str, List[str]] = None
    ) -> Dict[str, Any]:
        """
        Decompose with awareness of available agents.

        Args:
            task: Task to decompose
            agent_capabilities: Map of agent names to their capabilities

        Returns:
            Optimized decomposition for available agents
        """
        context = {
            "available_agents": self.available_agents,
            "capabilities": agent_capabilities or {}
        }

        result = await self.base_decomposer.decompose(task, context)

        # Validate agent hints against available agents
        for subtask in result.get("subtasks", []):
            if hasattr(subtask, "agent_hint"):
                if subtask.agent_hint not in self.available_agents:
                    subtask.agent_hint = "general"

        return result

    async def optimize_execution_plan(
        self,
        subtasks: List[DecomposedTask]
    ) -> Dict[str, Any]:
        """
        Optimize the execution plan for subtasks.

        Determines which tasks can run in parallel,
        which need to be sequential, and the optimal order.
        """
        # Build dependency graph
        task_map = {st.id: st for st in subtasks}
        no_deps = [st for st in subtasks if not st.dependencies]
        has_deps = [st for st in subtasks if st.dependencies]

        # Group tasks that can run in parallel
        parallel_groups = []
        remaining = list(has_deps)
        current_group = list(no_deps)

        while current_group or remaining:
            if current_group:
                parallel_groups.append([t.id for t in current_group])

            # Find tasks whose dependencies are satisfied
            completed_ids = {tid for group in parallel_groups for tid in group}
            next_group = []

            for task in remaining[:]:
                if all(dep in completed_ids for dep in task.dependencies):
                    next_group.append(task)
                    remaining.remove(task)

            current_group = next_group

            if not current_group and remaining:
                # Circular dependency or error - just add remaining
                parallel_groups.append([t.id for t in remaining])
                break

        return {
            "parallel_groups": parallel_groups,
            "total_groups": len(parallel_groups),
            "can_parallelize": any(len(g) > 1 for g in parallel_groups)
        }
