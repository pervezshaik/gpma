"""
Goal-Oriented Behavior Module

This module implements hierarchical goal management, enabling agents to
pursue complex objectives through goal decomposition, tracking, and
adaptive replanning.

THE GOAL-ORIENTED PARADIGM:
Instead of just executing tasks, goal-oriented agents:
1. Understand high-level objectives
2. Break them into achievable subgoals
3. Track progress toward each goal
4. Adapt when obstacles arise
5. Opportunistically pursue goals

GOAL HIERARCHY:
    Strategic Goal (long-term)
        └── Tactical Goal (medium-term)
            └── Operational Goal (short-term)
                └── Action (immediate)

EXAMPLE:
    Strategic: "Become proficient in Python"
    ├── Tactical: "Learn web development"
    │   ├── Operational: "Complete Flask tutorial"
    │   │   ├── Action: "Read chapter 1"
    │   │   ├── Action: "Code the example"
    │   │   └── Action: "Run the tests"
    │   └── Operational: "Build a REST API"
    └── Tactical: "Learn data science"
        └── ...

USAGE:
    from gpma.core.goal_manager import GoalManager, Goal

    manager = GoalManager(llm_provider)

    # Set a high-level goal
    goal = await manager.set_goal("Research and summarize AI trends for 2024")

    # Pursue the goal
    result = await manager.pursue_goal(goal)

    print(result.achieved)
    print(result.artifacts)  # The outputs produced
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum, auto
from datetime import datetime
import asyncio
import json
import logging
import uuid

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status of a goal."""
    PENDING = auto()       # Not yet started
    ACTIVE = auto()        # Currently being pursued
    BLOCKED = auto()       # Cannot proceed (waiting for something)
    ACHIEVED = auto()      # Successfully completed
    FAILED = auto()        # Could not be achieved
    ABANDONED = auto()     # Intentionally stopped


class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class GoalType(Enum):
    """Types of goals."""
    ACHIEVE = "achieve"      # Reach a specific state
    MAINTAIN = "maintain"    # Keep a condition true
    AVOID = "avoid"          # Prevent something
    OPTIMIZE = "optimize"    # Improve a metric
    EXPLORE = "explore"      # Learn/discover


@dataclass
class Goal:
    """
    Represents a goal the agent is pursuing.

    Goals are hierarchical - a goal can have subgoals, and
    achieving all subgoals leads to achieving the parent goal.
    """
    id: str
    description: str
    goal_type: GoalType = GoalType.ACHIEVE
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING

    # Hierarchy
    parent_id: Optional[str] = None
    subgoal_ids: List[str] = field(default_factory=list)

    # Conditions
    preconditions: List[str] = field(default_factory=list)  # What must be true to start
    success_conditions: List[str] = field(default_factory=list)  # What indicates success
    failure_conditions: List[str] = field(default_factory=list)  # What indicates failure

    # Progress tracking
    progress: float = 0.0  # 0-1
    attempts: int = 0
    max_attempts: int = 3

    # Outputs
    artifacts: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None

    # Execution hints
    suggested_approach: Optional[str] = None
    required_capabilities: List[str] = field(default_factory=list)
    blocked_by: List[str] = field(default_factory=list)  # Goal IDs blocking this

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "goal_type": self.goal_type.value,
            "priority": self.priority.name,
            "status": self.status.name,
            "parent_id": self.parent_id,
            "subgoal_ids": self.subgoal_ids,
            "progress": self.progress,
            "attempts": self.attempts,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "success_conditions": self.success_conditions
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        return cls(
            id=data["id"],
            description=data["description"],
            goal_type=GoalType(data.get("goal_type", "achieve")),
            priority=GoalPriority[data.get("priority", "MEDIUM")],
            status=GoalStatus[data.get("status", "PENDING")],
            parent_id=data.get("parent_id"),
            subgoal_ids=data.get("subgoal_ids", []),
            progress=data.get("progress", 0.0),
            attempts=data.get("attempts", 0),
            artifacts=data.get("artifacts", {}),
            notes=data.get("notes", []),
            success_conditions=data.get("success_conditions", [])
        )

    def is_actionable(self) -> bool:
        """Check if this goal can be acted upon directly."""
        return len(self.subgoal_ids) == 0 and self.status == GoalStatus.PENDING

    def is_blocked(self) -> bool:
        """Check if this goal is blocked."""
        return len(self.blocked_by) > 0 or self.status == GoalStatus.BLOCKED

    def update_progress(self, new_progress: float):
        """Update progress, clamping to [0, 1]."""
        self.progress = max(0.0, min(1.0, new_progress))

    def add_artifact(self, key: str, value: Any):
        """Add an artifact produced while pursuing this goal."""
        self.artifacts[key] = value

    def add_note(self, note: str):
        """Add a note about progress or observations."""
        self.notes.append(f"[{datetime.now().strftime('%H:%M:%S')}] {note}")


@dataclass
class GoalTree:
    """
    A tree structure of goals.

    Manages the hierarchy of goals and provides operations for
    traversing and manipulating the tree.
    """
    goals: Dict[str, Goal] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)  # Top-level goals

    def add_goal(self, goal: Goal, parent_id: Optional[str] = None):
        """Add a goal to the tree."""
        self.goals[goal.id] = goal

        if parent_id and parent_id in self.goals:
            goal.parent_id = parent_id
            self.goals[parent_id].subgoal_ids.append(goal.id)
        else:
            self.root_ids.append(goal.id)

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self.goals.get(goal_id)

    def get_subgoals(self, goal_id: str) -> List[Goal]:
        """Get all direct subgoals of a goal."""
        goal = self.goals.get(goal_id)
        if not goal:
            return []
        return [self.goals[sid] for sid in goal.subgoal_ids if sid in self.goals]

    def get_parent(self, goal_id: str) -> Optional[Goal]:
        """Get the parent of a goal."""
        goal = self.goals.get(goal_id)
        if goal and goal.parent_id:
            return self.goals.get(goal.parent_id)
        return None

    def get_ancestors(self, goal_id: str) -> List[Goal]:
        """Get all ancestors of a goal (parent, grandparent, etc.)."""
        ancestors = []
        current_id = goal_id

        while current_id:
            goal = self.goals.get(current_id)
            if goal and goal.parent_id:
                parent = self.goals.get(goal.parent_id)
                if parent:
                    ancestors.append(parent)
                current_id = goal.parent_id
            else:
                break

        return ancestors

    def get_descendants(self, goal_id: str) -> List[Goal]:
        """Get all descendants of a goal (children, grandchildren, etc.)."""
        descendants = []
        to_visit = [goal_id]

        while to_visit:
            current_id = to_visit.pop(0)
            goal = self.goals.get(current_id)
            if goal:
                for subgoal_id in goal.subgoal_ids:
                    if subgoal_id in self.goals:
                        descendants.append(self.goals[subgoal_id])
                        to_visit.append(subgoal_id)

        return descendants

    def get_actionable_goals(self) -> List[Goal]:
        """Get goals that can be acted upon (no subgoals, not blocked)."""
        actionable = []

        for goal in self.goals.values():
            if goal.is_actionable() and not goal.is_blocked():
                actionable.append(goal)

        # Sort by priority
        actionable.sort(key=lambda g: g.priority.value)
        return actionable

    def get_blocked_goals(self) -> List[Goal]:
        """Get all blocked goals."""
        return [g for g in self.goals.values() if g.is_blocked()]

    def update_parent_progress(self, goal_id: str):
        """Update parent goal progress based on subgoal completion."""
        goal = self.goals.get(goal_id)
        if not goal or not goal.parent_id:
            return

        parent = self.goals.get(goal.parent_id)
        if not parent or not parent.subgoal_ids:
            return

        # Calculate average progress of subgoals
        subgoals = self.get_subgoals(goal.parent_id)
        if subgoals:
            avg_progress = sum(sg.progress for sg in subgoals) / len(subgoals)
            parent.update_progress(avg_progress)

            # Recurse to update grandparent, etc.
            self.update_parent_progress(goal.parent_id)

    def mark_achieved(self, goal_id: str):
        """Mark a goal as achieved and update tree."""
        goal = self.goals.get(goal_id)
        if not goal:
            return

        goal.status = GoalStatus.ACHIEVED
        goal.progress = 1.0
        goal.completed_at = datetime.now()

        # Update parent progress
        self.update_parent_progress(goal_id)

        # Check if parent is now achieved (all subgoals done)
        if goal.parent_id:
            parent = self.goals.get(goal.parent_id)
            if parent:
                subgoals = self.get_subgoals(goal.parent_id)
                if all(sg.status == GoalStatus.ACHIEVED for sg in subgoals):
                    self.mark_achieved(goal.parent_id)

    def unblock_dependents(self, goal_id: str):
        """Unblock goals that were waiting for this goal."""
        for goal in self.goals.values():
            if goal_id in goal.blocked_by:
                goal.blocked_by.remove(goal_id)
                if not goal.blocked_by and goal.status == GoalStatus.BLOCKED:
                    goal.status = GoalStatus.PENDING


@dataclass
class GoalResult:
    """Result of pursuing a goal."""
    goal_id: str
    achieved: bool
    status: GoalStatus
    progress: float
    artifacts: Dict[str, Any]
    notes: List[str]
    subgoal_results: List["GoalResult"]
    total_time: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "achieved": self.achieved,
            "status": self.status.name,
            "progress": self.progress,
            "artifacts": self.artifacts,
            "notes": self.notes,
            "subgoal_results": [sr.to_dict() for sr in self.subgoal_results],
            "total_time": self.total_time,
            "error": self.error
        }


class GoalManager:
    """
    Manages goal-oriented behavior for agents.

    The GoalManager enables agents to:
    1. Set high-level goals
    2. Decompose them into subgoals
    3. Track progress
    4. Handle blockers
    5. Adapt plans

    USAGE:
        manager = GoalManager(llm_provider)

        # Create a goal
        goal = await manager.set_goal(
            "Build a REST API for user management",
            priority=GoalPriority.HIGH
        )

        # Pursue it
        result = await manager.pursue_goal(goal.id)

        # Check status
        status = manager.get_goal_status(goal.id)
    """

    def __init__(
        self,
        llm_provider=None,
        max_depth: int = 5,
        auto_decompose: bool = True
    ):
        """
        Initialize the goal manager.

        Args:
            llm_provider: LLM for goal decomposition and reasoning
            max_depth: Maximum goal hierarchy depth
            auto_decompose: Automatically decompose complex goals
        """
        self.llm = llm_provider
        self.max_depth = max_depth
        self.auto_decompose = auto_decompose

        self.goal_tree = GoalTree()
        self._action_handlers: Dict[str, Callable] = {}
        self._stop_requested = False

    async def set_goal(
        self,
        description: str,
        goal_type: GoalType = GoalType.ACHIEVE,
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_id: Optional[str] = None,
        success_conditions: List[str] = None,
        deadline: Optional[datetime] = None
    ) -> Goal:
        """
        Set a new goal.

        If auto_decompose is enabled and the goal is complex,
        it will be automatically broken into subgoals.

        Args:
            description: What to achieve
            goal_type: Type of goal
            priority: Priority level
            parent_id: ID of parent goal (for subgoals)
            success_conditions: How to know goal is achieved
            deadline: Optional deadline

        Returns:
            The created Goal
        """
        goal = Goal(
            id=str(uuid.uuid4())[:8],
            description=description,
            goal_type=goal_type,
            priority=priority,
            parent_id=parent_id,
            success_conditions=success_conditions or [],
            deadline=deadline
        )

        self.goal_tree.add_goal(goal, parent_id)
        logger.info(f"Goal set: {goal.id} - {description}")

        # Auto-decompose if enabled and complex
        if self.auto_decompose and self.llm:
            depth = len(self.goal_tree.get_ancestors(goal.id))
            logger.info(f"Goal depth: {depth}, auto_decompose: {self.auto_decompose}")
            if depth < self.max_depth:
                is_complex = await self._is_complex_goal(goal)
                logger.info(f"Is goal complex? {is_complex}")
                if is_complex:
                    await self.decompose_goal(goal.id)
                else:
                    logger.info("Goal not complex enough for decomposition")
            else:
                logger.info(f"Max depth reached ({self.max_depth})")
        else:
            logger.info(f"Auto-decompose disabled or no LLM: {self.auto_decompose}, {self.llm is not None}")

        return goal

    async def _is_complex_goal(self, goal: Goal) -> bool:
        """Determine if a goal needs decomposition."""
        # Simple heuristics
        description = goal.description.lower()

        # Goals with multiple actions are likely complex
        complex_indicators = [
            " and ", " then ", "multiple", "several", "various",
            "first,", "second,", "finally,", "steps", "phases",
            "create", "build", "implement", "design", "develop",
            "api", "system", "application", "service"
        ]

        if any(indicator in description for indicator in complex_indicators):
            return True

        # Short, simple descriptions are likely atomic
        if len(description.split()) < 4:
            return False
        
        # Goals with specific technical terms are likely complex
        technical_indicators = [
            "api", "rest", "database", "authentication", "crud",
            "frontend", "backend", "service", "microservice"
        ]
        
        if any(indicator in description for indicator in technical_indicators):
            return True

        # Use LLM if available for better judgment
        if self.llm:
            try:
                from ..llm.providers import Message, MessageRole, LLMConfig

                prompt = f"""Is this goal complex enough to need decomposition into subgoals?

GOAL: {goal.description}

Answer with JSON: {{"is_complex": true/false, "reason": "explanation"}}

A goal is complex if it:
- Requires multiple distinct steps
- Involves different types of activities
- Would take more than a few minutes
- Has multiple success criteria

A goal is simple if it:
- Is a single action
- Can be done in one step
- Is clearly defined and atomic
"""

                messages = [
                    Message(MessageRole.SYSTEM, "You assess goal complexity. Respond with JSON only."),
                    Message(MessageRole.USER, prompt)
                ]

                config = LLMConfig(temperature=0.2, max_tokens=256)
                response = await self.llm.chat(messages, config)

                import re
                json_match = re.search(r'\{[\s\S]*\}', response.content)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("is_complex", False)

            except Exception as e:
                logger.warning(f"Complexity check failed: {e}")

        return len(goal.description.split()) > 10

    async def decompose_goal(self, goal_id: str) -> List[Goal]:
        """
        Decompose a goal into subgoals.

        Uses LLM to intelligently break down the goal into
        achievable subgoals.

        Returns:
            List of created subgoals
        """
        goal = self.goal_tree.get_goal(goal_id)
        if not goal:
            return []

        if not self.llm:
            logger.warning("No LLM provider for goal decomposition")
            return []

        try:
            from ..llm.providers import Message, MessageRole, LLMConfig

            prompt = f"""Decompose this goal into 2-5 specific, actionable subgoals:

GOAL: {goal.description}
TYPE: {goal.goal_type.value}
{f"SUCCESS CONDITIONS: {goal.success_conditions}" if goal.success_conditions else ""}

Respond with JSON array:
[
    {{
        "description": "specific subgoal description",
        "goal_type": "achieve|maintain|avoid|optimize|explore",
        "priority": "CRITICAL|HIGH|MEDIUM|LOW",
        "success_conditions": ["condition 1", "condition 2"],
        "required_capabilities": ["capability1", "capability2"],
        "suggested_approach": "how to approach this subgoal"
    }}
]

Rules:
1. Each subgoal should be specific and measurable
2. Together, subgoals should achieve the parent goal
3. Order subgoals by logical sequence (dependencies implied by order)
4. Keep subgoals at a similar level of abstraction
"""

            messages = [
                Message(MessageRole.SYSTEM, "You decompose goals into subgoals. Respond with JSON only."),
                Message(MessageRole.USER, prompt)
            ]

            logger.info(f"Calling LLM for goal decomposition: {goal.description[:50]}...")
            response = await self.llm.chat(messages, LLMConfig(max_tokens=1500, temperature=0.3))
            logger.info(f"LLM response: {response.content[:200]}...")

            # Parse subgoals
            subgoals = self._parse_subgoals(response.content, goal_id)

            # Add to tree
            for subgoal in subgoals:
                self.goal_tree.add_goal(subgoal, goal_id)

            logger.info(f"Decomposed goal {goal_id} into {len(subgoals)} subgoals")
            return subgoals

        except Exception as e:
            logger.error(f"Goal decomposition failed: {e}")
            return []

    def _parse_subgoals(self, llm_response: str, parent_id: str) -> List[Goal]:
        """Parse LLM response into Goal objects."""
        import re

        try:
            # Clean the response - remove markdown code blocks
            cleaned_response = llm_response.strip()
            
            # Remove ```json and ``` markers
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]  # Remove ```json
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[3:]  # Remove ```
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]  # Remove trailing ```
            
            cleaned_response = cleaned_response.strip()
            
            # Try to extract JSON array - handle truncated responses
            json_match = re.search(r'\[[\s\S]*\]', cleaned_response)
            if json_match:
                json_str = json_match.group()
                
                # Try to fix common JSON issues
                # 1. Remove trailing commas
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                
                # 2. Try to close incomplete JSON
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
                
                if open_braces > 0:
                    json_str += '}' * open_braces
                if open_brackets > 0:
                    json_str += ']' * open_brackets
                
                try:
                    subgoals_data = json.loads(json_str)
                    subgoals = []
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON fix attempt failed: {e}")
                    # Try to extract individual objects
                    objects = re.findall(r'\{[^{}]*\}', json_str)
                    subgoals_data = []
                    for obj_str in objects:
                        try:
                            subgoals_data.append(json.loads(obj_str))
                        except:
                            pass
                    if not subgoals_data:
                        raise
                    subgoals = []

                for i, data in enumerate(subgoals_data):
                    subgoal = Goal(
                        id=f"{parent_id}_{i+1}",
                        description=data.get("description", ""),
                        goal_type=GoalType(data.get("goal_type", "achieve")),
                        priority=GoalPriority[data.get("priority", "MEDIUM")],
                        parent_id=parent_id,
                        success_conditions=data.get("success_conditions", []),
                        required_capabilities=data.get("required_capabilities", []),
                        suggested_approach=data.get("suggested_approach")
                    )
                    subgoals.append(subgoal)

                return subgoals

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to parse subgoals JSON: {e}")
            
            # Fallback: Try to extract bullet points or numbered lists
            try:
                lines = llm_response.strip().split('\n')
                subgoals = []
                
                for i, line in enumerate(lines):
                    line = line.strip()
                    # Look for bullet points or numbered items
                    if (line.startswith(('-', '*', '•')) or 
                        (line and line[0].isdigit() and ('.' in line or ')' in line))):
                        
                        # Clean the line to get description
                        desc = line
                        desc = re.sub(r'^[-*•\d\.\)]+\s*', '', desc)  # Remove bullet/number
                        desc = desc.strip()
                        
                        if desc and len(desc) > 10:  # Skip very short lines
                            subgoal = Goal(
                                id=f"{parent_id}_{i+1}",
                                description=desc,
                                goal_type=GoalType.ACHIEVE,
                                priority=GoalPriority.MEDIUM,
                                parent_id=parent_id
                            )
                            subgoals.append(subgoal)
                
                if subgoals:
                    logger.info(f"Extracted {len(subgoals)} subgoals from bullet points")
                    return subgoals
                    
            except Exception as e2:
                logger.warning(f"Fallback parsing also failed: {e2}")

        # Final fallback: Create a generic subgoal
        logger.warning("Using fallback subgoal generation")
        return [Goal(
            id=f"{parent_id}_1",
            description=f"Work on: {self.goal_tree.get_goal(parent_id).description}",
            goal_type=GoalType.ACHIEVE,
            priority=GoalPriority.MEDIUM,
            parent_id=parent_id
        )]

    async def pursue_goal(
        self,
        goal_id: str,
        action_executor: Optional[Callable] = None,
        max_iterations: int = 50
    ) -> GoalResult:
        """
        Pursue a goal until achieved, failed, or max iterations.

        This implements the goal pursuit loop:
        1. Find next actionable subgoal
        2. Execute action
        3. Update progress
        4. Handle blockers
        5. Repeat

        Args:
            goal_id: ID of goal to pursue
            action_executor: Function to execute actions (async)
            max_iterations: Maximum iterations

        Returns:
            GoalResult with outcome
        """
        import time
        start_time = time.time()

        goal = self.goal_tree.get_goal(goal_id)
        if not goal:
            return GoalResult(
                goal_id=goal_id,
                achieved=False,
                status=GoalStatus.FAILED,
                progress=0.0,
                artifacts={},
                notes=["Goal not found"],
                subgoal_results=[],
                total_time=0.0,
                error="Goal not found"
            )

        goal.status = GoalStatus.ACTIVE
        goal.started_at = datetime.now()
        self._stop_requested = False

        subgoal_results = []
        iterations = 0

        try:
            while iterations < max_iterations and not self._stop_requested:
                iterations += 1

                # Check if goal is achieved
                if await self._check_goal_achieved(goal):
                    self.goal_tree.mark_achieved(goal_id)
                    break

                # Find next actionable goal
                next_goal = self._get_next_actionable_goal(goal_id)

                if not next_goal:
                    # No actionable goals - check if blocked
                    blocked_goals = [
                        g for g in self.goal_tree.get_descendants(goal_id)
                        if g.is_blocked()
                    ]

                    if blocked_goals:
                        # Try to resolve blockers
                        await self._handle_blockers(blocked_goals)
                        continue
                    else:
                        # No more goals to pursue
                        break

                # Execute the actionable goal
                goal.add_note(f"Pursuing subgoal: {next_goal.description}")

                result = await self._execute_goal(next_goal, action_executor)
                subgoal_results.append(result)

                if result.achieved:
                    self.goal_tree.mark_achieved(next_goal.id)
                    self.goal_tree.unblock_dependents(next_goal.id)

                    # Merge artifacts up
                    goal.artifacts.update(result.artifacts)

                # Update progress
                self._update_goal_progress(goal_id)

        except Exception as e:
            logger.error(f"Goal pursuit error: {e}")
            goal.status = GoalStatus.FAILED
            goal.add_note(f"Failed with error: {str(e)}")

        # Determine final status
        final_achieved = goal.status == GoalStatus.ACHIEVED

        if not final_achieved and goal.status != GoalStatus.FAILED:
            if goal.progress >= 0.9:
                goal.status = GoalStatus.ACHIEVED
                final_achieved = True
            elif iterations >= max_iterations:
                goal.add_note("Reached max iterations")
            elif self._stop_requested:
                goal.add_note("Pursuit stopped by request")

        return GoalResult(
            goal_id=goal_id,
            achieved=final_achieved,
            status=goal.status,
            progress=goal.progress,
            artifacts=goal.artifacts,
            notes=goal.notes,
            subgoal_results=subgoal_results,
            total_time=time.time() - start_time
        )

    def _get_next_actionable_goal(self, root_goal_id: str) -> Optional[Goal]:
        """Get the next goal that can be acted upon."""
        # Get all descendants
        descendants = self.goal_tree.get_descendants(root_goal_id)

        # Filter to actionable
        actionable = [
            g for g in descendants
            if g.is_actionable() and not g.is_blocked()
        ]

        if not actionable:
            # Check if root itself is actionable
            root = self.goal_tree.get_goal(root_goal_id)
            if root and root.is_actionable() and not root.is_blocked():
                return root
            return None

        # Sort by priority
        actionable.sort(key=lambda g: g.priority.value)
        return actionable[0]

    async def _execute_goal(
        self,
        goal: Goal,
        action_executor: Optional[Callable]
    ) -> GoalResult:
        """Execute an actionable goal."""
        import time
        start_time = time.time()

        goal.status = GoalStatus.ACTIVE
        goal.attempts += 1

        try:
            if action_executor:
                # Use provided executor
                result = await action_executor(goal)

                if isinstance(result, dict):
                    goal.progress = result.get("progress", 1.0 if result.get("success") else 0.0)
                    goal.artifacts.update(result.get("artifacts", {}))

                    return GoalResult(
                        goal_id=goal.id,
                        achieved=result.get("success", False),
                        status=GoalStatus.ACHIEVED if result.get("success") else GoalStatus.FAILED,
                        progress=goal.progress,
                        artifacts=goal.artifacts,
                        notes=goal.notes,
                        subgoal_results=[],
                        total_time=time.time() - start_time
                    )

            # Default: mark as achieved if no executor
            goal.progress = 1.0
            return GoalResult(
                goal_id=goal.id,
                achieved=True,
                status=GoalStatus.ACHIEVED,
                progress=1.0,
                artifacts=goal.artifacts,
                notes=goal.notes,
                subgoal_results=[],
                total_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"Goal execution error: {e}")
            goal.status = GoalStatus.FAILED
            goal.add_note(f"Execution failed: {str(e)}")

            return GoalResult(
                goal_id=goal.id,
                achieved=False,
                status=GoalStatus.FAILED,
                progress=goal.progress,
                artifacts=goal.artifacts,
                notes=goal.notes,
                subgoal_results=[],
                total_time=time.time() - start_time,
                error=str(e)
            )

    async def _check_goal_achieved(self, goal: Goal) -> bool:
        """Check if a goal is achieved."""
        # Already achieved
        if goal.status == GoalStatus.ACHIEVED:
            return True

        # Check if all subgoals achieved
        if goal.subgoal_ids:
            subgoals = self.goal_tree.get_subgoals(goal.id)
            if all(sg.status == GoalStatus.ACHIEVED for sg in subgoals):
                return True

        # Check success conditions
        if goal.success_conditions:
            # TODO: Implement condition checking
            pass

        # Check progress
        return goal.progress >= 1.0

    async def _handle_blockers(self, blocked_goals: List[Goal]):
        """Try to resolve blockers for goals."""
        for goal in blocked_goals:
            if not goal.blocked_by:
                goal.status = GoalStatus.PENDING
                continue

            # Check if blocking goals are resolved
            for blocker_id in list(goal.blocked_by):
                blocker = self.goal_tree.get_goal(blocker_id)
                if blocker and blocker.status == GoalStatus.ACHIEVED:
                    goal.blocked_by.remove(blocker_id)

            if not goal.blocked_by:
                goal.status = GoalStatus.PENDING

    def _update_goal_progress(self, goal_id: str):
        """Update goal progress based on subgoal completion."""
        self.goal_tree.update_parent_progress(goal_id)

    def get_goal_status(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a goal."""
        goal = self.goal_tree.get_goal(goal_id)
        if not goal:
            return None

        subgoals = self.goal_tree.get_subgoals(goal_id)

        return {
            "id": goal.id,
            "description": goal.description,
            "status": goal.status.name,
            "progress": goal.progress,
            "attempts": goal.attempts,
            "subgoals": [
                {
                    "id": sg.id,
                    "description": sg.description,
                    "status": sg.status.name,
                    "progress": sg.progress
                }
                for sg in subgoals
            ],
            "artifacts": list(goal.artifacts.keys()),
            "notes": goal.notes[-5:]  # Last 5 notes
        }

    def get_active_goals(self) -> List[Goal]:
        """Get all currently active goals."""
        return [g for g in self.goal_tree.goals.values() if g.status == GoalStatus.ACTIVE]

    def get_all_goals(self) -> List[Dict[str, Any]]:
        """Get summary of all goals."""
        return [
            {
                "id": g.id,
                "description": g.description[:50] + "..." if len(g.description) > 50 else g.description,
                "status": g.status.name,
                "progress": g.progress,
                "priority": g.priority.name
            }
            for g in self.goal_tree.goals.values()
        ]

    def stop(self):
        """Request goal pursuit to stop."""
        self._stop_requested = True

    def register_action_handler(
        self,
        capability: str,
        handler: Callable
    ):
        """
        Register a handler for a specific capability.

        Handlers are called when pursuing goals that require
        that capability.
        """
        self._action_handlers[capability] = handler


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def pursue_goal(
    description: str,
    llm_provider,
    action_executor: Optional[Callable] = None
) -> GoalResult:
    """
    Convenience function to set and pursue a goal.

    Usage:
        result = await pursue_goal(
            "Research Python best practices and create a summary",
            llm_provider
        )
    """
    manager = GoalManager(llm_provider)
    goal = await manager.set_goal(description)
    return await manager.pursue_goal(goal.id, action_executor)


def create_goal(
    description: str,
    goal_type: GoalType = GoalType.ACHIEVE,
    priority: GoalPriority = GoalPriority.MEDIUM,
    success_conditions: List[str] = None
) -> Goal:
    """
    Convenience function to create a goal object.

    Usage:
        goal = create_goal(
            "Build a REST API",
            priority=GoalPriority.HIGH,
            success_conditions=["API responds to requests", "All endpoints work"]
        )
    """
    return Goal(
        id=str(uuid.uuid4())[:8],
        description=description,
        goal_type=goal_type,
        priority=priority,
        success_conditions=success_conditions or []
    )
