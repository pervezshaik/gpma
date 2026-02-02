"""
Agentic Capabilities Demo

This demo showcases the new agentic capabilities added to GPMA:
1. ReAct Loop - Reasoning and Acting cycle
2. Intelligent Planning - LLM-powered task decomposition
3. Self-Reflection - Output evaluation and correction
4. Goal-Oriented Behavior - Hierarchical goal pursuit
5. Agentic Agent - Enhanced agent with all capabilities

REQUIREMENTS:
- Ollama running with llama3.1 model (or another model)
- Or OpenAI API key for GPT-4

RUN:
    python -m gpma.examples.agentic_demo

DEMOS:
1. Basic ReAct Loop
2. Intelligent Planning
3. Self-Reflection and Correction
4. Goal Pursuit with Decomposition
5. Full Agentic Agent
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DEMO 1: Basic ReAct Loop
# ============================================================================

async def demo_react_loop():
    """
    Demonstrates the ReAct (Reasoning + Acting) loop.

    The loop iterates:
    1. Observe the current state
    2. Think about what to do
    3. Act on the decision
    4. Observe the result
    5. Reflect and potentially adjust
    """
    print("\n" + "="*60)
    print("DEMO 1: ReAct (Reasoning + Acting) Loop")
    print("="*60)

    try:
        from ..llm.providers import OllamaProvider
        from ..core.agentic_loop import AgenticLoop
        from ..tools.agentic_tools import create_demo_tools

        # Initialize LLM provider
        provider = OllamaProvider(model="glm-4.7-flash")

        # Get production-grade tools (search + calculator)
        # These replace the inline demo tools with:
        # - Safe calculator using AST parsing (no eval)
        # - Knowledge base with web search fallback
        tools = create_demo_tools()
        
        print(f"\nUsing production tools:")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:60]}...")

        # Create and run the agentic loop
        loop = AgenticLoop(
            llm_provider=provider,
            enable_reflection=True,
            verbose=True
        )

        print("\nRunning ReAct loop with goal: 'What is Python and calculate 25 * 4'")

        result = await loop.run(
            goal="Find information about Python programming and calculate 2 times 4",
            tools=tools,
            max_iterations=5
        )

        print(f"\n--- Result ---")
        print(f"Status: {result.status.name}")
        print(f"Achieved: {result.achieved}")
        print(f"Steps taken: {result.steps_taken}")
        print(f"Final answer: {result.final_answer}")

        if result.reasoning_trace:
            print(f"\nReasoning trace ({len(result.reasoning_trace)} steps):")
            for i, step in enumerate(result.reasoning_trace[:3], 1):
                thought = step.get("thought", {})
                print(f"  Step {i}: {thought.get('reasoning', '')[:100]}...")

        await provider.close()

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure Ollama is installed and running with llama3.1 model")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Demo 1 failed")


# ============================================================================
# DEMO 2: Intelligent Planning
# ============================================================================

async def demo_planning():
    """
    Demonstrates intelligent task planning.

    The planner:
    1. Analyzes the goal
    2. Decomposes into subtasks
    3. Determines dependencies
    4. Creates execution order
    """
    print("\n" + "="*60)
    print("DEMO 2: Intelligent Planning")
    print("="*60)

    try:
        from ..llm.providers import OllamaProvider
        from ..core.planner import TaskPlanner

        provider = OllamaProvider(model="llama3.1")

        planner = TaskPlanner(
            llm_provider=provider,
            enable_optimization=True,
            max_tasks=10
        )

        goal = "Build a user authentication system with login, registration, and password reset"

        print(f"\nPlanning goal: {goal}")
        print("\nCreating plan...")

        plan = await planner.plan(goal)

        print(f"\n--- Execution Plan ---")
        print(f"Plan ID: {plan.id}")
        print(f"Strategy: {plan.strategy.value}")
        print(f"Estimated time: {plan.estimated_total_time} seconds")
        print(f"\nTasks ({len(plan.tasks)}):")

        for task in plan.tasks:
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            print(f"  [{task.id}] {task.description}{deps}")
            print(f"       Action: {task.action}, Priority: {task.priority.name}")

        print(f"\nExecution order (parallel groups):")
        for i, group in enumerate(plan.execution_order):
            print(f"  Group {i+1}: {group}")

        if plan.reasoning:
            print(f"\nPlanning reasoning: {plan.reasoning[:200]}...")

        await provider.close()

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Demo 2 failed")


# ============================================================================
# DEMO 3: Self-Reflection and Correction
# ============================================================================

async def demo_reflection():
    """
    Demonstrates self-reflection and correction.

    The reflection engine:
    1. Evaluates output against criteria
    2. Assesses quality using LLM
    3. Identifies issues
    4. Generates corrections
    """
    print("\n" + "="*60)
    print("DEMO 3: Self-Reflection and Correction")
    print("="*60)

    try:
        from ..llm.providers import OllamaProvider
        from ..core.reflection import ReflectionEngine, SuccessCriteria

        provider = OllamaProvider(model="llama3.1")

        engine = ReflectionEngine(
            llm_provider=provider,
            max_correction_iterations=2,
            enable_llm_assessment=True
        )

        # Example output that needs improvement
        task = "Write a brief summary of machine learning"
        output = "Machine learning is AI. It uses data."  # Poor output

        # Define success criteria
        criteria = SuccessCriteria(
            must_contain=["algorithms", "training"],
            min_length=100,
            quality_threshold=0.7
        )

        print(f"\nTask: {task}")
        print(f"Original output: {output}")
        print(f"\nCriteria:")
        print(f"  - Must contain: {criteria.must_contain}")
        print(f"  - Min length: {criteria.min_length}")
        print(f"  - Quality threshold: {criteria.quality_threshold}")

        print("\nEvaluating and correcting...")

        result = await engine.evaluate_and_correct(
            output=output,
            task_description=task,
            criteria=criteria
        )

        print(f"\n--- Reflection Result ---")
        print(f"Passed: {result.passed}")
        print(f"Evaluation: {result.evaluation_result.name if result.evaluation_result else 'N/A'}")
        print(f"Correction applied: {result.correction_applied}")
        print(f"Iterations: {result.iterations}")

        if result.issues_found:
            print(f"\nIssues found:")
            for issue in result.issues_found[:5]:
                print(f"  - {issue}")

        if result.quality_assessment:
            qa = result.quality_assessment
            print(f"\nQuality scores:")
            print(f"  Overall: {qa.overall_score:.2f}")
            print(f"  Relevance: {qa.relevance_score:.2f}")
            print(f"  Completeness: {qa.completeness_score:.2f}")

        if result.corrected_output:
            print(f"\nCorrected output:")
            print(f"  {result.corrected_output[:300]}...")

        print(f"\nImprovement: {result.improvement_description}")

        await provider.close()

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Demo 3 failed")


# ============================================================================
# DEMO 4: Goal-Oriented Behavior
# ============================================================================

async def demo_goal_pursuit():
    """
    Demonstrates goal-oriented behavior with decomposition.

    The goal manager:
    1. Sets high-level goals
    2. Decomposes into subgoals
    3. Tracks progress
    4. Handles blockers
    """
    print("\n" + "="*60)
    print("DEMO 4: Goal-Oriented Behavior")
    print("="*60)

    try:
        from ..llm.providers import OllamaProvider
        from ..core.goal_manager import GoalManager, GoalPriority

        provider = OllamaProvider(model="glm-4.7-flash")

        manager = GoalManager(
            llm_provider=provider,
            max_depth=3,
            auto_decompose=True
        )

        # Set a goal that will be decomposed
        goal_description = "Create a REST API for a todo application"

        print(f"\nSetting goal: {goal_description}")

        goal = await manager.set_goal(
            description=goal_description,
            priority=GoalPriority.HIGH,
            success_conditions=[
                "API endpoints for CRUD operations",
                "Input validation",
                "Error handling"
            ]
        )

        print(f"\n--- Goal Created ---")
        print(f"ID: {goal.id}")
        print(f"Status: {goal.status.name}")
        print(f"Priority: {goal.priority.name}")

        # Show decomposition
        subgoals = manager.goal_tree.get_subgoals(goal.id)
        if subgoals:
            print(f"\nDecomposed into {len(subgoals)} subgoals:")
            for sg in subgoals:
                print(f"  [{sg.id}] {sg.description}")
                print(f"       Type: {sg.goal_type.value}, Priority: {sg.priority.name}")

        # Show goal tree status
        all_goals = manager.get_all_goals()
        print(f"\nGoal tree ({len(all_goals)} total goals):")
        for g in all_goals:
            print(f"  {g['id']}: {g['description']} [{g['status']}]")

        await provider.close()

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Demo 4 failed")


# ============================================================================
# DEMO 5: Full Agentic Agent
# ============================================================================

async def demo_agentic_agent():
    """
    Demonstrates the full AgenticAgent with all capabilities.

    The agentic agent combines:
    - ReAct loop
    - Planning
    - Reflection
    - Goal management
    """
    print("\n" + "="*60)
    print("DEMO 5: Full Agentic Agent")
    print("="*60)

    try:
        from ..llm.providers import OllamaProvider
        from ..core.agentic_agent import (
            SimpleAgenticAgent,
            AgentConfig,
            AgentMode
        )
        from ..core.base_agent import AgentCapability

        provider = OllamaProvider(model="glm-4.7-flash")

        # Define agent tools using AgenticTool format
        from ..core.agentic_loop import AgenticTool
        
        async def search_tool(query: str) -> str:
            # Simulated search
            return f"Search results for '{query}': Python best practices include PEP 8 style guide, docstrings, type hints, unit testing with pytest, virtual environments, and modular code organization."

        async def analyze_tool(text: str) -> str:
            # Simulated analysis
            return f"Analysis of text: Key themes identified include code organization, naming conventions, testing strategies, and documentation practices."

        async def summarize_tool(text: str) -> str:
            # Simulated summarization
            return f"Summary: {text[:100]}... (condensed to key points about Python best practices)"

        # Configure agent
        config = AgentConfig(
            mode=AgentMode.PROACTIVE,
            enable_planning=True,
            enable_reflection=True,
            enable_goal_decomposition=True,
            max_iterations=10,
            quality_threshold=0.7,
            verbose=True
        )

        # Create actions (functions that accept keyword arguments)
        async def search_action(**kwargs) -> str:
            query = kwargs.get("query", "")
            return await search_tool(query)

        async def analyze_action(**kwargs) -> str:
            text = kwargs.get("text", "")
            return await analyze_tool(text)

        async def summarize_action(**kwargs) -> str:
            text = kwargs.get("text", "")
            return await summarize_tool(text)

        # Create agent
        agent = SimpleAgenticAgent(
            name="ResearchAgent",
            llm_provider=provider,
            config=config,
            actions={
                "search": search_action,
                "analyze": analyze_action,
                "summarize": summarize_action
            },
            capability_list=[
                AgentCapability("research", "Research topics using search", ["search", "find", "research"]),
                AgentCapability("analyze", "Analyze content", ["analyze", "examine"]),
                AgentCapability("summarize", "Summarize content", ["summarize", "brief"])
            ]
        )

        print(f"\nAgent: {agent.name}")
        print(f"Mode: {config.mode.name}")
        print(f"Capabilities: {[c.name for c in agent.capabilities]}")

        # Pursue a goal
        goal = "Research information about Python programming best practices"

        print(f"\nPursuing goal: {goal}")
        print("-" * 40)

        result = await agent.pursue_goal(goal)

        print(f"\n--- Agentic Result ---")
        print(f"Success: {result.success}")
        print(f"Goal achieved: {result.goal_achieved}")
        print(f"Iterations: {result.iterations}")
        print(f"Corrections made: {result.corrections_made}")
        print(f"Total time: {result.total_time:.2f}s")

        if result.data:
            print(f"\nFinal output:")
            output = str(result.data)
            print(f"  {output[:300]}...")

        if result.reasoning_trace:
            print(f"\nReasoning trace ({len(result.reasoning_trace)} steps):")
            for i, thought in enumerate(result.reasoning_trace[:3], 1):
                print(f"  Step {i}: {thought.action_chosen} (confidence: {thought.confidence:.2f})")

        # Show agent stats
        stats = agent.get_stats()
        print(f"\nAgent stats:")
        print(f"  Tasks processed: {stats['tasks_processed']}")
        print(f"  Reasoning steps: {stats['reasoning_steps_total']}")
        print(f"  Reflections: {stats['reflections_total']}")

        await provider.close()

    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Demo 5 failed")


# ============================================================================
# DEMO 6: Complex Multi-Phase Research Project
# ============================================================================

async def demo_complex_research():
    """
    Demonstrates a complex agentic solution with:
    - Multi-phase workflow (Plan → Research → Analyze → Synthesize → Report)
    - Multiple specialized tools
    - Progress tracking with checkpoints
    - Error recovery and adaptive replanning
    - Final deliverable generation
    
    This simulates a real-world research assistant that can:
    1. Break down a complex research question
    2. Gather information from multiple sources
    3. Cross-reference and validate findings
    4. Generate a structured report with citations
    """
    print("\n" + "="*60)
    print("DEMO 6: Complex Multi-Phase Research Project")
    print("="*60)
    
    try:
        from ..llm.providers import OllamaProvider
        from ..core.agentic_loop import AgenticLoop, AgenticTool
        from datetime import datetime
        
        provider = OllamaProvider(model="glm-4.7-flash")
        
        # =====================================================================
        # PHASE 1: Define Specialized Tools
        # =====================================================================
        
        # Knowledge base for the demo
        research_db = {
            "ai_trends_2024": {
                "title": "AI Trends 2024",
                "content": "Key AI trends in 2024 include: 1) Multimodal AI systems combining text, image, and audio. 2) Small Language Models (SLMs) for edge deployment. 3) AI agents with autonomous capabilities. 4) Retrieval-Augmented Generation (RAG) for enterprise. 5) AI governance and regulation frameworks.",
                "source": "TechReview Annual Report",
                "date": "2024-01"
            },
            "llm_architectures": {
                "title": "LLM Architecture Evolution",
                "content": "Modern LLM architectures have evolved from basic transformers to include: Mixture of Experts (MoE) for efficiency, State Space Models (Mamba) for long context, and hybrid architectures combining attention with recurrence. Key innovations include Flash Attention, Grouped Query Attention, and Rotary Position Embeddings.",
                "source": "AI Research Quarterly",
                "date": "2024-02"
            },
            "agent_frameworks": {
                "title": "AI Agent Frameworks Comparison",
                "content": "Popular agent frameworks include: LangChain (flexible, large ecosystem), AutoGPT (autonomous goal pursuit), CrewAI (multi-agent collaboration), and LangGraph (stateful workflows). Key capabilities: tool use, memory, planning, and reflection. Challenges: reliability, cost, and latency.",
                "source": "Developer Survey 2024",
                "date": "2024-03"
            },
            "enterprise_ai": {
                "title": "Enterprise AI Adoption",
                "content": "Enterprise AI adoption patterns show: 78% using AI for customer service, 65% for data analysis, 52% for content generation. Key concerns: data privacy (89%), accuracy (76%), integration complexity (68%). ROI typically seen within 12-18 months for well-scoped projects.",
                "source": "Gartner Enterprise Report",
                "date": "2024-01"
            }
        }
        
        findings = []  # Store research findings
        report_sections = {}  # Store report sections
        
        async def search_research_db(query: str, topic: str = "") -> str:
            """Search the research database for relevant information."""
            query_lower = query.lower()
            results = []
            
            for key, data in research_db.items():
                if (query_lower in data["title"].lower() or 
                    query_lower in data["content"].lower() or
                    any(word in data["content"].lower() for word in query_lower.split())):
                    results.append(data)
            
            if not results:
                return f"No results found for '{query}'. Try broader terms like 'AI', 'LLM', 'agents', or 'enterprise'."
            
            output = f"Found {len(results)} result(s) for '{query}':\n\n"
            for r in results:
                output += f"📄 {r['title']} ({r['source']}, {r['date']})\n"
                output += f"   {r['content'][:200]}...\n\n"
                findings.append({"query": query, "result": r})
            
            return output
        
        async def analyze_findings(aspect: str) -> str:
            """Analyze collected findings for patterns and insights."""
            if not findings:
                return "No findings to analyze yet. Use search_research_db first."
            
            # Simulate analysis
            analysis = f"Analysis of {len(findings)} findings for '{aspect}':\n\n"
            
            # Extract key themes
            all_content = " ".join([f["result"]["content"] for f in findings])
            
            themes = []
            if "agent" in all_content.lower():
                themes.append("AI Agents and Autonomy")
            if "llm" in all_content.lower() or "language model" in all_content.lower():
                themes.append("Large Language Model Evolution")
            if "enterprise" in all_content.lower():
                themes.append("Enterprise Adoption Patterns")
            if "framework" in all_content.lower():
                themes.append("Development Frameworks")
            
            analysis += f"Key Themes Identified: {', '.join(themes) if themes else 'General AI trends'}\n"
            analysis += f"Sources Consulted: {len(set(f['result']['source'] for f in findings))}\n"
            analysis += f"Time Range: {min(f['result']['date'] for f in findings)} to {max(f['result']['date'] for f in findings)}\n"
            
            return analysis
        
        async def synthesize_section(section_name: str, content_focus: str) -> str:
            """Synthesize findings into a report section."""
            if not findings:
                return "No findings to synthesize. Gather research first."
            
            # Create section content
            section_content = f"## {section_name}\n\n"
            
            relevant_findings = [f for f in findings if content_focus.lower() in f["result"]["content"].lower()]
            if not relevant_findings:
                relevant_findings = findings[:2]  # Use first 2 if no match
            
            for f in relevant_findings:
                section_content += f"According to {f['result']['source']}, {f['result']['content'][:150]}...\n\n"
            
            report_sections[section_name] = section_content
            return f"Section '{section_name}' created with {len(relevant_findings)} citations."
        
        async def generate_report(title: str) -> str:
            """Generate the final research report."""
            if not report_sections:
                return "No sections created yet. Use synthesize_section first."
            
            report = f"# {title}\n"
            report += f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
            report += "---\n\n"
            
            # Executive Summary
            report += "## Executive Summary\n\n"
            report += f"This report synthesizes findings from {len(findings)} research queries "
            report += f"across {len(set(f['result']['source'] for f in findings))} sources.\n\n"
            
            # Add all sections
            for section_name, content in report_sections.items():
                report += content + "\n"
            
            # References
            report += "## References\n\n"
            sources = set()
            for f in findings:
                sources.add(f"{f['result']['source']} ({f['result']['date']})")
            for i, source in enumerate(sources, 1):
                report += f"{i}. {source}\n"
            
            return report
        
        async def save_checkpoint(phase: str, status: str) -> str:
            """Save a progress checkpoint."""
            checkpoint = {
                "phase": phase,
                "status": status,
                "findings_count": len(findings),
                "sections_count": len(report_sections),
                "timestamp": datetime.now().isoformat()
            }
            return f"✓ Checkpoint saved: {phase} - {status}"
        
        # Define tools
        tools = [
            AgenticTool(
                name="search_research_db",
                description="Search the research database for information on AI topics. Use keywords like 'AI trends', 'LLM', 'agents', 'enterprise', 'frameworks'.",
                parameters={
                    "query": {"type": "string", "description": "Search query", "required": True},
                    "topic": {"type": "string", "description": "Specific topic area", "required": False}
                },
                function=search_research_db
            ),
            AgenticTool(
                name="analyze_findings",
                description="Analyze collected research findings to identify patterns, themes, and insights.",
                parameters={
                    "aspect": {"type": "string", "description": "Aspect to analyze (e.g., 'trends', 'challenges', 'opportunities')", "required": True}
                },
                function=analyze_findings
            ),
            AgenticTool(
                name="synthesize_section",
                description="Create a report section by synthesizing findings. Use after gathering enough research.",
                parameters={
                    "section_name": {"type": "string", "description": "Name of the report section", "required": True},
                    "content_focus": {"type": "string", "description": "Focus area for this section", "required": True}
                },
                function=synthesize_section
            ),
            AgenticTool(
                name="generate_report",
                description="Generate the final research report with all sections. Use as the final step.",
                parameters={
                    "title": {"type": "string", "description": "Report title", "required": True}
                },
                function=generate_report
            ),
            AgenticTool(
                name="save_checkpoint",
                description="Save a progress checkpoint to track workflow status.",
                parameters={
                    "phase": {"type": "string", "description": "Current phase name", "required": True},
                    "status": {"type": "string", "description": "Status description", "required": True}
                },
                function=save_checkpoint
            )
        ]
        
        # =====================================================================
        # PHASE 2: Create the Agentic Loop with Enhanced Configuration
        # =====================================================================
        
        loop = AgenticLoop(
            llm_provider=provider,
            enable_reflection=True,
            verbose=True
        )
        
        # Complex research goal
        research_goal = """
        Conduct comprehensive research on "The State of AI Agents in 2024" and produce a structured report.
        
        Required steps:
        1. Search for information on AI trends, agent frameworks, and enterprise adoption
        2. Analyze the findings to identify key themes and patterns
        3. Synthesize findings into report sections (Introduction, Key Trends, Challenges, Conclusion)
        4. Generate the final report with proper citations
        
        The final deliverable should be a well-structured research report.
        """
        
        print(f"\n{'='*60}")
        print("RESEARCH PROJECT: The State of AI Agents in 2024")
        print(f"{'='*60}")
        print("\nObjective: Generate a comprehensive research report")
        print("\nPhases:")
        print("  1. 🔍 Research - Gather information from multiple sources")
        print("  2. 📊 Analysis - Identify patterns and themes")
        print("  3. 📝 Synthesis - Create report sections")
        print("  4. 📄 Report - Generate final deliverable")
        print(f"\n{'-'*60}")
        print("Starting autonomous research process...")
        print(f"{'-'*60}\n")
        
        # Run the agentic loop
        result = await loop.run(
            goal=research_goal,
            tools=tools,
            max_iterations=12  # Allow more iterations for complex task
        )
        
        # =====================================================================
        # PHASE 3: Display Results
        # =====================================================================
        
        print(f"\n{'='*60}")
        print("RESEARCH PROJECT COMPLETE")
        print(f"{'='*60}")
        
        print(f"\nStatus: {'✅ SUCCESS' if result.achieved else '⚠️ PARTIAL'}")
        print(f"Steps Taken: {result.steps_taken}")
        print(f"Total Time: {result.total_time:.2f}s")
        
        print(f"\n📊 Research Statistics:")
        print(f"  • Queries executed: {len(findings)}")
        print(f"  • Sources consulted: {len(set(f['result']['source'] for f in findings)) if findings else 0}")
        print(f"  • Report sections: {len(report_sections)}")
        
        # Generate the final report from collected data
        # (in case the LLM completed without calling generate_report)
        final_report = await generate_report("The State of AI Agents in 2024")
        
        print(f"\n{'='*60}")
        print("📄 FINAL RESEARCH REPORT")
        print(f"{'='*60}\n")
        print(final_report)
        
        print(f"\n{'='*60}")
        print(f"🔍 Reasoning Trace ({len(result.reasoning_trace)} steps):")
        for i, trace in enumerate(result.reasoning_trace[:5], 1):
            tool_name = trace.get('tool_name', 'think')
            print(f"  Step {i}: {tool_name}")
        if len(result.reasoning_trace) > 5:
            print(f"  ... and {len(result.reasoning_trace) - 5} more steps")
        
        await provider.close()
        
    except ImportError as e:
        print(f"Import error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Demo 6 failed")


# ============================================================================
# DEMO 7: New Developer Experience Features
# ============================================================================

async def demo_new_features():
    """
    Demonstrates the new GPMA developer experience improvements:
    - AgentBuilder: Fluent API for agent creation
    - @auto_tool: Automatic tool creation from functions
    - Templates: Pre-built agent configurations
    - Console UI: Rich visual output
    - Observability: Real-time event tracking
    """
    print("\n" + "="*60)
    print("DEMO 7: New Developer Experience Features")
    print("="*60)
    
    try:
        from ..llm.providers import OllamaProvider
        from ..core.agentic_agent import AgentBuilder, AgentMode
        from ..core.console_ui import AgentConsole, ProgressBar, Symbols, colorize, Colors
        from ..tools.agentic_tools import auto_tool
        from ..templates import ResearchAgentTemplate
        
        provider = OllamaProvider(model="glm-4.7-flash")
        console = AgentConsole()
        
        # =====================================================================
        # FEATURE 1: @auto_tool Decorator
        # =====================================================================
        print("\n" + "-"*60)
        print(f"{Symbols.TOOL} FEATURE 1: @auto_tool Decorator")
        print("-"*60)
        print("Create tools automatically from functions with type hints:\n")
        
        @auto_tool("Search for information on a topic")
        async def smart_search(query: str, max_results: int = 5) -> str:
            """Search and return relevant information.
            
            Args:
                query: The search query
                max_results: Maximum results to return
            """
            return f"Found {max_results} results for '{query}': AI trends include autonomous agents, multimodal models, and enterprise adoption."
        
        @auto_tool("Analyze text for key insights")
        async def smart_analyze(text: str) -> str:
            """Analyze text content.
            
            Args:
                text: Text to analyze
            """
            return f"Analysis: Key themes identified - innovation, automation, scalability."
        
        print(f"  Created tool: {smart_search.name}")
        print(f"    Description: {smart_search.description}")
        print(f"    Parameters: {list(smart_search.parameters.keys())}")
        print(f"\n  Created tool: {smart_analyze.name}")
        print(f"    Description: {smart_analyze.description}")
        
        # =====================================================================
        # FEATURE 2: AgentBuilder Fluent API
        # =====================================================================
        print("\n" + "-"*60)
        print(f"{Symbols.ROCKET} FEATURE 2: AgentBuilder Fluent API")
        print("-"*60)
        print("Create agents with minimal code using fluent builder:\n")
        
        print("  # Old way (verbose):")
        print("  config = AgentConfig(mode=AgentMode.PROACTIVE, ...)")
        print("  agent = SimpleAgenticAgent(name=..., config=config, actions={...})")
        print()
        print("  # New way (fluent):")
        print('  agent = (AgentBuilder("MyAgent")')
        print('      .with_llm(provider)')
        print('      .add_tool("search", search_func, "Search the web")')
        print('      .enable_reflection()')
        print('      .build())')
        print()
        
        # Build an agent using the new API
        agent = (AgentBuilder("DemoAgent")
            .with_llm(provider)
            .with_mode(AgentMode.PROACTIVE)
            .add_tool("search", lambda **kw: f"Results for: {kw.get('query', '')}", "Search for information")
            .add_tool("analyze", lambda **kw: f"Analysis of: {kw.get('text', '')[:50]}", "Analyze content")
            .enable_reflection()
            .with_max_iterations(5)
            .verbose(True)
            .build())
        
        print(f"  {Symbols.CHECK} Built agent: {agent.name}")
        print(f"  {Symbols.CHECK} Mode: {agent.config.mode.name}")
        print(f"  {Symbols.CHECK} Tools: {list(agent._tools.keys())}")
        
        # =====================================================================
        # FEATURE 3: Agent Templates
        # =====================================================================
        print("\n" + "-"*60)
        print(f"{Symbols.PLAN} FEATURE 3: Agent Templates")
        print("-"*60)
        print("Use pre-built templates for common agent types:\n")
        
        print("  Available templates:")
        print("    - ResearchAgentTemplate: Search, analyze, summarize")
        print("    - CodingAgentTemplate: Generate, review, test code")
        print("    - DataAnalystTemplate: Load, analyze, visualize data")
        print("    - WriterAgentTemplate: Draft, edit, style content")
        print()
        
        # Create agent from template
        research_agent = ResearchAgentTemplate.create(provider, name="QuickResearcher")
        print(f"  {Symbols.CHECK} Created from template: {research_agent.name}")
        print(f"  {Symbols.CHECK} Pre-configured tools: {list(research_agent._tools.keys())}")
        
        # =====================================================================
        # FEATURE 4: Console UI
        # =====================================================================
        print("\n" + "-"*60)
        print(f"{Symbols.GOAL} FEATURE 4: Rich Console UI")
        print("-"*60)
        print("Beautiful, informative console output:\n")
        
        # Demo progress bar
        print("  Progress Bar Demo:")
        bar = ProgressBar(total=5, width=30, show_eta=False, prefix="    ")
        bar.start()
        for i in range(5):
            import asyncio
            await asyncio.sleep(0.2)
            bar.update(i + 1, f"Step {i + 1}")
        bar.finish("Complete!")
        
        # Demo colors and symbols
        print("\n  Available Symbols:")
        symbols_demo = [
            (Symbols.THINKING, "Thinking"),
            (Symbols.ACTION, "Action"),
            (Symbols.SUCCESS, "Success"),
            (Symbols.FAILURE, "Failure"),
            (Symbols.GOAL, "Goal"),
            (Symbols.REFLECT, "Reflect"),
        ]
        for sym, name in symbols_demo:
            print(f"    {sym} {name}")
        
        # =====================================================================
        # FEATURE 5: Live Agent Execution with Console
        # =====================================================================
        print("\n" + "-"*60)
        print(f"{Symbols.ROCKET} FEATURE 5: Live Agent Execution")
        print("-"*60)
        
        # Use console to show agent execution
        console.start_agent("QuickResearcher", "Research AI agent frameworks")
        
        console.start_iteration(1, 3)
        console.show_thinking("Analyzing the research goal to identify key topics...")
        console.show_action("search", {"query": "AI agent frameworks 2024"})
        console.show_result("Found information on LangChain, AutoGPT, CrewAI", success=True)
        
        console.start_iteration(2, 3)
        console.show_thinking("Analyzing the search results for key insights...")
        console.show_action("analyze", {"text": "Framework comparison data..."})
        console.show_result("Identified key themes: tool use, memory, planning", success=True)
        
        console.start_iteration(3, 3)
        console.show_thinking("Synthesizing findings into a summary...")
        console.show_reflection("Good coverage of major frameworks", quality=0.85)
        
        console.complete("Research complete with 3 key findings!")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "="*60)
        print("SUMMARY: New Developer Experience Features")
        print("="*60)
        print("""
  1. @auto_tool    - Create tools from functions automatically
  2. AgentBuilder  - Fluent API reduces boilerplate by 80%
  3. Templates     - Pre-built agents for common use cases
  4. Console UI    - Rich visual output for monitoring
  5. Observability - Event-based tracking (see observability.py)
  
  Import examples:
    from gpma.core import AgentBuilder, AgentConsole
    from gpma.tools import auto_tool
    from gpma.templates import ResearchAgentTemplate
""")
        
        await provider.close()
        
    except ImportError as e:
        print(f"Import error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# MAIN
# ============================================================================

# Available demos
DEMOS = [
    ("ReAct Loop", demo_react_loop, "Autonomous reasoning and acting cycle"),
    ("Intelligent Planning", demo_planning, "LLM-powered task decomposition"),
    ("Self-Reflection", demo_reflection, "Output evaluation and correction"),
    ("Goal-Oriented Behavior", demo_goal_pursuit, "Hierarchical goal pursuit"),
    ("Full Agentic Agent", demo_agentic_agent, "Agent with all capabilities"),
    ("Complex Research Project", demo_complex_research, "Multi-phase autonomous research"),
    ("New DX Features", demo_new_features, "AgentBuilder, Templates, Console UI")
]


def print_menu():
    """Print the demo menu."""
    print("="*60)
    print("GPMA AGENTIC CAPABILITIES DEMO")
    print("="*60)
    print("\nAvailable demos:")
    for i, (name, _, description) in enumerate(DEMOS, 1):
        print(f"  {i}. {name} - {description}")
    print(f"  0. Run ALL demos")
    print(f"  q. Quit")
    print("\n" + "-"*60)
    print("Note: These demos require Ollama with llama3.1 model")
    print("Install: https://ollama.ai")
    print("Pull model: ollama pull llama3.1")
    print("-"*60)


async def run_single_demo(demo_num: int):
    """Run a single demo by number (1-indexed)."""
    if demo_num < 1 or demo_num > len(DEMOS):
        print(f"Invalid demo number: {demo_num}. Choose 1-{len(DEMOS)}")
        return

    name, demo_func, _ = DEMOS[demo_num - 1]
    try:
        await demo_func()
    except Exception as e:
        print(f"\n{name} demo failed: {e}")
        logger.exception(f"Demo {demo_num} failed")


async def run_all_demos():
    """Run all demos sequentially."""
    for i, (name, demo_func, _) in enumerate(DEMOS, 1):
        try:
            await demo_func()
        except Exception as e:
            print(f"\n{name} demo failed: {e}")

        print("\n")
        await asyncio.sleep(1)  # Brief pause between demos

    print_completion_message()


def print_completion_message():
    """Print the completion message."""
    print("="*60)
    print("DEMOS COMPLETE")
    print("="*60)
    print("\nThe agentic capabilities transform GPMA from a task executor")
    print("into a true autonomous agent system capable of:")
    print("  - Reasoning about goals")
    print("  - Planning multi-step approaches")
    print("  - Self-correcting outputs")
    print("  - Adapting to obstacles")
    print("\nSee PROFESSIONAL_UPGRADE_PLAN.md for the complete roadmap.")


async def main(demo_number: int = None):
    """
    Run demos.

    Args:
        demo_number: Specific demo to run (1-5), 0 for all, None for interactive
    """
    print_menu()

    # If demo number provided as argument, run it directly
    if demo_number is not None:
        if demo_number == 0:
            print("\nRunning ALL demos...")
            await run_all_demos()
        else:
            print(f"\nRunning demo {demo_number}...")
            await run_single_demo(demo_number)
        return

    # Interactive mode
    while True:
        try:
            choice = input("\nEnter demo number (1-5, 0=all, q=quit): ").strip().lower()

            if choice == 'q':
                print("Goodbye!")
                break

            demo_num = int(choice)

            if demo_num == 0:
                print("\nRunning ALL demos...")
                await run_all_demos()
                break
            elif 1 <= demo_num <= len(DEMOS):
                await run_single_demo(demo_num)

                # Ask if user wants to run another
                another = input("\nRun another demo? (y/n): ").strip().lower()
                if another != 'y':
                    print_completion_message()
                    break
                print_menu()
            else:
                print(f"Please enter a number between 0 and {len(DEMOS)}")

        except ValueError:
            print("Please enter a valid number or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            # Non-interactive mode, just exit
            break


if __name__ == "__main__":
    import sys

    # Check for command line argument
    demo_num = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ['-h', '--help']:
            print("Usage: python -m gpma.examples.agentic_demo [DEMO_NUMBER]")
            print("\nDEMO_NUMBER:")
            print("  1 - ReAct Loop")
            print("  2 - Intelligent Planning")
            print("  3 - Self-Reflection")
            print("  4 - Goal-Oriented Behavior")
            print("  5 - Full Agentic Agent")
            print("  6 - Complex Research Project")
            print("  7 - New DX Features (AgentBuilder, Templates, Console UI)")
            print("  0 - Run ALL demos")
            print("  (no argument) - Interactive mode")
            sys.exit(0)
        try:
            demo_num = int(arg)
            if demo_num < 0 or demo_num > len(DEMOS):
                print(f"Error: Demo number must be 0-{len(DEMOS)}, got {demo_num}")
                sys.exit(1)
        except ValueError:
            print(f"Error: Invalid argument '{arg}'. Use -h for help.")
            sys.exit(1)

    asyncio.run(main(demo_num))
