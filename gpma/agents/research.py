"""
Research Agent

An agent specialized in research, analysis, and information synthesis.

CAPABILITIES:
- Search for information across multiple sources
- Summarize content
- Compare and analyze data
- Answer questions based on gathered information

LEARNING POINTS:
- This agent orchestrates other agents (uses WebBrowserAgent)
- It demonstrates agent collaboration
- It adds higher-level intelligence (summarization, analysis)
"""

from typing import Any, Dict, List, Optional
import re

from ..core.base_agent import BaseAgent, AgentCapability, TaskResult, Tool
from ..core.message_bus import Message, MessageType

import logging

logger = logging.getLogger(__name__)


class ResearchAgent(BaseAgent):
    """
    Agent for research and analysis tasks.

    This agent can:
    1. Search for information on a topic
    2. Fetch and analyze multiple sources
    3. Summarize findings
    4. Answer questions based on research

    COLLABORATION PATTERN:
    ResearchAgent uses WebBrowserAgent for fetching content.
    It coordinates the research workflow:

    1. User: "Research topic X"
    2. ResearchAgent: Generates search queries
    3. ResearchAgent -> WebBrowserAgent: Fetch search results
    4. ResearchAgent -> WebBrowserAgent: Fetch top pages
    5. ResearchAgent: Synthesize and summarize
    6. Return combined research

    EXAMPLE:
        agent = ResearchAgent()
        result = await agent.run_task({
            "action": "research",
            "input": "What are the latest developments in AI?"
        })
    """

    def __init__(self, name: str = None):
        super().__init__(name or "Researcher")

        # Reference to collaborating agents (set by orchestrator or manually)
        self._web_agent = None

        # Research settings
        self.max_sources = 5
        self.max_content_length = 10000

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="research",
                description="Research topics and gather information",
                keywords=["research", "investigate", "study", "learn about", "find out"],
                priority=3
            ),
            AgentCapability(
                name="summarize",
                description="Summarize text or web content",
                keywords=["summarize", "summary", "brief", "overview", "tldr"],
                priority=2
            ),
            AgentCapability(
                name="analyze",
                description="Analyze and compare information",
                keywords=["analyze", "compare", "evaluate", "assess", "review"],
                priority=2
            ),
            AgentCapability(
                name="question_answer",
                description="Answer questions based on web research",
                keywords=["what", "who", "when", "where", "why", "how", "question"],
                priority=1
            )
        ]

    def set_web_agent(self, web_agent: BaseAgent) -> None:
        """
        Set the web agent for collaboration.

        In a full system, this would be done by the orchestrator.
        """
        self._web_agent = web_agent

    async def process(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a research-related task.

        Actions:
        - research: Full research workflow
        - summarize: Summarize given content or URL
        - analyze: Analyze content for insights
        - answer: Answer a question
        """
        action = task.get("action", "research")
        input_data = task.get("input", "")
        context = task.get("context", {})

        try:
            if action == "research":
                return await self._handle_research(input_data, context)
            elif action == "summarize":
                return await self._handle_summarize(input_data, context)
            elif action == "analyze":
                return await self._handle_analyze(input_data, context)
            elif action in ["answer", "question"]:
                return await self._handle_question(input_data, context)
            else:
                # Infer action from input
                if "?" in input_data or any(q in input_data.lower() for q in ["what", "who", "when", "where", "why", "how"]):
                    return await self._handle_question(input_data, context)
                elif "summarize" in input_data.lower():
                    return await self._handle_summarize(input_data, context)
                else:
                    return await self._handle_research(input_data, context)

        except Exception as e:
            logger.error(f"ResearchAgent error: {e}")
            return TaskResult(
                success=False,
                data=None,
                error=str(e)
            )

    async def _handle_research(self, topic: str, context: Dict) -> TaskResult:
        """
        Conduct comprehensive research on a topic.

        Workflow:
        1. Generate search queries
        2. Fetch search results
        3. Select best sources
        4. Fetch and analyze each source
        5. Synthesize findings
        """
        logger.info(f"Researching: {topic}")

        # Step 1: Generate search queries
        queries = self._generate_search_queries(topic)

        # Step 2: Search and collect results
        all_results = []
        for query in queries[:3]:  # Limit queries
            search_result = await self._do_search(query)
            if search_result:
                all_results.extend(search_result.get("results", []))

        # Step 3: Deduplicate and select best sources
        seen_urls = set()
        unique_results = []
        for result in all_results:
            url = result.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)

        sources = unique_results[:self.max_sources]

        # Step 4: Fetch and analyze each source
        source_analyses = []
        for source in sources:
            analysis = await self._analyze_source(source["url"])
            if analysis:
                source_analyses.append({
                    "title": source.get("title", ""),
                    "url": source["url"],
                    "snippet": source.get("snippet", ""),
                    "analysis": analysis
                })

        # Step 5: Synthesize findings
        synthesis = self._synthesize_research(topic, source_analyses)

        return TaskResult(
            success=True,
            data={
                "topic": topic,
                "queries_used": queries[:3],
                "sources_analyzed": len(source_analyses),
                "sources": source_analyses,
                "synthesis": synthesis
            },
            metadata={"total_results_found": len(all_results)}
        )

    async def _handle_summarize(self, input_data: str, context: Dict) -> TaskResult:
        """
        Summarize content from a URL or provided text.
        """
        # Check if input is a URL
        if input_data.startswith(("http://", "https://")):
            # Fetch the content first
            page_data = await self._fetch_page(input_data)
            if not page_data:
                return TaskResult(
                    success=False,
                    data=None,
                    error="Failed to fetch URL for summarization"
                )
            text = page_data.get("text", "")
            title = page_data.get("title", "")
        else:
            # Use provided text
            text = context.get("content", input_data)
            title = context.get("title", "")

        # Generate summary
        summary = self._generate_summary(text)

        return TaskResult(
            success=True,
            data={
                "title": title,
                "original_length": len(text),
                "summary": summary,
                "summary_length": len(summary)
            }
        )

    async def _handle_analyze(self, input_data: str, context: Dict) -> TaskResult:
        """
        Analyze content for key insights.
        """
        # Get content
        if input_data.startswith(("http://", "https://")):
            page_data = await self._fetch_page(input_data)
            text = page_data.get("text", "") if page_data else ""
        else:
            text = context.get("content", input_data)

        # Perform analysis
        analysis = {
            "word_count": len(text.split()),
            "sentence_count": len(re.split(r'[.!?]+', text)),
            "key_topics": self._extract_key_topics(text),
            "key_facts": self._extract_key_facts(text),
            "summary": self._generate_summary(text, max_length=500)
        }

        return TaskResult(
            success=True,
            data=analysis
        )

    async def _handle_question(self, question: str, context: Dict) -> TaskResult:
        """
        Answer a question by researching it.
        """
        # First, do research on the question
        research_result = await self._handle_research(question, context)

        if not research_result.success:
            return research_result

        # Format as an answer
        synthesis = research_result.data.get("synthesis", "")
        sources = research_result.data.get("sources", [])

        # Create answer
        answer = {
            "question": question,
            "answer": synthesis,
            "confidence": "medium" if sources else "low",
            "sources": [
                {"title": s["title"], "url": s["url"]}
                for s in sources[:3]
            ]
        }

        return TaskResult(
            success=True,
            data=answer
        )

    # Helper methods
    def _generate_search_queries(self, topic: str) -> List[str]:
        """
        Generate multiple search queries for a topic.

        This is a simple implementation - a real system might use
        an LLM to generate better queries.
        """
        queries = [topic]

        # Add variations
        if "?" in topic:
            # It's a question, also search without the question mark
            queries.append(topic.rstrip("?"))

        # Add "latest" prefix for news-like queries
        if not any(word in topic.lower() for word in ["latest", "recent", "new"]):
            queries.append(f"latest {topic}")

        # Add "explained" suffix for how/what questions
        if any(word in topic.lower() for word in ["what is", "how to", "why"]):
            queries.append(f"{topic} explained")

        return queries

    async def _do_search(self, query: str) -> Optional[Dict]:
        """
        Perform a web search.

        Uses the web agent if available, otherwise uses tools directly.
        """
        if self._web_agent:
            result = await self._web_agent.run_task({
                "action": "search",
                "input": query
            })
            return result.data if result.success else None
        else:
            # Use tools directly
            from ..tools.web_tools import search_web
            results = await search_web(query, num_results=5)
            return {
                "query": query,
                "results": [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in results
                ]
            }

    async def _fetch_page(self, url: str) -> Optional[Dict]:
        """Fetch a web page."""
        if self._web_agent:
            result = await self._web_agent.run_task({
                "action": "fetch",
                "input": url
            })
            return result.data if result.success else None
        else:
            from ..tools.web_tools import fetch_url
            page = await fetch_url(url)
            if page.status_code != 0:
                return {"title": page.title, "text": page.text}
            return None

    async def _analyze_source(self, url: str) -> Optional[Dict]:
        """Analyze a single source."""
        page_data = await self._fetch_page(url)
        if not page_data:
            return None

        text = page_data.get("text", "")[:self.max_content_length]

        return {
            "key_points": self._extract_key_facts(text)[:5],
            "summary": self._generate_summary(text, max_length=300),
            "word_count": len(text.split())
        }

    def _synthesize_research(self, topic: str, sources: List[Dict]) -> str:
        """
        Synthesize research from multiple sources.

        This is a simple implementation - a real system would use
        an LLM for better synthesis.
        """
        if not sources:
            return f"No information found about: {topic}"

        # Collect all summaries
        summaries = [s.get("analysis", {}).get("summary", "") for s in sources if s.get("analysis")]

        # Combine into synthesis
        synthesis = f"Research on '{topic}':\n\n"
        synthesis += "Key findings from sources:\n"

        for i, source in enumerate(sources[:3], 1):
            title = source.get("title", "Unknown source")
            analysis = source.get("analysis", {})
            summary = analysis.get("summary", "No summary available")
            synthesis += f"\n{i}. {title}:\n   {summary[:200]}...\n"

        # Add overall summary
        synthesis += f"\nBased on {len(sources)} sources analyzed."

        return synthesis

    def _generate_summary(self, text: str, max_length: int = 1000) -> str:
        """
        Generate a summary of text.

        Simple extractive summarization - takes first sentences.
        A real implementation would use NLP or an LLM.
        """
        if not text:
            return ""

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Take sentences until we reach max length
        summary = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > max_length:
                break
            summary.append(sentence)
            current_length += len(sentence)

        return " ".join(summary)

    def _extract_key_topics(self, text: str) -> List[str]:
        """
        Extract key topics from text.

        Simple implementation using capitalized words.
        A real implementation would use NLP.
        """
        # Find capitalized words/phrases
        words = text.split()
        topics = []

        for word in words:
            # Clean the word
            clean = re.sub(r'[^a-zA-Z]', '', word)
            if clean and clean[0].isupper() and len(clean) > 3:
                if clean not in topics:
                    topics.append(clean)

        return topics[:10]

    def _extract_key_facts(self, text: str) -> List[str]:
        """
        Extract key facts/statements from text.

        Simple implementation - takes sentences with numbers or key phrases.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)

        facts = []
        key_indicators = ["is", "are", "was", "were", "has", "have", "will", "can"]

        for sentence in sentences:
            # Look for factual statements
            has_number = bool(re.search(r'\d+', sentence))
            has_indicator = any(ind in sentence.lower().split() for ind in key_indicators)

            if (has_number or has_indicator) and len(sentence) > 20 and len(sentence) < 200:
                facts.append(sentence.strip())

        return facts[:10]
