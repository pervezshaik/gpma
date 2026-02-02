"""
Self-Correction and Reflection Module

This module enables agents to evaluate their outputs and improve them
through reflection and self-correction.

THE REFLECTION PROCESS:
1. Evaluate - Check if output meets criteria
2. Assess - Use LLM to assess quality
3. Identify - Find specific issues
4. Correct - Generate improved version
5. Verify - Confirm correction is better

WHY REFLECTION MATTERS:
- LLMs can generate incorrect or suboptimal outputs
- Simple retry doesn't improve quality
- Reflection identifies WHAT to fix and HOW
- Self-correction leads to better results

REFLECTION STRATEGIES:
1. Criteria-based: Check against explicit success criteria
2. Quality-based: LLM assesses overall quality
3. Comparative: Compare against examples or alternatives
4. Iterative: Multiple rounds of refinement

USAGE:
    from gpma.core.reflection import ReflectionEngine, SuccessCriteria

    engine = ReflectionEngine(llm_provider)

    # Define success criteria
    criteria = SuccessCriteria(
        must_contain=["conclusion", "recommendation"],
        max_length=1000,
        quality_threshold=0.7
    )

    # Evaluate and potentially correct
    result = await engine.evaluate_and_correct(
        output="The analysis shows...",
        task_description="Analyze sales data and provide recommendations",
        criteria=criteria
    )

    print(result.passed)
    print(result.corrected_output)  # Improved version if correction was needed
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum, auto
from datetime import datetime
import asyncio
import json
import logging
import re

logger = logging.getLogger(__name__)


class EvaluationResult(Enum):
    """Result of an evaluation."""
    PASS = auto()
    FAIL = auto()
    PARTIAL = auto()
    UNKNOWN = auto()


class CorrectionStrategy(Enum):
    """Strategy for correcting outputs."""
    REGENERATE = "regenerate"      # Generate from scratch
    REFINE = "refine"              # Improve existing output
    EXPAND = "expand"              # Add missing content
    CONDENSE = "condense"          # Remove excess content
    RESTRUCTURE = "restructure"    # Change organization
    FIX_ERRORS = "fix_errors"      # Fix specific errors


@dataclass
class SuccessCriteria:
    """
    Defines what makes an output successful.

    Multiple criteria can be combined. An output passes if it
    meets the minimum threshold of criteria.
    """
    # Content requirements
    must_contain: List[str] = field(default_factory=list)  # Required keywords/phrases
    must_not_contain: List[str] = field(default_factory=list)  # Forbidden content
    min_length: int = 0
    max_length: int = 0  # 0 = no limit

    # Quality thresholds (0-1)
    quality_threshold: float = 0.7
    relevance_threshold: float = 0.7
    completeness_threshold: float = 0.7

    # Format requirements
    required_format: Optional[str] = None  # e.g., "json", "markdown", "list"
    required_sections: List[str] = field(default_factory=list)

    # Custom validators
    custom_validators: List[Callable[[str], bool]] = field(default_factory=list)

    # Passing threshold (what percentage of checks must pass)
    pass_threshold: float = 0.8  # 80% of checks must pass


@dataclass
class CriterionCheck:
    """Result of checking a single criterion."""
    criterion: str
    passed: bool
    details: str
    severity: str = "medium"  # low, medium, high


@dataclass
class QualityAssessment:
    """
    Comprehensive quality assessment of an output.
    """
    overall_score: float  # 0-1
    relevance_score: float
    completeness_score: float
    clarity_score: float
    accuracy_score: float

    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "relevance_score": self.relevance_score,
            "completeness_score": self.completeness_score,
            "clarity_score": self.clarity_score,
            "accuracy_score": self.accuracy_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "suggestions": self.suggestions
        }


@dataclass
class ReflectionResult:
    """
    Complete result of reflection and potential correction.
    """
    passed: bool
    original_output: str
    evaluation_result: EvaluationResult
    criteria_checks: List[CriterionCheck]
    quality_assessment: Optional[QualityAssessment]
    issues_found: List[str]
    correction_applied: bool
    corrected_output: Optional[str]
    correction_strategy: Optional[CorrectionStrategy]
    improvement_description: str
    iterations: int
    total_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "evaluation_result": self.evaluation_result.name,
            "criteria_checks": [
                {"criterion": c.criterion, "passed": c.passed, "details": c.details}
                for c in self.criteria_checks
            ],
            "quality_assessment": self.quality_assessment.to_dict() if self.quality_assessment else None,
            "issues_found": self.issues_found,
            "correction_applied": self.correction_applied,
            "improvement_description": self.improvement_description,
            "iterations": self.iterations,
            "total_time": self.total_time
        }


class ReflectionEngine:
    """
    Engine for evaluating and improving agent outputs through reflection.

    The engine can:
    1. Check outputs against explicit criteria
    2. Use LLM to assess quality
    3. Identify specific issues
    4. Generate corrections
    5. Verify improvements

    USAGE:
        engine = ReflectionEngine(llm_provider)

        result = await engine.evaluate_and_correct(
            output="Generated content...",
            task_description="Write a summary of...",
            criteria=SuccessCriteria(must_contain=["conclusion"], quality_threshold=0.7)
        )

        if result.correction_applied:
            use(result.corrected_output)
        else:
            use(result.original_output)
    """

    def __init__(
        self,
        llm_provider,
        max_correction_iterations: int = 3,
        enable_llm_assessment: bool = True
    ):
        """
        Initialize the reflection engine.

        Args:
            llm_provider: LLM for quality assessment and correction
            max_correction_iterations: Max attempts to correct
            enable_llm_assessment: Use LLM for quality assessment
        """
        self.llm = llm_provider
        self.max_iterations = max_correction_iterations
        self.enable_llm_assessment = enable_llm_assessment

    async def evaluate(
        self,
        output: str,
        task_description: str,
        criteria: Optional[SuccessCriteria] = None
    ) -> Tuple[EvaluationResult, List[CriterionCheck], Optional[QualityAssessment]]:
        """
        Evaluate an output against criteria.

        Returns:
            Tuple of (result, criterion_checks, quality_assessment)
        """
        criteria = criteria or SuccessCriteria()
        checks: List[CriterionCheck] = []

        # Check content requirements
        checks.extend(self._check_content_criteria(output, criteria))

        # Check format requirements
        checks.extend(self._check_format_criteria(output, criteria))

        # Run custom validators
        checks.extend(self._run_custom_validators(output, criteria))

        # LLM quality assessment
        quality_assessment = None
        if self.enable_llm_assessment:
            quality_assessment = await self._assess_quality(output, task_description)

            # Add quality-based checks
            if quality_assessment:
                if quality_assessment.overall_score < criteria.quality_threshold:
                    checks.append(CriterionCheck(
                        criterion="quality_threshold",
                        passed=False,
                        details=f"Quality score {quality_assessment.overall_score:.2f} below threshold {criteria.quality_threshold}",
                        severity="high"
                    ))
                else:
                    checks.append(CriterionCheck(
                        criterion="quality_threshold",
                        passed=True,
                        details=f"Quality score {quality_assessment.overall_score:.2f} meets threshold"
                    ))

                if quality_assessment.relevance_score < criteria.relevance_threshold:
                    checks.append(CriterionCheck(
                        criterion="relevance_threshold",
                        passed=False,
                        details=f"Relevance score {quality_assessment.relevance_score:.2f} below threshold",
                        severity="high"
                    ))
                else:
                    checks.append(CriterionCheck(
                        criterion="relevance_threshold",
                        passed=True,
                        details=f"Relevance score meets threshold"
                    ))

        # Determine overall result
        if not checks:
            return EvaluationResult.PASS, checks, quality_assessment

        passed_count = sum(1 for c in checks if c.passed)
        pass_ratio = passed_count / len(checks) if checks else 1.0

        if pass_ratio >= criteria.pass_threshold:
            result = EvaluationResult.PASS
        elif pass_ratio >= 0.5:
            result = EvaluationResult.PARTIAL
        else:
            result = EvaluationResult.FAIL

        return result, checks, quality_assessment

    def _check_content_criteria(
        self,
        output: str,
        criteria: SuccessCriteria
    ) -> List[CriterionCheck]:
        """Check content-related criteria."""
        checks = []
        output_lower = output.lower()

        # Must contain
        for phrase in criteria.must_contain:
            found = phrase.lower() in output_lower
            checks.append(CriterionCheck(
                criterion=f"must_contain:{phrase}",
                passed=found,
                details=f"'{phrase}' {'found' if found else 'not found'} in output",
                severity="high" if not found else "low"
            ))

        # Must not contain
        for phrase in criteria.must_not_contain:
            found = phrase.lower() in output_lower
            checks.append(CriterionCheck(
                criterion=f"must_not_contain:{phrase}",
                passed=not found,
                details=f"'{phrase}' {'found (violation)' if found else 'not found (good)'}",
                severity="high" if found else "low"
            ))

        # Length checks
        output_length = len(output)

        if criteria.min_length > 0:
            meets_min = output_length >= criteria.min_length
            checks.append(CriterionCheck(
                criterion="min_length",
                passed=meets_min,
                details=f"Length {output_length} {'meets' if meets_min else 'below'} minimum {criteria.min_length}",
                severity="medium" if not meets_min else "low"
            ))

        if criteria.max_length > 0:
            meets_max = output_length <= criteria.max_length
            checks.append(CriterionCheck(
                criterion="max_length",
                passed=meets_max,
                details=f"Length {output_length} {'within' if meets_max else 'exceeds'} maximum {criteria.max_length}",
                severity="medium" if not meets_max else "low"
            ))

        return checks

    def _check_format_criteria(
        self,
        output: str,
        criteria: SuccessCriteria
    ) -> List[CriterionCheck]:
        """Check format-related criteria."""
        checks = []

        # Required format
        if criteria.required_format:
            format_check = self._check_format(output, criteria.required_format)
            checks.append(CriterionCheck(
                criterion=f"format:{criteria.required_format}",
                passed=format_check[0],
                details=format_check[1],
                severity="high" if not format_check[0] else "low"
            ))

        # Required sections
        for section in criteria.required_sections:
            # Look for section headers in various formats
            patterns = [
                rf"#{1,3}\s*{re.escape(section)}",  # Markdown headers
                rf"\*\*{re.escape(section)}\*\*",    # Bold
                rf"^{re.escape(section)}:",          # Label format
                rf"{re.escape(section)}\n[-=]+"     # Underlined
            ]

            found = any(re.search(p, output, re.IGNORECASE | re.MULTILINE) for p in patterns)
            checks.append(CriterionCheck(
                criterion=f"required_section:{section}",
                passed=found,
                details=f"Section '{section}' {'found' if found else 'not found'}",
                severity="medium" if not found else "low"
            ))

        return checks

    def _check_format(self, output: str, required_format: str) -> Tuple[bool, str]:
        """Check if output matches required format."""
        if required_format.lower() == "json":
            try:
                json.loads(output)
                return True, "Valid JSON format"
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON: {str(e)}"

        elif required_format.lower() == "markdown":
            # Check for markdown indicators
            md_patterns = [r'^#+\s', r'\*\*.*\*\*', r'\*.*\*', r'```', r'\[.*\]\(.*\)']
            has_markdown = any(re.search(p, output, re.MULTILINE) for p in md_patterns)
            if has_markdown:
                return True, "Contains markdown formatting"
            return False, "No markdown formatting detected"

        elif required_format.lower() == "list":
            # Check for list indicators
            list_patterns = [r'^[-*•]\s', r'^\d+\.\s']
            has_list = any(re.search(p, output, re.MULTILINE) for p in list_patterns)
            if has_list:
                return True, "Contains list formatting"
            return False, "No list formatting detected"

        return True, f"Format '{required_format}' not validated"

    def _run_custom_validators(
        self,
        output: str,
        criteria: SuccessCriteria
    ) -> List[CriterionCheck]:
        """Run custom validator functions."""
        checks = []

        for i, validator in enumerate(criteria.custom_validators):
            try:
                passed = validator(output)
                checks.append(CriterionCheck(
                    criterion=f"custom_validator_{i}",
                    passed=passed,
                    details=f"Custom validator {'passed' if passed else 'failed'}"
                ))
            except Exception as e:
                checks.append(CriterionCheck(
                    criterion=f"custom_validator_{i}",
                    passed=False,
                    details=f"Validator error: {str(e)}",
                    severity="high"
                ))

        return checks

    async def _assess_quality(
        self,
        output: str,
        task_description: str
    ) -> Optional[QualityAssessment]:
        """Use LLM to assess output quality."""
        try:
            from ..llm.providers import Message, MessageRole, LLMConfig

            prompt = f"""Assess the quality of this output for the given task.

TASK: {task_description}

OUTPUT:
{output[:3000]}  # Truncate for prompt length

Rate on a scale of 0.0 to 1.0 and provide analysis:

{{
    "overall_score": 0.0-1.0,
    "relevance_score": 0.0-1.0 (how relevant to the task),
    "completeness_score": 0.0-1.0 (how complete the response is),
    "clarity_score": 0.0-1.0 (how clear and well-written),
    "accuracy_score": 0.0-1.0 (how accurate/correct),
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "suggestions": ["specific improvement suggestions"]
}}
"""

            messages = [
                Message(MessageRole.SYSTEM, "You are a quality assessment expert. Respond only with valid JSON."),
                Message(MessageRole.USER, prompt)
            ]

            config = LLMConfig(temperature=0.2, max_tokens=1024)
            response = await self.llm.chat(messages, config)

            # Parse response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                data = json.loads(json_match.group())
                return QualityAssessment(
                    overall_score=float(data.get("overall_score", 0.5)),
                    relevance_score=float(data.get("relevance_score", 0.5)),
                    completeness_score=float(data.get("completeness_score", 0.5)),
                    clarity_score=float(data.get("clarity_score", 0.5)),
                    accuracy_score=float(data.get("accuracy_score", 0.5)),
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    suggestions=data.get("suggestions", [])
                )

        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")

        return None

    async def evaluate_and_correct(
        self,
        output: str,
        task_description: str,
        criteria: Optional[SuccessCriteria] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ReflectionResult:
        """
        Evaluate an output and correct it if needed.

        This is the main method that combines evaluation with
        iterative correction.

        Args:
            output: The output to evaluate
            task_description: Description of the original task
            criteria: Success criteria
            context: Additional context for correction

        Returns:
            ReflectionResult with evaluation and any corrections
        """
        import time
        start_time = time.time()

        criteria = criteria or SuccessCriteria()
        current_output = output
        iterations = 0
        correction_applied = False
        correction_strategy = None
        improvement_description = ""

        # Initial evaluation
        eval_result, checks, quality = await self.evaluate(
            current_output, task_description, criteria
        )

        issues_found = [c.details for c in checks if not c.passed]

        # If passed, return early
        if eval_result == EvaluationResult.PASS:
            return ReflectionResult(
                passed=True,
                original_output=output,
                evaluation_result=eval_result,
                criteria_checks=checks,
                quality_assessment=quality,
                issues_found=issues_found,
                correction_applied=False,
                corrected_output=None,
                correction_strategy=None,
                improvement_description="Output meets all criteria",
                iterations=0,
                total_time=time.time() - start_time
            )

        # Try to correct
        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            # Determine correction strategy
            strategy = self._determine_correction_strategy(checks, quality)
            correction_strategy = strategy

            # Generate correction
            corrected = await self._generate_correction(
                current_output,
                task_description,
                checks,
                quality,
                strategy,
                context
            )

            if corrected and corrected != current_output:
                current_output = corrected
                correction_applied = True

                # Re-evaluate
                eval_result, checks, quality = await self.evaluate(
                    current_output, task_description, criteria
                )

                issues_found = [c.details for c in checks if not c.passed]

                if eval_result == EvaluationResult.PASS:
                    improvement_description = f"Corrected after {iterations} iteration(s) using {strategy.value}"
                    break
            else:
                # Correction didn't produce different output
                break

        # Final result
        passed = eval_result == EvaluationResult.PASS

        if correction_applied and not passed:
            improvement_description = f"Partial improvement after {iterations} iteration(s), some issues remain"
        elif not correction_applied:
            improvement_description = "Could not generate effective correction"

        return ReflectionResult(
            passed=passed,
            original_output=output,
            evaluation_result=eval_result,
            criteria_checks=checks,
            quality_assessment=quality,
            issues_found=issues_found,
            correction_applied=correction_applied,
            corrected_output=current_output if correction_applied else None,
            correction_strategy=correction_strategy,
            improvement_description=improvement_description,
            iterations=iterations,
            total_time=time.time() - start_time
        )

    def _determine_correction_strategy(
        self,
        checks: List[CriterionCheck],
        quality: Optional[QualityAssessment]
    ) -> CorrectionStrategy:
        """Determine the best strategy for correction."""
        failed_checks = [c for c in checks if not c.passed]

        # Check for specific patterns
        has_length_issue = any("length" in c.criterion for c in failed_checks)
        has_missing_content = any("must_contain" in c.criterion for c in failed_checks)
        has_format_issue = any("format" in c.criterion for c in failed_checks)
        has_section_issue = any("section" in c.criterion for c in failed_checks)

        # Determine strategy based on issues
        if has_format_issue or has_section_issue:
            return CorrectionStrategy.RESTRUCTURE
        elif has_missing_content:
            return CorrectionStrategy.EXPAND
        elif has_length_issue and any("max_length" in c.criterion for c in failed_checks):
            return CorrectionStrategy.CONDENSE
        elif quality and quality.overall_score < 0.5:
            return CorrectionStrategy.REGENERATE
        else:
            return CorrectionStrategy.REFINE

    async def _generate_correction(
        self,
        output: str,
        task_description: str,
        checks: List[CriterionCheck],
        quality: Optional[QualityAssessment],
        strategy: CorrectionStrategy,
        context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Generate a corrected version of the output."""
        try:
            from ..llm.providers import Message, MessageRole, LLMConfig

            # Build correction prompt based on strategy
            failed_checks = [c for c in checks if not c.passed]
            issues = "\n".join([f"- {c.details}" for c in failed_checks])

            suggestions = ""
            if quality and quality.suggestions:
                suggestions = "\nSuggestions:\n" + "\n".join([f"- {s}" for s in quality.suggestions])

            strategy_instructions = {
                CorrectionStrategy.REGENERATE: "Completely rewrite the output from scratch, addressing all issues.",
                CorrectionStrategy.REFINE: "Improve the output while maintaining its structure.",
                CorrectionStrategy.EXPAND: "Add the missing content while keeping existing good content.",
                CorrectionStrategy.CONDENSE: "Shorten the output while keeping the essential information.",
                CorrectionStrategy.RESTRUCTURE: "Reorganize the output to match the required format/structure.",
                CorrectionStrategy.FIX_ERRORS: "Fix the specific errors identified."
            }

            prompt = f"""Correct this output based on the identified issues.

ORIGINAL TASK: {task_description}

CURRENT OUTPUT:
{output[:2500]}

ISSUES TO FIX:
{issues}
{suggestions}

CORRECTION STRATEGY: {strategy.value}
{strategy_instructions[strategy]}

{f"ADDITIONAL CONTEXT: {json.dumps(context)}" if context else ""}

Provide ONLY the corrected output, no explanations:
"""

            messages = [
                Message(MessageRole.SYSTEM, "You are an expert editor. Output ONLY the corrected content."),
                Message(MessageRole.USER, prompt)
            ]

            config = LLMConfig(temperature=0.4, max_tokens=3000)
            response = await self.llm.chat(messages, config)

            return response.content.strip()

        except Exception as e:
            logger.error(f"Correction generation failed: {e}")
            return None

    async def reflect_on_action(
        self,
        action: str,
        action_result: Any,
        intended_outcome: str,
        actual_outcome: str
    ) -> Dict[str, Any]:
        """
        Reflect on an action and its outcome.

        This is useful for learning from actions in the agentic loop.

        Returns dict with:
        - success: Whether action achieved intended outcome
        - lesson: What was learned
        - should_retry: Whether to try again
        - alternative: Alternative approach if retry
        """
        try:
            from ..llm.providers import Message, MessageRole, LLMConfig

            prompt = f"""Reflect on this action and its outcome:

ACTION: {action}
INTENDED OUTCOME: {intended_outcome}
ACTUAL OUTCOME: {actual_outcome}
RESULT DATA: {str(action_result)[:500]}

Analyze and respond with JSON:
{{
    "success": true/false (did it achieve the intended outcome?),
    "partial_success": true/false (did it achieve anything useful?),
    "lesson": "what can be learned from this",
    "should_retry": true/false,
    "alternative_approach": "what to try differently (if should_retry)",
    "insights": ["any useful observations"]
}}
"""

            messages = [
                Message(MessageRole.SYSTEM, "You are a reflective analyst. Respond with JSON only."),
                Message(MessageRole.USER, prompt)
            ]

            config = LLMConfig(temperature=0.3, max_tokens=512)
            response = await self.llm.chat(messages, config)

            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                return json.loads(json_match.group())

        except Exception as e:
            logger.warning(f"Reflection failed: {e}")

        # Fallback
        return {
            "success": False,
            "partial_success": False,
            "lesson": "Reflection could not be completed",
            "should_retry": False,
            "alternative_approach": None,
            "insights": []
        }


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def evaluate_output(
    output: str,
    task_description: str,
    llm_provider,
    criteria: Optional[SuccessCriteria] = None
) -> ReflectionResult:
    """
    Convenience function to evaluate and correct an output.

    Usage:
        result = await evaluate_output(
            "Generated text...",
            "Write a summary",
            llm_provider,
            SuccessCriteria(must_contain=["conclusion"])
        )
    """
    engine = ReflectionEngine(llm_provider)
    return await engine.evaluate_and_correct(output, task_description, criteria)


def create_criteria(
    must_contain: List[str] = None,
    must_not_contain: List[str] = None,
    min_length: int = 0,
    max_length: int = 0,
    quality_threshold: float = 0.7,
    required_format: Optional[str] = None,
    required_sections: List[str] = None
) -> SuccessCriteria:
    """
    Convenience function to create success criteria.

    Usage:
        criteria = create_criteria(
            must_contain=["conclusion", "recommendation"],
            quality_threshold=0.8,
            required_format="markdown"
        )
    """
    return SuccessCriteria(
        must_contain=must_contain or [],
        must_not_contain=must_not_contain or [],
        min_length=min_length,
        max_length=max_length,
        quality_threshold=quality_threshold,
        required_format=required_format,
        required_sections=required_sections or []
    )
