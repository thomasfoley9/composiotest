"""Composite scorer for overall benchmark evaluation."""

from dataclasses import dataclass, field
from typing import Any, Optional

from .tool_selection import ToolSelectionScore, ToolSelectionEvaluator
from .execution_quality import ExecutionScore, ExecutionEvaluator
from .plan_quality import PlanningScore, PlanEvaluator
from ..simulator.runner import SimulationResult, SimulationTrace


@dataclass
class BenchmarkScore:
    """Complete benchmark score for a simulation."""
    tool_selection: ToolSelectionScore
    execution: ExecutionScore
    planning: PlanningScore
    overall_score: float
    grade: str
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "BenchmarkScore":
        """Create an empty score."""
        return cls(
            tool_selection=ToolSelectionScore.empty(),
            execution=ExecutionScore.empty(),
            planning=PlanningScore.empty(),
            overall_score=0.0,
            grade="F",
            details={},
        )


class CompositeScorer:
    """Combines all evaluators into a composite score."""

    # Grade thresholds
    GRADE_THRESHOLDS = [
        (0.95, "A+"),
        (0.90, "A"),
        (0.85, "A-"),
        (0.80, "B+"),
        (0.75, "B"),
        (0.70, "B-"),
        (0.65, "C+"),
        (0.60, "C"),
        (0.55, "C-"),
        (0.50, "D+"),
        (0.45, "D"),
        (0.40, "D-"),
        (0.0, "F"),
    ]

    def __init__(
        self,
        expected_tools: Optional[list[str]] = None,
        expected_params: Optional[dict[str, Any]] = None,
        required_keywords: Optional[list[str]] = None,
        irrelevant_tools: Optional[list[str]] = None,
        required_output_fields: Optional[list[str]] = None,
        expected_flow: Optional[list[str]] = None,
        require_connection_check: bool = False,
        max_tool_calls: int = 5,
        max_steps: int = 6,
        weights: Optional[dict[str, float]] = None,
    ):
        """Initialize the composite scorer.

        Args:
            expected_tools: Expected tool slugs
            expected_params: Expected execution parameters
            required_keywords: Required search query keywords
            irrelevant_tools: Tools to avoid
            required_output_fields: Required output fields
            expected_flow: Expected operation sequence
            require_connection_check: Whether connection check is required
            max_tool_calls: Maximum efficient tool calls
            max_steps: Maximum steps for minimal plan
            weights: Custom weights for each component
        """
        self.tool_selection_evaluator = ToolSelectionEvaluator(
            expected_tools=expected_tools,
            required_keywords=required_keywords,
            irrelevant_tools=irrelevant_tools,
        )
        self.execution_evaluator = ExecutionEvaluator(
            expected_params=expected_params,
            required_output_fields=required_output_fields,
            max_tool_calls=max_tool_calls,
        )
        self.plan_evaluator = PlanEvaluator(
            expected_flow=expected_flow,
            require_connection_check=require_connection_check,
            max_steps=max_steps,
        )

        # Default weights
        self.weights = weights or {
            "tool_selection": 0.40,
            "execution": 0.35,
            "planning": 0.25,
        }

    def score(self, result: SimulationResult) -> BenchmarkScore:
        """Score a simulation result.

        Args:
            result: The simulation result to score

        Returns:
            BenchmarkScore with all components
        """
        trace = result.trace

        # Evaluate each component
        tool_selection = self.tool_selection_evaluator.evaluate(trace)
        execution = self.execution_evaluator.evaluate(trace)
        planning = self.plan_evaluator.evaluate(trace)

        # Calculate overall score
        overall = (
            self.weights["tool_selection"] * tool_selection.score
            + self.weights["execution"] * execution.score
            + self.weights["planning"] * planning.score
        )

        # Determine grade
        grade = self._calculate_grade(overall)

        # Compile details
        details = {
            "model": result.trace.model,
            "model_id": result.trace.model_id,
            "provider": result.trace.provider,
            "success": result.success,
            "error": result.error,
            "metrics": {
                "total_time_ms": result.metrics.total_time_ms,
                "search_time_ms": result.metrics.search_time_ms,
                "execution_time_ms": result.metrics.execution_time_ms,
                "total_tokens": result.metrics.total_tokens,
                "estimated_cost_usd": result.metrics.estimated_cost_usd,
            },
            "search_queries": trace.search_queries,
            "tools_discovered": [
                t.get("tool_slug", "") for t in trace.tools_discovered
            ],
            "tools_selected": trace.tools_selected,
        }

        return BenchmarkScore(
            tool_selection=tool_selection,
            execution=execution,
            planning=planning,
            overall_score=overall,
            grade=grade,
            details=details,
        )

    def _calculate_grade(self, score: float) -> str:
        """Calculate letter grade from numeric score."""
        for threshold, grade in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"


def score_simulation(
    result: SimulationResult,
    expected_tools: Optional[list[str]] = None,
    expected_params: Optional[dict[str, Any]] = None,
    required_keywords: Optional[list[str]] = None,
    **kwargs,
) -> BenchmarkScore:
    """Convenience function to score a simulation.

    Args:
        result: Simulation result to score
        expected_tools: Expected tool slugs
        expected_params: Expected parameters
        required_keywords: Required search keywords
        **kwargs: Additional scorer arguments

    Returns:
        BenchmarkScore
    """
    scorer = CompositeScorer(
        expected_tools=expected_tools,
        expected_params=expected_params,
        required_keywords=required_keywords,
        **kwargs,
    )
    return scorer.score(result)


def compare_scores(scores: dict[str, BenchmarkScore]) -> dict[str, Any]:
    """Compare scores across multiple models.

    Args:
        scores: Dict mapping model names to scores

    Returns:
        Comparison summary with rankings
    """
    if not scores:
        return {}

    # Calculate rankings for each dimension
    rankings = {
        "overall": sorted(
            scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True,
        ),
        "tool_selection": sorted(
            scores.items(),
            key=lambda x: x[1].tool_selection.score,
            reverse=True,
        ),
        "execution": sorted(
            scores.items(),
            key=lambda x: x[1].execution.score,
            reverse=True,
        ),
        "planning": sorted(
            scores.items(),
            key=lambda x: x[1].planning.score,
            reverse=True,
        ),
    }

    # Find best in each category
    best = {
        "overall": rankings["overall"][0][0] if rankings["overall"] else None,
        "tool_selection": rankings["tool_selection"][0][0] if rankings["tool_selection"] else None,
        "execution": rankings["execution"][0][0] if rankings["execution"] else None,
        "planning": rankings["planning"][0][0] if rankings["planning"] else None,
    }

    # Find fastest and cheapest
    by_speed = sorted(
        scores.items(),
        key=lambda x: x[1].details.get("metrics", {}).get("total_time_ms", float("inf")),
    )
    by_cost = sorted(
        scores.items(),
        key=lambda x: x[1].details.get("metrics", {}).get("estimated_cost_usd", float("inf")),
    )

    best["fastest"] = by_speed[0][0] if by_speed else None
    best["cheapest"] = by_cost[0][0] if by_cost else None

    return {
        "rankings": {k: [(m, s.overall_score) for m, s in v] for k, v in rankings.items()},
        "best": best,
        "summary": {
            model: {
                "overall_score": score.overall_score,
                "grade": score.grade,
                "tool_selection": score.tool_selection.score,
                "execution": score.execution.score,
                "planning": score.planning.score,
            }
            for model, score in scores.items()
        },
    }
