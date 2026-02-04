"""Evaluator for tool selection quality."""

from dataclasses import dataclass
from typing import Optional

from ..simulator.runner import SimulationTrace


@dataclass
class ToolSelectionScore:
    """Score for tool selection quality."""
    found_relevant_tools: bool
    selected_correct_tool: bool
    avoided_irrelevant_tools: bool
    proper_search_query: bool
    score: float
    details: dict[str, str]

    @classmethod
    def empty(cls) -> "ToolSelectionScore":
        """Create an empty score."""
        return cls(
            found_relevant_tools=False,
            selected_correct_tool=False,
            avoided_irrelevant_tools=True,
            proper_search_query=False,
            score=0.0,
            details={},
        )


class ToolSelectionEvaluator:
    """Evaluates how well a model selects tools."""

    def __init__(
        self,
        expected_tools: Optional[list[str]] = None,
        required_keywords: Optional[list[str]] = None,
        irrelevant_tools: Optional[list[str]] = None,
    ):
        """Initialize the evaluator.

        Args:
            expected_tools: List of expected tool slugs
            required_keywords: Keywords that should appear in search queries
            irrelevant_tools: Tools that should NOT be selected
        """
        self.expected_tools = expected_tools or []
        self.required_keywords = required_keywords or []
        self.irrelevant_tools = irrelevant_tools or []

    def evaluate(self, trace: SimulationTrace) -> ToolSelectionScore:
        """Evaluate tool selection from a simulation trace.

        Args:
            trace: The simulation trace to evaluate

        Returns:
            ToolSelectionScore with detailed breakdown
        """
        details = {}

        # Check if relevant tools were found
        found_relevant = self._check_found_relevant(trace, details)

        # Check if correct tools were selected
        selected_correct = self._check_selected_correct(trace, details)

        # Check if irrelevant tools were avoided
        avoided_irrelevant = self._check_avoided_irrelevant(trace, details)

        # Check if search queries were well-formed
        proper_query = self._check_proper_query(trace, details)

        # Calculate composite score
        weights = {
            "found_relevant": 0.25,
            "selected_correct": 0.35,
            "avoided_irrelevant": 0.20,
            "proper_query": 0.20,
        }

        score = (
            weights["found_relevant"] * (1.0 if found_relevant else 0.0)
            + weights["selected_correct"] * (1.0 if selected_correct else 0.0)
            + weights["avoided_irrelevant"] * (1.0 if avoided_irrelevant else 0.0)
            + weights["proper_query"] * (1.0 if proper_query else 0.0)
        )

        return ToolSelectionScore(
            found_relevant_tools=found_relevant,
            selected_correct_tool=selected_correct,
            avoided_irrelevant_tools=avoided_irrelevant,
            proper_search_query=proper_query,
            score=score,
            details=details,
        )

    def _check_found_relevant(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if search returned relevant tools."""
        if not trace.tools_discovered:
            details["found_relevant"] = "No tools discovered"
            return False

        # Get slugs of discovered tools
        discovered_slugs = {
            t.get("tool_slug", "").upper()
            for t in trace.tools_discovered
        }

        # Check if any expected tools were found
        expected_upper = {t.upper() for t in self.expected_tools}
        found = discovered_slugs.intersection(expected_upper)

        if found:
            details["found_relevant"] = f"Found: {', '.join(found)}"
            return True
        else:
            details["found_relevant"] = (
                f"Expected {expected_upper}, got {discovered_slugs}"
            )
            return False

    def _check_selected_correct(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if correct tools were selected for execution."""
        if not trace.tools_selected:
            details["selected_correct"] = "No tools selected"
            return False

        selected_upper = {t.upper() for t in trace.tools_selected}
        expected_upper = {t.upper() for t in self.expected_tools}

        # All expected tools should be selected
        missing = expected_upper - selected_upper
        if missing:
            details["selected_correct"] = f"Missing: {', '.join(missing)}"
            return False

        # Check if at least the expected tools are selected
        matched = selected_upper.intersection(expected_upper)
        if matched:
            details["selected_correct"] = f"Selected: {', '.join(matched)}"
            return True

        details["selected_correct"] = (
            f"Expected {expected_upper}, selected {selected_upper}"
        )
        return False

    def _check_avoided_irrelevant(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if irrelevant tools were avoided."""
        if not self.irrelevant_tools:
            details["avoided_irrelevant"] = "No irrelevant tools defined"
            return True

        selected_upper = {t.upper() for t in trace.tools_selected}
        irrelevant_upper = {t.upper() for t in self.irrelevant_tools}

        selected_irrelevant = selected_upper.intersection(irrelevant_upper)

        if selected_irrelevant:
            details["avoided_irrelevant"] = (
                f"Selected irrelevant: {', '.join(selected_irrelevant)}"
            )
            return False

        details["avoided_irrelevant"] = "All selections were relevant"
        return True

    def _check_proper_query(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if search queries were well-formed."""
        if not trace.search_queries:
            details["proper_query"] = "No search queries made"
            return False

        if not self.required_keywords:
            details["proper_query"] = "No required keywords defined"
            return True

        # Check if any query contains required keywords
        all_queries = " ".join(trace.search_queries).lower()
        required_lower = [k.lower() for k in self.required_keywords]

        found_keywords = [k for k in required_lower if k in all_queries]
        missing_keywords = [k for k in required_lower if k not in all_queries]

        if missing_keywords:
            details["proper_query"] = (
                f"Missing keywords: {', '.join(missing_keywords)}. "
                f"Queries: {trace.search_queries}"
            )
            return False

        details["proper_query"] = f"Found keywords: {', '.join(found_keywords)}"
        return True


def evaluate_tool_selection(
    trace: SimulationTrace,
    expected_tools: Optional[list[str]] = None,
    required_keywords: Optional[list[str]] = None,
    irrelevant_tools: Optional[list[str]] = None,
) -> ToolSelectionScore:
    """Convenience function to evaluate tool selection.

    Args:
        trace: Simulation trace to evaluate
        expected_tools: Expected tool slugs
        required_keywords: Required query keywords
        irrelevant_tools: Tools to avoid

    Returns:
        ToolSelectionScore
    """
    evaluator = ToolSelectionEvaluator(
        expected_tools=expected_tools,
        required_keywords=required_keywords,
        irrelevant_tools=irrelevant_tools,
    )
    return evaluator.evaluate(trace)
