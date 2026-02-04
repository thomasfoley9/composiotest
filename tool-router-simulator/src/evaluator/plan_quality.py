"""Evaluator for planning quality."""

from dataclasses import dataclass
from typing import Optional

from ..simulator.runner import SimulationTrace


@dataclass
class PlanningScore:
    """Score for planning quality."""
    logical_step_order: bool
    handled_prerequisites: bool
    minimal_steps: bool
    score: float
    details: dict[str, str]

    @classmethod
    def empty(cls) -> "PlanningScore":
        """Create an empty score."""
        return cls(
            logical_step_order=False,
            handled_prerequisites=False,
            minimal_steps=True,
            score=0.0,
            details={},
        )


class PlanEvaluator:
    """Evaluates the quality of the model's planning."""

    # Expected order of Tool Router operations
    EXPECTED_FLOW = [
        "RUBE_SEARCH_TOOLS",
        "RUBE_MANAGE_CONNECTIONS",  # Optional
        "RUBE_MULTI_EXECUTE_TOOL",
    ]

    def __init__(
        self,
        expected_flow: Optional[list[str]] = None,
        require_connection_check: bool = False,
        max_steps: int = 6,
    ):
        """Initialize the evaluator.

        Args:
            expected_flow: Expected sequence of operations
            require_connection_check: Whether MANAGE_CONNECTIONS is required
            max_steps: Maximum number of steps for minimal plan
        """
        self.expected_flow = expected_flow or self.EXPECTED_FLOW
        self.require_connection_check = require_connection_check
        self.max_steps = max_steps

    def evaluate(self, trace: SimulationTrace) -> PlanningScore:
        """Evaluate planning quality from a simulation trace.

        Args:
            trace: The simulation trace to evaluate

        Returns:
            PlanningScore with detailed breakdown
        """
        details = {}

        # Check logical step order
        logical_order = self._check_step_order(trace, details)

        # Check if prerequisites were handled
        handled_prereqs = self._check_prerequisites(trace, details)

        # Check if plan was minimal
        minimal = self._check_minimal(trace, details)

        # Calculate composite score
        weights = {
            "logical_order": 0.40,
            "handled_prereqs": 0.35,
            "minimal": 0.25,
        }

        score = (
            weights["logical_order"] * (1.0 if logical_order else 0.0)
            + weights["handled_prereqs"] * (1.0 if handled_prereqs else 0.0)
            + weights["minimal"] * (1.0 if minimal else 0.0)
        )

        return PlanningScore(
            logical_step_order=logical_order,
            handled_prerequisites=handled_prereqs,
            minimal_steps=minimal,
            score=score,
            details=details,
        )

    def _check_step_order(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if steps were in logical order."""
        if not trace.tool_interactions:
            details["step_order"] = "No tool interactions"
            return False

        # Get the sequence of tool calls
        tool_sequence = [i.tool_name for i in trace.tool_interactions]

        # Build expected sequence (removing optional steps not present)
        actual_flow = []
        for tool in tool_sequence:
            if tool in self.expected_flow and tool not in actual_flow:
                actual_flow.append(tool)

        # Check order
        search_idx = -1
        execute_idx = -1
        connection_idx = -1

        for i, tool in enumerate(tool_sequence):
            if tool == "RUBE_SEARCH_TOOLS" and search_idx == -1:
                search_idx = i
            elif tool == "RUBE_MULTI_EXECUTE_TOOL" and execute_idx == -1:
                execute_idx = i
            elif tool == "RUBE_MANAGE_CONNECTIONS" and connection_idx == -1:
                connection_idx = i

        # Search should come before execute
        if search_idx == -1:
            details["step_order"] = "No search performed"
            return False

        if execute_idx == -1:
            details["step_order"] = "No execution performed"
            return False

        if search_idx > execute_idx:
            details["step_order"] = "Search came after execution"
            return False

        # If connection check happened, it should be between search and execute
        if connection_idx != -1:
            if connection_idx < search_idx or connection_idx > execute_idx:
                details["step_order"] = "Connection check in wrong position"
                # This is a soft error, still logical
                pass

        details["step_order"] = f"Order: {' -> '.join(tool_sequence[:5])}..."
        return True

    def _check_prerequisites(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if prerequisites were handled."""
        tool_names = {i.tool_name for i in trace.tool_interactions}

        # Search is always a prerequisite for execution
        if "RUBE_MULTI_EXECUTE_TOOL" in tool_names:
            if "RUBE_SEARCH_TOOLS" not in tool_names:
                details["prerequisites"] = "Executed without searching first"
                return False

        # Check if connection was verified when required
        if self.require_connection_check:
            if "RUBE_MANAGE_CONNECTIONS" not in tool_names:
                details["prerequisites"] = "Connection check required but not performed"
                return False

        # Check if search was performed before execution
        search_performed = False
        for interaction in trace.tool_interactions:
            if interaction.tool_name == "RUBE_SEARCH_TOOLS":
                search_performed = True
            elif interaction.tool_name == "RUBE_MULTI_EXECUTE_TOOL":
                if not search_performed:
                    details["prerequisites"] = "Execution before search"
                    return False

        details["prerequisites"] = "All prerequisites handled"
        return True

    def _check_minimal(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if the plan was minimal."""
        num_steps = len(trace.tool_interactions)

        if num_steps == 0:
            details["minimal"] = "No steps taken"
            return False

        if num_steps <= self.max_steps:
            details["minimal"] = f"{num_steps} steps (max: {self.max_steps})"
            return True

        details["minimal"] = f"Too many steps: {num_steps} (max: {self.max_steps})"
        return False


def evaluate_planning(
    trace: SimulationTrace,
    expected_flow: Optional[list[str]] = None,
    require_connection_check: bool = False,
    max_steps: int = 6,
) -> PlanningScore:
    """Convenience function to evaluate planning quality.

    Args:
        trace: Simulation trace to evaluate
        expected_flow: Expected sequence of operations
        require_connection_check: Whether connection check is required
        max_steps: Maximum steps for minimal plan

    Returns:
        PlanningScore
    """
    evaluator = PlanEvaluator(
        expected_flow=expected_flow,
        require_connection_check=require_connection_check,
        max_steps=max_steps,
    )
    return evaluator.evaluate(trace)
