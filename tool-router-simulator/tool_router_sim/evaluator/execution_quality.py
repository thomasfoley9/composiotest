"""Evaluator for tool execution quality."""

from dataclasses import dataclass
from typing import Any, Optional

from ..simulator.runner import SimulationTrace


@dataclass
class ExecutionScore:
    """Score for execution quality."""
    correct_parameters: bool
    successful_execution: bool
    complete_result: bool
    efficient_execution: bool
    score: float
    details: dict[str, str]

    @classmethod
    def empty(cls) -> "ExecutionScore":
        """Create an empty score."""
        return cls(
            correct_parameters=False,
            successful_execution=False,
            complete_result=False,
            efficient_execution=True,
            score=0.0,
            details={},
        )


class ExecutionEvaluator:
    """Evaluates how well a model executes tools."""

    def __init__(
        self,
        expected_params: Optional[dict[str, Any]] = None,
        required_output_fields: Optional[list[str]] = None,
        max_tool_calls: int = 5,
    ):
        """Initialize the evaluator.

        Args:
            expected_params: Expected parameters for tool execution
            required_output_fields: Fields that should be in the output
            max_tool_calls: Maximum efficient number of tool calls
        """
        self.expected_params = expected_params or {}
        self.required_output_fields = required_output_fields or []
        self.max_tool_calls = max_tool_calls

    def evaluate(self, trace: SimulationTrace) -> ExecutionScore:
        """Evaluate execution quality from a simulation trace.

        Args:
            trace: The simulation trace to evaluate

        Returns:
            ExecutionScore with detailed breakdown
        """
        details = {}

        # Check if parameters were correct
        correct_params = self._check_parameters(trace, details)

        # Check if execution was successful
        successful = self._check_success(trace, details)

        # Check if result was complete
        complete = self._check_completeness(trace, details)

        # Check if execution was efficient
        efficient = self._check_efficiency(trace, details)

        # Calculate composite score
        weights = {
            "correct_params": 0.30,
            "successful": 0.35,
            "complete": 0.20,
            "efficient": 0.15,
        }

        score = (
            weights["correct_params"] * (1.0 if correct_params else 0.0)
            + weights["successful"] * (1.0 if successful else 0.0)
            + weights["complete"] * (1.0 if complete else 0.0)
            + weights["efficient"] * (1.0 if efficient else 0.0)
        )

        return ExecutionScore(
            correct_parameters=correct_params,
            successful_execution=successful,
            complete_result=complete,
            efficient_execution=efficient,
            score=score,
            details=details,
        )

    def _check_parameters(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if parameters were correct."""
        if not trace.tools_executed:
            details["parameters"] = "No tools executed"
            return False

        if not self.expected_params:
            details["parameters"] = "No expected params defined"
            return True

        # Check each executed tool's parameters
        all_correct = True
        param_issues = []

        for tool_exec in trace.tools_executed:
            tool_args = tool_exec.get("arguments", {})

            for param_name, expected_value in self.expected_params.items():
                actual_value = tool_args.get(param_name)

                if actual_value is None:
                    param_issues.append(f"Missing: {param_name}")
                    all_correct = False
                elif actual_value != expected_value:
                    # Check for partial matches (e.g., query contains expected)
                    if isinstance(expected_value, str) and isinstance(actual_value, str):
                        if expected_value.lower() in actual_value.lower():
                            continue
                    param_issues.append(
                        f"{param_name}: expected '{expected_value}', got '{actual_value}'"
                    )
                    all_correct = False

        if param_issues:
            details["parameters"] = "; ".join(param_issues)
        else:
            details["parameters"] = "All parameters correct"

        return all_correct

    def _check_success(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if tool execution was successful."""
        # Look for RUBE_MULTI_EXECUTE_TOOL results
        execute_interactions = [
            i for i in trace.tool_interactions
            if i.tool_name == "RUBE_MULTI_EXECUTE_TOOL"
        ]

        if not execute_interactions:
            details["success"] = "No execution calls made"
            return False

        # Check if all executions succeeded
        all_success = True
        errors = []

        for interaction in execute_interactions:
            if not interaction.success:
                all_success = False
                errors.append(interaction.error or "Unknown error")
            elif interaction.result:
                # Check for errors in result
                if isinstance(interaction.result, dict):
                    result_errors = interaction.result.get("errors", [])
                    if result_errors:
                        all_success = False
                        errors.extend(str(e) for e in result_errors)

        if errors:
            details["success"] = f"Errors: {'; '.join(errors)}"
        else:
            details["success"] = "All executions successful"

        return all_success

    def _check_completeness(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if the result was complete."""
        if not self.required_output_fields:
            details["completeness"] = "No required fields defined"
            return True

        # Look for execution results
        execute_interactions = [
            i for i in trace.tool_interactions
            if i.tool_name == "RUBE_MULTI_EXECUTE_TOOL"
        ]

        if not execute_interactions:
            details["completeness"] = "No execution results"
            return False

        # Check for required fields in any result
        found_fields = set()

        for interaction in execute_interactions:
            if interaction.result:
                self._extract_fields(interaction.result, found_fields)

        required_set = set(self.required_output_fields)
        missing = required_set - found_fields

        if missing:
            details["completeness"] = f"Missing fields: {', '.join(missing)}"
            return False

        details["completeness"] = f"Found all required fields: {', '.join(found_fields)}"
        return True

    def _extract_fields(self, data: Any, fields: set) -> None:
        """Recursively extract field names from data."""
        if isinstance(data, dict):
            for key, value in data.items():
                fields.add(key)
                self._extract_fields(value, fields)
        elif isinstance(data, list):
            for item in data:
                self._extract_fields(item, fields)

    def _check_efficiency(
        self,
        trace: SimulationTrace,
        details: dict[str, str],
    ) -> bool:
        """Check if execution was efficient."""
        # Count total tool calls
        total_calls = len(trace.tool_interactions)

        if total_calls <= self.max_tool_calls:
            details["efficiency"] = f"{total_calls} calls (max: {self.max_tool_calls})"
            return True

        details["efficiency"] = (
            f"Too many calls: {total_calls} (max: {self.max_tool_calls})"
        )
        return False


def evaluate_execution(
    trace: SimulationTrace,
    expected_params: Optional[dict[str, Any]] = None,
    required_output_fields: Optional[list[str]] = None,
    max_tool_calls: int = 5,
) -> ExecutionScore:
    """Convenience function to evaluate execution quality.

    Args:
        trace: Simulation trace to evaluate
        expected_params: Expected parameters
        required_output_fields: Required output fields
        max_tool_calls: Maximum efficient calls

    Returns:
        ExecutionScore
    """
    evaluator = ExecutionEvaluator(
        expected_params=expected_params,
        required_output_fields=required_output_fields,
        max_tool_calls=max_tool_calls,
    )
    return evaluator.evaluate(trace)
