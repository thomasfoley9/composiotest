"""Benchmark results reporter with rich terminal output."""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.box import DOUBLE_EDGE, ROUNDED

from ..evaluator.scorer import BenchmarkScore, compare_scores
from ..simulator.runner import SimulationResult


class ReportFormat(str, Enum):
    """Output format for reports."""
    TERMINAL = "terminal"
    JSON = "json"
    MARKDOWN = "markdown"


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    task: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    results: dict[str, SimulationResult] = field(default_factory=dict)
    scores: dict[str, BenchmarkScore] = field(default_factory=dict)
    comparison: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkReporter:
    """Reporter for benchmark results with rich terminal output."""

    def __init__(self, console: Optional[Console] = None):
        """Initialize the reporter.

        Args:
            console: Rich console for output (creates new if not provided)
        """
        self.console = console or Console()

    def create_report(
        self,
        task: str,
        results: dict[str, SimulationResult],
        scores: dict[str, BenchmarkScore],
    ) -> BenchmarkReport:
        """Create a benchmark report.

        Args:
            task: The task that was benchmarked
            results: Dict mapping model names to results
            scores: Dict mapping model names to scores

        Returns:
            BenchmarkReport
        """
        comparison = compare_scores(scores)

        return BenchmarkReport(
            task=task,
            results=results,
            scores=scores,
            comparison=comparison,
        )

    def print_report(self, report: BenchmarkReport) -> None:
        """Print a report to the terminal.

        Args:
            report: The report to print
        """
        # Title panel
        self.console.print()
        self.console.print(Panel(
            f"[bold cyan]COMPOSIO TOOL ROUTER - MODEL BENCHMARK RESULTS[/bold cyan]\n"
            f"[dim]Task: \"{report.task}\"[/dim]",
            box=DOUBLE_EDGE,
            padding=(1, 2),
        ))

        # Results table
        self._print_results_table(report)

        # Best in category
        self._print_best_in_category(report)

        # Detailed analysis
        self._print_detailed_analysis(report)

    def _print_results_table(self, report: BenchmarkReport) -> None:
        """Print the main results table."""
        table = Table(
            title="",
            box=ROUNDED,
            show_header=True,
            header_style="bold white",
        )

        table.add_column("Model", style="cyan", min_width=20)
        table.add_column("Tool Selection", justify="center", min_width=14)
        table.add_column("Execution", justify="center", min_width=10)
        table.add_column("Speed", justify="right", min_width=8)
        table.add_column("Cost", justify="right", min_width=8)
        table.add_column("Grade", justify="center", min_width=6)

        # Sort by overall score
        sorted_scores = sorted(
            report.scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True,
        )

        for model, score in sorted_scores:
            # Get metrics
            metrics = score.details.get("metrics", {})
            total_time = metrics.get("total_time_ms", 0)
            cost = metrics.get("estimated_cost_usd", 0)

            # Format values
            tool_sel_pct = f"{score.tool_selection.score * 100:.0f}%"
            exec_pct = f"{score.execution.score * 100:.0f}%"
            speed = f"{total_time / 1000:.1f}s" if total_time > 0 else "N/A"
            cost_str = f"${cost:.4f}" if cost > 0 else "N/A"

            # Color the grade
            grade_color = self._get_grade_color(score.grade)
            grade_styled = f"[{grade_color}]{score.grade}[/{grade_color}]"

            table.add_row(
                model,
                tool_sel_pct,
                exec_pct,
                speed,
                cost_str,
                grade_styled,
            )

        self.console.print(table)

    def _print_best_in_category(self, report: BenchmarkReport) -> None:
        """Print best in category section."""
        best = report.comparison.get("best", {})

        if not best:
            return

        self.console.print()

        lines = []

        if best.get("tool_selection"):
            score = report.scores.get(best["tool_selection"])
            if score:
                lines.append(
                    f"[yellow]🏆 BEST TOOL SELECTION:[/yellow] "
                    f"{best['tool_selection']} ({score.tool_selection.score * 100:.0f}%)"
                )

        if best.get("fastest"):
            score = report.scores.get(best["fastest"])
            if score:
                time_ms = score.details.get("metrics", {}).get("total_time_ms", 0)
                lines.append(
                    f"[cyan]⚡ FASTEST:[/cyan] "
                    f"{best['fastest']} ({time_ms / 1000:.1f}s)"
                )

        if best.get("cheapest"):
            score = report.scores.get(best["cheapest"])
            if score:
                cost = score.details.get("metrics", {}).get("estimated_cost_usd", 0)
                lines.append(
                    f"[green]💰 CHEAPEST:[/green] "
                    f"{best['cheapest']} (${cost:.4f})"
                )

        if best.get("overall"):
            lines.append(f"[magenta]⭐ BEST OVERALL:[/magenta] {best['overall']}")

        for line in lines:
            self.console.print(line)

    def _print_detailed_analysis(self, report: BenchmarkReport) -> None:
        """Print detailed analysis for each model."""
        self.console.print()
        self.console.print("[bold]DETAILED ANALYSIS:[/bold]")
        self.console.print("─" * 60)

        for model, score in report.scores.items():
            self.console.print(f"\n[bold cyan]{model}:[/bold cyan]")

            # Tool selection details
            ts = score.tool_selection
            if ts.found_relevant_tools:
                self.console.print(f"  [green]✓[/green] Found relevant tools")
            else:
                self.console.print(f"  [red]✗[/red] {ts.details.get('found_relevant', 'No tools found')}")

            if ts.selected_correct_tool:
                self.console.print(f"  [green]✓[/green] Selected correct tool")
            else:
                self.console.print(f"  [red]✗[/red] {ts.details.get('selected_correct', 'Wrong selection')}")

            # Search queries
            queries = score.details.get("search_queries", [])
            if queries:
                self.console.print(f"  [dim]Search query: \"{queries[0]}\"[/dim]")

            # Tools selected
            selected = score.details.get("tools_selected", [])
            if selected:
                self.console.print(f"  [dim]Selected: {', '.join(selected)}[/dim]")

            # Execution details
            ex = score.execution
            if ex.successful_execution:
                self.console.print(f"  [green]✓[/green] Execution successful")
            else:
                self.console.print(f"  [red]✗[/red] {ex.details.get('success', 'Execution failed')}")

            if ex.correct_parameters:
                self.console.print(f"  [green]✓[/green] Parameters correct")
            else:
                self.console.print(f"  [yellow]~[/yellow] {ex.details.get('parameters', 'Parameter issues')}")

    def _get_grade_color(self, grade: str) -> str:
        """Get color for a grade."""
        if grade.startswith("A"):
            return "green"
        elif grade.startswith("B"):
            return "cyan"
        elif grade.startswith("C"):
            return "yellow"
        elif grade.startswith("D"):
            return "orange3"
        else:
            return "red"

    def export_json(self, report: BenchmarkReport, path: str | Path) -> None:
        """Export report to JSON.

        Args:
            report: The report to export
            path: Path to save to
        """
        path = Path(path)

        # Convert to serializable format
        data = {
            "task": report.task,
            "timestamp": report.timestamp,
            "scores": {
                model: {
                    "overall_score": score.overall_score,
                    "grade": score.grade,
                    "tool_selection": {
                        "score": score.tool_selection.score,
                        "found_relevant_tools": score.tool_selection.found_relevant_tools,
                        "selected_correct_tool": score.tool_selection.selected_correct_tool,
                        "details": score.tool_selection.details,
                    },
                    "execution": {
                        "score": score.execution.score,
                        "successful_execution": score.execution.successful_execution,
                        "correct_parameters": score.execution.correct_parameters,
                        "details": score.execution.details,
                    },
                    "planning": {
                        "score": score.planning.score,
                        "logical_step_order": score.planning.logical_step_order,
                        "details": score.planning.details,
                    },
                    "details": score.details,
                }
                for model, score in report.scores.items()
            },
            "comparison": report.comparison,
            "metadata": report.metadata,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        self.console.print(f"[green]Report exported to {path}[/green]")

    def export_markdown(self, report: BenchmarkReport, path: str | Path) -> None:
        """Export report to Markdown.

        Args:
            report: The report to export
            path: Path to save to
        """
        path = Path(path)

        lines = [
            "# Composio Tool Router - Model Benchmark Results",
            "",
            f"**Task:** {report.task}",
            f"**Timestamp:** {report.timestamp}",
            "",
            "## Results Summary",
            "",
            "| Model | Tool Selection | Execution | Speed | Cost | Grade |",
            "|-------|----------------|-----------|-------|------|-------|",
        ]

        sorted_scores = sorted(
            report.scores.items(),
            key=lambda x: x[1].overall_score,
            reverse=True,
        )

        for model, score in sorted_scores:
            metrics = score.details.get("metrics", {})
            total_time = metrics.get("total_time_ms", 0)
            cost = metrics.get("estimated_cost_usd", 0)

            lines.append(
                f"| {model} | "
                f"{score.tool_selection.score * 100:.0f}% | "
                f"{score.execution.score * 100:.0f}% | "
                f"{total_time / 1000:.1f}s | "
                f"${cost:.4f} | "
                f"{score.grade} |"
            )

        # Best in category
        best = report.comparison.get("best", {})
        if best:
            lines.extend([
                "",
                "## Best in Category",
                "",
            ])
            if best.get("tool_selection"):
                lines.append(f"- **Best Tool Selection:** {best['tool_selection']}")
            if best.get("fastest"):
                lines.append(f"- **Fastest:** {best['fastest']}")
            if best.get("cheapest"):
                lines.append(f"- **Cheapest:** {best['cheapest']}")
            if best.get("overall"):
                lines.append(f"- **Best Overall:** {best['overall']}")

        # Detailed analysis
        lines.extend([
            "",
            "## Detailed Analysis",
            "",
        ])

        for model, score in report.scores.items():
            lines.append(f"### {model}")
            lines.append("")

            ts = score.tool_selection
            lines.append(f"- Found relevant tools: {'✓' if ts.found_relevant_tools else '✗'}")
            lines.append(f"- Selected correct tool: {'✓' if ts.selected_correct_tool else '✗'}")

            queries = score.details.get("search_queries", [])
            if queries:
                lines.append(f"- Search query: `{queries[0]}`")

            selected = score.details.get("tools_selected", [])
            if selected:
                lines.append(f"- Tools selected: {', '.join(selected)}")

            ex = score.execution
            lines.append(f"- Execution successful: {'✓' if ex.successful_execution else '✗'}")
            lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

        self.console.print(f"[green]Report exported to {path}[/green]")


def print_quick_summary(
    task: str,
    scores: dict[str, BenchmarkScore],
    console: Optional[Console] = None,
) -> None:
    """Print a quick summary of benchmark results.

    Args:
        task: The task description
        scores: Dict mapping model names to scores
        console: Rich console (creates new if not provided)
    """
    console = console or Console()

    console.print(f"\n[bold]Results for:[/bold] {task}")
    console.print("-" * 40)

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1].overall_score,
        reverse=True,
    )

    for i, (model, score) in enumerate(sorted_scores, 1):
        grade_color = "green" if score.grade.startswith("A") else "yellow" if score.grade.startswith(("B", "C")) else "red"
        console.print(
            f"{i}. {model}: [{grade_color}]{score.grade}[/{grade_color}] "
            f"({score.overall_score * 100:.0f}%)"
        )
