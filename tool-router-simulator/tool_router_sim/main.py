"""CLI entry point for the Tool Router Simulator."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import (
    ALL_MODELS,
    VERCEL_MODELS,
    GROQ_MODELS,
    Provider,
    list_all_model_names,
)
from .composio_client import create_client, ComposioClient
from .simulator.runner import SimulationResult
from .simulator.vercel_runner import create_vercel_runner
from .simulator.groq_runner import create_groq_runner
from .evaluator.scorer import CompositeScorer, BenchmarkScore
from .benchmark.tasks import (
    BenchmarkTask,
    TaskSuite,
    load_benchmark_suite,
    get_builtin_suite,
    get_all_builtin_tasks,
)
from .benchmark.reporter import BenchmarkReporter, BenchmarkReport, ReportFormat

console = Console()


def get_runner_for_model(model_name: str, client: ComposioClient):
    """Create the appropriate runner for a model."""
    if model_name in VERCEL_MODELS:
        return create_vercel_runner(client, model_name)
    elif model_name in GROQ_MODELS:
        return create_groq_runner(client, model_name)
    else:
        # Try to determine by model config
        config = ALL_MODELS.get(model_name)
        if config:
            if config.provider == Provider.VERCEL:
                return create_vercel_runner(client, model_name)
            else:
                return create_groq_runner(client, model_name)
        raise ValueError(f"Unknown model: {model_name}")


def run_simulation(
    task: str,
    model_name: str,
    client: ComposioClient,
    task_def: Optional[BenchmarkTask] = None,
) -> tuple[SimulationResult, BenchmarkScore]:
    """Run a simulation for a single model.

    Args:
        task: The task to run
        model_name: Name of the model to use
        client: Composio client
        task_def: Optional task definition with expected values

    Returns:
        Tuple of (SimulationResult, BenchmarkScore)
    """
    runner = get_runner_for_model(model_name, client)
    result = runner.simulate(task)

    # Create scorer with task expectations if available
    if task_def:
        scorer = CompositeScorer(
            expected_tools=task_def.expected_tools,
            expected_params=task_def.expected_params,
            required_keywords=task_def.required_keywords,
            irrelevant_tools=task_def.irrelevant_tools,
            required_output_fields=task_def.required_output_fields,
            expected_flow=task_def.expected_flow,
            require_connection_check=task_def.require_connection_check,
        )
    else:
        scorer = CompositeScorer()

    score = scorer.score(result)
    return result, score


def run_all_models(
    task: str,
    models: list[str],
    client: ComposioClient,
    task_def: Optional[BenchmarkTask] = None,
    max_workers: int = 5,
) -> dict[str, tuple[SimulationResult, BenchmarkScore]]:
    """Run simulations across multiple models in parallel.

    Args:
        task: The task to run
        models: List of model names
        client: Composio client
        task_def: Optional task definition
        max_workers: Maximum parallel workers

    Returns:
        Dict mapping model names to (result, score) tuples
    """
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task_id = progress.add_task("Running simulations...", total=len(models))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for model_name in models:
                future = executor.submit(
                    run_simulation, task, model_name, client, task_def
                )
                futures[future] = model_name

            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    result, score = future.result()
                    results[model_name] = (result, score)
                except Exception as e:
                    console.print(f"[red]Error running {model_name}: {e}[/red]")
                progress.advance(task_id)

    return results


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Composio Tool Router Model Simulator.

    Benchmark how different LLM models perform when using Composio's Tool Router.
    """
    pass


@cli.command()
@click.option("--task", "-t", required=True, help="The task to run")
@click.option(
    "--models", "-m",
    help="Comma-separated list of models to test (default: all)",
)
@click.option("--dry-run", is_flag=True, help="Run without calling APIs")
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Export results to JSON file",
)
@click.option("--parallel", "-p", default=5, help="Max parallel workers")
def run(task: str, models: Optional[str], dry_run: bool, output: Optional[str], parallel: int):
    """Run a single task across models.

    Example:
        python -m tool_router_sim run --task "List my unread emails"
    """
    console.print(f"\n[bold]Task:[/bold] {task}")
    console.print(f"[bold]Dry run:[/bold] {dry_run}")

    # Parse models
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = list_all_model_names()

    console.print(f"[bold]Models:[/bold] {', '.join(model_list)}")

    # Create client
    client = create_client(dry_run=dry_run)

    # Run simulations
    results = run_all_models(task, model_list, client, max_workers=parallel)

    # Create report
    sim_results = {m: r for m, (r, _) in results.items()}
    scores = {m: s for m, (_, s) in results.items()}

    reporter = BenchmarkReporter(console)
    report = reporter.create_report(task, sim_results, scores)
    reporter.print_report(report)

    # Export if requested
    if output:
        output_path = Path(output)
        if output_path.suffix == ".json":
            reporter.export_json(report, output_path)
        elif output_path.suffix == ".md":
            reporter.export_markdown(report, output_path)
        else:
            reporter.export_json(report, output_path.with_suffix(".json"))


@cli.command()
@click.option(
    "--suite", "-s",
    required=True,
    help="Benchmark suite name (simple_tasks, multi_step_tasks, edge_cases) or path to JSON file",
)
@click.option(
    "--models", "-m",
    help="Comma-separated list of models (default: all)",
)
@click.option("--dry-run", is_flag=True, help="Run without calling APIs")
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="results",
    help="Directory for results",
)
@click.option("--parallel", "-p", default=5, help="Max parallel workers")
def benchmark(suite: str, models: Optional[str], dry_run: bool, output_dir: str, parallel: int):
    """Run a complete benchmark suite.

    Example:
        python -m tool_router_sim benchmark --suite simple_tasks
    """
    # Load suite
    suite_path = Path(suite)
    if suite_path.exists():
        task_suite = load_benchmark_suite(suite_path)
    else:
        task_suite = get_builtin_suite(suite)
        if not task_suite:
            console.print(f"[red]Unknown suite: {suite}[/red]")
            console.print("Available: simple_tasks, multi_step_tasks, edge_cases")
            sys.exit(1)

    console.print(f"\n[bold]Suite:[/bold] {task_suite.name}")
    console.print(f"[bold]Tasks:[/bold] {len(task_suite.tasks)}")

    # Parse models
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = list_all_model_names()

    console.print(f"[bold]Models:[/bold] {', '.join(model_list)}")
    console.print(f"[bold]Dry run:[/bold] {dry_run}")

    # Create client and reporter
    client = create_client(dry_run=dry_run)
    reporter = BenchmarkReporter(console)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run each task
    all_reports = []

    for task_def in task_suite.tasks:
        console.print(f"\n[bold cyan]Running task: {task_def.id}[/bold cyan]")
        console.print(f"[dim]{task_def.task}[/dim]")

        results = run_all_models(
            task_def.task,
            model_list,
            client,
            task_def=task_def,
            max_workers=parallel,
        )

        sim_results = {m: r for m, (r, _) in results.items()}
        scores = {m: s for m, (_, s) in results.items()}

        report = reporter.create_report(task_def.task, sim_results, scores)
        report.metadata["task_id"] = task_def.id
        report.metadata["difficulty"] = task_def.difficulty.value
        all_reports.append(report)

        reporter.print_report(report)

        # Save individual report
        report_file = output_path / f"{task_def.id}.json"
        reporter.export_json(report, report_file)

    # Print summary
    console.print("\n" + "=" * 60)
    console.print("[bold]BENCHMARK SUMMARY[/bold]")
    console.print("=" * 60)

    # Aggregate scores by model
    model_totals: dict[str, list[float]] = {}
    for report in all_reports:
        for model, score in report.scores.items():
            if model not in model_totals:
                model_totals[model] = []
            model_totals[model].append(score.overall_score)

    # Print average scores
    console.print("\n[bold]Average Scores by Model:[/bold]")
    sorted_models = sorted(
        model_totals.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True,
    )
    for model, scores_list in sorted_models:
        avg = sum(scores_list) / len(scores_list)
        console.print(f"  {model}: {avg * 100:.1f}%")


@cli.command()
@click.option(
    "--models", "-m",
    help="Comma-separated list of models (default: all)",
)
@click.option("--dry-run", is_flag=True, help="Run without calling APIs")
def interactive(models: Optional[str], dry_run: bool):
    """Interactive mode for running tasks.

    Example:
        python -m tool_router_sim interactive
    """
    console.print("\n[bold cyan]Tool Router Simulator - Interactive Mode[/bold cyan]")
    console.print("[dim]Type 'quit' or 'exit' to leave, 'models' to list models[/dim]\n")

    # Parse models
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = list_all_model_names()

    client = create_client(dry_run=dry_run)
    reporter = BenchmarkReporter(console)

    while True:
        try:
            task = console.input("[bold green]Task>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not task:
            continue

        if task.lower() in ("quit", "exit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if task.lower() == "models":
            console.print("\n[bold]Available models:[/bold]")
            console.print("\n[cyan]Vercel (AI Gateway):[/cyan]")
            for name, config in VERCEL_MODELS.items():
                console.print(f"  {name}: {config.description}")
            console.print("\n[cyan]Groq (Fast Inference):[/cyan]")
            for name, config in GROQ_MODELS.items():
                console.print(f"  {name}: {config.description}")
            console.print()
            continue

        if task.lower() == "help":
            console.print("\n[bold]Commands:[/bold]")
            console.print("  models - List available models")
            console.print("  quit/exit - Exit interactive mode")
            console.print("  <any task> - Run the task across all models")
            console.print()
            continue

        # Run the task
        console.print()
        results = run_all_models(task, model_list, client, max_workers=5)

        sim_results = {m: r for m, (r, _) in results.items()}
        scores = {m: s for m, (_, s) in results.items()}

        report = reporter.create_report(task, sim_results, scores)
        reporter.print_report(report)
        console.print()


@cli.command()
def list_models():
    """List all available models."""
    console.print("\n[bold]Available Models[/bold]\n")

    console.print("[cyan]Vercel AI Gateway:[/cyan]")
    for name, config in VERCEL_MODELS.items():
        console.print(f"  [bold]{name}[/bold]")
        console.print(f"    ID: {config.model_id}")
        console.print(f"    Description: {config.description}")
        console.print(f"    Cost: ${config.input_cost_per_1k}/1k in, ${config.output_cost_per_1k}/1k out")
        console.print()

    console.print("[cyan]Groq (Fast Inference):[/cyan]")
    for name, config in GROQ_MODELS.items():
        console.print(f"  [bold]{name}[/bold]")
        console.print(f"    ID: {config.model_id}")
        console.print(f"    Description: {config.description}")
        console.print(f"    Cost: ${config.input_cost_per_1k}/1k in, ${config.output_cost_per_1k}/1k out")
        console.print()


@cli.command()
def list_tasks():
    """List all built-in benchmark tasks."""
    console.print("\n[bold]Built-in Benchmark Tasks[/bold]\n")

    all_tasks = get_all_builtin_tasks()

    for task in all_tasks.tasks:
        console.print(f"[bold cyan]{task.id}[/bold cyan] [{task.difficulty.value}]")
        console.print(f"  {task.task}")
        if task.expected_tools:
            console.print(f"  [dim]Expected tools: {', '.join(task.expected_tools)}[/dim]")
        console.print()


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
