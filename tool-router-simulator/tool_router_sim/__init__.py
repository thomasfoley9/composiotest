"""Composio Tool Router Model Simulator.

A benchmarking tool for testing how different LLM models perform
when using Composio's Tool Router system.
"""

__version__ = "0.1.0"

from .composio_client import ComposioClient, create_client
from .config import (
    ALL_MODELS,
    VERCEL_MODELS,
    GROQ_MODELS,
    ModelConfig,
    Provider,
)
from .simulator.runner import SimulationRunner, SimulationResult, SimulationTrace
from .simulator.vercel_runner import VercelRunner, create_vercel_runner
from .simulator.groq_runner import GroqRunner, create_groq_runner
from .evaluator.scorer import CompositeScorer, BenchmarkScore
from .benchmark.tasks import BenchmarkTask, TaskSuite, load_benchmark_suite

__all__ = [
    # Version
    "__version__",
    # Client
    "ComposioClient",
    "create_client",
    # Config
    "ALL_MODELS",
    "VERCEL_MODELS",
    "GROQ_MODELS",
    "ModelConfig",
    "Provider",
    # Runners
    "SimulationRunner",
    "SimulationResult",
    "SimulationTrace",
    "VercelRunner",
    "GroqRunner",
    "create_vercel_runner",
    "create_groq_runner",
    # Evaluation
    "CompositeScorer",
    "BenchmarkScore",
    # Benchmarks
    "BenchmarkTask",
    "TaskSuite",
    "load_benchmark_suite",
]
