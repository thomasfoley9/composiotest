"""Model configurations and pricing for the Tool Router Simulator."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    """Model provider types."""
    VERCEL = "vercel"
    GROQ = "groq"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: Provider
    model_id: str
    input_cost_per_1k: float  # USD per 1000 input tokens
    output_cost_per_1k: float  # USD per 1000 output tokens
    max_tokens: int = 4096
    supports_tools: bool = True
    description: str = ""


# Vercel AI Gateway Models
VERCEL_MODELS = {
    "claude-sonnet-4": ModelConfig(
        name="Claude Sonnet 4",
        provider=Provider.VERCEL,
        model_id="anthropic/claude-sonnet-4-20250514",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        description="Anthropic's balanced model with excellent tool use",
    ),
    "claude-haiku-4.5": ModelConfig(
        name="Claude Haiku 4.5",
        provider=Provider.VERCEL,
        model_id="anthropic/claude-haiku-4-5-20251001",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
        description="Fast and cost-effective Claude model",
    ),
    "gpt-4o": ModelConfig(
        name="GPT-4o",
        provider=Provider.VERCEL,
        model_id="openai/gpt-4o",
        input_cost_per_1k=0.005,
        output_cost_per_1k=0.015,
        description="OpenAI's flagship multimodal model",
    ),
    "gpt-4o-mini": ModelConfig(
        name="GPT-4o Mini",
        provider=Provider.VERCEL,
        model_id="openai/gpt-4o-mini",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0006,
        description="Cost-effective OpenAI model",
    ),
    "gemini-2.0-flash": ModelConfig(
        name="Gemini 2.0 Flash",
        provider=Provider.VERCEL,
        model_id="google/gemini-2.0-flash",
        input_cost_per_1k=0.00035,
        output_cost_per_1k=0.0015,
        description="Google's fast multimodal model",
    ),
}

# Groq Models (fast inference)
GROQ_MODELS = {
    "llama-3.3-70b": ModelConfig(
        name="Llama 3.3 70B",
        provider=Provider.GROQ,
        model_id="llama-3.3-70b-versatile",
        input_cost_per_1k=0.00059,
        output_cost_per_1k=0.00079,
        description="Meta's large versatile model on Groq",
    ),
    "llama-3.1-8b": ModelConfig(
        name="Llama 3.1 8B",
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.00008,
        description="Fast, small Llama model",
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        provider=Provider.GROQ,
        model_id="mixtral-8x7b-32768",
        input_cost_per_1k=0.00024,
        output_cost_per_1k=0.00024,
        description="Mistral's mixture of experts model",
    ),
    "gemma2-9b": ModelConfig(
        name="Gemma 2 9B",
        provider=Provider.GROQ,
        model_id="gemma2-9b-it",
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0002,
        description="Google's instruction-tuned Gemma model",
    ),
}

# All models combined
ALL_MODELS = {**VERCEL_MODELS, **GROQ_MODELS}


def get_model_config(model_name: str) -> Optional[ModelConfig]:
    """Get configuration for a model by name or ID."""
    # Direct lookup
    if model_name in ALL_MODELS:
        return ALL_MODELS[model_name]

    # Search by model_id
    for config in ALL_MODELS.values():
        if config.model_id == model_name:
            return config

    return None


def get_models_by_provider(provider: Provider) -> dict[str, ModelConfig]:
    """Get all models for a specific provider."""
    if provider == Provider.VERCEL:
        return VERCEL_MODELS
    elif provider == Provider.GROQ:
        return GROQ_MODELS
    return {}


def list_all_model_names() -> list[str]:
    """Get a list of all available model names."""
    return list(ALL_MODELS.keys())


def calculate_cost(
    model: ModelConfig,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Calculate the cost for a model run."""
    input_cost = (input_tokens / 1000) * model.input_cost_per_1k
    output_cost = (output_tokens / 1000) * model.output_cost_per_1k
    return input_cost + output_cost


# Tool Router system prompt
TOOL_ROUTER_SYSTEM_PROMPT = """You are an AI assistant with access to Composio Tool Router.

Available tools:
- RUBE_SEARCH_TOOLS: Search for tools matching a use case. Use this to discover available tools.
- RUBE_MULTI_EXECUTE_TOOL: Execute one or more discovered tools in parallel.
- RUBE_MANAGE_CONNECTIONS: Check and manage app connections/authentication.
- RUBE_REMOTE_WORKBENCH: Python execution environment for data processing.

For any task:
1. First use RUBE_SEARCH_TOOLS to find relevant tools for the task
2. Check that required connections are active using RUBE_MANAGE_CONNECTIONS
3. Execute the appropriate tools using RUBE_MULTI_EXECUTE_TOOL
4. Return the results to the user

Be precise in your search queries and tool selections. Only execute tools that are necessary for the task."""


# Default settings
DEFAULT_TEMPERATURE = 0.0  # Deterministic for benchmarking
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 120
