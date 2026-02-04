"""Model configurations and pricing for the Tool Router Simulator."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Provider(str, Enum):
    """Model provider types."""
    VERCEL = "vercel"
    GROQ = "groq"
    BEDROCK = "bedrock"


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

# Groq Models (fast inference) - All available models
GROQ_MODELS = {
    # Llama 3.3 Models
    "llama-3.3-70b": ModelConfig(
        name="Llama 3.3 70B",
        provider=Provider.GROQ,
        model_id="llama-3.3-70b-versatile",
        input_cost_per_1k=0.00059,
        output_cost_per_1k=0.00079,
        max_tokens=32768,
        description="Meta's large versatile model - best overall",
    ),
    "llama-3.3-70b-specdec": ModelConfig(
        name="Llama 3.3 70B SpecDec",
        provider=Provider.GROQ,
        model_id="llama-3.3-70b-specdec",
        input_cost_per_1k=0.00059,
        output_cost_per_1k=0.00099,
        max_tokens=8192,
        description="Speculative decoding variant - faster inference",
    ),
    # Llama 3.2 Models
    "llama-3.2-90b-vision": ModelConfig(
        name="Llama 3.2 90B Vision",
        provider=Provider.GROQ,
        model_id="llama-3.2-90b-vision-preview",
        input_cost_per_1k=0.0009,
        output_cost_per_1k=0.0009,
        max_tokens=8192,
        description="Vision-capable large model",
    ),
    "llama-3.2-11b-vision": ModelConfig(
        name="Llama 3.2 11B Vision",
        provider=Provider.GROQ,
        model_id="llama-3.2-11b-vision-preview",
        input_cost_per_1k=0.00018,
        output_cost_per_1k=0.00018,
        max_tokens=8192,
        description="Vision-capable compact model",
    ),
    "llama-3.2-3b": ModelConfig(
        name="Llama 3.2 3B",
        provider=Provider.GROQ,
        model_id="llama-3.2-3b-preview",
        input_cost_per_1k=0.00006,
        output_cost_per_1k=0.00006,
        max_tokens=8192,
        description="Ultra-compact Llama model",
    ),
    "llama-3.2-1b": ModelConfig(
        name="Llama 3.2 1B",
        provider=Provider.GROQ,
        model_id="llama-3.2-1b-preview",
        input_cost_per_1k=0.00004,
        output_cost_per_1k=0.00004,
        max_tokens=8192,
        description="Smallest Llama model - fastest",
    ),
    # Llama 3.1 Models
    "llama-3.1-405b": ModelConfig(
        name="Llama 3.1 405B",
        provider=Provider.GROQ,
        model_id="llama-3.1-405b-reasoning",
        input_cost_per_1k=0.00,  # Free tier
        output_cost_per_1k=0.00,
        max_tokens=32768,
        description="Largest Llama - best reasoning",
    ),
    "llama-3.1-70b": ModelConfig(
        name="Llama 3.1 70B",
        provider=Provider.GROQ,
        model_id="llama-3.1-70b-versatile",
        input_cost_per_1k=0.00059,
        output_cost_per_1k=0.00079,
        max_tokens=32768,
        description="Large versatile Llama 3.1",
    ),
    "llama-3.1-8b": ModelConfig(
        name="Llama 3.1 8B",
        provider=Provider.GROQ,
        model_id="llama-3.1-8b-instant",
        input_cost_per_1k=0.00005,
        output_cost_per_1k=0.00008,
        max_tokens=8192,
        description="Fast, small Llama model",
    ),
    # Mixtral Models
    "mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        provider=Provider.GROQ,
        model_id="mixtral-8x7b-32768",
        input_cost_per_1k=0.00024,
        output_cost_per_1k=0.00024,
        max_tokens=32768,
        description="Mistral's mixture of experts",
    ),
    # Gemma Models
    "gemma2-9b": ModelConfig(
        name="Gemma 2 9B",
        provider=Provider.GROQ,
        model_id="gemma2-9b-it",
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0002,
        max_tokens=8192,
        description="Google's instruction-tuned Gemma",
    ),
    # DeepSeek Models
    "deepseek-r1-distill-llama-70b": ModelConfig(
        name="DeepSeek R1 Distill 70B",
        provider=Provider.GROQ,
        model_id="deepseek-r1-distill-llama-70b",
        input_cost_per_1k=0.00075,
        output_cost_per_1k=0.00099,
        max_tokens=8192,
        description="DeepSeek reasoning model distilled to Llama",
    ),
    # Qwen Models
    "qwen-2.5-72b": ModelConfig(
        name="Qwen 2.5 72B",
        provider=Provider.GROQ,
        model_id="qwen-2.5-72b",
        input_cost_per_1k=0.0009,
        output_cost_per_1k=0.0009,
        max_tokens=32768,
        description="Alibaba's large language model",
    ),
    "qwen-2.5-32b": ModelConfig(
        name="Qwen 2.5 32B",
        provider=Provider.GROQ,
        model_id="qwen-2.5-32b",
        input_cost_per_1k=0.00079,
        output_cost_per_1k=0.00079,
        max_tokens=32768,
        description="Mid-size Qwen model",
    ),
    "qwen-qwq-32b": ModelConfig(
        name="Qwen QwQ 32B",
        provider=Provider.GROQ,
        model_id="qwen-qwq-32b",
        input_cost_per_1k=0.00029,
        output_cost_per_1k=0.00039,
        max_tokens=32768,
        description="Qwen reasoning model",
    ),
    # Mistral Models
    "mistral-saba-24b": ModelConfig(
        name="Mistral Saba 24B",
        provider=Provider.GROQ,
        model_id="mistral-saba-24b",
        input_cost_per_1k=0.00079,
        output_cost_per_1k=0.00079,
        max_tokens=32768,
        description="Mistral's efficient 24B model",
    ),
}

# AWS Bedrock Models
BEDROCK_MODELS = {
    # Anthropic Claude Models
    "bedrock-claude-3.5-sonnet": ModelConfig(
        name="Claude 3.5 Sonnet",
        provider=Provider.BEDROCK,
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=8192,
        description="Anthropic's balanced Claude on Bedrock",
    ),
    "bedrock-claude-3.5-haiku": ModelConfig(
        name="Claude 3.5 Haiku",
        provider=Provider.BEDROCK,
        model_id="anthropic.claude-3-5-haiku-20241022-v1:0",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.005,
        max_tokens=8192,
        description="Fast Claude on Bedrock",
    ),
    "bedrock-claude-3-opus": ModelConfig(
        name="Claude 3 Opus",
        provider=Provider.BEDROCK,
        model_id="anthropic.claude-3-opus-20240229-v1:0",
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        max_tokens=4096,
        description="Most capable Claude on Bedrock",
    ),
    # Amazon Titan Models
    "bedrock-titan-premier": ModelConfig(
        name="Titan Text Premier",
        provider=Provider.BEDROCK,
        model_id="amazon.titan-text-premier-v1:0",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        max_tokens=3072,
        description="Amazon's premier text model",
    ),
    "bedrock-titan-express": ModelConfig(
        name="Titan Text Express",
        provider=Provider.BEDROCK,
        model_id="amazon.titan-text-express-v1",
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0006,
        max_tokens=8192,
        description="Fast Amazon Titan model",
    ),
    "bedrock-titan-lite": ModelConfig(
        name="Titan Text Lite",
        provider=Provider.BEDROCK,
        model_id="amazon.titan-text-lite-v1",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.0002,
        max_tokens=4096,
        description="Lightweight Amazon Titan",
    ),
    # Meta Llama Models on Bedrock
    "bedrock-llama-3.2-90b": ModelConfig(
        name="Llama 3.2 90B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-2-90b-instruct-v1:0",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.002,
        max_tokens=8192,
        description="Large Llama on Bedrock",
    ),
    "bedrock-llama-3.2-11b": ModelConfig(
        name="Llama 3.2 11B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-2-11b-instruct-v1:0",
        input_cost_per_1k=0.00035,
        output_cost_per_1k=0.00035,
        max_tokens=8192,
        description="Compact Llama on Bedrock",
    ),
    "bedrock-llama-3.2-3b": ModelConfig(
        name="Llama 3.2 3B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-2-3b-instruct-v1:0",
        input_cost_per_1k=0.00015,
        output_cost_per_1k=0.00015,
        max_tokens=8192,
        description="Small Llama on Bedrock",
    ),
    "bedrock-llama-3.2-1b": ModelConfig(
        name="Llama 3.2 1B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-2-1b-instruct-v1:0",
        input_cost_per_1k=0.0001,
        output_cost_per_1k=0.0001,
        max_tokens=8192,
        description="Smallest Llama on Bedrock",
    ),
    "bedrock-llama-3.1-405b": ModelConfig(
        name="Llama 3.1 405B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-1-405b-instruct-v1:0",
        input_cost_per_1k=0.00532,
        output_cost_per_1k=0.016,
        max_tokens=8192,
        description="Largest Llama on Bedrock",
    ),
    "bedrock-llama-3.1-70b": ModelConfig(
        name="Llama 3.1 70B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-1-70b-instruct-v1:0",
        input_cost_per_1k=0.00099,
        output_cost_per_1k=0.00099,
        max_tokens=8192,
        description="Large Llama 3.1 on Bedrock",
    ),
    "bedrock-llama-3.1-8b": ModelConfig(
        name="Llama 3.1 8B",
        provider=Provider.BEDROCK,
        model_id="meta.llama3-1-8b-instruct-v1:0",
        input_cost_per_1k=0.0003,
        output_cost_per_1k=0.0006,
        max_tokens=8192,
        description="Compact Llama 3.1 on Bedrock",
    ),
    # Mistral Models on Bedrock
    "bedrock-mistral-large": ModelConfig(
        name="Mistral Large 2",
        provider=Provider.BEDROCK,
        model_id="mistral.mistral-large-2407-v1:0",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.009,
        max_tokens=8192,
        description="Mistral's flagship on Bedrock",
    ),
    "bedrock-mistral-small": ModelConfig(
        name="Mistral Small",
        provider=Provider.BEDROCK,
        model_id="mistral.mistral-small-2402-v1:0",
        input_cost_per_1k=0.001,
        output_cost_per_1k=0.003,
        max_tokens=8192,
        description="Efficient Mistral on Bedrock",
    ),
    "bedrock-mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        provider=Provider.BEDROCK,
        model_id="mistral.mixtral-8x7b-instruct-v0:1",
        input_cost_per_1k=0.00045,
        output_cost_per_1k=0.0007,
        max_tokens=8192,
        description="Mixtral MoE on Bedrock",
    ),
    # Cohere Models
    "bedrock-cohere-command-r-plus": ModelConfig(
        name="Command R+",
        provider=Provider.BEDROCK,
        model_id="cohere.command-r-plus-v1:0",
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=4096,
        description="Cohere's enterprise model",
    ),
    "bedrock-cohere-command-r": ModelConfig(
        name="Command R",
        provider=Provider.BEDROCK,
        model_id="cohere.command-r-v1:0",
        input_cost_per_1k=0.0005,
        output_cost_per_1k=0.0015,
        max_tokens=4096,
        description="Cohere's efficient model",
    ),
    # AI21 Labs Models
    "bedrock-jamba-1.5-large": ModelConfig(
        name="Jamba 1.5 Large",
        provider=Provider.BEDROCK,
        model_id="ai21.jamba-1-5-large-v1:0",
        input_cost_per_1k=0.002,
        output_cost_per_1k=0.008,
        max_tokens=4096,
        description="AI21's large hybrid model",
    ),
    "bedrock-jamba-1.5-mini": ModelConfig(
        name="Jamba 1.5 Mini",
        provider=Provider.BEDROCK,
        model_id="ai21.jamba-1-5-mini-v1:0",
        input_cost_per_1k=0.0002,
        output_cost_per_1k=0.0004,
        max_tokens=4096,
        description="AI21's efficient model",
    ),
    # Amazon Nova Models
    "bedrock-nova-pro": ModelConfig(
        name="Nova Pro",
        provider=Provider.BEDROCK,
        model_id="amazon.nova-pro-v1:0",
        input_cost_per_1k=0.0008,
        output_cost_per_1k=0.0032,
        max_tokens=5120,
        description="Amazon's Nova Pro model",
    ),
    "bedrock-nova-lite": ModelConfig(
        name="Nova Lite",
        provider=Provider.BEDROCK,
        model_id="amazon.nova-lite-v1:0",
        input_cost_per_1k=0.00006,
        output_cost_per_1k=0.00024,
        max_tokens=5120,
        description="Amazon's Nova Lite model",
    ),
    "bedrock-nova-micro": ModelConfig(
        name="Nova Micro",
        provider=Provider.BEDROCK,
        model_id="amazon.nova-micro-v1:0",
        input_cost_per_1k=0.000035,
        output_cost_per_1k=0.00014,
        max_tokens=5120,
        description="Amazon's fastest Nova model",
    ),
}

# All models combined
ALL_MODELS = {**VERCEL_MODELS, **GROQ_MODELS, **BEDROCK_MODELS}


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
    elif provider == Provider.BEDROCK:
        return BEDROCK_MODELS
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
