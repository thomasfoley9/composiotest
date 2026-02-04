"""Groq runner for fast inference simulations."""

from typing import Any, Optional

from .runner import SimulationRunner
from ..config import ModelConfig, Provider
from ..composio_client import ComposioClient


class GroqRunner(SimulationRunner):
    """Simulation runner using Groq via Composio."""

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "groq"

    def run_chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Run a chat completion through Groq.

        Args:
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            Raw response from the model
        """
        result = self.client.chat_completion(
            provider="groq",
            model=self.model_config.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=tools,
        )

        if not result.success:
            raise RuntimeError(f"Groq chat failed: {result.error}")

        return result.data


def create_groq_runner(
    client: ComposioClient,
    model_name: str,
    **kwargs,
) -> GroqRunner:
    """Factory to create a Groq runner.

    Args:
        client: Composio client
        model_name: Model name (e.g., "llama-3.3-70b")
        **kwargs: Additional arguments for the runner

    Returns:
        Configured GroqRunner
    """
    from ..config import GROQ_MODELS, get_model_config

    # Look up the model config
    if model_name in GROQ_MODELS:
        model_config = GROQ_MODELS[model_name]
    else:
        model_config = get_model_config(model_name)
        if model_config is None or model_config.provider != Provider.GROQ:
            raise ValueError(
                f"Unknown Groq model: {model_name}. "
                f"Available: {list(GROQ_MODELS.keys())}"
            )

    return GroqRunner(
        client=client,
        model_config=model_config,
        **kwargs,
    )
