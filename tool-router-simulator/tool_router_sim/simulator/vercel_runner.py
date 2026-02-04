"""Vercel AI Gateway runner for simulations."""

from typing import Any, Optional

from .runner import SimulationRunner
from ..config import ModelConfig, Provider
from ..composio_client import ComposioClient


class VercelRunner(SimulationRunner):
    """Simulation runner using Vercel AI Gateway via Composio."""

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "vercel"

    def run_chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Run a chat completion through Vercel AI Gateway.

        Args:
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            Raw response from the model
        """
        result = self.client.chat_completion(
            provider="vercel",
            model=self.model_config.model_id,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tools=tools,
        )

        if not result.success:
            raise RuntimeError(f"Vercel chat failed: {result.error}")

        return result.data


def create_vercel_runner(
    client: ComposioClient,
    model_name: str,
    **kwargs,
) -> VercelRunner:
    """Factory to create a Vercel runner.

    Args:
        client: Composio client
        model_name: Model name (e.g., "claude-sonnet-4")
        **kwargs: Additional arguments for the runner

    Returns:
        Configured VercelRunner
    """
    from ..config import VERCEL_MODELS, get_model_config

    # Look up the model config
    if model_name in VERCEL_MODELS:
        model_config = VERCEL_MODELS[model_name]
    else:
        model_config = get_model_config(model_name)
        if model_config is None or model_config.provider != Provider.VERCEL:
            raise ValueError(
                f"Unknown Vercel model: {model_name}. "
                f"Available: {list(VERCEL_MODELS.keys())}"
            )

    return VercelRunner(
        client=client,
        model_config=model_config,
        **kwargs,
    )
