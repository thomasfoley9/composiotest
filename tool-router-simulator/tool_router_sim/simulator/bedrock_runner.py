"""Bedrock runner for AWS Bedrock model simulations."""

from typing import Any, Optional

from .runner import SimulationRunner
from ..config import ModelConfig, Provider
from ..composio_client import ComposioClient, ToolName


class BedrockRunner(SimulationRunner):
    """Simulation runner using AWS Bedrock via Composio."""

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return "bedrock"

    def run_chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Run a chat completion through AWS Bedrock.

        Args:
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            Raw response from the model
        """
        # Use Composio's Bedrock integration
        # This uses COMPOSIO_SEARCH_BEDROCK_CHAT or similar
        result = self.client.run_tool(
            "COMPOSIO_SEARCH_BEDROCK_CONVERSE",
            {
                "model": self.model_config.model_id,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "tools": tools or [],
            }
        )

        if not result.success:
            raise RuntimeError(f"Bedrock chat failed: {result.error}")

        return result.data


def create_bedrock_runner(
    client: ComposioClient,
    model_name: str,
    **kwargs,
) -> BedrockRunner:
    """Factory to create a Bedrock runner.

    Args:
        client: Composio client
        model_name: Model name (e.g., "bedrock-claude-3.5-sonnet")
        **kwargs: Additional arguments for the runner

    Returns:
        Configured BedrockRunner
    """
    from ..config import BEDROCK_MODELS, get_model_config

    # Look up the model config
    if model_name in BEDROCK_MODELS:
        model_config = BEDROCK_MODELS[model_name]
    else:
        model_config = get_model_config(model_name)
        if model_config is None or model_config.provider != Provider.BEDROCK:
            raise ValueError(
                f"Unknown Bedrock model: {model_name}. "
                f"Available: {list(BEDROCK_MODELS.keys())}"
            )

    return BedrockRunner(
        client=client,
        model_config=model_config,
        **kwargs,
    )
