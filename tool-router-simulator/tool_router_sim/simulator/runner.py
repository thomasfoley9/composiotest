"""Base simulation runner for Tool Router benchmarks."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import (
    ModelConfig,
    TOOL_ROUTER_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)
from ..composio_client import ComposioClient, ToolCall


@dataclass
class ToolInteraction:
    """Record of a single tool interaction during simulation."""
    tool_name: str
    parameters: dict[str, Any]
    result: Any
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class SimulationTrace:
    """Complete trace of a simulation run."""
    model: str
    model_id: str
    provider: str
    task: str
    messages: list[dict[str, str]] = field(default_factory=list)
    tool_interactions: list[ToolInteraction] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    tools_discovered: list[dict[str, Any]] = field(default_factory=list)
    tools_selected: list[str] = field(default_factory=list)
    tools_executed: list[dict[str, Any]] = field(default_factory=list)
    final_response: Optional[str] = None
    session_id: Optional[str] = None

    def add_tool_interaction(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        result: Any,
        success: bool,
        error: Optional[str] = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Add a tool interaction to the trace."""
        self.tool_interactions.append(ToolInteraction(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            success=success,
            error=error,
            duration_ms=duration_ms,
        ))


@dataclass
class PerformanceMetrics:
    """Performance metrics for a simulation."""
    total_time_ms: float = 0.0
    search_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    chat_time_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    num_api_calls: int = 0


@dataclass
class SimulationResult:
    """Complete result of a simulation run."""
    success: bool
    trace: SimulationTrace
    metrics: PerformanceMetrics
    error: Optional[str] = None
    raw_responses: list[Any] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self.trace.model

    @property
    def provider(self) -> str:
        """Get the provider name."""
        return self.trace.provider


class SimulationRunner(ABC):
    """Abstract base class for simulation runners."""

    def __init__(
        self,
        client: ComposioClient,
        model_config: ModelConfig,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the simulation runner.

        Args:
            client: Composio client for API calls
            model_config: Configuration for the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            system_prompt: Optional custom system prompt
        """
        self.client = client
        self.model_config = model_config
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or TOOL_ROUTER_SYSTEM_PROMPT

    @property
    @abstractmethod
    def provider(self) -> str:
        """Get the provider name."""
        pass

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Get Tool Router tool definitions for function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "RUBE_SEARCH_TOOLS",
                    "description": "Search for tools matching a use case. Returns available tools with their parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "use_case": {
                                "type": "string",
                                "description": "Natural language description of what you want to accomplish",
                            },
                        },
                        "required": ["use_case"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "RUBE_MULTI_EXECUTE_TOOL",
                    "description": "Execute one or more tools. Can run multiple tools in parallel.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tools": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "tool_slug": {
                                            "type": "string",
                                            "description": "The tool identifier from search results",
                                        },
                                        "arguments": {
                                            "type": "object",
                                            "description": "Arguments to pass to the tool",
                                        },
                                    },
                                    "required": ["tool_slug"],
                                },
                                "description": "List of tools to execute",
                            },
                        },
                        "required": ["tools"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "RUBE_MANAGE_CONNECTIONS",
                    "description": "Check and manage app connections/authentication.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["list", "check", "create"],
                                "description": "Action to perform",
                            },
                            "app_name": {
                                "type": "string",
                                "description": "App name for check/create actions",
                            },
                        },
                        "required": ["action"],
                    },
                },
            },
        ]

    def build_messages(self, task: str) -> list[dict[str, str]]:
        """Build the initial message list for a task."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

    @abstractmethod
    def run_chat(
        self,
        messages: list[dict[str, str]],
        tools: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Run a chat completion with the model.

        Args:
            messages: Chat messages
            tools: Optional tool definitions

        Returns:
            Raw response from the model
        """
        pass

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        trace: SimulationTrace,
    ) -> dict[str, Any]:
        """Handle a tool call from the model.

        Args:
            tool_call: Tool call from model response
            trace: Simulation trace to update

        Returns:
            Tool result to add to messages
        """
        import json

        tool_name = tool_call.get("function", {}).get("name", "")
        arguments_str = tool_call.get("function", {}).get("arguments", "{}")

        try:
            arguments = json.loads(arguments_str)
        except json.JSONDecodeError:
            arguments = {}

        # Track the tool call
        if tool_name == "RUBE_SEARCH_TOOLS":
            use_case = arguments.get("use_case", "")
            trace.search_queries.append(use_case)

            # Execute the actual search
            result = self.client.search_tools(use_case)

            if result.success and result.data:
                tools_found = result.data.get("tools", [])
                trace.tools_discovered.extend(tools_found)
                trace.session_id = result.data.get("session_id")

            trace.add_tool_interaction(
                tool_name=tool_name,
                parameters=arguments,
                result=result.data,
                success=result.success,
                error=result.error,
                duration_ms=result.duration_ms,
            )

            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result.data) if result.success else result.error,
            }

        elif tool_name == "RUBE_MULTI_EXECUTE_TOOL":
            tools_to_execute = arguments.get("tools", [])
            trace.tools_executed.extend(tools_to_execute)
            trace.tools_selected.extend([t.get("tool_slug", "") for t in tools_to_execute])

            # Execute the tools
            result = self.client.execute_tools(
                tools=tools_to_execute,
                session_id=trace.session_id,
            )

            trace.add_tool_interaction(
                tool_name=tool_name,
                parameters=arguments,
                result=result.data,
                success=result.success,
                error=result.error,
                duration_ms=result.duration_ms,
            )

            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result.data) if result.success else result.error,
            }

        elif tool_name == "RUBE_MANAGE_CONNECTIONS":
            result = self.client.manage_connections(
                action=arguments.get("action", "list"),
                app_name=arguments.get("app_name"),
            )

            trace.add_tool_interaction(
                tool_name=tool_name,
                parameters=arguments,
                result=result.data,
                success=result.success,
                error=result.error,
                duration_ms=result.duration_ms,
            )

            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps(result.data) if result.success else result.error,
            }

        else:
            return {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": json.dumps({"error": f"Unknown tool: {tool_name}"}),
            }

    def simulate(
        self,
        task: str,
        max_turns: int = 10,
    ) -> SimulationResult:
        """Run a complete simulation for a task.

        Args:
            task: The task to perform
            max_turns: Maximum number of conversation turns

        Returns:
            SimulationResult with trace and metrics
        """
        import json

        start_time = time.time()

        trace = SimulationTrace(
            model=self.model_config.name,
            model_id=self.model_config.model_id,
            provider=self.provider,
            task=task,
        )

        metrics = PerformanceMetrics()
        raw_responses = []

        messages = self.build_messages(task)
        trace.messages = messages.copy()
        tools = self.get_tool_definitions()

        try:
            for turn in range(max_turns):
                # Run chat completion
                chat_start = time.time()
                response = self.run_chat(messages, tools)
                chat_duration = (time.time() - chat_start) * 1000
                metrics.chat_time_ms += chat_duration
                metrics.num_api_calls += 1

                raw_responses.append(response)

                # Extract usage info if available
                usage = response.get("usage", {})
                metrics.input_tokens += usage.get("prompt_tokens", 0)
                metrics.output_tokens += usage.get("completion_tokens", 0)
                metrics.total_tokens += usage.get("total_tokens", 0)

                # Get the assistant message
                choices = response.get("choices", [])
                if not choices:
                    break

                assistant_message = choices[0].get("message", {})
                messages.append(assistant_message)
                trace.messages.append(assistant_message)

                # Check for tool calls
                tool_calls = assistant_message.get("tool_calls", [])

                if not tool_calls:
                    # No tool calls, we're done
                    trace.final_response = assistant_message.get("content", "")
                    break

                # Handle each tool call
                for tool_call in tool_calls:
                    tool_result = self.handle_tool_call(tool_call, trace)
                    messages.append(tool_result)
                    trace.messages.append(tool_result)

                    # Track timing for specific tools
                    tool_name = tool_call.get("function", {}).get("name", "")
                    if tool_name == "RUBE_SEARCH_TOOLS":
                        metrics.search_time_ms += trace.tool_interactions[-1].duration_ms
                    elif tool_name == "RUBE_MULTI_EXECUTE_TOOL":
                        metrics.execution_time_ms += trace.tool_interactions[-1].duration_ms

            # Calculate total time and cost
            metrics.total_time_ms = (time.time() - start_time) * 1000
            metrics.estimated_cost_usd = self._calculate_cost(metrics)

            return SimulationResult(
                success=True,
                trace=trace,
                metrics=metrics,
                raw_responses=raw_responses,
            )

        except Exception as e:
            metrics.total_time_ms = (time.time() - start_time) * 1000

            return SimulationResult(
                success=False,
                trace=trace,
                metrics=metrics,
                error=str(e),
                raw_responses=raw_responses,
            )

    def _calculate_cost(self, metrics: PerformanceMetrics) -> float:
        """Calculate estimated cost for the simulation."""
        input_cost = (metrics.input_tokens / 1000) * self.model_config.input_cost_per_1k
        output_cost = (metrics.output_tokens / 1000) * self.model_config.output_cost_per_1k
        return input_cost + output_cost
