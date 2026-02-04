"""Wrapper for Composio Tool Router tools."""

import time
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum

try:
    from composio import ComposioToolSet, Action
    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False


class ToolName(str, Enum):
    """Composio Tool Router tool names."""
    SEARCH_TOOLS = "RUBE_SEARCH_TOOLS"
    MULTI_EXECUTE = "RUBE_MULTI_EXECUTE_TOOL"
    MANAGE_CONNECTIONS = "RUBE_MANAGE_CONNECTIONS"
    REMOTE_WORKBENCH = "RUBE_REMOTE_WORKBENCH"
    VERCEL_CHAT = "COMPOSIO_SEARCH_VERCEL_AI_CHAT"
    GROQ_CHAT = "COMPOSIO_SEARCH_GROQ_CHAT"


@dataclass
class ToolCall:
    """Represents a single tool call."""
    tool_name: str
    parameters: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    duration_ms: float = 0.0
    raw_response: Optional[dict] = None


@dataclass
class SessionState:
    """Tracks state across Tool Router interactions."""
    session_id: Optional[str] = None
    memory: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    search_results: list[dict] = field(default_factory=list)


class ComposioClient:
    """Client for interacting with Composio tools."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Composio client.

        Args:
            api_key: Optional API key. If not provided, uses COMPOSIO_API_KEY env var.
        """
        self._api_key = api_key
        self._toolset: Optional[Any] = None
        self._session = SessionState()

    @property
    def toolset(self) -> Any:
        """Lazy initialization of Composio toolset."""
        if self._toolset is None:
            if not COMPOSIO_AVAILABLE:
                raise RuntimeError(
                    "composio-core is not installed. "
                    "Install with: pip install composio-core"
                )
            self._toolset = ComposioToolSet(api_key=self._api_key)
        return self._toolset

    @property
    def session(self) -> SessionState:
        """Get the current session state."""
        return self._session

    def reset_session(self) -> None:
        """Reset the session state."""
        self._session = SessionState()

    def run_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        timeout: float = 120.0,
    ) -> ToolResult:
        """Execute a Composio tool.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters to pass to the tool
            timeout: Timeout in seconds

        Returns:
            ToolResult with success status and data/error
        """
        start_time = time.time()

        # Record the call
        self._session.tool_calls.append(ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            timestamp=start_time,
        ))

        try:
            # Get the action
            action = getattr(Action, tool_name, None)
            if action is None:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tool_name}",
                    duration_ms=(time.time() - start_time) * 1000,
                )

            # Execute the tool
            result = self.toolset.execute_action(
                action=action,
                params=parameters,
            )

            duration_ms = (time.time() - start_time) * 1000

            # Parse the result
            if isinstance(result, tuple):
                data, error = result
                if error:
                    return ToolResult(
                        success=False,
                        data=data,
                        error=str(error),
                        duration_ms=duration_ms,
                        raw_response=result,
                    )
                return ToolResult(
                    success=True,
                    data=data,
                    duration_ms=duration_ms,
                    raw_response=result,
                )
            else:
                return ToolResult(
                    success=True,
                    data=result,
                    duration_ms=duration_ms,
                    raw_response=result,
                )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                duration_ms=duration_ms,
            )

    def search_tools(
        self,
        use_case: str,
        generate_session: bool = True,
    ) -> ToolResult:
        """Search for tools matching a use case.

        Args:
            use_case: Natural language description of the task
            generate_session: Whether to generate a new session ID

        Returns:
            ToolResult containing discovered tools
        """
        params = {
            "queries": [{"use_case": use_case}],
            "session": {"generate_id": generate_session},
        }

        result = self.run_tool(ToolName.SEARCH_TOOLS.value, params)

        if result.success and result.data:
            # Extract session ID if present
            if isinstance(result.data, dict):
                if "session_id" in result.data:
                    self._session.session_id = result.data["session_id"]
                self._session.search_results.append(result.data)

        return result

    def execute_tools(
        self,
        tools: list[dict[str, Any]],
        session_id: Optional[str] = None,
    ) -> ToolResult:
        """Execute one or more tools.

        Args:
            tools: List of tool specifications with tool_slug and arguments
            session_id: Session ID from search_tools (uses stored if not provided)

        Returns:
            ToolResult containing execution results
        """
        sid = session_id or self._session.session_id
        if not sid:
            return ToolResult(
                success=False,
                data=None,
                error="No session ID available. Run search_tools first.",
            )

        params = {
            "tools": tools,
            "session_id": sid,
            "memory": self._session.memory,
        }

        return self.run_tool(ToolName.MULTI_EXECUTE.value, params)

    def manage_connections(
        self,
        action: str = "list",
        app_name: Optional[str] = None,
    ) -> ToolResult:
        """Manage app connections.

        Args:
            action: Action to perform (list, check, create)
            app_name: App name for check/create actions

        Returns:
            ToolResult containing connection info
        """
        params = {"action": action}
        if app_name:
            params["app_name"] = app_name

        return self.run_tool(ToolName.MANAGE_CONNECTIONS.value, params)

    def chat_completion(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: Optional[list[dict]] = None,
    ) -> ToolResult:
        """Run a chat completion through Vercel or Groq.

        Args:
            provider: Either "vercel" or "groq"
            model: Model ID to use
            messages: Chat messages
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            tools: Optional tool definitions for function calling

        Returns:
            ToolResult containing the model's response
        """
        tool_name = (
            ToolName.VERCEL_CHAT.value
            if provider == "vercel"
            else ToolName.GROQ_CHAT.value
        )

        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            params["tools"] = tools

        return self.run_tool(tool_name, params)


class MockComposioClient(ComposioClient):
    """Mock client for dry-run and testing."""

    def __init__(self):
        super().__init__()
        self._mock_responses: dict[str, Any] = {}

    def set_mock_response(self, tool_name: str, response: Any) -> None:
        """Set a mock response for a tool."""
        self._mock_responses[tool_name] = response

    def run_tool(
        self,
        tool_name: str,
        parameters: dict[str, Any],
        timeout: float = 120.0,
    ) -> ToolResult:
        """Return mock response instead of calling real API."""
        start_time = time.time()

        self._session.tool_calls.append(ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            timestamp=start_time,
        ))

        # Simulate some latency
        time.sleep(0.1)

        duration_ms = (time.time() - start_time) * 1000

        if tool_name in self._mock_responses:
            return ToolResult(
                success=True,
                data=self._mock_responses[tool_name],
                duration_ms=duration_ms,
            )

        # Default mock responses
        if tool_name == ToolName.SEARCH_TOOLS.value:
            return ToolResult(
                success=True,
                data={
                    "session_id": "mock-session-123",
                    "tools": [
                        {
                            "tool_slug": "GMAIL_FETCH_EMAILS",
                            "description": "Fetch emails from Gmail",
                            "parameters": ["max_results", "query", "user_id"],
                        }
                    ],
                },
                duration_ms=duration_ms,
            )

        if tool_name == ToolName.MULTI_EXECUTE.value:
            return ToolResult(
                success=True,
                data={
                    "results": [
                        {"tool": "GMAIL_FETCH_EMAILS", "status": "success", "data": []}
                    ]
                },
                duration_ms=duration_ms,
            )

        if tool_name in (ToolName.VERCEL_CHAT.value, ToolName.GROQ_CHAT.value):
            return ToolResult(
                success=True,
                data={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "I'll help you with that task.",
                                "tool_calls": [],
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                },
                duration_ms=duration_ms,
            )

        return ToolResult(
            success=True,
            data={"mock": True, "tool": tool_name},
            duration_ms=duration_ms,
        )


def create_client(
    dry_run: bool = False,
    api_key: Optional[str] = None,
) -> ComposioClient:
    """Factory function to create appropriate client.

    Args:
        dry_run: If True, returns a mock client
        api_key: Optional API key for real client

    Returns:
        ComposioClient or MockComposioClient instance
    """
    if dry_run:
        return MockComposioClient()
    return ComposioClient(api_key=api_key)
