"""Custom error types for MDAPFlow-MCP."""

from __future__ import annotations

from typing import Any


class MCPError(Exception):
    """Error type surfaced through the FastMCP server."""

    def __init__(
        self,
        code: int,
        message: str,
        *,
        hint: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        error_data: dict[str, Any] = {}
        if hint:
            error_data["hint"] = hint
        if data:
            error_data.update(data)
        self.data = error_data


class ToolValidationError(MCPError):
    """Exception raised for invalid tool inputs."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(4000, message, hint=hint)


class ToolInternalError(MCPError):
    """Exception raised for failures inside a tool implementation."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(5000, message, hint=hint)


class MDAPExecutionError(ToolInternalError):
    """Exception raised for MDAP execution failures."""

    pass


class LLMProviderError(ToolInternalError):
    """Exception raised for LLM provider failures."""

    pass


class VotingConvergenceError(MDAPExecutionError):
    """Exception raised when voting fails to converge within max rounds."""

    pass
