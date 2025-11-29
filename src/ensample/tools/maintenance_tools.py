"""MCP maintenance and health check tools."""

from __future__ import annotations

import time
from typing import Any

from mcp.server.fastmcp import FastMCP

from ..config import (
    Settings,
    create_default_ensemble_config,
    create_default_red_flag_config,
)
from ..observability import instrumented_tool


class MaintenanceTools:
    """Maintenance and health check tools."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._start_time = time.time()

    def register(self, server: FastMCP) -> None:
        """Register maintenance tools with the FastMCP server."""

        @instrumented_tool(server, name="ensample.ping")
        async def ping() -> dict[str, Any]:
            """Health check and server status."""
            uptime_seconds = int(time.time() - self._start_time)
            uptime_str = self._format_uptime(uptime_seconds)

            # Check if default configs can be loaded
            config_status = "loaded"
            try:
                # Just verify the functions work, don't assign to variables
                create_default_ensemble_config()
                create_default_red_flag_config()
            except Exception as e:
                config_status = f"error: {e}"

            return {
                "status": "ok",
                "message": "Ensample server is running",
                "uptime": uptime_str,
                "mdap_config_loaded": config_status == "loaded",
                "config_status": config_status,
                "version": "0.1.0",
            }

        @instrumented_tool(server, name="ensample.server_info")
        async def server_info() -> dict[str, Any]:
            """Get detailed server information."""
            return {
                "server_name": "Ensample",
                "version": "0.1.0",
                "description": "Ensemble-based LLM Orchestration with Massively Decomposed Agentic Processes",
                "max_concurrent_llm_calls": self.settings.MDAP_MAX_CONCURRENT_LLM_CALLS,
                "max_voting_rounds": self.settings.MDAP_MAX_VOTING_ROUNDS,
                "default_voting_k": self.settings.MDAP_DEFAULT_VOTING_K,
                "log_level": self.settings.MDAP_LOG_LEVEL,
                "otel_endpoint_configured": bool(self.settings.OTEL_EXPORTER_OTLP_ENDPOINT),
            }

    def _format_uptime(self, seconds: int) -> str:
        """Format uptime in human-readable format."""
        days, remainder = divmod(seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        if seconds or not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)
