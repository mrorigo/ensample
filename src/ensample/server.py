"""FastMCP server bootstrap for Ensample."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from .config import Settings
from .exceptions import MCPError
from .mdap_engine import MDAPEngine
from .observability import LOGGER, configure_logging, configure_tracing
from .tools.maintenance_tools import MaintenanceTools
from .tools.mdap_tools import MDAPTools

configure_logging()
configure_tracing()


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Lifespan manager for server startup and shutdown."""
    LOGGER.info("Starting Ensample server", extra={"version": "0.1.0"})

    # Initialize settings
    settings = Settings()

    # Initialize MDAP engine
    mdap_engine = MDAPEngine(settings)

    # Initialize tool managers
    mdap_tools = MDAPTools(mdap_engine)
    maintenance_tools = MaintenanceTools(settings)

    # Register tools
    mdap_tools.register(server)
    maintenance_tools.register(server)

    # Store references for external access
    server.mdap_engine = mdap_engine  # type: ignore[attr-defined]
    server.settings = settings  # type: ignore[attr-defined]

    LOGGER.info("Ensample server initialized", extra={
        "max_concurrent_calls": settings.MDAP_MAX_CONCURRENT_LLM_CALLS,
        "max_voting_rounds": settings.MDAP_MAX_VOTING_ROUNDS,
    })

    try:
        yield {
            "settings": settings,
            "mdap_engine": mdap_engine,
            "mdap_tools": mdap_tools,
            "maintenance_tools": maintenance_tools,
        }
    finally:
        LOGGER.info("Ensample server shutdown")


server = FastMCP("Ensample", lifespan=lifespan)


def main() -> None:
    """Entry point used by `uv run ensample`."""
    transport = os.environ.get("MDAP_SERVER_TRANSPORT", "stdio").lower()
    try:
        if transport == "stdio":
            server.run()
        elif transport == "sse":
            asyncio.run(server.run_sse_async())
        elif transport in {"streamable-http", "http"}:
            asyncio.run(server.run_streamable_http_async())
        else:
            raise MCPError(4000, f"Unsupported transport: {transport}")
    except MCPError as exc:
        LOGGER.error("MCP runtime error: %s", exc, extra={"code": getattr(exc, 'code', 'n/a')})
        raise
