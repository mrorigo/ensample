"""MCP tools for MDAP execution."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from ..mdap_engine import MDAPEngine
from ..models import EnsembleConfig, MDAPInput, RedFlagConfig
from ..observability import instrumented_tool


class MDAPTools:
    """MDAP execution tools."""

    def __init__(self, mdap_engine: MDAPEngine) -> None:
        self.mdap_engine = mdap_engine

    def register(self, server: FastMCP) -> None:
        """Register MDAP tools with the FastMCP server."""

        @instrumented_tool(server, name="mdapflow.execute_llm_role")
        async def execute_llm_role(
            prompt: str = Field(..., description="The natural language prompt for the LLM role or function."),
            role_name: str = Field(..., description="A descriptive identifier for the LLM role or function being performed by the client (e.g., 'DocumentSummarizer', 'CodeGenerator', 'IntentClassifier'). Used for logging and metrics."),
            ensemble_config: dict[str, Any] | None = Field(None, description="Optional: overrides default ensemble configuration."),
            voting_k: int = Field(3, ge=0, description="The 'k' value for 'first-to-ahead-by-k' voting. k=0 means greedy/no voting beyond first valid response, k=1 means simple majority."),
            red_flag_config: dict[str, Any] | None = Field(None, description="Optional: overrides default red-flagging configuration."),
            output_parser_schema: dict[str, Any] | None = Field(None, description="JSON schema or other structured definition for parsing LLM output to extract a canonical response. If absent, output is treated as plain text."),
            fast_path_enabled: bool = Field(False, description="If true, short-circuits voting if a 'sufficiently confident' response is found early (e.g., first valid response if k=0)."),
            client_request_id: str | None = Field(None, description="Client-defined ID for the overall request/task this MDAP execution is part of."),
            client_sub_step_id: str | None = Field(None, description="Client-defined ID for a specific sub-step within the request/task."),
        ) -> dict[str, Any]:
            """Execute LLM role with MDAP reliability guarantees."""

            # Convert dict configs to proper models
            ensemble_config_obj = None
            if ensemble_config:
                try:
                    ensemble_config_obj = EnsembleConfig.model_validate(ensemble_config)
                except Exception as e:
                    raise ValueError(f"Invalid ensemble_config: {e}") from e

            red_flag_config_obj = None
            if red_flag_config:
                try:
                    red_flag_config_obj = RedFlagConfig.model_validate(red_flag_config)
                except Exception as e:
                    raise ValueError(f"Invalid red_flag_config: {e}") from e

            # Create MDAP input
            mdap_input = MDAPInput(
                prompt=prompt,
                role_name=role_name,
                ensemble_config=ensemble_config_obj,
                voting_k=voting_k,
                red_flag_config=red_flag_config_obj,
                output_parser_schema=output_parser_schema,
                fast_path_enabled=fast_path_enabled,
                client_request_id=client_request_id,
                client_sub_step_id=client_sub_step_id,
            )

            # Execute MDAP - the mdap_engine.execute_llm_role is already async
            result = await self.mdap_engine.execute_llm_role(mdap_input)

            # Convert result to dict for JSON serialization
            return result.model_dump()
