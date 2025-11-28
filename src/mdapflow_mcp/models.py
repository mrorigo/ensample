"""Pydantic models that mirror the MDAPFlow-MCP specification."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# --- LLM Configuration Models ---
class LLMConfig(BaseModel):
    """Configuration for a single LLM model in the ensemble."""
    provider: str = Field(..., description="Name of the LLM provider (e.g., 'openai', 'anthropic', 'openrouter').")
    model: str = Field(..., description="Specific model name (e.g., 'gpt-4o', 'claude-3-opus-20240229', 'x-ai/grok-1.0').")
    api_key_env_var: str | None = Field(None, description="Environment variable name for the API key. If not provided, will look for a default for the provider.")
    base_url: str | None = Field(None, description="Custom base URL for the LLM API endpoint.")
    temperature: float = Field(0.1, ge=0.0, le=2.0, description="Sampling temperature for text generation.")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Cumulative probability for nucleus sampling.")
    max_tokens: int | None = Field(None, ge=0, description="Maximum number of tokens to generate.")
    stop_sequences: list[str] | None = Field(None, description="Sequences at which to stop generation.")
    # Add any other provider-specific parameters as a generic dict
    extra_params: dict[str, Any] | None = Field(None, description="Provider-specific parameters not covered by standard fields.")


class EnsembleConfig(BaseModel):
    """Configuration for the ensemble of LLM models used in MDAP."""
    models: list[LLMConfig] = Field(..., min_length=1, description="List of LLM configurations to be used in the ensemble.")


# --- Red-Flagging Models ---
class RedFlagRule(BaseModel):
    """A single rule for identifying problematic LLM outputs."""
    type: Literal["regex", "keyword", "json_parse_error", "length_exceeds"] = Field(..., description="Type of red-flag rule.")
    value: str | None = Field(None, description="Regex pattern, keyword, or length threshold (as string 'N_TOKENS').")
    message: str = Field(..., description="Message to log when this red flag is hit.")


class RedFlagConfig(BaseModel):
    """Configuration for applying red-flag rules."""
    rules: list[RedFlagRule] = Field(default_factory=list, description="List of red-flag rules to apply to LLM outputs.")
    enabled: bool = Field(True, description="Whether red-flagging is enabled.")


# --- MDAP Execution Input ---
class MDAPInput(BaseModel):
    """Input for the `execute_llm_role` tool."""
    prompt: str = Field(..., description="The natural language prompt for the LLM role or function.")
    role_name: str = Field(..., description="A descriptive identifier for the LLM role or function being performed by the client (e.g., 'DocumentSummarizer', 'CodeGenerator', 'IntentClassifier'). Used for logging and metrics.")
    # MDAP-specific parameters
    ensemble_config: EnsembleConfig | None = Field(None, description="Optional: overrides default ensemble configuration.")
    voting_k: int = Field(3, ge=0, description="The 'k' value for 'first-to-ahead-by-k' voting. k=0 means greedy/no voting beyond first valid response, k=1 means simple majority.")
    red_flag_config: RedFlagConfig | None = Field(None, description="Optional: overrides default red-flagging configuration.")
    output_parser_schema: dict[str, Any] | None = Field(None, description="JSON schema or other structured definition for parsing LLM output to extract a canonical response. If absent, output is treated as plain text.")
    fast_path_enabled: bool = Field(False, description="If true, short-circuits voting if a 'sufficiently confident' response is found early (e.g., first valid response if k=0).")
    # Contextual fields for logging/tracing specific to the client's workflow
    client_request_id: str | None = Field(None, description="Client-defined ID for the overall request/task this MDAP execution is part of.")
    client_sub_step_id: str | None = Field(None, description="Client-defined ID for a specific sub-step within the request/task.")


# --- MDAP Execution Output ---
class MDAPMetrics(BaseModel):
    """Metrics collected during MDAP execution."""
    total_llm_calls: int = Field(..., ge=0, description="Total number of LLM API calls made.")
    voting_rounds: int = Field(..., ge=0, description="Number of voting rounds completed (including initial samples).")
    red_flags_hit: dict[str, int] = Field(default_factory=dict, description="Count of each type of red flag hit.")
    valid_responses_per_round: list[int] = Field(default_factory=list, description="Number of valid responses collected in each voting round.")
    winning_response_votes: int = Field(..., ge=0, description="Number of votes for the winning response.")
    time_taken_ms: int = Field(..., ge=0, description="Total time taken for MDAP execution in milliseconds.")
    estimated_llm_cost_usd: float = Field(0.0, ge=0.0, description="Estimated total LLM API cost for this MDAP execution.")


class MDAPOutput(BaseModel):
    """Output from the `execute_llm_role` tool."""
    final_response: str = Field(..., description="The chosen, high-confidence LLM output after MDAP processing.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="A measure of confidence (e.g., winning votes / total valid votes in final round, or 1.0 for fast-path).")
    mdap_metrics: MDAPMetrics = Field(..., description="Detailed metrics about the MDAP execution.")
    error_message: str | None = Field(None, description="Error message if MDAP execution failed or timed out.")


# --- Additional utility models ---
class TokenUsage(BaseModel):
    """Detailed token usage information."""
    prompt_tokens: int | None = Field(None, ge=0, description="Number of tokens used for the prompt.")
    completion_tokens: int | None = Field(None, ge=0, description="Number of tokens used for the completion.")
    total_tokens: int | None = Field(None, ge=0, description="Total number of tokens used.")
    estimated: bool = Field(False, description="Whether these token counts are estimated rather than actual.")


class LLMResponse(BaseModel):
    """Represents a raw LLM response."""
    response: str
    llm_config: LLMConfig
    cost_estimate: float = 0.0
    latency_ms: int = 0
    tokens_used: TokenUsage | None = Field(None, description="Detailed token usage information.")


class ParsedResponse(BaseModel):
    """Represents a parsed and validated LLM response."""
    raw_response: LLMResponse
    parsed_content: str | dict[str, Any]
    red_flags_hit: list[str] = Field(default_factory=list)
    is_valid: bool = True
    parse_error: str | None = None
