# Ensample: Massively Decomposed Agentic Processes (MDAPs) Execution Engine Specification

## 1. Introduction & Purpose

`Ensample` is a specialized Model Context Protocol (MCP) server designed to provide **highly reliable LLM-driven responses**. It centralizes and operationalizes the principles of **Massively Decomposed Agentic Processes (MDAPs)**, as outlined in "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025).

Clients of `Ensample` (which could be any agentic system, orchestration layer, or application) can delegate complex LLM interactions, ensemble sampling, voting, and red-flagging to this server. This allows client systems to abstract away the intricacies of achieving high-assurance LLM outputs and focus on their core business logic.

**Key Objectives:**
*   **Decouple LLM Reliability from Client Logic:** Abstract away the complexity of MDAP execution, allowing client systems to focus on their primary task workflows.
*   **Achieve Near-Zero-Error LLM Outputs:** Systematically reduce the effective error rate of individual LLM decisions through ensemble voting and stringent quality checks.
*   **Enhance Scalability:** Allow for independent scaling of LLM inference workers and parallel execution of ensemble calls.
*   **Provide Configurable Reliability:** Offer dynamic control over MDAP parameters (e.g., voting threshold, ensemble composition, red-flag rules) to adapt to task-specific reliability and cost requirements.
*   **Ensure Observability:** Provide detailed telemetry into the MDAP execution process, including LLM calls, voting rounds, and error detection.

## 2. Architecture Overview (FastMCP Server)

`Ensample` is implemented as a **FastMCP server** designed for high-performance, asynchronous MDAP execution.

```mermaid
graph TD
    subgraph MDAPFlowMCP_Server [Ensample Server]
        direction TB
        MCPServerCore[MCP Server Core - FastMCP]
        ToolExecutor[Tool Executor - `execute_llm_role` Logic]

        subgraph MDAP_Engine [MDAP Engine]
            direction LR
            EnsembleManager[Ensemble Manager]
            LLMProviderInterface[LLM Provider Interface (OpenAI, Anthropic, etc.)]
            RedFlaggingEngine[Red-Flagging Engine]
            OutputParser[Output Parser]
            VotingMechanism[Voting Mechanism (First-to-ahead-by-k)]
            FastPathController[Fast-Path Controller]
        end

        MCPServerCore --> ToolExecutor
        ToolExecutor --> MDAP_Engine
        MDAP_Engine --> LLMProviderInterface
    end
    Client[Client System] -->|MCP Protocol - `mdapflow.execute_llm_role`| MCPServerCore
    LLMProviderInterface -.->|External LLM APIs| LLMs[Diverse LLM Models]
```

### 2.1 Architectural Principles

1.  **MDAP-First Reliability:** Every LLM decision processed by `Ensample` is subjected to MDAP principles unless explicitly overridden by a fast-path directive.
2.  **Configurable Ensemble:** Supports dynamic configuration of LLM ensembles (mix of models, temperatures, etc.) to optimize for cost, performance, and error decorrelation.
3.  **Parallel Execution:** Designed to execute multiple LLM calls for ensemble voting in parallel to minimize wall-clock time.
4.  **Proactive Error Mitigation:** Integrates a Red-Flagging Engine to discard unreliable LLM outputs early, improving the efficiency and accuracy of voting.
5.  **Dynamic Adaptability:** Allows client systems to dynamically adjust MDAP parameters (e.g., `voting_k`, `fast_path_enabled`) based on task complexity or user-defined policies.
6.  **Transparency:** Provides rich metadata and metrics on MDAP execution to the client and for observability.

## 3. Common Data Types (Pydantic Models)

These are the core schemas used in tool inputs and outputs for `Ensample`.

```python
from pydantic import BaseModel, Field, conlist, NonNegativeInt, HttpUrl, confloat
from typing import List, Literal, Optional, Dict, Any, Union

# --- LLM Configuration Models ---
class LLMConfig(BaseModel):
    """Configuration for a single LLM model in the ensemble."""
    provider: str = Field(..., description="Name of the LLM provider (e.g., 'openai', 'anthropic', 'openrouter').")
    model: str = Field(..., description="Specific model name (e.g., 'gpt-4o', 'claude-3-opus-20240229', 'x-ai/grok-1.0').")
    api_key_env_var: Optional[str] = Field(None, description="Environment variable name for the API key. If not provided, will look for a default for the provider.")
    base_url: Optional[HttpUrl] = Field(None, description="Custom base URL for the LLM API endpoint.")
    temperature: confloat(ge=0.0, le=2.0) = Field(0.1, description="Sampling temperature for text generation.")
    top_p: confloat(ge=0.0, le=1.0) = Field(1.0, description="Cumulative probability for nucleus sampling.")
    max_tokens: Optional[NonNegativeInt] = Field(None, description="Maximum number of tokens to generate.")
    stop_sequences: Optional[List[str]] = Field(None, description="Sequences at which to stop generation.")
    # Add any other provider-specific parameters as a generic dict
    extra_params: Optional[Dict[str, Any]] = Field(None, description="Provider-specific parameters not covered by standard fields.")

class EnsembleConfig(BaseModel):
    """Configuration for the ensemble of LLM models used in MDAP."""
    models: conlist(LLMConfig, min_length=1) = Field(..., description="List of LLM configurations to be used in the ensemble.")

# --- Red-Flagging Models ---
class RedFlagRule(BaseModel):
    """A single rule for identifying problematic LLM outputs."""
    type: Literal["regex", "keyword", "json_parse_error", "length_exceeds"] = Field(..., description="Type of red-flag rule.")
    value: Optional[str] = Field(None, description="Regex pattern, keyword, or length threshold (as string 'N_TOKENS').")
    message: str = Field(..., description="Message to log when this red flag is hit.")

class RedFlagConfig(BaseModel):
    """Configuration for applying red-flag rules."""
    rules: List[RedFlagRule] = Field([], description="List of red-flag rules to apply to LLM outputs.")
    enabled: bool = Field(True, description="Whether red-flagging is enabled.")

# --- MDAP Execution Input ---
class MDAPInput(BaseModel):
    """Input for the `execute_llm_role` tool."""
    prompt: str = Field(..., description="The natural language prompt for the LLM role or function.")
    role_name: str = Field(..., description="A descriptive identifier for the LLM role or function being performed by the client (e.g., 'DocumentSummarizer', 'CodeGenerator', 'IntentClassifier'). Used for logging and metrics.")
    # MDAP-specific parameters
    ensemble_config: Optional[EnsembleConfig] = Field(None, description="Optional: overrides default ensemble configuration.")
    voting_k: NonNegativeInt = Field(3, description="The 'k' value for 'first-to-ahead-by-k' voting. k=0 means greedy/no voting beyond first valid response, k=1 means simple majority.")
    red_flag_config: Optional[RedFlagConfig] = Field(None, description="Optional: overrides default red-flagging configuration.")
    output_parser_schema: Optional[Dict[str, Any]] = Field(None, description="JSON schema or other structured definition for parsing LLM output to extract a canonical response. If absent, output is treated as plain text.")
    fast_path_enabled: bool = Field(False, description="If true, short-circuits voting if a 'sufficiently confident' response is found early (e.g., first valid response if k=0).")
    # Contextual fields for logging/tracing specific to the client's workflow
    client_request_id: Optional[str] = Field(None, description="Client-defined ID for the overall request/task this MDAP execution is part of.")
    client_sub_step_id: Optional[str] = Field(None, description="Client-defined ID for a specific sub-step within the request/task.")

# --- MDAP Execution Output ---
class MDAPMetrics(BaseModel):
    """Metrics collected during MDAP execution."""
    total_llm_calls: NonNegativeInt = Field(..., description="Total number of LLM API calls made.")
    voting_rounds: NonNegativeInt = Field(..., description="Number of voting rounds completed (including initial samples).")
    red_flags_hit: Dict[str, NonNegativeInt] = Field({}, description="Count of each type of red flag hit.")
    valid_responses_per_round: List[NonNegativeInt] = Field([], description="Number of valid responses collected in each voting round.")
    winning_response_votes: NonNegativeInt = Field(..., description="Number of votes for the winning response.")
    time_taken_ms: NonNegativeInt = Field(..., description="Total time taken for MDAP execution in milliseconds.")
    estimated_llm_cost_usd: confloat(ge=0.0) = Field(0.0, description="Estimated total LLM API cost for this MDAP execution.")

class MDAPOutput(BaseModel):
    """Output from the `execute_llm_role` tool."""
    final_response: str = Field(..., description="The chosen, high-confidence LLM output after MDAP processing.")
    confidence_score: confloat(ge=0.0, le=1.0) = Field(..., description="A measure of confidence (e.g., winning votes / total valid votes in final round, or 1.0 for fast-path).")
    mdap_metrics: MDAPMetrics = Field(..., description="Detailed metrics about the MDAP execution.")
    error_message: Optional[str] = Field(None, description="Error message if MDAP execution failed or timed out.")

```

## 4. Server Tools (MCP Protocol)

`Ensample` exposes the following tools to its clients.

### 4.1 `mdapflow.execute_llm_role`
The primary tool for invoking LLM roles with MDAP reliability.

**Parameters (Input: `MDAPInput`):**
*   `prompt` (string): The natural language prompt for the LLM role or function.
*   `role_name` (string): A descriptive identifier for the LLM role or function.
*   `ensemble_config` (optional `EnsembleConfig`): Overrides default ensemble.
*   `voting_k` (optional `NonNegativeInt`): `k` value for voting.
*   `red_flag_config` (optional `RedFlagConfig`): Overrides default red-flagging.
*   `output_parser_schema` (optional `Dict[str, Any]`): For structured output parsing.
*   `fast_path_enabled` (bool): Enables short-circuiting voting.
*   `client_request_id` (optional string): Client-defined ID for the overall request.
*   `client_sub_step_id` (optional string): Client-defined ID for a specific sub-step.

**Returns (Output: `MDAPOutput`):**
*   `final_response` (string): The chosen LLM output.
*   `confidence_score` (float): Confidence measure.
*   `mdap_metrics` (`MDAPMetrics`): Detailed execution metrics.
*   `error_message` (optional string): If an error occurred.

### 4.2 `mdapflow.ping`
Standard MCP health check tool.

**Parameters (Input: None):**
**Returns (Output: `Dict[str, Any]`):**
*   `status` (string): "ok"
*   `message` (string): Human-readable status.
*   `uptime` (string): Server uptime.
*   `mdap_config_loaded` (bool): Whether default MDAP configuration was loaded successfully.

## 5. Internal Logic & Workflow

`Ensample` orchestrates a sophisticated internal workflow to achieve reliable LLM outputs.

### 5.1 LLM Provider Interface
*   **Purpose:** Manages connections and API calls to diverse external LLM providers (e.g., OpenAI, Anthropic, Together.ai, custom endpoints).
*   **Implementation:** An abstraction layer that handles authentication (via environment variables like `OPENAI_API_KEY`), dynamic selection based on `LLMConfig`, rate limiting, retries, and formatting requests/responses to/from `LLMConfig` parameters.

### 5.2 Ensemble Management
*   **Purpose:** Configures and dispatches LLM calls across a heterogeneous ensemble of models.
*   **Implementation:** Takes an `EnsembleConfig` (either default or provided in `MDAPInput`). For each voting round, it selects and dispatches parallel calls to LLMs from the ensemble to generate new samples. It prioritizes using different models for initial samples to maximize error decorrelation.

### 5.3 Red-Flagging Engine
*   **Purpose:** Filters out unreliable or malformed LLM outputs *before* they enter the voting pool, increasing the efficiency and accuracy of voting.
*   **Implementation:** Applies `RedFlagConfig` rules to each raw LLM response.
    *   **`json_parse_error`:** Attempts to parse output against `output_parser_schema` (if provided). Flags if parsing fails.
    *   **`regex`:** Checks if the output matches a predefined regex pattern (e.g., for refusal phrases, unwanted content).
    *   **`keyword`:** Checks for presence of specific keywords.
    *   **`length_exceeds`:** Checks if output token count exceeds a threshold (e.g., "N_TOKENS" from rule value).
*   **Action:** Responses hitting a red flag are discarded, and new samples are requested to replace them in the current voting round.

### 5.4 Output Parser
*   **Purpose:** Canonicalizes the structured output of LLMs into a consistent format for voting comparison.
*   **Implementation:** If `output_parser_schema` is provided in `MDAPInput`, attempts to parse the LLM's raw string response into a structured object (e.g., a Pydantic model, a JSON object). If successful, this structured representation is used for voting. If `output_parser_schema` is absent, the raw string is used. This parser can also handle potential "repairing" logic (e.g., fixing minor JSON syntax errors) if configured, to increase the effective valid response rate.

### 5.5 Voting Mechanism ("First-to-ahead-by-`k`")
*   **Purpose:** Statistically amplifies the probability of selecting a correct LLM response from multiple samples, forming the core of MDAP reliability.
*   **Implementation:**
    *   **Parallel Sampling:** In each round, dispatches parallel calls to LLMs (from the ensemble) to generate multiple responses.
    *   **Validation:** Each raw response is subjected to Red-Flagging and Output Parsing. Only valid, parsed responses are counted as votes.
    *   **Vote Counting:** Tracks votes for each unique valid response.
    *   **Convergence:** Continues sampling in rounds until one response has accumulated `k` more votes than any other alternative.
    *   **Tie-breaking:** If `k=0` (greedy) or `k=1` (simple majority) and a tie occurs at the end of a round, a tie-breaking rule is applied (e.g., oldest winning response, random choice, shortest response).

### 5.6 Fast-Path Controller
*   **Purpose:** Allows dynamic adjustment of MDAP rigor based on the `fast_path_enabled` flag, enabling cost-performance optimization for lower-risk scenarios.
*   **Implementation:**
    *   If `fast_path_enabled` is true:
        *   For `voting_k=0`: Returns the *first valid, non-red-flagged response* (after parsing) immediately, effectively acting as a single, validated LLM call.
        *   For `voting_k=1`: Returns the first response that reaches 1 vote and has no other contenders yet, or the highest voted response after a single round if no single winner.
        *   For `voting_k > 1`: Might dynamically lower `k` or reduce ensemble size if initial responses show very strong consensus. This logic can be configurable.
    *   The `confidence_score` for fast-path results is typically `1.0` or near `1.0` to reflect an early strong signal, despite fewer samples.

## 6. Configuration and Observability

`Ensample` is highly configurable via environment variables and extensively instrumented for observability.

### 6.1 Configuration

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MDAP_DEFAULT_ENSEMBLE_CONFIG_PATH` | Path to a JSON file defining the default `EnsembleConfig`. | `None` |
| `MDAP_DEFAULT_RED_FLAG_CONFIG_PATH` | Path to a JSON file defining the default `RedFlagConfig`. | `None` |
| `MDAP_DEFAULT_VOTING_K` | Default `k` value if not specified in `MDAPInput`. | `3` |
| `MDAP_MAX_CONCURRENT_LLM_CALLS` | Maximum number of parallel LLM calls `Ensample` will make. | `10` |
| `MDAP_MAX_VOTING_ROUNDS` | Maximum voting rounds before MDAP times out and returns an error. | `20` |
| `MDAP_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, etc.). | `INFO` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | Enables **OpenTelemetry** distributed tracing. | `unset` |
| `LLM_PROVIDER_OPENAI_API_KEY` | Environment variable for OpenAI API key. | `OPENAI_API_KEY` |
| `LLM_PROVIDER_ANTHROPIC_API_KEY` | Environment variable for Anthropic API key. | `ANTHROPIC_API_KEY` |
| `LLM_PROVIDER_OPENROUTER_API_KEY` | Environment variable for OpenRouter API key. | `OPENROUTER_API_KEY` |
| `LLM_PROVIDER_CUSTOM_BASE_URL` | Base URL for custom LLM endpoints (e.g., for local models). | `None` |
| `LLM_PROVIDER_DEFAULT_MAX_TOKENS` | Default max tokens for LLM generation if not specified in `LLMConfig`. | `2048` |

### 6.2 Observability
*   **Distributed Tracing (OpenTelemetry):** Every `mdapflow.execute_llm_role` call triggers a new OTEL span. This span is instrumented to show:
    *   Sub-spans for each LLM API call (including chosen model, prompt length, response length, latency).
    *   Spans for Red-Flagging (indicating if a flag was hit and which rule).
    *   Spans for Output Parsing (indicating success/failure).
    *   Metadata (tags) for the `role_name`, `voting_k`, `fast_path_enabled`, `client_request_id`, `client_sub_step_id`, and round metrics.
*   **Structured Logging:** Detailed JSON logs provide step-by-step information on MDAP execution, including votes per candidate, red flags hit, and LLM call details.
*   **Metrics:** `MDAPMetrics` are returned with each output, providing granular performance data. The server also exposes Prometheus-compatible metrics for:
    *   `mdap_execution_total`: Total count of MDAP executions.
    *   `mdap_execution_success_total`: Total successful MDAP executions.
    *   `mdap_execution_failed_total`: Total failed MDAP executions.
    *   `mdap_llm_calls_total`: Total LLM API calls made.
    *   `mdap_red_flags_hit_total{rule_type}`: Counter for each type of red flag hit.
    *   `mdap_voting_rounds_histogram`: Histogram of voting rounds per execution.
    *   `mdap_execution_latency_ms_histogram`: Histogram of end-to-end MDAP execution latency.
    *   `mdap_estimated_cost_usd_total`: Cumulative estimated LLM API cost.

## 7. Future Considerations

*   **Adaptive `k` Values:** Implement dynamic adjustment of `k` based on real-time LLM confidence scores, or historical per-role performance, informed by external client policy.
*   **Semantic Validation:** Introduce a lightweight LLM or rule-based system for basic semantic validation of outputs, beyond just format (e.g., does the proposed code snippet actually make sense in context?), potentially as a configurable Red-Flag rule.
*   **Fine-Grained Cost Control:** Allow for explicit cost caps per `mdapflow.execute_llm_role` call, with dynamic adjustments to `k` or ensemble selection to stay within budget.
*   **Complex Output Structures:** Support for MDAP voting on multi-part outputs (e.g., code + explanation + verification steps) by decomposing voting to sub-components.
*   **Client-Configurable Error Handling:** Allow clients to register custom callback functions (or webhooks) for specific MDAP failure modes (e.g., non-convergence after max rounds), enabling external human-in-the-loop review for difficult LLM decisions.
