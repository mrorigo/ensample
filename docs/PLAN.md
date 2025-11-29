# Ensample: Detailed Implementation Plan

## 1. Introduction & Project Goals

This document provides a detailed, phased implementation plan for `Ensample`, a standalone Model Context Protocol (MCP) server dedicated to executing Massively Decomposed Agentic Processes (MDAPs) for highly reliable LLM-driven responses. This plan translates the `Ensample` specification into actionable development phases, ensuring a robust, scalable, and observable implementation.

**Key Objectives:**
*   Implement all core components of the `Ensample` specification, including LLM provider integration, ensemble management, red-flagging, output parsing, the "first-to-ahead-by-k" voting mechanism, and the fast-path controller.
*   Ensure the server is performant, scalable, and resilient for production use by any client system requiring high-assurance LLM outputs.
*   Integrate comprehensive observability (OpenTelemetry tracing, structured logging, Prometheus metrics) to provide deep insights into MDAP execution.
*   Deliver a fully testable and containerizable service.

## 2. Core Principles & Architecture

The implementation will strictly adhere to the architectural principles defined in the `Ensample` specification, focusing on modularity, asynchronous processing, and robust error handling at every layer. The architecture shown in the spec guides the component breakdown.

## 3. High-Level Phased Implementation Plan

Development will proceed in six focused phases to manage complexity, ensure a logical build order, and facilitate incremental testing and delivery.

*   **Phase 1: Foundation & MCP Core.**
    *   **Goal:** Establish the MCP server scaffold, core data models, and basic infrastructure.
    *   **Deliverables:** A running `FastMCP` server, all Pydantic models, configuration system, and the `ping` tool.

*   **Phase 2: LLM Provider Integration & Ensemble Management.**
    *   **Goal:** Enable communication with external LLM providers and manage a diverse ensemble of models.
    *   **Deliverables:** An `LLMProviderInterface` supporting multiple providers, `EnsembleManager` for parallel dispatch, and initial LLM call functionality.

*   **Phase 3: Output Validation & Parsing.**
    *   **Goal:** Implement mechanisms to filter unreliable LLM outputs and standardize valid ones.
    *   **Deliverables:** `RedFlaggingEngine` (regex, keyword, length, JSON error checks), `OutputParser` (JSON schema validation), and associated `RedFlagConfig` and `output_parser_schema` handling.

*   **Phase 4: Core MDAP: Voting Mechanism.**
    *   **Goal:** Build the central "first-to-ahead-by-k" voting logic that drives MDAP reliability.
    *   **Deliverables:** `VotingMechanism` with parallel sampling, vote counting, convergence detection, and tie-breaking rules.

*   **Phase 5: MDAP Orchestration & Advanced Control.**
    *   **Goal:** Integrate all MDAP components into the main `execute_llm_role` tool, including the fast-path logic.
    *   **Deliverables:** Fully functional `mdapflow.execute_llm_role` tool, `FastPathController` implementation, and robust error handling for MDAP execution.

*   **Phase 6: Production Hardening & Observability.**
    *   **Goal:** Prepare the server for production deployment with comprehensive monitoring and security.
    *   **Deliverables:** OpenTelemetry tracing, Prometheus metrics, structured logging, `Dockerfile`, and performance optimizations.

## 4. Detailed Design & Implementation Plan

### Phase 1: Foundation & MCP Core

1.  **Project Scaffolding:**
    *   Initialize a Python project with `uv` for dependency management.
    *   Establish project structure: `Ensample/`, `src/mdapflow_mcp/`, `tests/`, `pyproject.toml`.
    *   Create base modules: `observability.py` (for structured logging setup), `exceptions.py` (custom exceptions), `config.py` (Pydantic `BaseSettings` for env vars).

2.  **MCP Server Core (`src/mdapflow_mcp/server.py`):**
    *   **Technology:** `mcp-sdk` (`FastMCP`), `asyncio`.
    *   **Implementation:**
        *   Instantiate `FastMCP` with a `lifespan` manager for graceful startup/shutdown.
        *   Implement centralized exception handling to catch `MDAPFlowError` and return structured MCP errors.

3.  **Pydantic Models (`src/mdapflow_mcp/models.py`):**
    *   **Technology:** `pydantic`.
    *   **Implementation:** Transcribe all data types from the spec: `LLMConfig`, `EnsembleConfig`, `RedFlagRule`, `RedFlagConfig`, `MDAPInput`, `MDAPOutput`, `MDAPMetrics`. Ensure proper validation for `conlist`, `NonNegativeInt`, `HttpUrl`, `confloat`.

4.  **Configuration Management (`src/mdapflow_mcp/config.py`):**
    *   Implement loading of all `MDAP_` and `LLM_PROVIDER_` environment variables into a `Settings` object.
    *   Implement logic to load default `EnsembleConfig` and `RedFlagConfig` from JSON files if paths are specified via environment variables.

5.  **Tool: `mdapflow.ping` (`src/mdapflow_mcp/tools/maintenance.py`):**
    *   Implement the `ping` tool, returning basic server status and confirmation of default MDAP config loading.

### Phase 2: LLM Provider Integration & Ensemble Management

1.  **LLM Provider Interface (`src/mdapflow_mcp/llm_provider.py`):**
    *   **Technology:** `httpx` (for async HTTP requests), `openai`, `anthropic` client libraries, custom API wrappers.
    *   **Implementation:**
        *   Define an abstract `BaseLLMClient` with a `generate(prompt, config)` method.
        *   Implement concrete `OpenAIClient`, `AnthropicClient`, `OpenRouterClient`, and `CustomAPIClient` classes based on `LLMConfig`.
        *   Handle API key retrieval (from env vars), rate limiting, exponential backoff retries, and basic error mapping.
        *   Implement a factory function to get the correct LLM client instance based on `LLMConfig.provider`.
        *   Add a cost estimation function for each provider/model.

2.  **Ensemble Manager (`src/mdapflow_mcp/ensemble_manager.py`):**
    *   **Technology:** `asyncio`, `LLMProviderInterface`.
    *   **Implementation:**
        *   `dispatch_ensemble_calls(prompt, ensemble_config, num_calls_per_model)`: Dispatches `num_calls_per_model` parallel LLM calls for each model in the ensemble using `asyncio.gather`.
        *   Initial logic for selecting models to call in a round, aiming for diversity.
        *   Aggregate estimated costs from individual LLM calls.

### Phase 3: Output Validation & Parsing

1.  **Red-Flagging Engine (`src/mdapflow_mcp/red_flagging_engine.py`):**
    *   **Technology:** `re` (Python regex module), `jsonschema` (for JSON validation).
    *   **Implementation:**
        *   `apply_rules(response_text, red_flag_config, output_parser_schema)`: Iterates through `RedFlagRule`s.
        *   **`regex` & `keyword`:** Simple string/regex matching.
        *   **`length_exceeds`:** Tokenizes response (e.g., simple whitespace tokenization or a lightweight tokenizer) and checks length.
        *   **`json_parse_error`:** Attempts to parse `response_text` as JSON and then validate against `output_parser_schema` if provided. If parsing/validation fails, it flags the response.
        *   Returns a list of hit red flags or `None` if no flags are hit.

2.  **Output Parser (`src/mdapflow_mcp/output_parser.py`):**
    *   **Technology:** `json`, `jsonschema` (optional for validation).
    *   **Implementation:**
        *   `parse_output(response_text, output_parser_schema)`: Attempts to parse `response_text` into a canonical format.
        *   If `output_parser_schema` is present: Attempts `json.loads` and then `jsonschema.validate`. Returns the parsed object or raises a `ParsingError`.
        *   If `output_parser_schema` is absent: Returns `response_text` as is.
        *   Consider an optional "repairing parser" for minor JSON syntax issues if robustness is desired.

### Phase 4: Core MDAP: Voting Mechanism

1.  **Voting Mechanism (`src/mdapflow_mcp/voting_mechanism.py`):**
    *   **Technology:** `asyncio`, `collections.Counter`.
    *   **Implementation:**
        *   `run_voting(prompt, ensemble_manager, red_flag_engine, output_parser, mdap_input)`: This is the core loop.
        *   **Looping Rounds:** In each round, dispatches a configured number of parallel LLM calls (via `EnsembleManager`).
        *   **Validation & Parsing:** Each raw LLM response is processed by `RedFlaggingEngine` and `OutputParser`. Invalid/red-flagged responses are discarded.
        *   **Vote Counting:** Stores valid, canonicalized responses and their vote counts (using `collections.Counter`).
        *   **Convergence Check:** After each round, checks if any response has reached `k` votes ahead of all others.
        *   **Tie-breaking:** Implements a configurable tie-breaking strategy if `voting_k` is low (e.g., shortest response, first-seen).
        *   **Metrics Collection:** Gathers `MDAPMetrics` during execution (total calls, rounds, red flags, etc.).

### Phase 5: MDAP Orchestration & Advanced Control

1.  **MDAP Execution Logic (`src/mdapflow_mcp/mdap_engine.py`):**
    *   **Technology:** Integrates components from Phases 2-4.
    *   **Implementation:** This module will contain the `execute_llm_role` function, which is the top-level orchestration of the MDAP process.
        *   Takes `MDAPInput` as input.
        *   Initializes `LLMProviderInterface`, `EnsembleManager`, `RedFlaggingEngine`, `OutputParser` based on default or provided configs.
        *   Handles overall timing and error reporting for the `MDAPOutput`.

2.  **Fast-Path Controller (`src/mdapflow_mcp/fast_path_controller.py`):**
    *   **Implementation:**
        *   `check_fast_path(mdap_input, current_votes, current_responses)`: This function is called within the `VotingMechanism`'s loop.
        *   If `mdap_input.fast_path_enabled` is true:
            *   **`voting_k=0`:** Immediately returns the first valid response received if any.
            *   **`voting_k=1`:** Returns the first response to get 1 vote if it has no competition yet.
            *   More advanced logic for higher `k`: Could trigger if a single response overwhelmingly dominates early in the voting (e.g., 80% of votes after 2 rounds).
        *   Returns the winning response and appropriate confidence score if fast-path convergence is met.

3.  **Tool: `mdapflow.execute_llm_role` (`src/mdapflow_mcp/tools/mdap_tools.py`):**
    *   Implement the MCP tool interface for `mdapflow.execute_llm_role`, calling the `mdap_engine.execute_llm_role` function.
    *   Ensure all input parameters are correctly mapped and output `MDAPOutput` is returned.

### Phase 6: Production Hardening & Observability

1.  **OpenTelemetry Integration (`src/mdapflow_mcp/observability.py`):**
    *   **Technology:** `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `opentelemetry-instrumentation-httpx`.
    *   **Implementation:**
        *   Configure OTEL `TracerProvider` at server startup (listening to `OTEL_EXPORTER_OTLP_ENDPOINT`).
        *   Use `trace.get_current_span()` to propagate `client_request_id` and `client_sub_step_id` as span attributes.
        *   Instrument all LLM API calls with detailed sub-spans (model, prompt/response length, latency, cost).
        *   Add custom spans for `RedFlaggingEngine` (indicating rules hit), `OutputParser`, `VotingMechanism` rounds.
        *   Add attributes to spans for `role_name`, `voting_k`, `fast_path_enabled`.

2.  **Structured Logging (`src/mdapflow_mcp/observability.py`):**
    *   **Technology:** `python-json-logger`.
    *   **Implementation:** Configure standard Python logging to output JSON format. Ensure all logs include `trace_id`, `span_id`, `role_name`, `client_request_id`, `client_sub_step_id` where applicable. Log MDAP execution details (votes, red flags) at `DEBUG` or `INFO` levels.

3.  **Prometheus Metrics (`src/mdapflow_mcp/metrics.py`):**
    *   **Technology:** `prometheus_client`.
    *   **Implementation:**
        *   Register standard Prometheus metrics: `Counter` for total/success/failed executions, `Histogram` for latency and voting rounds, `Gauge` for current LLM provider health.
        *   Expose a `/metrics` endpoint.
        *   Update metrics within `mdap_engine.execute_llm_role` and `voting_mechanism.run_voting`.

4.  **Performance Optimization:**
    *   **Implementation:** Review and ensure all blocking I/O (e.g., `httpx` for LLM calls, `re` for complex regex if it can block) is done asynchronously or wrapped in `asyncio.to_thread` to prevent event loop blocking. Profile critical sections to identify bottlenecks.

5.  **Containerization & Security:**
    *   **Implementation:** Create a `Dockerfile` for production deployment. Ensure the image runs as a non-root user, adheres to minimal privileges, and clearly defines required environment variables. Implement any necessary input sanitization to prevent prompt injection or abuse.

## 5. Technology Stack Summary

| Component | Technology / Library | Rationale |
| :--- | :--- | :--- |
| **MCP Server** | `mcp-sdk` (`FastMCP`) | High-performance, `asyncio`-native, standardized MCP framework. |
| **Data Validation** | `pydantic` | Strict schema enforcement, crucial for reliable inter-service communication. |
| **Asynchronous HTTP** | `httpx` | Robust, modern HTTP client for non-blocking API calls. |
| **LLM Clients** | `openai`, `anthropic`, custom wrappers | Direct interaction with LLM providers. |
| **Regular Expressions** | `re` | Core Python module for pattern matching. |
| **JSON Schema Validation** | `jsonschema` | Programmatic validation of structured JSON outputs. |
| **Concurrency** | `asyncio` | Native Python framework for efficient concurrent operations. |
| **Observability** | `OpenTelemetry SDK`, `python-json-logger`, `prometheus_client` | Vendor-neutral standards for tracing, logging, and metrics. |
| **Containerization** | `Docker` | Standard for packaging and deploying services. |

## 6. Testing Strategy

*   **Unit Tests:** Each module (e.g., `llm_provider.py`, `red_flagging_engine.py`, `output_parser.py`, `voting_mechanism.py`) will be thoroughly tested in isolation with mocked dependencies (e.g., mocked LLM responses, mocked API calls). This includes edge cases for voting, various red-flag scenarios, and parser failures.
*   **Integration Tests:** Each MCP tool will be tested end-to-end. For example, `mdapflow.execute_llm_role` will be tested with a full flow from prompt input, through ensemble calls, red-flagging, voting, and fast-path scenarios. This will involve using a local test `FastMCP` client to interact with the server.
*   **Performance Tests:** Load testing will be conducted to assess `Ensample`'s performance under various loads, concurrent requests, and different MDAP configurations (e.g., varying `voting_k`, ensemble sizes).
*   **Observability Validation:** Tests will verify that OpenTelemetry traces and Prometheus metrics are correctly emitted and contain the expected data and attributes for various execution paths, including successful runs, red-flag hits, and failures.
