# Developer Agent Guide: Ensample

Ensample is a specialized high-reliability LLM orchestration server. Its purpose is to transform probabilistic LLM outputs into deterministic, high-confidence results using ensemble methods.

## Core Component Map

- **`MDAPEngine`**: The central orchestrator. It manages the execution lifecycle of a single tool request.
- **`EnsembleManager`**: Dispatches parallel API calls across distinct models via LiteLLM.
- **`VotingMechanism`**: Implements "first-to-ahead-by-k" logic. It tracks consensus across rounds until a winner emerges.
- **`RedFlaggingEngine`**: Performs pre-voting quality checks (regex, keyword filters, length, and JSON validation).
- **`PricingService`**: Provides real-time and cached API cost calculations, predominantly via OpenRouter.
- **`FastPathController`**: Optimizes for latency by enabling early exit for high-confidence or greedy (k=0) scenarios.

## Tooling Interface

- **`ensample.execute_llm_role`**: The primary entry point. It requires a `prompt` and a `role_name`, and returns a verified `final_response` along with precision metrics.
- **`ensample.ping` / `ensample.server_info`**: Standard MCP maintenance tools for health and versioning.

## Development Priorities

1.  **Reliability Above All**: Every change must ensure that the "consensus" logic remains robust.
2.  **Observability**: Maintain deep instrumentation. Every LLM call and red-flag event should be visible via OpenTelemetry and Prometheus.
3.  **Agnosticism**: Keep the provider interface generic. We use LiteLLM to ensure we are never locked into a single provider.
4.  **Performance**: Optimize for concurrency. Ensure `asyncio` is used effectively to keep ensemble latency minimal.

## Project Guidelines

- **Package Management**: Use `uv`.
- **Typing**: Strict type hints are mandatory.
- **Testing**: Maintain >70% coverage. All core reliability logic (voting, parsing) must have unit tests.
