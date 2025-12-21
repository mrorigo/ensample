# Ensample

**Ensemble-based LLM Orchestration with Massively Decomposed Agentic Processes (MDAP)**

Ensample is a high-performance Model Context Protocol (MCP) server designed to deliver **near-zero-error LLM outputs** through ensemble sampling, multi-model voting, and intelligent red-flagging. It implements a production-grade orchestration engine with advanced cost tracking, dynamic pricing, and comprehensive observability.

---

## 🌟 Key Features

- **Massively Decomposed Agentic Processes (MDAP)**: Implements a pipeline-based approach for high-reliability LLM execution.
- **Dynamic Pricing Service**: Real-time cost calculation using live pricing from OpenRouter API with smart caching and static fallbacks.
- **Ensemble Voting Mechanism**: Robust "first-to-ahead-by-k" algorithm to achieve consensus across diverse LLM providers.
- **Intelligent Red-Flagging**: Quality filtering using regex, keywords, length thresholds, and JSON schema validation.
- **Fast-Path Optimization**: Early termination for high-confidence queries to reduce latency and API costs.
- **Enhanced Observability**: Built-in Prometheus metrics and OpenTelemetry tracing for monitoring performance, costs, and quality.
- **Multi-Provider LiteLLM Integration**: Seamless support for OpenAI, Anthropic, Google, DeepSeek, Mistral, and many more.

---

## 🏗️ Architecture

Ensample orchestrates the flow from request to high-confidence response:

```text
Client Request
    ↓
MDAP Engine (Main Orchestrator)
    ↓
Ensemble Manager (Parallel Dispatch via LiteLLM)
    ↓
Red-Flagging Engine (Quality Filtering & Validation)
    ↓
Output Parser (Structured JSON Schema Enforcement)
    ↓
Voting Mechanism (First-to-Ahead-by-K Convergence)
    ↓
Dynamic Pricing Service (Real-time Cost Calculation)
    ↓
Final Result + Detailed Performance & Cost Metrics
```

---

## 💰 Dynamic Pricing & Cost Tracking

Ensample features a dedicated **Dynamic Pricing Service** that ensures accurate cost reporting across all providers.

### How it works:
1. **Real-time Retrieval**: When a request is made, Ensample can fetch live pricing from the [OpenRouter API](https://openrouter.ai/docs#models) for the requested models.
2. **Smart Caching**: Pricing information is cached with a configurable TTL (Default: 1 hour) to minimize API overhead.
3. **Static Fallbacks**: Includes a comprehensive built-in pricing table for major models (GPT-4o, Claude 3.5, Gemini 2.5, DeepSeek V3, etc.) used when dynamic fetching is unavailable.
4. **Token-Level Accuracy**: Costs are calculated based on actual `prompt_tokens` and `completion_tokens` returned by the provider.

### Supported Models for Dynamic Pricing:
- **OpenAI**: GPT-4o, GPT-4o-mini, GPT-3.5
- **Anthropic**: Claude 3.5 Sonnet, Opus, Haiku
- **Google**: Gemini 2.0/2.5 Pro/Flash
- **DeepSeek**: V3, R1
- **Meta/Mistral**: All models via OpenRouter or Together AI

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ensample

# Sync dependencies using uv
uv sync
```

### Environment Configuration

Set your provider API keys and optional configuration:

```bash
# Provider Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENROUTER_API_KEY="sk-or-..."

# Optional Settings
export MDAP_DEFAULT_VOTING_K=3
export MDAP_MAX_VOTING_ROUNDS=10
export MDAP_LOG_LEVEL=INFO
```

### Running the Server

```bash
# Start the STDIO server
uv run ensample

# Or run with HTTP/SSE transport
export MDAP_SERVER_TRANSPORT=sse
uv run ensample
```

---

## 🔧 Configuration

### MDAP Parameters (`MDAPInput`)

| Parameter              | Type   | Default    | Description                                                                        |
| ---------------------- | ------ | ---------- | ---------------------------------------------------------------------------------- |
| `prompt`               | `str`  | *Required* | The natural language prompt.                                                       |
| `role_name`            | `str`  | *Required* | Identifier for the task (used in metrics/logging).                                 |
| `voting_k`             | `int`  | `3`        | Convergence threshold. `k=3` means one response needs 3 more votes than any other. |
| `fast_path_enabled`    | `bool` | `False`    | Enables early exit for high-confidence/greedy results.                             |
| `ensemble_config`      | `dict` | `None`     | Custom list of models, temperatures, and parameters.                               |
| `red_flag_config`      | `dict` | `None`     | Custom validation rules (keywords, regex, length).                                 |
| `output_parser_schema` | `dict` | `None`     | JSON Schema for structured output validation.                                      |

---

## 📊 Observability

Ensample provides deep insights into your LLM operations:

### Prometheus Metrics
- `mdap_estimated_cost_usd_total`: Cumulative cost across all executions.
- `mdap_provider_cost_usd_total`: Cost breakdown per provider and model.
- `mdap_tokens_prompt_total` / `mdap_tokens_completion_total`: Granular token tracking.
- `mdap_red_flags_hit_total`: Analysis of quality rule violations.
- `mdap_execution_latency_ms_histogram`: Performance distribution.

### OpenTelemetry Tracing
- Detailed spans for every LLM call, including model version, token counts, and specific pricing sources used.
- Correlation of `client_request_id` across the entire MDAP pipeline.

---

## 🛠️ MCP Tools

### `ensample.execute_llm_role`
The primary tool for executing reliable LLM tasks.

**Example Usage:**
```json
{
  "prompt": "Summarize the latest financial report.",
  "role_name": "Summarizer",
  "voting_k": 2,
  "fast_path_enabled": true,
  "output_parser_schema": {
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
    }
  }
}
```

### `ensample.ping`
Health check returning uptime and configuration status.

### `ensample.server_info`
Detailed version and capability information.

---

## 🔒 Production Hardening

- **Security**: Run as non-root, use secure environment variables for API keys, and implement network-level access control.
- **Reliability**: Use horizontal scaling with multiple instances behind a load balancer for high availability.
- **Monitoring**: Integrate with Grafana dashboards using the provided Prometheus metrics and OpenTelemetry spans.
- **Budgeting**: Set `MDAP_MAX_VOTING_ROUNDS` and choose cost-effective ensembles to control costs.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Ensure all tests pass (`uv run pytest`).
4. Submit a Pull Request with detailed descriptions of changes.

---

## 🧪 Testing

Ensample maintains a rigorous test suite with **~73% coverage** across the board.

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/ensample --cov-report=term-missing
```

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Ensample** — *Bringing industrial-grade reliability to LLM orchestration.*