# Ensample

**Ensemble-based LLM Orchestration with Massively Decomposed Agentic Processes**

Ensample is a specialized Model Context Protocol (MCP) server designed to provide **highly reliable LLM-driven responses** by implementing ensemble sampling, voting, and red-flagging to achieve near-zero-error LLM outputs with comprehensive observability and production-grade reliability.

## 🌟 Key Features

- **High-Reliability LLM Outputs**: Uses ensemble voting and red-flagging to minimize errors
- **Configurable MDAP Parameters**: Control voting thresholds, ensemble composition, and quality checks
- **Fast-Path Optimization**: Early termination for high-confidence scenarios
- **Multi-Provider LLM Support**: Works with OpenAI, Anthropic, Google, Mistral, and custom providers via LiteLLM
- **Comprehensive Observability**: OpenTelemetry tracing, structured logging, and Prometheus metrics
- **Production-Ready**: Containerized deployment with health checks and monitoring
- **Multiple Transport Support**: STDIO, Server-Sent Events, and HTTP protocols
- **Test Coverage**: 73% test coverage with comprehensive unit and integration tests

## 🏗️ Architecture

Ensample implements a sophisticated MDAP pipeline using FastMCP:

```
Client Request
    ↓
FastMCP Server (FastMCP framework)
    ↓
MDAP Engine Orchestration
    ↓
Ensemble Manager (Parallel LLM Calls via LiteLLM)
    ↓
Red-Flagging Engine (Quality Filtering)
    ↓
Output Parser (Structured Validation with JSON Schema)
    ↓
Voting Mechanism (First-to-Ahead-by-K)
    ↓
Fast-Path Controller (Early Termination)
    ↓
High-Confidence Result + Comprehensive Metrics
```

### Core Components

- **MDAPEngine**: Main orchestration engine with configuration management
- **EnsembleManager**: Parallel LLM call dispatching with diversity-based model selection
- **VotingMechanism**: Implements "first-to-ahead-by-k" convergence algorithm
- **RedFlaggingEngine**: Quality filtering with keyword, regex, length, and JSON validation rules
- **OutputParser**: Structured output validation with JSON schema support and repair capabilities
- **FastPathController**: Early termination optimization for high-confidence scenarios
- **LLMProviderInterface**: Multi-provider LLM integration via LiteLLM
- **MetricsCollector**: Comprehensive Prometheus metrics with token and cost tracking

## 🚀 Quick Start

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd ensample
   uv sync
   ```

2. **Configure environment variables**:
   ```bash
   # LLM Provider API Keys
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export OPENROUTER_API_KEY="your-openrouter-key"
   
   # Optional: Custom configuration
   export MDAP_DEFAULT_VOTING_K=3
   export MDAP_MAX_CONCURRENT_LLM_CALLS=10
   export MDAP_MAX_VOTING_ROUNDS=20
   export MDAP_LOG_LEVEL=INFO
   export MDAP_SERVER_TRANSPORT=stdio
   ```

3. **Run the server**:
   ```bash
   uv run ensample
   ```

### Docker Deployment

```bash
docker build -t ensample .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  ensample
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MDAP_DEFAULT_VOTING_K` | Default voting threshold | `3` |
| `MDAP_MAX_CONCURRENT_LLM_CALLS` | Max parallel LLM calls | `10` |
| `MDAP_MAX_VOTING_ROUNDS` | Max voting rounds before timeout | `20` |
| `MDAP_LOG_LEVEL` | Logging verbosity | `INFO` |
| `MDAP_SERVER_TRANSPORT` | Transport protocol (`stdio`, `sse`, `http`) | `stdio` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OpenTelemetry endpoint | `unset` |
| `LLM_PROVIDER_*_API_KEY` | Provider-specific API keys | `required` |

### Default Configuration Files

Create JSON configuration files for custom ensemble and red-flag settings:

**ensemble-config.json**:
```json
{
  "models": [
    {
      "provider": "openai",
      "model": "gpt-4o-mini",
      "temperature": 0.1,
      "max_tokens": 2048
    },
    {
      "provider": "anthropic", 
      "model": "claude-3-haiku-20240307",
      "temperature": 0.1,
      "max_tokens": 2048
    }
  ]
}
```

**red-flag-config.json**:
```json
{
  "enabled": true,
  "rules": [
    {
      "type": "keyword",
      "value": "i cannot|i can't|i don't know|i'm sorry",
      "message": "LLM refused to provide response"
    },
    {
      "type": "length_exceeds",
      "value": "5",
      "message": "Response too short, likely an error"
    },
    {
      "type": "regex",
      "value": "^[\\s\\w]*$",
      "message": "Response contains only whitespace and common characters"
    }
  ]
}
```

## 📚 API Usage

### MCP Tools

#### `ensample.execute_llm_role`

Execute an LLM role with MDAP reliability guarantees:

```python
# Python MCP client example
import asyncio
from mcp.client import create_client

async def main():
    async with create_client("stdio", command="uv", args=["run", "ensample"]) as client:
        # Execute MDAP with custom ensemble
        result = await client.call_tool(
            "ensample.execute_llm_role",
            {
                "prompt": "Generate a Python function to calculate fibonacci numbers",
                "role_name": "CodeGenerator",
                "voting_k": 3,
                "fast_path_enabled": True,
                "ensemble_config": {
                    "models": [
                        {
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "temperature": 0.1
                        },
                        {
                            "provider": "anthropic",
                            "model": "claude-3-haiku-20240307", 
                            "temperature": 0.1
                        }
                    ]
                },
                "output_parser_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "explanation": {"type": "string"}
                    }
                }
            }
        )
        
        print(f"Result: {result.content[0].text}")
        print(f"Confidence: {result.metadata['confidence_score']}")
        print(f"Metrics: {result.metadata['mdap_metrics']}")

asyncio.run(main())
```

**Response**:
```json
{
  "final_response": "{\n  \"code\": \"def fibonacci(n):\\n    if n <= 1:\\n        return n\\n    return fibonacci(n-1) + fibonacci(n-2)\",\n  \"explanation\": \"This function uses recursion to calculate Fibonacci numbers.\"\n}",
  "confidence_score": 0.95,
  "mdap_metrics": {
    "total_llm_calls": 4,
    "voting_rounds": 2,
    "red_flags_hit": {},
    "valid_responses_per_round": [2, 2],
    "winning_response_votes": 3,
    "time_taken_ms": 1800,
    "estimated_llm_cost_usd": 0.008
  }
}
```

#### `ensample.ping`

Health check and server status:

```json
{}
```

**Response**:
```json
{
  "status": "ok",
  "message": "Ensample server is running",
  "uptime": "2h 15m 30s",
  "mdap_config_loaded": true,
  "config_status": "loaded",
  "version": "0.1.0"
}
```

#### `ensample.server_info`

Detailed server information:

```json
{}
```

**Response**:
```json
{
  "server_name": "Ensample",
  "version": "0.1.0",
  "description": "Ensemble-based LLM Orchestration with Massively Decomposed Agentic Processes",
  "max_concurrent_llm_calls": 10,
  "max_voting_rounds": 20,
  "default_voting_k": 3,
  "log_level": "INFO",
  "otel_endpoint_configured": false
}
```

## 🎯 MDAP Parameters

### Voting Configuration

- **`voting_k`**: Controls convergence criteria
  - `k=0`: Greedy mode - returns first valid response
  - `k=1`: Simple majority - needs >50% agreement
  - `k>1`: First-to-ahead-by-k - needs `k` vote advantage

### Ensemble Configuration

- **`models`**: List of LLM configurations with provider, model, temperature, max_tokens
- **`temperature`**: Sampling temperature (0.0-2.0)
- **`max_tokens`**: Maximum response length
- **`stop_sequences`**: Early termination sequences
- **`extra_params`**: Provider-specific parameters

### Red-Flag Rules

- **`keyword`**: Filter responses containing specific terms (case-insensitive, supports pipe-separated values)
- **`regex`**: Filter responses matching regex patterns
- **`length_exceeds`**: Filter responses below token threshold
- **`json_parse_error`**: Filter responses failing JSON validation against schema

### Fast-Path Optimization

Enable `fast_path_enabled` to optimize for latency and cost in high-confidence scenarios.

**When to use it:**
- **Low-Risk Queries**: Tasks where a single correct answer is likely and verification is easy.
- **Latency-Sensitive Apps**: When user experience demands sub-second responses.
- **Cost control**: To avoid full ensemble execution for simple prompts.

**How it works:**
1. **Initial Check**: Starts with a smaller initial ensemble (max 2 models).
2. **Immediate Validation**: Checks if conditions are met after the first round.
   - **Greedy (`k=0`)**: Returns the first valid response immediately.
   - **Majority (`k=1`)**: Returns immediately if a response has >50% vote share.
   - **Consensus (`k>1`)**: Returns if a "Strong Consensus" (80% agreement with 3+ votes) or "K-Advantage" is met early.
3. **Fallback**: If fast-path criteria aren't met, seamlessly transitions to full MDAP voting.

## 📊 Observability

### Metrics (Prometheus)

Key metrics available at `/metrics` endpoint:

- `mdap_execution_total`: Total MDAP executions
- `mdap_execution_success_total`: Successful executions  
- `mdap_execution_failed_total`: Failed executions
- `mdap_llm_calls_total`: LLM API calls made by provider/model
- `mdap_tokens_prompt_total`: Total prompt tokens by provider/model
- `mdap_tokens_completion_total`: Total completion tokens by provider/model
- `mdap_voting_rounds_histogram`: Distribution of voting rounds
- `mdap_execution_latency_ms_histogram`: Execution latency
- `mdap_estimated_cost_usd_total`: Cumulative cost
- `mdap_provider_cost_usd_total`: Cost breakdown by provider/model
- `mdap_red_flags_hit_total`: Red flag detection by rule type

### Tracing (OpenTelemetry)

Distributed tracing with:
- MDAP execution spans with role names and client IDs
- LLM call spans with model, timing, and cost details
- Red-flagging and parsing spans with rule hit details
- Custom attributes for role names, voting parameters, and confidence scores

### Structured Logging

JSON-formatted logs with:
- Trace IDs for request correlation
- MDAP execution details and performance metrics
- LLM provider performance and error tracking
- Red-flag detection events with rule specifics
- Cost tracking and token usage analytics

## 🔒 Production Deployment

### Security Considerations

1. **API Key Management**: Use secure secret management (Docker secrets, Kubernetes secrets)
2. **Network Security**: Deploy behind VPN/firewall with TLS termination
3. **Resource Limits**: Set appropriate CPU/memory limits in container orchestration
4. **Rate Limiting**: Implement client rate limiting and circuit breakers

### High Availability

1. **Horizontal Scaling**: Deploy multiple instances behind load balancer
2. **Transport Options**: Use HTTP/SSE for web deployments, STDIO for local integration
3. **Health Monitoring**: Implement health checks and auto-recovery
4. **Circuit Breakers**: Handle provider failures with fallback strategies

### Cost Optimization

1. **Ensemble Sizing**: Use minimal effective ensemble size (2-3 models typically sufficient)
2. **Fast-Path Tuning**: Enable for low-risk scenarios to reduce LLM calls
3. **Model Selection**: Choose cost-effective models for ensemble roles
4. **Cost Alerts**: Monitor and alert on API costs via Prometheus metrics

## 🧪 Testing

### Test Coverage

```bash
# Run all tests with coverage
uv run pytest --cov=src/ensample --cov-report=html

# Run specific test categories
uv run pytest tests/test_mdap_engine.py -v
uv run pytest tests/test_ensemble_manager.py -v
uv run pytest tests/test_voting_mechanism.py -v
```

### Current Test Status

- **267 tests passing** with **73% overall coverage**
- **Excellent coverage** on core components:
  - Ensemble Manager: 100%
  - Output Parser: 97%
  - Red Flagging Engine: 97%
  - MDAP Engine: 95%
  - Metrics System: 99%

## 🔧 Development

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run linting and type checking
uv run ruff check .
uv run mypy src/

# Run tests
uv run pytest

# Run with specific test markers
uv run pytest -m "unit"
uv run pytest -m "not slow"
```

### Architecture Decisions

- **FastMCP Framework**: Chosen for high-performance, async-native MCP implementation
- **LiteLLM Integration**: Provides broad LLM provider coverage with unified interface
- **Pydantic Models**: Strict schema validation for configuration and data exchange
- **Async/Await Pattern**: Full async implementation for concurrent LLM call handling
- **Modular Design**: Clear separation of concerns with injectable dependencies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Implement changes with comprehensive tests
4. Ensure all tests pass and coverage doesn't decrease
5. Run linting and type checking
6. Submit a pull request

### Code Quality Standards

- **Type Hints**: All code must include proper type annotations
- **Test Coverage**: New features must include comprehensive tests
- **Documentation**: All public APIs must be documented
- **Error Handling**: Robust error handling with appropriate logging

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Based on "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
- Built with [FastMCP](https://github.com/microsoft/fastmcp) for high-performance MCP
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM support
- Inspired by ensemble methods in machine learning for improved reliability

## 📞 Support

- **Issues**: GitHub Issues for bug reports and feature requests
- **Discussions**: GitHub Discussions for questions and community support  
- **Documentation**: This README and inline code documentation
- **Examples**: Check the `examples/` directory for integration patterns

---

**Ensample** - Making LLM outputs reliably correct through the power of ensemble voting and intelligent quality filtering.