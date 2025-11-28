# MDAPFlow-MCP

**Massively Decomposed Agentic Processes (MDAPs) Execution Engine - MCP Server**

MDAPFlow-MCP is a specialized Model Context Protocol (MCP) server designed to provide **highly reliable LLM-driven responses** by implementing the principles of Massively Decomposed Agentic Processes (MDAPs). It centralizes ensemble sampling, voting, and red-flagging to achieve near-zero-error LLM outputs.

## 🌟 Key Features

- **High-Reliability LLM Outputs**: Uses ensemble voting and red-flagging to minimize errors
- **Configurable MDAP Parameters**: Control voting thresholds, ensemble composition, and quality checks
- **Fast-Path Optimization**: Early termination for high-confidence scenarios
- **Multi-Provider LLM Support**: Works with OpenAI, Anthropic, OpenRouter, and custom providers via LiteLLM
- **Comprehensive Observability**: OpenTelemetry tracing, structured logging, and Prometheus metrics
- **Production-Ready**: Containerized deployment with health checks and monitoring

## 🏗️ Architecture

MDAPFlow-MCP implements a sophisticated MDAP pipeline:

```
Client Request
    ↓
MDAP Engine Orchestration
    ↓
Ensemble Manager (Parallel LLM Calls)
    ↓
Red-Flagging Engine (Quality Filtering)
    ↓
Output Parser (Structured Validation)
    ↓
Voting Mechanism (First-to-Ahead-by-K)
    ↓
Fast-Path Controller (Early Termination)
    ↓
High-Confidence Result + Metrics
```

## 🚀 Quick Start

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd mdapflow-mcp
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
   export MDAP_MAX_VOTING_ROUNDS=20
   export MDAP_LOG_LEVEL=INFO
   ```

3. **Run the server**:
   ```bash
   uv run mdapflow-mcp
   ```

### Docker Deployment

```bash
docker build -t mdapflow-mcp .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-key" \
  -e ANTHROPIC_API_KEY="your-key" \
  mdapflow-mcp
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MDAP_DEFAULT_VOTING_K` | Default voting threshold | `3` |
| `MDAP_MAX_CONCURRENT_LLM_CALLS` | Max parallel LLM calls | `10` |
| `MDAP_MAX_VOTING_ROUNDS` | Max voting rounds before timeout | `20` |
| `MDAP_LOG_LEVEL` | Logging verbosity | `INFO` |
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
    }
  ]
}
```

## 📚 API Usage

### MCP Tools

#### `mdapflow.execute_llm_role`

Execute an LLM role with MDAP reliability guarantees:

```json
{
  "prompt": "Summarize the following text: [text content]",
  "role_name": "DocumentSummarizer",
  "voting_k": 3,
  "fast_path_enabled": false,
  "output_parser_schema": {
    "type": "object",
    "properties": {
      "summary": {"type": "string"},
      "confidence": {"type": "number"}
    }
  }
}
```

**Response**:
```json
{
  "final_response": "{\"summary\": \"The text discusses...\", \"confidence\": 0.95}",
  "confidence_score": 0.95,
  "mdap_metrics": {
    "total_llm_calls": 6,
    "voting_rounds": 2,
    "red_flags_hit": {},
    "valid_responses_per_round": [3, 3],
    "winning_response_votes": 4,
    "time_taken_ms": 2500,
    "estimated_llm_cost_usd": 0.012
  }
}
```

#### `mdapflow.ping`

Health check and server status:

```json
{}
```

**Response**:
```json
{
  "status": "ok",
  "message": "MDAPFlow-MCP server is running",
  "uptime": "2h 15m 30s",
  "mdap_config_loaded": true,
  "config_status": "loaded",
  "version": "0.1.0"
}
```

### Client Integration

**Python Example**:
```python
import asyncio
from mcp.client import create_client

async def main():
    async with create_client("stdio", command="uv", args=["run", "mdapflow-mcp"]) as client:
        # Execute MDAP
        result = await client.call_tool(
            "mdapflow.execute_llm_role",
            {
                "prompt": "Generate a Python function to calculate fibonacci numbers",
                "role_name": "CodeGenerator",
                "voting_k": 3,
                "fast_path_enabled": True
            }
        )
        
        print(f"Result: {result.content[0].text}")
        print(f"Confidence: {result.metadata['confidence_score']}")

asyncio.run(main())
```

## 🎯 MDAP Parameters

### Voting Configuration

- **`voting_k`**: Controls convergence criteria
  - `k=0`: Greedy mode - returns first valid response
  - `k=1`: Simple majority - needs >50% agreement
  - `k>1`: First-to-ahead-by-k - needs `k` vote advantage

### Ensemble Configuration

- **`models`**: List of LLM configurations
- **`temperature`**: Sampling temperature (0.0-2.0)
- **`max_tokens`**: Maximum response length
- **`stop_sequences`**: Early termination sequences

### Red-Flag Rules

- **`keyword`**: Filter responses containing specific terms
- **`regex`**: Filter responses matching patterns
- **`length_exceeds`**: Filter responses below token threshold
- **`json_parse_error`**: Filter responses failing JSON validation

## 📊 Observability

### Metrics (Prometheus)

Key metrics available at `/metrics`:

- `mdap_execution_total`: Total MDAP executions
- `mdap_execution_success_total`: Successful executions  
- `mdap_execution_failed_total`: Failed executions
- `mdap_llm_calls_total`: LLM API calls made
- `mdap_voting_rounds_histogram`: Distribution of voting rounds
- `mdap_execution_latency_ms_histogram`: Execution latency
- `mdap_estimated_cost_usd_total`: Cumulative cost

### Tracing (OpenTelemetry)

Distributed tracing with:
- MDAP execution spans
- LLM call spans with model/timing details
- Red-flagging and parsing spans
- Custom attributes for role names and client IDs

### Structured Logging

JSON-formatted logs with:
- Trace IDs for request correlation
- MDAP execution details
- LLM provider performance
- Red-flag detection events

## 🔒 Production Deployment

### Security Considerations

1. **API Key Management**: Use secure secret management
2. **Network Security**: Deploy behind VPN/firewall
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Rate Limiting**: Implement client rate limiting

### High Availability

1. **Horizontal Scaling**: Deploy multiple instances
2. **Load Balancing**: Distribute requests across instances
3. **Health Monitoring**: Implement health checks
4. **Circuit Breakers**: Handle provider failures

### Cost Optimization

1. **Ensemble Sizing**: Use minimal effective ensemble size
2. **Fast-Path Tuning**: Enable for low-risk scenarios
3. **Caching**: Implement result caching where appropriate
4. **Cost Alerts**: Monitor and alert on API costs

## 🧪 Testing

### Unit Tests
```bash
uv run pytest tests/unit/ -v
```

### Integration Tests  
```bash
uv run pytest tests/integration/ -v
```

### Load Testing
```bash
uv run pytest tests/load/ -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

# Run linting and type checking
uv run ruff check .
uv run mypy src/

# Run tests
uv run pytest
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Based on "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025)
- Built with [FastMCP](https://github.com/microsoft/fastmcp) for high-performance MCP
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for multi-provider LLM support

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions  
- **Documentation**: This README and inline code docs