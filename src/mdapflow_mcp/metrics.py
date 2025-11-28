"""Prometheus metrics for MDAPFlow-MCP."""

from __future__ import annotations

from typing import Any
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from prometheus_client.core import CollectorRegistry

# Create custom registry
REGISTRY = CollectorRegistry()

# MDAP Execution Metrics
mdap_execution_total = Counter(
    'mdap_execution_total',
    'Total number of MDAP executions',
    registry=REGISTRY
)

mdap_execution_success_total = Counter(
    'mdap_execution_success_total',
    'Total number of successful MDAP executions',
    registry=REGISTRY
)

mdap_execution_failed_total = Counter(
    'mdap_execution_failed_total',
    'Total number of failed MDAP executions',
    registry=REGISTRY
)

# LLM Call Metrics
mdap_llm_calls_total = Counter(
    'mdap_llm_calls_total',
    'Total number of LLM API calls made',
    ['provider', 'model'],
    registry=REGISTRY
)

# Token Usage Metrics - Comprehensive
mdap_tokens_prompt_total = Counter(
    'mdap_tokens_prompt_total',
    'Total number of prompt tokens processed',
    ['provider', 'model'],
    registry=REGISTRY
)

mdap_tokens_completion_total = Counter(
    'mdap_tokens_completion_total',
    'Total number of completion tokens generated',
    ['provider', 'model'],
    registry=REGISTRY
)

mdap_tokens_total_execution_counter = Counter(
    'mdap_tokens_total_execution_counter',
    'Total tokens processed per MDAP execution',
    ['role_name', 'provider', 'model'],
    registry=REGISTRY
)

# Token Histograms for Distribution Analysis
mdap_tokens_per_execution_histogram = Histogram(
    'mdap_tokens_per_execution_histogram',
    'Distribution of total tokens per MDAP execution',
    buckets=[100, 500, 1000, 2000, 5000, 10000, 25000, 50000, 100000],
    registry=REGISTRY
)

mdap_prompt_tokens_histogram = Histogram(
    'mdap_prompt_tokens_histogram',
    'Distribution of prompt tokens per LLM call',
    ['provider', 'model'],
    buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000],
    registry=REGISTRY
)

mdap_completion_tokens_histogram = Histogram(
    'mdap_completion_tokens_histogram',
    'Distribution of completion tokens per LLM call',
    ['provider', 'model'],
    buckets=[10, 50, 100, 500, 1000, 2000, 5000, 10000, 25000],
    registry=REGISTRY
)

# Cost Metrics - Enhanced
mdap_estimated_cost_usd_total = Counter(
    'mdap_estimated_cost_usd_total',
    'Cumulative estimated LLM API cost in USD',
    registry=REGISTRY
)

mdap_cost_per_execution_histogram = Histogram(
    'mdap_cost_per_execution_histogram',
    'Distribution of cost per MDAP execution in USD',
    ['role_name'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    registry=REGISTRY
)

mdap_cost_per_token_gauge = Counter(
    'mdap_cost_per_token_gauge',
    'Average cost per token in USD',
    ['provider', 'model', 'token_type'],
    registry=REGISTRY
)

# Provider-specific cost tracking
mdap_provider_cost_usd_total = Counter(
    'mdap_provider_cost_usd_total',
    'Cumulative cost by provider and model',
    ['provider', 'model'],
    registry=REGISTRY
)

# Token Efficiency Metrics
mdap_estimated_token_percentage = Counter(
    'mdap_estimated_token_percentage',
    'Percentage of tokens that were estimated vs actual',
    ['provider', 'model', 'estimation_type'],
    registry=REGISTRY
)

# Red Flag and Quality Metrics
mdap_red_flags_hit_total = Counter(
    'mdap_red_flags_hit_total',
    'Total count of each type of red flag hit',
    ['rule_type'],
    registry=REGISTRY
)

mdap_voting_rounds_histogram = Histogram(
    'mdap_voting_rounds_histogram',
    'Distribution of voting rounds per MDAP execution',
    buckets=[1, 2, 3, 5, 10, 20, 50],
    registry=REGISTRY
)

mdap_execution_latency_ms_histogram = Histogram(
    'mdap_execution_latency_ms_histogram',
    'Distribution of end-to-end MDAP execution latency in milliseconds',
    buckets=[100, 500, 1000, 2000, 5000, 10000, 30000, 60000],
    registry=REGISTRY
)

# Gauges for Current State
mdap_active_executions = Gauge(
    'mdap_active_executions',
    'Current number of active MDAP executions',
    registry=REGISTRY
)

mdap_provider_health = Gauge(
    'mdap_provider_health',
    'Health status of LLM providers (1=healthy, 0=unhealthy)',
    ['provider'],
    registry=REGISTRY
)

# Current token usage gauges
mdap_current_token_usage = Gauge(
    'mdap_current_token_usage',
    'Current token usage in active executions',
    ['token_type', 'provider', 'model'],
    registry=REGISTRY
)


class MetricsCollector:
    """Enhanced collector for MDAPFlow-MCP metrics with comprehensive token and cost tracking."""

    def __init__(self) -> None:
        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0

    def record_execution_start(self) -> None:
        """Record start of MDAP execution."""
        mdap_execution_total.inc()
        mdap_active_executions.inc()
        self._execution_count += 1

    def record_execution_success(
        self,
        latency_ms: int,
        voting_rounds: int,
        llm_calls: int,
        cost_usd: float,
        total_tokens: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        provider: str | None = None,
        model: str | None = None,
        role_name: str = "unknown"
    ) -> None:
        """Record successful MDAP execution with comprehensive token and cost tracking."""
        mdap_execution_success_total.inc()
        mdap_active_executions.dec()

        # Record histograms
        mdap_voting_rounds_histogram.observe(voting_rounds)
        mdap_execution_latency_ms_histogram.observe(latency_ms)
        
        # Token distribution histograms
        if total_tokens > 0:
            mdap_tokens_per_execution_histogram.observe(total_tokens)
        
        # Cost distribution
        if cost_usd > 0:
            mdap_cost_per_execution_histogram.labels(role_name=role_name).observe(cost_usd)

        # Record counters
        if provider and model:
            # LLM calls
            mdap_llm_calls_total.labels(provider=provider, model=model).inc(llm_calls)
            
            # Token totals by provider/model
            if prompt_tokens > 0:
                mdap_tokens_prompt_total.labels(provider=provider, model=model).inc(prompt_tokens)
                mdap_prompt_tokens_histogram.labels(provider=provider, model=model).observe(prompt_tokens)
            
            if completion_tokens > 0:
                mdap_tokens_completion_total.labels(provider=provider, model=model).inc(completion_tokens)
                mdap_completion_tokens_histogram.labels(provider=provider, model=model).observe(completion_tokens)
                
            # Provider cost tracking
            mdap_provider_cost_usd_total.labels(provider=provider, model=model).inc(cost_usd)
            
            # Token efficiency metrics
            estimated_prompt = prompt_tokens == 0  # Simplified estimation tracking
            estimated_completion = completion_tokens == 0
            
            if estimated_prompt:
                mdap_estimated_token_percentage.labels(
                    provider=provider, model=model, estimation_type="prompt"
                ).inc(100)
            if estimated_completion:
                mdap_estimated_token_percentage.labels(
                    provider=provider, model=model, estimation_type="completion"
                ).inc(100)

        # Global counters
        mdap_estimated_cost_usd_total.inc(cost_usd)
        
        # Per-execution token tracking
        if total_tokens > 0:
            mdap_tokens_total_execution_counter.labels(
                role_name=role_name, provider=provider or "unknown", model=model or "unknown"
            ).inc(total_tokens)

        self._success_count += 1

    def record_execution_failure(self, latency_ms: int) -> None:
        """Record failed MDAP execution."""
        mdap_execution_failed_total.inc()
        mdap_active_executions.dec()
        mdap_execution_latency_ms_histogram.observe(latency_ms)

        self._failure_count += 1

    def record_red_flag_hit(self, rule_type: str) -> None:
        """Record red flag hit."""
        mdap_red_flags_hit_total.labels(rule_type=rule_type).inc()

    def record_llm_call_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        estimated: bool = False
    ) -> None:
        """Record token usage for a single LLM call."""
        if prompt_tokens > 0:
            mdap_tokens_prompt_total.labels(provider=provider, model=model).inc(prompt_tokens)
            mdap_prompt_tokens_histogram.labels(provider=provider, model=model).observe(prompt_tokens)
        
        if completion_tokens > 0:
            mdap_tokens_completion_total.labels(provider=provider, model=model).inc(completion_tokens)
            mdap_completion_tokens_histogram.labels(provider=provider, model=model).observe(completion_tokens)
            
        # Record estimation status
        if estimated:
            if prompt_tokens > 0:
                mdap_estimated_token_percentage.labels(
                    provider=provider, model=model, estimation_type="prompt"
                ).inc(1)
            if completion_tokens > 0:
                mdap_estimated_token_percentage.labels(
                    provider=provider, model=model, estimation_type="completion"
                ).inc(1)

    def record_llm_call_cost(
        self,
        provider: str,
        model: str,
        cost_usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0
    ) -> None:
        """Record cost and token ratios for cost analysis."""
        if cost_usd > 0:
            mdap_provider_cost_usd_total.labels(provider=provider, model=model).inc(cost_usd)
            
        # Calculate and record cost per token if tokens available
        if input_tokens > 0 and cost_usd > 0:
            cost_per_input_token = cost_usd / input_tokens
            mdap_cost_per_token_gauge.labels(
                provider=provider, model=model, token_type="input"
            ).inc(cost_per_input_token)
            
        if output_tokens > 0 and cost_usd > 0:
            cost_per_output_token = cost_usd / output_tokens
            mdap_cost_per_token_gauge.labels(
                provider=provider, model=model, token_type="output"
            ).inc(cost_per_output_token)

    def update_provider_health(self, provider: str, healthy: bool) -> None:
        """Update provider health status."""
        mdap_provider_health.labels(provider=provider).set(1 if healthy else 0)

    def get_metrics_text(self) -> tuple[bytes, str]:
        """Get Prometheus metrics in text format."""
        return generate_latest(REGISTRY), CONTENT_TYPE_LATEST

    def get_token_summary(self) -> dict[str, Any]:
        """Get summary of token usage metrics."""
        return {
            "total_prompt_tokens": "Available via Prometheus metrics",
            "total_completion_tokens": "Available via Prometheus metrics", 
            "estimated_token_percentage": "Available via Prometheus metrics",
            "note": "Use /metrics endpoint for detailed metrics"
        }

    def get_cost_summary(self) -> dict[str, Any]:
        """Get summary of cost metrics."""
        return {
            "total_estimated_cost_usd": "Available via Prometheus metrics",
            "provider_breakdown": "Available via Prometheus metrics",
            "note": "Use /metrics endpoint for detailed metrics"
        }


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_handler():
    """Get HTTP handler for metrics endpoint."""
    def metrics_handler():
        metrics_text, content_type = metrics_collector.get_metrics_text()
        return metrics_text, 200, {'Content-Type': content_type}

    return metrics_handler
