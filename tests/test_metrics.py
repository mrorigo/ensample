"""Comprehensive tests for MetricsCollector - Prometheus metrics and observability."""

import pytest
from unittest.mock import patch, MagicMock

from src.mdapflow_mcp.metrics import (
    MetricsCollector,
    metrics_collector,
    get_metrics_handler,
    REGISTRY,
    mdap_execution_total,
    mdap_execution_success_total,
    mdap_execution_failed_total,
    mdap_llm_calls_total,
    mdap_tokens_prompt_total,
    mdap_tokens_completion_total,
    mdap_estimated_cost_usd_total,
    mdap_red_flags_hit_total,
    mdap_voting_rounds_histogram,
    mdap_execution_latency_ms_histogram,
    mdap_active_executions,
    mdap_provider_health,
    mdap_provider_cost_usd_total,
)


class TestMetricsCollectorInitialization:
    """Test MetricsCollector initialization and basic state."""

    def test_initialization(self):
        """Test basic initialization."""
        collector = MetricsCollector()
        
        assert collector._execution_count == 0
        assert collector._success_count == 0
        assert collector._failure_count == 0

    def test_initialization_multiple_instances(self):
        """Test that instances maintain separate state."""
        collector1 = MetricsCollector()
        collector2 = MetricsCollector()
        
        collector1._execution_count = 10
        collector1._success_count = 5
        collector1._failure_count = 3
        
        assert collector1._execution_count == 10
        assert collector1._success_count == 5
        assert collector1._failure_count == 3
        
        # Other instance should be unchanged
        assert collector2._execution_count == 0
        assert collector2._success_count == 0
        assert collector2._failure_count == 0


class TestExecutionRecording:
    """Test recording of MDAP execution metrics."""

    def test_record_execution_start(self):
        """Test recording execution start."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_total.inc') as mock_total_inc, \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.inc') as mock_active_inc:
            
            collector.record_execution_start()
            
            mock_total_inc.assert_called_once()
            mock_active_inc.assert_called_once()
            assert collector._execution_count == 1

    def test_record_execution_success_minimal(self):
        """Test recording successful execution with minimal data."""
        collector = MetricsCollector()
        collector._execution_count = 1
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_success_total.inc') as mock_success_inc, \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec') as mock_active_dec, \
             patch('src.mdapflow_mcp.metrics.mdap_voting_rounds_histogram.observe') as mock_rounds_obs, \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe') as mock_latency_obs, \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_execution_histogram.labels') as mock_cost_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_cost_usd_total.inc') as mock_cost_inc:
            
            # Set up the histogram mock chain
            mock_cost_hist.return_value.observe = MagicMock()
            
            collector.record_execution_success(
                latency_ms=1000,
                voting_rounds=3,
                llm_calls=5,
                cost_usd=0.05,
                total_tokens=100
            )
            
            mock_success_inc.assert_called_once()
            mock_active_dec.assert_called_once()
            mock_rounds_obs.assert_called_once_with(3)
            mock_latency_obs.assert_called_once_with(1000)
            mock_cost_inc.assert_called_once_with(0.05)
            
            assert collector._success_count == 1

    def test_record_execution_success_with_tokens(self):
        """Test recording successful execution with token data."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_success_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec'), \
             patch('src.mdapflow_mcp.metrics.mdap_voting_rounds_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_per_execution_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_execution_histogram.labels') as mock_cost_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_cost_usd_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_llm_calls_total.labels') as mock_calls, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_prompt_total.labels') as mock_prompt, \
             patch('src.mdapflow_mcp.metrics.mdap_prompt_tokens_histogram.labels') as mock_prompt_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_completion_total.labels') as mock_completion, \
             patch('src.mdapflow_mcp.metrics.mdap_completion_tokens_histogram.labels') as mock_completion_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_provider_cost_usd_total.labels') as mock_provider_cost, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_token_percentage.labels') as mock_estimated, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_total_execution_counter.labels') as mock_total_counter:
            
            # Set up the mock chains
            mock_cost_hist.return_value.observe = MagicMock()
            mock_calls.return_value.inc = MagicMock()
            mock_prompt.return_value.inc = MagicMock()
            mock_prompt.return_value.observe = MagicMock()
            mock_completion.return_value.inc = MagicMock()
            mock_completion.return_value.observe = MagicMock()
            mock_provider_cost.return_value.inc = MagicMock()
            mock_estimated.return_value.inc = MagicMock()
            mock_total_counter.return_value.inc = MagicMock()
            
            collector.record_execution_success(
                latency_ms=2000,
                voting_rounds=2,
                llm_calls=3,
                cost_usd=0.10,
                total_tokens=1500,
                prompt_tokens=800,
                completion_tokens=700,
                provider="openai",
                model="gpt-4o",
                role_name="test_role"
            )
            
            # Verify all the token-related calls were made
            mock_calls.return_value.inc.assert_called_once_with(3)
            mock_prompt.return_value.inc.assert_called_once_with(800)
            mock_completion.return_value.inc.assert_called_once_with(700)
            mock_provider_cost.return_value.inc.assert_called_once_with(0.10)

    def test_record_execution_success_with_zero_cost(self):
        """Test recording successful execution with zero cost."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_success_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec'), \
             patch('src.mdapflow_mcp.metrics.mdap_voting_rounds_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_execution_histogram.labels') as mock_cost_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_cost_usd_total.inc') as mock_cost_inc:
            
            # Set up the histogram mock chain
            mock_cost_hist.return_value.observe = MagicMock()
            
            collector.record_execution_success(
                latency_ms=500,
                voting_rounds=1,
                llm_calls=1,
                cost_usd=0.0,
                total_tokens=100,
                provider="test",
                model="test"
            )
            
            # Should not increment cost metrics when cost is 0
            mock_cost_inc.assert_called_once_with(0.0)
            mock_cost_hist.return_value.observe.assert_not_called()

    def test_record_execution_failure(self):
        """Test recording failed execution."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_failed_total.inc') as mock_failed_inc, \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec') as mock_active_dec, \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe') as mock_latency_obs:
            
            collector.record_execution_failure(latency_ms=5000)
            
            mock_failed_inc.assert_called_once()
            mock_active_dec.assert_called_once()
            mock_latency_obs.assert_called_once_with(5000)
            assert collector._failure_count == 1


class TestRedFlagRecording:
    """Test red flag hit recording."""

    def test_record_red_flag_hit(self):
        """Test recording red flag hit."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_red_flags_hit_total.labels') as mock_labels:
            mock_labels.return_value.inc = MagicMock()
            
            collector.record_red_flag_hit("keyword")
            
            mock_labels.assert_called_once_with(rule_type="keyword")
            mock_labels.return_value.inc.assert_called_once()


class TestTokenRecording:
    """Test token usage recording for individual LLM calls."""

    def test_record_llm_call_tokens_prompt_only(self):
        """Test recording token usage with only prompt tokens."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_tokens_prompt_total.labels') as mock_prompt, \
             patch('src.mdapflow_mcp.metrics.mdap_prompt_tokens_histogram.labels') as mock_prompt_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_token_percentage.labels') as mock_estimated:
            
            # Set up mock chains
            mock_prompt.return_value.inc = MagicMock()
            mock_prompt_hist.return_value.observe = MagicMock()
            mock_estimated.return_value.inc = MagicMock()
            
            collector.record_llm_call_tokens(
                provider="openai",
                model="gpt-4o",
                prompt_tokens=100,
                completion_tokens=0,
                estimated=False
            )
            
            mock_prompt.return_value.inc.assert_called_once_with(100)
            mock_prompt_hist.return_value.observe.assert_called_once_with(100)
            mock_estimated.assert_not_called()

    def test_record_llm_call_tokens_completion_only(self):
        """Test recording token usage with only completion tokens."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_tokens_completion_total.labels') as mock_completion, \
             patch('src.mdapflow_mcp.metrics.mdap_completion_tokens_histogram.labels') as mock_completion_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_token_percentage.labels') as mock_estimated:
            
            # Set up mock chains
            mock_completion.return_value.inc = MagicMock()
            mock_completion_hist.return_value.observe = MagicMock()
            mock_estimated.return_value.inc = MagicMock()
            
            collector.record_llm_call_tokens(
                provider="anthropic",
                model="claude-3",
                prompt_tokens=0,
                completion_tokens=150,
                estimated=True
            )
            
            mock_completion.return_value.inc.assert_called_once_with(150)
            mock_completion_hist.return_value.observe.assert_called_once_with(150)
            mock_estimated.return_value.inc.assert_called_once_with(1)

    def test_record_llm_call_tokens_both(self):
        """Test recording token usage with both prompt and completion tokens."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_tokens_prompt_total.labels') as mock_prompt, \
             patch('src.mdapflow_mcp.metrics.mdap_prompt_tokens_histogram.labels') as mock_prompt_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_completion_total.labels') as mock_completion, \
             patch('src.mdapflow_mcp.metrics.mdap_completion_tokens_histogram.labels') as mock_completion_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_token_percentage.labels') as mock_estimated:
            
            # Set up mock chains
            mock_prompt.return_value.inc = MagicMock()
            mock_prompt_hist.return_value.observe = MagicMock()
            mock_completion.return_value.inc = MagicMock()
            mock_completion_hist.return_value.observe = MagicMock()
            mock_estimated.return_value.inc = MagicMock()
            
            collector.record_llm_call_tokens(
                provider="openai",
                model="gpt-4o",
                prompt_tokens=200,
                completion_tokens=100,
                estimated=False
            )
            
            # Verify both prompt and completion tokens are recorded
            mock_prompt.return_value.inc.assert_called_once_with(200)
            mock_prompt_hist.return_value.observe.assert_called_once_with(200)
            mock_completion.return_value.inc.assert_called_once_with(100)
            mock_completion_hist.return_value.observe.assert_called_once_with(100)
            mock_estimated.assert_not_called()

    def test_record_llm_call_tokens_zero_tokens(self):
        """Test recording token usage with zero tokens."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_tokens_prompt_total.labels') as mock_prompt, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_completion_total.labels') as mock_completion, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_token_percentage.labels') as mock_estimated:
            
            collector.record_llm_call_tokens(
                provider="test",
                model="test",
                prompt_tokens=0,
                completion_tokens=0,
                estimated=True
            )
            
            # Should not call any metrics when tokens are 0
            mock_prompt.assert_not_called()
            mock_completion.assert_not_called()
            mock_estimated.assert_not_called()


class TestCostRecording:
    """Test cost recording for individual LLM calls."""

    def test_record_llm_call_cost_with_tokens(self):
        """Test recording cost with token data."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_provider_cost_usd_total.labels') as mock_provider_cost, \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_token_gauge.labels') as mock_cost_per_token:
            
            # Set up mock chains
            mock_provider_cost.return_value.inc = MagicMock()
            mock_cost_per_token.return_value.inc = MagicMock()
            
            collector.record_llm_call_cost(
                provider="openai",
                model="gpt-4o",
                cost_usd=0.05,
                input_tokens=1000,
                output_tokens=500
            )
            
            mock_provider_cost.return_value.inc.assert_called_once_with(0.05)
            
            # Should calculate and record cost per token ratios
            assert mock_cost_per_token.call_count == 2
            mock_cost_per_token.return_value.inc.assert_any_call(0.05 / 1000)  # Input cost per token
            mock_cost_per_token.return_value.inc.assert_any_call(0.05 / 500)   # Output cost per token

    def test_record_llm_call_cost_zero_cost(self):
        """Test recording cost with zero cost."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_provider_cost_usd_total.labels') as mock_provider_cost, \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_token_gauge.labels') as mock_cost_per_token:
            
            collector.record_llm_call_cost(
                provider="test",
                model="test",
                cost_usd=0.0,
                input_tokens=100,
                output_tokens=50
            )
            
            # Should not record provider cost when cost is 0
            mock_provider_cost.assert_not_called()
            mock_cost_per_token.assert_not_called()

    def test_record_llm_call_cost_no_tokens(self):
        """Test recording cost without token data."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_provider_cost_usd_total.labels') as mock_provider_cost, \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_token_gauge.labels') as mock_cost_per_token:
            
            collector.record_llm_call_cost(
                provider="test",
                model="test",
                cost_usd=0.02
            )
            
            # Should only record provider cost
            mock_provider_cost.return_value.inc.assert_called_once_with(0.02)
            mock_cost_per_token.assert_not_called()


class TestProviderHealth:
    """Test provider health status updates."""

    def test_update_provider_health_healthy(self):
        """Test updating provider health to healthy."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_provider_health.labels') as mock_labels:
            mock_labels.return_value.set = MagicMock()
            
            collector.update_provider_health("openai", True)
            
            mock_labels.assert_called_once_with(provider="openai")
            mock_labels.return_value.set.assert_called_once_with(1)

    def test_update_provider_health_unhealthy(self):
        """Test updating provider health to unhealthy."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_provider_health.labels') as mock_labels:
            mock_labels.return_value.set = MagicMock()
            
            collector.update_provider_health("anthropic", False)
            
            mock_labels.assert_called_once_with(provider="anthropic")
            mock_labels.return_value.set.assert_called_once_with(0)


class TestMetricsExport:
    """Test metrics export and summary functionality."""

    def test_get_metrics_text(self):
        """Test getting metrics in Prometheus text format."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.generate_latest') as mock_generate, \
             patch('src.mdapflow_mcp.metrics.CONTENT_TYPE_LATEST', "text/plain; version=1.0.0; charset=utf-8"):
            
            mock_generate.return_value = b"metric1\nmetric2\n"
            
            metrics_text, content_type = collector.get_metrics_text()
            
            mock_generate.assert_called_once_with(REGISTRY)
            assert metrics_text == b"metric1\nmetric2\n"
            assert content_type == "text/plain; version=1.0.0; charset=utf-8"

    def test_get_token_summary(self):
        """Test getting token usage summary."""
        collector = MetricsCollector()
        
        summary = collector.get_token_summary()
        
        assert "total_prompt_tokens" in summary
        assert "total_completion_tokens" in summary
        assert "estimated_token_percentage" in summary
        assert "note" in summary
        assert summary["total_prompt_tokens"] == "Available via Prometheus metrics"

    def test_get_cost_summary(self):
        """Test getting cost summary."""
        collector = MetricsCollector()
        
        summary = collector.get_cost_summary()
        
        assert "total_estimated_cost_usd" in summary
        assert "provider_breakdown" in summary
        assert "note" in summary
        assert summary["total_estimated_cost_usd"] == "Available via Prometheus metrics"


class TestMetricsHandler:
    """Test metrics HTTP handler."""

    def test_get_metrics_handler(self):
        """Test getting metrics HTTP handler."""
        handler = get_metrics_handler()
        
        # Should return a callable
        assert callable(handler)
        
        # Test calling the handler
        result = handler()
        
        # Should return actual metrics data
        assert len(result) == 3
        assert result[1] == 200
        assert "Content-Type" in result[2]
        assert result[2]["Content-Type"] == "text/plain; version=1.0.0; charset=utf-8"
        assert isinstance(result[0], bytes)
        assert b"mdap_execution_total" in result[0]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_record_execution_success_negative_cost(self):
        """Test handling negative cost gracefully."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_success_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec'), \
             patch('src.mdapflow_mcp.metrics.mdap_voting_rounds_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_cost_usd_total.inc') as mock_cost_inc:
            
            # Should handle negative cost (though unlikely in practice)
            collector.record_execution_success(
                latency_ms=1000,
                voting_rounds=1,
                llm_calls=1,
                cost_usd=-0.01,  # Negative cost
                total_tokens=100
            )
            
            # Should still increment the counter (though negative)
            mock_cost_inc.assert_called_once_with(-0.01)

    def test_record_execution_success_large_numbers(self):
        """Test handling large numbers in metrics."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_success_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec'), \
             patch('src.mdapflow_mcp.metrics.mdap_voting_rounds_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_per_execution_histogram.observe'), \
             patch('src.mdapflow_mcp.metrics.mdap_cost_per_execution_histogram.labels') as mock_cost_hist, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_cost_usd_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_llm_calls_total.labels') as mock_calls, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_prompt_total.labels') as mock_prompt, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_completion_total.labels') as mock_completion, \
             patch('src.mdapflow_mcp.metrics.mdap_provider_cost_usd_total.labels') as mock_provider_cost, \
             patch('src.mdapflow_mcp.metrics.mdap_estimated_token_percentage.labels') as mock_estimated, \
             patch('src.mdapflow_mcp.metrics.mdap_tokens_total_execution_counter.labels') as mock_total_counter:
            
            # Set up all the mock chains properly
            mock_cost_hist.return_value.observe = MagicMock()
            mock_calls.return_value.inc = MagicMock()
            mock_prompt.return_value.inc = MagicMock()
            mock_prompt.return_value.observe = MagicMock()
            mock_completion.return_value.inc = MagicMock()
            mock_completion.return_value.observe = MagicMock()
            mock_provider_cost.return_value.inc = MagicMock()
            mock_estimated.return_value.inc = MagicMock()
            mock_total_counter.return_value.inc = MagicMock()
            
            # Should handle large numbers gracefully
            collector.record_execution_success(
                latency_ms=3600000,  # 1 hour in ms
                voting_rounds=100,
                llm_calls=1000,
                cost_usd=1000.0,
                total_tokens=1000000,
                prompt_tokens=500000,
                completion_tokens=500000,
                provider="test",
                model="test"
            )
            
            # Just verify no exceptions are raised

    def test_record_execution_failure_large_latency(self):
        """Test recording failure with large latency."""
        collector = MetricsCollector()
        
        with patch('src.mdapflow_mcp.metrics.mdap_execution_failed_total.inc'), \
             patch('src.mdapflow_mcp.metrics.mdap_active_executions.dec'), \
             patch('src.mdapflow_mcp.metrics.mdap_execution_latency_ms_histogram.observe') as mock_latency_obs:
            
            collector.record_execution_failure(latency_ms=3600000)  # 1 hour
            
            mock_latency_obs.assert_called_once_with(3600000)


class TestGlobalMetricsCollector:
    """Test the global metrics collector instance."""

    def test_global_metrics_collector_exists(self):
        """Test that global metrics collector exists."""
        assert metrics_collector is not None
        assert isinstance(metrics_collector, MetricsCollector)

    def test_global_metrics_collector_is_singleton(self):
        """Test that global metrics collector is the same instance."""
        from src.mdapflow_mcp.metrics import metrics_collector as another_ref
        
        assert metrics_collector is another_ref


class TestRegistryAndMetrics:
    """Test that all defined metrics exist in the registry."""

    def test_registry_exists(self):
        """Test that registry is properly created."""
        assert REGISTRY is not None

    def test_all_counters_defined(self):
        """Test that all expected counters are defined."""
        # These should not raise exceptions
        assert mdap_execution_total is not None
        assert mdap_execution_success_total is not None
        assert mdap_execution_failed_total is not None
        assert mdap_llm_calls_total is not None
        assert mdap_tokens_prompt_total is not None
        assert mdap_tokens_completion_total is not None
        assert mdap_estimated_cost_usd_total is not None
        assert mdap_red_flags_hit_total is not None
        assert mdap_provider_cost_usd_total is not None

    def test_all_gauges_defined(self):
        """Test that all expected gauges are defined."""
        assert mdap_active_executions is not None
        assert mdap_provider_health is not None

    def test_all_histograms_defined(self):
        """Test that all expected histograms are defined."""
        assert mdap_voting_rounds_histogram is not None
        assert mdap_execution_latency_ms_histogram is not None