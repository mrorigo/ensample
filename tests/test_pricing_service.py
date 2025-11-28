"""Comprehensive tests for the pricing service module."""

import asyncio
import pytest_asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mdapflow_mcp.pricing_service import PricingService, PricingInfo, get_pricing_service
from mdapflow_mcp.models import TokenUsage


@pytest_asyncio.fixture
async def pricing_service():
    """Create a fresh pricing service for each test."""
    service = PricingService()
    yield service
    await service.close()


class TestPricingService:
    """Test cases for PricingService."""

    @pytest.mark.asyncio
    async def test_initialize_with_static_pricing_openai(self, pricing_service):
        """Test that static pricing is initialized correctly for OpenAI models."""
        # Test GPT-5.1 (current top model)
        pricing = await pricing_service.get_pricing("openai", "gpt-5.1")
        assert pricing.input_rate == 1.25e-6
        assert pricing.output_rate == 10.00e-6
        assert pricing.source == "static"

    @pytest.mark.asyncio
    async def test_initialize_with_static_pricing_gpt4o_mini(self, pricing_service):
        """Test GPT-4o-mini pricing."""
        pricing = await pricing_service.get_pricing("openai", "gpt-4o-mini")
        assert pricing.input_rate == 0.15e-6
        assert pricing.output_rate == 0.60e-6
        assert pricing.source == "static"

    @pytest.mark.asyncio
    async def test_initialize_with_static_pricing_anthropic(self, pricing_service):
        """Test Anthropic model pricing with current models."""
        # Test Claude Opus 4.5 (latest)
        pricing = await pricing_service.get_pricing("anthropic", "claude-opus-4.5")
        assert pricing.input_rate == 5.00e-6
        assert pricing.output_rate == 25.00e-6
        assert pricing.source == "static"

        # Test Claude Sonnet 4.5
        pricing = await pricing_service.get_pricing("anthropic", "claude-sonnet-4.5")
        assert pricing.input_rate == 3.00e-6
        assert pricing.output_rate == 15.00e-6
        assert pricing.source == "static"

        # Test Claude Haiku 4.5
        pricing = await pricing_service.get_pricing("anthropic", "claude-haiku-4.5")
        assert pricing.input_rate == 1.00e-6
        assert pricing.output_rate == 5.00e-6
        assert pricing.source == "static"

    @pytest.mark.asyncio
    async def test_initialize_with_static_pricing_google(self, pricing_service):
        """Test Google Gemini model pricing."""
        # Test Gemini 2.5 Pro
        pricing = await pricing_service.get_pricing("google", "gemini-2.5-pro")
        assert pricing.input_rate == 1.25e-6
        assert pricing.output_rate == 10.00e-6
        assert pricing.source == "static"

        # Test Gemini 2.5 Flash
        pricing = await pricing_service.get_pricing("google", "gemini-2.5-flash")
        assert pricing.input_rate == 0.30e-6
        assert pricing.output_rate == 2.5e-6
        assert pricing.source == "static"

    @pytest.mark.asyncio
    async def test_initialize_with_static_pricing_deepseek(self, pricing_service):
        """Test DeepSeek model pricing."""
        pricing = await pricing_service.get_pricing("deepseek", "deepseek-v3.1-terminus")
        assert pricing.input_rate == 0.216e-6
        assert pricing.output_rate == 0.80e-6
        assert pricing.source == "static"

    @pytest.mark.asyncio
    async def test_initialize_with_static_pricing_mistral(self, pricing_service):
        """Test Mistral model pricing."""
        # Test Mixtral 8x22B
        pricing = await pricing_service.get_pricing("mistralai", "mixtral-8x22b-instruct")
        assert pricing.input_rate == 0.65e-6
        assert pricing.output_rate == 0.65e-6
        assert pricing.source == "static"

        # Test Mistral 7B
        pricing = await pricing_service.get_pricing("mistralai", "mistral-7b-instruct")
        assert pricing.input_rate == 0.13e-6
        assert pricing.output_rate == 0.13e-6
        assert pricing.source == "static"

    @pytest.mark.asyncio
    async def test_cache_functionality(self, pricing_service):
        """Test that caching works correctly."""
        # First call should cache the result
        pricing1 = await pricing_service.get_pricing("openai", "gpt-4o")
        
        # Second call should use cached result
        pricing2 = await pricing_service.get_pricing("openai", "gpt-4o")
        
        # Should be the same object reference (cached)
        assert pricing1 == pricing2
        assert pricing1.source == "static"

    @pytest.mark.asyncio
    async def test_unknown_model_fallback(self, pricing_service):
        """Test fallback pricing for unknown models."""
        pricing = await pricing_service.get_pricing("unknown", "model-xyz")
        
        # Should use default fallback pricing
        assert pricing.input_rate == 0.000001
        assert pricing.output_rate == 0.000001
        assert pricing.source == "default_fallback"

    @pytest.mark.asyncio
    async def test_cache_statistics(self, pricing_service):
        """Test cache statistics functionality."""
        # Get some pricing to populate cache
        await pricing_service.get_pricing("openai", "gpt-5.1")
        await pricing_service.get_pricing("anthropic", "claude-sonnet-4")
        await pricing_service.get_pricing("google", "gemini-2.5-pro")
        
        stats = await pricing_service.get_cache_stats()
        
        assert "total_entries" in stats
        assert "valid_entries" in stats
        assert "sources" in stats
        assert stats["total_entries"] >= 3
        assert "static" in stats["sources"]

    @pytest.mark.asyncio
    async def test_cache_clearing(self, pricing_service):
        """Test cache clearing functionality."""
        # Populate cache
        await pricing_service.get_pricing("openai", "gpt-5.1")
        
        stats_before = await pricing_service.get_cache_stats()
        assert stats_before["total_entries"] > 0
        
        # Clear cache
        await pricing_service.clear_cache()
        
        stats_after = await pricing_service.get_cache_stats()
        assert stats_after["total_entries"] == 0

    @pytest.mark.asyncio
    async def test_pricing_info_named_tuple(self):
        """Test PricingInfo named tuple functionality."""
        pricing = PricingInfo(0.001, 0.002, "USD", 1234567890.0, "test_source")
        
        assert pricing.input_rate == 0.001
        assert pricing.output_rate == 0.002
        assert pricing.currency == "USD"
        assert pricing.last_updated == 1234567890.0
        assert pricing.source == "test_source"


class TestGlobalPricingService:
    """Test cases for global pricing service functions."""

    @pytest.mark.asyncio
    async def test_get_pricing_service_singleton(self):
        """Test that global pricing service is a singleton."""
        service1 = await get_pricing_service()
        service2 = await get_pricing_service()
        
        # Should be the same instance
        assert service1 is service2

    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test that the global service initializes correctly."""
        service = await get_pricing_service()
        
        # Should have static pricing available
        pricing = await service.get_pricing("anthropic", "claude-haiku-4.5")
        assert pricing.input_rate > 0
        assert pricing.output_rate > 0


class TestPricingIntegration:
    """Integration tests with other components."""

    @pytest.mark.asyncio
    async def test_token_usage_cost_calculation_gpt4o_mini(self):
        """Test cost calculation with TokenUsage objects for GPT-4o-mini."""
        from mdapflow_mcp.llm_provider import LiteLLMClient
        from mdapflow_mcp.models import LLMConfig
        
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
        )
        client = LiteLLMClient(config)
        
        token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated=False
        )
        
        cost = await client._estimate_cost("openai/gpt-4o-mini", token_usage)
        
        # Cost should be calculated based on actual rates (0.15e-6, 0.60e-6)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Should be reasonable for 150 tokens
        expected_cost_approx = (100 * 0.15e-6) + (50 * 0.60e-6)
        assert abs(cost - expected_cost_approx) < 0.00001

    @pytest.mark.asyncio
    async def test_token_usage_cost_calculation_claude_opus(self):
        """Test cost calculation with TokenUsage objects for Claude Opus."""
        from mdapflow_mcp.llm_provider import LiteLLMClient
        from mdapflow_mcp.models import LLMConfig
        
        config = LLMConfig(
            provider="anthropic",
            model="claude-opus-4.5",
        )
        client = LiteLLMClient(config)
        
        token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated=False
        )
        
        cost = await client._estimate_cost("anthropic/claude-opus-4.5", token_usage)
        
        # Cost should be calculated based on actual rates (5.00e-6, 25.00e-6)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Should be higher than cheaper models
        expected_cost_approx = (100 * 5.00e-6) + (50 * 25.00e-6)
        assert abs(cost - expected_cost_approx) < 0.00001

    @pytest.mark.asyncio
    async def test_token_usage_cost_calculation_gemini(self):
        """Test cost calculation with TokenUsage objects for Gemini."""
        from mdapflow_mcp.llm_provider import LiteLLMClient
        from mdapflow_mcp.models import LLMConfig
        
        config = LLMConfig(
            provider="google",
            model="gemini-2.5-flash",
        )
        client = LiteLLMClient(config)
        
        token_usage = TokenUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            estimated=False
        )
        
        cost = await client._estimate_cost("google/gemini-2.5-flash", token_usage)
        
        # Cost should be calculated based on actual rates (0.30e-6, 2.5e-6)
        assert cost > 0
        assert isinstance(cost, float)
        
        expected_cost_approx = (200 * 0.30e-6) + (100 * 2.5e-6)
        assert abs(cost - expected_cost_approx) < 0.00001

    @pytest.mark.asyncio
    async def test_none_token_usage_handling(self):
        """Test handling of None token usage."""
        from mdapflow_mcp.llm_provider import LiteLLMClient
        from mdapflow_mcp.models import LLMConfig
        
        config = LLMConfig(
            provider="openai",
            model="gpt-5.1",
        )
        client = LiteLLMClient(config)
        
        cost = await client._estimate_cost("openai/gpt-5.1", None)
        
        # Should return 0 for None token usage
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_estimated_vs_actual_pricing(self):
        """Test pricing for estimated vs actual token usage."""
        from mdapflow_mcp.llm_provider import LiteLLMClient
        from mdapflow_mcp.models import LLMConfig
        
        config = LLMConfig(
            provider="deepseek",
            model="deepseek-v3.1-terminus",
        )
        client = LiteLLMClient(config)
        
        # Test with estimated tokens
        estimated_usage = TokenUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            estimated=True
        )
        
        # Test with actual tokens (simulated)
        actual_usage = TokenUsage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            estimated=False
        )
        
        # Cost should be the same regardless of estimated flag
        # (flag is for informational purposes)
        estimated_cost = await client._estimate_cost("deepseek/deepseek-v3.1-terminus", estimated_usage)
        actual_cost = await client._estimate_cost("deepseek/deepseek-v3.1-terminus", actual_usage)
        
        # Both should calculate cost based on actual token counts
        assert estimated_cost > 0
        assert actual_cost > 0
        assert abs(estimated_cost - actual_cost) < 0.00001

    @pytest.mark.asyncio
    async def test_mistral_cost_calculation(self):
        """Test cost calculation for Mistral models."""
        from mdapflow_mcp.llm_provider import LiteLLMClient
        from mdapflow_mcp.models import LLMConfig
        
        config = LLMConfig(
            provider="mistralai",
            model="mistral-7b-instruct",
        )
        client = LiteLLMClient(config)
        
        token_usage = TokenUsage(
            prompt_tokens=150,
            completion_tokens=75,
            total_tokens=225,
            estimated=False
        )
        
        cost = await client._estimate_cost("mistralai/mistral-7b-instruct", token_usage)
        
        # Cost should be calculated based on actual rates (0.13e-6 for both input/output)
        assert cost > 0
        assert isinstance(cost, float)
        
        # Should be very low cost for open-source model
        expected_cost_approx = 225 * 0.13e-6
        assert abs(cost - expected_cost_approx) < 0.00001
        assert cost < 0.001  # Should be very cheap