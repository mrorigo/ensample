"""Comprehensive tests for the LLM provider module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from ensample.llm_provider import LiteLLMClient, LLMProviderInterface
from ensample.models import LLMConfig, TokenUsage, LLMResponse


class TestLiteLLMClient:
    """Test cases for LiteLLMClient."""

    @pytest.fixture
    def llm_config_openai(self):
        """Create a basic OpenAI LLM configuration."""
        return LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
        )

    @pytest.fixture
    def llm_config_anthropic(self):
        """Create a basic Anthropic LLM configuration."""
        return LLMConfig(
            provider="anthropic",
            model="claude-3-haiku-20240307",
        )

    @pytest.fixture
    def llm_config_custom(self):
        """Create a custom LLM configuration."""
        return LLMConfig(
            provider="custom",
            model="my-model",
            api_key_env_var="CUSTOM_API_KEY",
            base_url="https://custom.api.com",
            temperature=0.7,
            max_tokens=1000,
        )

    def test_client_initialization_openai(self, llm_config_openai):
        """Test client initialization with OpenAI config."""
        client = LiteLLMClient(llm_config_openai)
        assert client.config.provider == "openai"
        assert client.config.model == "gpt-4o-mini"

    def test_client_initialization_custom(self, llm_config_custom):
        """Test client initialization with custom config."""
        client = LiteLLMClient(llm_config_custom)
        assert client.config.provider == "custom"
        assert client.config.model == "my-model"
        assert client.config.api_key_env_var == "CUSTOM_API_KEY"
        assert client.config.base_url == "https://custom.api.com"
        assert client.config.temperature == 0.7
        assert client.config.max_tokens == 1000

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key-123'})
    def test_api_key_retrieval_openai(self, llm_config_openai):
        """Test API key retrieval for OpenAI."""
        client = LiteLLMClient(llm_config_openai)
        api_key = client._get_api_key()
        assert api_key == "test-key-123"

    @patch.dict('os.environ', {'CUSTOM_API_KEY': 'custom-key-456'})
    def test_api_key_retrieval_custom_env_var(self, llm_config_custom):
        """Test API key retrieval using custom environment variable."""
        client = LiteLLMClient(llm_config_custom)
        api_key = client._get_api_key()
        assert api_key == "custom-key-456"

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'anthropic-key-789'})
    def test_api_key_retrieval_anthropic_fallback(self, llm_config_anthropic):
        """Test API key retrieval for Anthropic using fallback."""
        client = LiteLLMClient(llm_config_anthropic)
        api_key = client._get_api_key()
        assert api_key == "anthropic-key-789"

    def test_api_key_retrieval_no_env_var(self, llm_config_openai):
        """Test API key retrieval when no environment variable is set."""
        client = LiteLLMClient(llm_config_openai)
        api_key = client._get_api_key()
        assert api_key is None

    def test_build_litellm_params_basic(self, llm_config_openai):
        """Test building LiteLLM parameters with basic config."""
        client = LiteLLMClient(llm_config_openai)
        params = client._build_litellm_params("Test prompt")
        
        assert params["model"] == "openai/gpt-4o-mini"
        assert params["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert params["temperature"] == 0.1
        assert params["top_p"] == 1.0

    def test_build_litellm_params_custom(self, llm_config_custom):
        """Test building LiteLLM parameters with custom config."""
        client = LiteLLMClient(llm_config_custom)
        params = client._build_litellm_params("Test prompt")
        
        assert params["model"] == "custom/my-model"
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1000
        assert params["stop"] is None  # default value

    def test_build_litellm_params_with_stop_sequences(self):
        """Test building parameters with stop sequences."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            stop_sequences=["STOP", "END"]
        )
        client = LiteLLMClient(config)
        params = client._build_litellm_params("Test prompt")
        
        assert params["stop"] == ["STOP", "END"]

    def test_build_litellm_params_with_api_key(self):
        """Test building parameters with API key."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var="TEST_API_KEY"
        )
        
        with patch.dict('os.environ', {'TEST_API_KEY': 'test-key'}):
            client = LiteLLMClient(config)
            params = client._build_litellm_params("Test prompt")
            
            assert params["api_key"] == "test-key"

    def test_build_litellm_params_with_base_url(self, llm_config_custom):
        """Test building parameters with custom base URL."""
        client = LiteLLMClient(llm_config_custom)
        params = client._build_litellm_params("Test prompt")
        
        assert params["base_url"] == "https://custom.api.com"

    def test_build_litellm_params_with_extra_params(self):
        """Test building parameters with extra parameters."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            extra_params={"frequency_penalty": 0.5, "presence_penalty": 0.3}
        )
        client = LiteLLMClient(config)
        params = client._build_litellm_params("Test prompt")
        
        assert params["frequency_penalty"] == 0.5
        assert params["presence_penalty"] == 0.3

    def test_estimate_tokens_basic(self, llm_config_openai):
        """Test token estimation for basic text."""
        client = LiteLLMClient(llm_config_openai)
        
        # Simple text
        tokens = client._estimate_tokens("Hello world", "openai/gpt-4o")
        assert tokens > 0
        
        # Empty text should return minimum of 1
        tokens = client._estimate_tokens("", "openai/gpt-4o")
        assert tokens == 1

    def test_estimate_tokens_long_text(self, llm_config_openai):
        """Test token estimation for longer text."""
        client = LiteLLMClient(llm_config_openai)
        
        long_text = "This is a very long text. " * 100  # 100 repetitions
        tokens = client._estimate_tokens(long_text, "openai/gpt-4o")
        
        # Should estimate roughly 1.3 tokens per word
        expected_approx = int(len(long_text.split()) * 1.3)
        assert abs(tokens - expected_approx) < 50  # Allow some variance

    def test_fallback_cost_calculation_with_token_usage(self, llm_config_openai):
        """Test fallback cost calculation with valid token usage."""
        client = LiteLLMClient(llm_config_openai)
        
        token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated=False
        )
        
        cost = client._fallback_cost_calculation("openai/gpt-4o-mini", token_usage)
        
        # Expected cost: 100 * 0.00000015 + 50 * 0.0000006
        expected_cost = (100 * 0.00000015) + (50 * 0.0000006)
        assert abs(cost - expected_cost) < 0.00001

    def test_fallback_cost_calculation_none_token_usage(self, llm_config_openai):
        """Test fallback cost calculation with None token usage."""
        client = LiteLLMClient(llm_config_openai)
        
        cost = client._fallback_cost_calculation("openai/gpt-4o-mini", None)
        assert cost == 0.0

    def test_fallback_cost_calculation_unknown_model(self, llm_config_openai):
        """Test fallback cost calculation for unknown model."""
        client = LiteLLMClient(llm_config_openai)
        
        token_usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated=False
        )
        
        cost = client._fallback_cost_calculation("unknown/provider-model", token_usage)
        
        # Should use default rate of 0.000001 for unknown models
        expected_cost = 150 * 0.000001
        assert abs(cost - expected_cost) < 0.00001

    def test_fallback_cost_calculation_partial_token_data(self, llm_config_openai):
        """Test fallback cost calculation with partial token data."""
        client = LiteLLMClient(llm_config_openai)
        
        token_usage = TokenUsage(
            prompt_tokens=100,  # Only prompt tokens set
            completion_tokens=None,
            total_tokens=None,
            estimated=False
        )
        
        cost = client._fallback_cost_calculation("openai/gpt-4o-mini", token_usage)
        
        # Should only use prompt tokens, completion should be 0
        expected_cost = 100 * 0.00000015
        assert abs(cost - expected_cost) < 0.00001

    @pytest.mark.asyncio
    async def test_extract_token_usage_complete_data(self, llm_config_openai):
        """Test token usage extraction with complete data."""
        client = LiteLLMClient(llm_config_openai)
        
        # Mock response with usage data
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        
        mock_response = MagicMock()
        mock_response.usage = mock_usage
        
        token_usage = client._extract_token_usage(mock_response)
        
        assert token_usage is not None
        assert token_usage.prompt_tokens == 100
        assert token_usage.completion_tokens == 50
        assert token_usage.total_tokens == 150
        assert token_usage.estimated is False

    @pytest.mark.asyncio
    async def test_extract_token_usage_no_usage_data(self, llm_config_openai):
        """Test token usage extraction when no usage data is available."""
        client = LiteLLMClient(llm_config_openai)
        
        mock_response = MagicMock()
        mock_response.usage = None
        
        token_usage = client._extract_token_usage(mock_response)
        assert token_usage is None

    @pytest.mark.asyncio
    async def test_extract_response_content_with_choices(self, llm_config_openai):
        """Test response content extraction with choices format."""
        client = LiteLLMClient(llm_config_openai)
        
        mock_choice = MagicMock()
        mock_choice.message.content = "This is the response content"
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        
        content, token_usage = client._extract_response_content(mock_response)
        
        assert content == "This is the response content"
        assert token_usage is not None  # Should create estimated usage

    @pytest.mark.asyncio
    async def test_extract_response_content_fallback_formats(self, llm_config_openai):
        """Test response content extraction with different response formats."""
        client = LiteLLMClient(llm_config_openai)
        
        # Test with text attribute
        mock_response = MagicMock()
        mock_response.text = "Text-based response"
        
        content, token_usage = client._extract_response_content(mock_response)
        
        assert content == "Text-based response"
        
        # Test with content attribute
        mock_response = MagicMock()
        mock_response.content = "Content-based response"
        del mock_response.text
        
        content, token_usage = client._extract_response_content(mock_response)
        
        assert content == "Content-based response"


class TestLLMProviderInterface:
    """Test cases for LLMProviderInterface."""

    @pytest.fixture
    def provider_interface(self):
        """Create a provider interface for testing."""
        return LLMProviderInterface()

    def test_get_client_creates_new_client(self, provider_interface):
        """Test that get_client creates a new client when needed."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        
        client = provider_interface.get_client(config)
        
        assert isinstance(client, LiteLLMClient)
        assert client.config.provider == "openai"
        assert client.config.model == "gpt-4o-mini"

    def test_get_client_reuses_existing_client(self, provider_interface):
        """Test that get_client reuses existing clients."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        
        client1 = provider_interface.get_client(config)
        client2 = provider_interface.get_client(config)
        
        # Should be the same instance
        assert client1 is client2

    def test_get_client_different_configs_different_clients(self, provider_interface):
        """Test that different configs create different clients."""
        config1 = LLMConfig(provider="openai", model="gpt-4o-mini")
        config2 = LLMConfig(provider="anthropic", model="claude-3-haiku")
        
        client1 = provider_interface.get_client(config1)
        client2 = provider_interface.get_client(config2)
        
        # Should be different instances
        assert client1 is not client2
        assert client1.config.provider == "openai"
        assert client2.config.provider == "anthropic"

    @pytest.mark.asyncio
    async def test_estimate_cost_with_dynamic_pricing(self, provider_interface):
        """Test cost estimation uses dynamic pricing."""
        config1 = LLMConfig(provider="openai", model="gpt-4o-mini")
        config2 = LLMConfig(provider="anthropic", model="claude-3-haiku")
        
        configs = [config1, config2]
        prompt = "This is a test prompt with some words"
        
        cost = await provider_interface.estimate_cost(configs, prompt)
        
        # Should return a positive cost estimate
        assert cost > 0
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_estimate_cost_empty_configs(self, provider_interface):
        """Test cost estimation with empty configs."""
        configs = []
        prompt = "Test prompt"
        
        cost = await provider_interface.estimate_cost(configs, prompt)
        
        # Should return 0 for empty configs
        assert cost == 0.0


class TestIntegration:
    """Integration tests for LLM provider components."""

    @pytest.mark.asyncio
    async def test_full_cost_calculation_workflow(self):
        """Test the complete cost calculation workflow."""
        from ensample.llm_provider import LiteLLMClient
        from ensample.models import LLMConfig, TokenUsage
        
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
        )
        client = LiteLLMClient(config)
        
        # Test with realistic token usage
        token_usage = TokenUsage(
            prompt_tokens=150,  # ~200 words
            completion_tokens=75,  # ~100 words
            total_tokens=225,
            estimated=False
        )
        
        cost = await client._estimate_cost("openai/gpt-4o-mini", token_usage)
        
        # Verify cost calculation is reasonable
        assert cost > 0
        assert cost < 0.01  # Should be less than 1 cent for typical usage
        
        # Verify it matches expected calculation
        expected_cost = (150 * 0.00000015) + (75 * 0.0000006)
        assert abs(cost - expected_cost) < 0.00001

    @pytest.mark.asyncio
    async def test_multiple_provider_cost_comparison(self):
        """Test cost comparison across different providers."""
        from ensample.llm_provider import LLMProviderInterface
        
        # Create interface with multiple providers
        provider = LLMProviderInterface()
        
        config_openai = LLMConfig(provider="openai", model="gpt-4o-mini")
        config_anthropic = LLMConfig(provider="anthropic", model="claude-3-haiku-20240307")
        
        prompt = "What is the capital of France?"
        
        cost_openai = await provider.estimate_cost([config_openai], prompt)
        cost_anthropic = await provider.estimate_cost([config_anthropic], prompt)
        
        # Both should provide cost estimates
        assert cost_openai > 0
        assert cost_anthropic > 0
        
        # Costs may differ between providers
        # (This is expected due to different pricing models)