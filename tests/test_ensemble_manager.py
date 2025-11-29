"""Comprehensive tests for EnsembleManager - parallel LLM ensemble execution."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from src.ensample.ensemble_manager import EnsembleManager
from src.ensample.llm_provider import LLMProviderInterface
from src.ensample.models import EnsembleConfig, LLMConfig, LLMResponse


class TestEnsembleManagerInitialization:
    """Test EnsembleManager initialization."""

    def test_initialization(self):
        """Test basic initialization."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        
        manager = EnsembleManager(provider_interface)
        
        assert manager.provider_interface is provider_interface
        assert manager._call_history == []

    def test_initialization_with_custom_provider(self):
        """Test initialization with custom provider interface."""
        mock_provider = MagicMock(spec=LLMProviderInterface)
        
        manager = EnsembleManager(mock_provider)
        
        assert manager.provider_interface is mock_provider


class TestModelSelection:
    """Test model selection for voting rounds."""

    def test_select_models_fewer_than_max(self):
        """Test selection when there are fewer models than max_calls."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Create ensemble with 2 models
        config = EnsembleConfig(
            models=[
                LLMConfig(provider="openai", model="gpt-4o"),
                LLMConfig(provider="anthropic", model="claude-3")
            ]
        )
        
        # Request 5 calls (more than available models)
        selected = manager.select_models_for_round(config, max_calls=5)
        
        # Should return all models
        assert len(selected) == 2
        assert selected == config.models

    def test_select_models_equal_to_max(self):
        """Test selection when models equal max_calls."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        config = EnsembleConfig(
            models=[
                LLMConfig(provider="openai", model="gpt-4o"),
                LLMConfig(provider="anthropic", model="claude-3"),
                LLMConfig(provider="openai", model="gpt-3.5")
            ]
        )
        
        selected = manager.select_models_for_round(config, max_calls=3)
        
        # Should return all models
        assert len(selected) == 3
        assert all(model in config.models for model in selected)
        assert len([m for m in config.models if m in selected]) == 3

    def test_select_models_diversity_selection(self):
        """Test diversity-based model selection."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Create ensemble with 5 models
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="openai", model="gpt-3.5"),
            LLMConfig(provider="google", model="gemini-pro"),
            LLMConfig(provider="mistral", model="mistral-large")
        ]
        config = EnsembleConfig(models=models)
        
        # Add some models to call history to test diversity
        manager._call_history = [models[0], models[1], models[0]]  # gpt-4o called twice, claude-3 once
        
        # Request 3 calls - should prefer unused models
        selected = manager.select_models_for_round(config, max_calls=3)
        
        # Should include unused models (gpt-3.5, gemini-pro, mistral-large)
        unused_models = [models[2], models[3], models[4]]
        assert len(selected) == 3
        assert all(model in unused_models for model in selected)
        # Compare using model identifiers since LLMConfig objects are not hashable
        selected_ids = {f"{m.provider}/{m.model}" for m in selected}
        expected_ids = {f"{m.provider}/{m.model}" for m in unused_models}
        assert selected_ids == expected_ids

    def test_select_models_with_reuse_when_insufficient_unused(self):
        """Test model selection when not enough unused models exist."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="openai", model="gpt-3.5")
        ]
        config = EnsembleConfig(models=models)
        
        # Add all but one model to history
        manager._call_history = [models[0], models[1]]
        
        # Request 3 calls but only 1 unused model available
        selected = manager.select_models_for_round(config, max_calls=3)
        
        # Should include the unused model plus reuse others
        assert len(selected) == 3
        assert models[2] in selected  # The unused model
        # Should also include some of the used models due to reuse
        assert any(model in selected for model in [models[0], models[1]])

    def test_select_models_single_model(self):
        """Test selection with single model ensemble."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        config = EnsembleConfig(
            models=[LLMConfig(provider="openai", model="gpt-4o")]
        )
        
        selected = manager.select_models_for_round(config, max_calls=1)
        
        assert len(selected) == 1
        assert selected[0] == config.models[0]

    def test_select_models_empty_history(self):
        """Test selection with empty call history."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3")
        ]
        config = EnsembleConfig(models=models)
        
        manager._call_history = []
        
        selected = manager.select_models_for_round(config, max_calls=2)
        
        # Should select randomly from all models
        assert len(selected) == 2
        assert all(model in models for model in selected)

    def test_select_models_request_one_from_many(self):
        """Test selecting single model from larger ensemble."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="openai", model="gpt-3.5"),
            LLMConfig(provider="google", model="gemini-pro"),
            LLMConfig(provider="mistral", model="mistral-large")
        ]
        config = EnsembleConfig(models=models)
        
        # Request just 1 call
        selected = manager.select_models_for_round(config, max_calls=1)
        
        assert len(selected) == 1
        assert selected[0] in models


class TestEnsembleStats:
    """Test ensemble statistics gathering."""

    def test_get_ensemble_stats_empty_history(self):
        """Test stats with no call history."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 0
        assert stats["unique_models_used"] == 0
        assert stats["model_usage"] == {}

    def test_get_ensemble_stats_single_model(self):
        """Test stats with single model calls."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        model = LLMConfig(provider="openai", model="gpt-4o")
        manager._call_history = [model, model, model]  # 3 calls to same model
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 3
        assert stats["unique_models_used"] == 1
        assert stats["model_usage"] == {"openai/gpt-4o": 3}

    def test_get_ensemble_stats_multiple_models(self):
        """Test stats with multiple models."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="openai", model="gpt-3.5")
        ]
        
        # Mixed usage pattern
        manager._call_history = [
            models[0], models[1], models[0],  # gpt-4o: 2 calls, claude-3: 1 call
            models[2], models[1], models[2]   # gpt-3.5: 2 calls, claude-3: 1 more (total: 2)
        ]
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 6
        assert stats["unique_models_used"] == 3
        assert stats["model_usage"] == {
            "openai/gpt-4o": 2,
            "anthropic/claude-3": 2,
            "openai/gpt-3.5": 2
        }

    def test_get_ensemble_stats_different_providers(self):
        """Test stats with different provider combinations."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="openai", model="gpt-3.5"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="google", model="gemini-pro")
        ]
        
        manager._call_history = [models[0], models[2], models[1], models[3]]
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 4
        assert stats["unique_models_used"] == 4
        assert len(stats["model_usage"]) == 4

    def test_get_ensemble_stats_no_duplicates(self):
        """Test that stats correctly handle unique model identification."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Same model added multiple times
        model = LLMConfig(provider="openai", model="gpt-4o")
        manager._call_history = [model] * 5
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 5
        assert stats["unique_models_used"] == 1
        assert stats["model_usage"] == {"openai/gpt-4o": 5}


class TestDispatchEnsembleCalls:
    """Test async ensemble call dispatching."""

    @pytest.mark.asyncio
    async def test_dispatch_ensemble_calls_basic(self):
        """Test basic ensemble call dispatching."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Mock provider interface
        mock_client = AsyncMock()
        mock_client.generate.return_value = LLMResponse(
            response="test response",
            llm_config=LLMConfig(provider="openai", model="gpt-4o"),
            cost_estimate=0.05
        )
        provider_interface.get_client.return_value = mock_client
        
        config = EnsembleConfig(
            models=[
                LLMConfig(provider="openai", model="gpt-4o"),
                LLMConfig(provider="anthropic", model="claude-3")
            ]
        )
        
        responses = await manager.dispatch_ensemble_calls(
            prompt="test prompt",
            ensemble_config=config,
            num_calls_per_model=1
        )
        
        # Should have 2 responses (one from each model)
        assert len(responses) == 2
        assert all(isinstance(resp, LLMResponse) for resp in responses)
        
        # Verify client was called for each model
        assert provider_interface.get_client.call_count == 2
        
        # Verify call history was updated
        assert len(manager._call_history) == 2

    @pytest.mark.asyncio
    async def test_dispatch_ensemble_calls_multiple_calls_per_model(self):
        """Test dispatching multiple calls per model."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Mock different responses for different calls
        responses = [
            LLMResponse(response=f"response {i}", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05)
            for i in range(4)  # 2 models * 2 calls each = 4 responses
        ]
        
        mock_client = AsyncMock()
        mock_client.generate.side_effect = responses
        provider_interface.get_client.return_value = mock_client
        
        config = EnsembleConfig(
            models=[
                LLMConfig(provider="openai", model="gpt-4o"),
                LLMConfig(provider="anthropic", model="claude-3")
            ]
        )
        
        result_responses = await manager.dispatch_ensemble_calls(
            prompt="test prompt",
            ensemble_config=config,
            num_calls_per_model=2
        )
        
        # Should have 4 responses (2 models * 2 calls each)
        assert len(result_responses) == 4
        assert all(isinstance(resp, LLMResponse) for resp in result_responses)
        
        # Verify total calls made
        assert mock_client.generate.call_count == 4
        
        # Verify call history (4 entries)
        assert len(manager._call_history) == 4

    @pytest.mark.asyncio
    async def test_dispatch_ensemble_calls_with_failures(self):
        """Test handling of failed LLM calls."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Mock some failing and some succeeding calls using side_effect list
        error_response = Exception("API Error")
        success_response = LLMResponse(
            response="successful response",
            llm_config=LLMConfig(provider="openai", model="gpt-4o"),
            cost_estimate=0.05
        )
        
        # Alternate between success and failure
        side_effect = [success_response, error_response, success_response, error_response]
        
        mock_client = AsyncMock()
        mock_client.generate.side_effect = side_effect
        provider_interface.get_client.return_value = mock_client
        
        config = EnsembleConfig(
            models=[
                LLMConfig(provider="openai", model="gpt-4o"),
                LLMConfig(provider="anthropic", model="claude-3")
            ]
        )
        
        responses = await manager.dispatch_ensemble_calls(
            prompt="test prompt",
            ensemble_config=config,
            num_calls_per_model=2  # 4 total calls, 2 should succeed
        )
        
        # Should have 2 successful responses
        assert len(responses) == 2
        assert all(isinstance(resp, LLMResponse) for resp in responses)
        
        # Verify only successful calls were recorded in history
        assert len(manager._call_history) == 2

    @pytest.mark.asyncio
    async def test_dispatch_ensemble_calls_all_fail(self):
        """Test handling when all calls fail."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Mock all calls failing
        mock_client = AsyncMock()
        mock_client.generate.side_effect = Exception("All API errors")
        provider_interface.get_client.return_value = mock_client
        
        config = EnsembleConfig(
            models=[LLMConfig(provider="openai", model="gpt-4o")]
        )
        
        responses = await manager.dispatch_ensemble_calls(
            prompt="test prompt",
            ensemble_config=config,
            num_calls_per_model=1
        )
        
        # Should have no responses
        assert len(responses) == 0
        
        # Call history should be empty (no successful calls)
        assert len(manager._call_history) == 0

    @pytest.mark.asyncio
    async def test_dispatch_ensemble_calls_single_model(self):
        """Test dispatching with single model."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        mock_client = AsyncMock()
        mock_client.generate.return_value = LLMResponse(
            response="single response",
            llm_config=LLMConfig(provider="openai", model="gpt-4o"),
            cost_estimate=0.05
        )
        provider_interface.get_client.return_value = mock_client
        
        config = EnsembleConfig(
            models=[LLMConfig(provider="openai", model="gpt-4o")]
        )
        
        responses = await manager.dispatch_ensemble_calls(
            prompt="test prompt",
            ensemble_config=config,
            num_calls_per_model=3
        )
        
        # Should have 3 responses
        assert len(responses) == 3
        assert all(isinstance(resp, LLMResponse) for resp in responses)
        
        # Verify client was called 3 times
        assert mock_client.generate.call_count == 3
        
        # Verify call history (3 entries for same model)
        assert len(manager._call_history) == 3
        assert all(config == manager._call_history[0] for config in manager._call_history)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_select_models_max_calls_zero(self):
        """Test model selection with max_calls=0."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        config = EnsembleConfig(
            models=[LLMConfig(provider="openai", model="gpt-4o")]
        )
        
        selected = manager.select_models_for_round(config, max_calls=0)
        
        # Should return empty list when max_calls is 0
        assert selected == []

    def test_select_models_negative_calls(self):
        """Test model selection with negative max_calls."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        config = EnsembleConfig(
            models=[LLMConfig(provider="openai", model="gpt-4o")]
        )
        
        # Should handle gracefully - negative values should be treated as 0
        selected = manager.select_models_for_round(config, max_calls=-1)
        
        # With negative max_calls, should return empty list
        assert selected == []

    def test_get_ensemble_stats_malformed_history(self):
        """Test stats with unusual call history."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Add some models to history
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3")
        ]
        manager._call_history = models
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 2
        assert stats["unique_models_used"] == 2
        assert "openai/gpt-4o" in stats["model_usage"]
        assert "anthropic/claude-3" in stats["model_usage"]

    def test_manager_state_persistence(self):
        """Test that manager state persists across operations."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Add to call history manually
        model = LLMConfig(provider="openai", model="gpt-4o")
        manager._call_history.append(model)
        
        # Check stats reflect the manual addition
        stats = manager.get_ensemble_stats()
        assert stats["total_calls"] == 1
        
        # Select models should consider the history
        config = EnsembleConfig(models=[model])
        selected = manager.select_models_for_round(config, max_calls=1)
        
        # Should still work even with history
        assert len(selected) == 1

    def test_select_models_single_from_multiple_with_history(self):
        """Test selecting 1 model from multiple when history has some usage."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="openai", model="gpt-3.5")
        ]
        config = EnsembleConfig(models=models)
        
        # Add one model to history
        manager._call_history = [models[0]]
        
        # Request 1 call - should prefer unused model
        selected = manager.select_models_for_round(config, max_calls=1)
        
        # Should select one of the unused models
        assert len(selected) == 1
        assert selected[0] in [models[1], models[2]]  # Not the one in history

    def test_get_ensemble_stats_provider_distribution(self):
        """Test that stats correctly show provider distribution."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        # Create models from different providers
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="openai", model="gpt-3.5"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="google", model="gemini-pro")
        ]
        
        # Create uneven usage across providers
        manager._call_history = [
            models[0], models[0],  # OpenAI: 2 calls
            models[1],             # OpenAI: 1 more (total: 3)
            models[2],             # Anthropic: 1 call
            models[3]              # Google: 1 call
        ]
        
        stats = manager.get_ensemble_stats()
        
        assert stats["total_calls"] == 5
        assert stats["unique_models_used"] == 4
        
        # Check provider distribution
        model_usage = stats["model_usage"]
        assert model_usage["openai/gpt-4o"] == 2
        assert model_usage["openai/gpt-3.5"] == 1
        assert model_usage["anthropic/claude-3"] == 1
        assert model_usage["google/gemini-pro"] == 1

    def test_select_models_all_used_recently(self):
        """Test selection when all models have been used recently."""
        provider_interface = MagicMock(spec=LLMProviderInterface)
        manager = EnsembleManager(provider_interface)
        
        models = [
            LLMConfig(provider="openai", model="gpt-4o"),
            LLMConfig(provider="anthropic", model="claude-3"),
            LLMConfig(provider="google", model="gemini-pro")
        ]
        config = EnsembleConfig(models=models)
        
        # Add all models to recent history
        manager._call_history = models.copy()
        
        # Request 2 calls - should reuse some models
        selected = manager.select_models_for_round(config, max_calls=2)
        
        # Should return 2 models (may reuse some from history)
        assert len(selected) == 2
        assert all(model in models for model in selected)