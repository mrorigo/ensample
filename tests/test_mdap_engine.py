"""Comprehensive tests for MDAPEngine - main orchestration component."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

from src.mdapflow_mcp.mdap_engine import MDAPEngine
from src.mdapflow_mcp.models import (
    MDAPInput,
    MDAPOutput,
    MDAPMetrics,
    LLMConfig,
    EnsembleConfig,
    RedFlagConfig,
    RedFlagRule,
    ParsedResponse,
    LLMResponse,
    TokenUsage,
)
from src.mdapflow_mcp.config import Settings


# Mock OpenTelemetry components for testing
@pytest.fixture
def mock_tracer():
    """Mock OpenTelemetry tracer."""
    mock_span = MagicMock()
    mock_span.set_attribute = MagicMock()
    mock_span.record_exception = MagicMock()
    mock_span.set_status = MagicMock()
    
    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    mock_tracer.start_as_current_span.return_value.__exit__.return_value = None
    
    with patch('src.mdapflow_mcp.mdap_engine.TRACER', mock_tracer):
        yield mock_tracer


class TestMDAPEngineInitialization:
    """Test MDAPEngine initialization and default configuration loading."""

    def test_engine_initialization_default_settings(self):
        """Test engine initialization with default settings."""
        engine = MDAPEngine()
        
        assert engine.settings is not None
        assert engine.provider_interface is not None
        assert engine.ensemble_manager is not None
        assert engine.red_flag_engine is not None
        assert engine.output_parser is not None
        assert engine.voting_mechanism is not None
        assert engine.fast_path_controller is not None

    def test_engine_initialization_custom_settings(self):
        """Test engine initialization with custom settings."""
        custom_settings = Settings()
        engine = MDAPEngine(settings=custom_settings)
        
        assert engine.settings is custom_settings

    @patch('src.mdapflow_mcp.mdap_engine.load_ensemble_config')
    @patch('src.mdapflow_mcp.mdap_engine.create_default_ensemble_config')
    @patch('src.mdapflow_mcp.mdap_engine.create_default_red_flag_config')
    def test_load_default_configs_success(self, mock_create_red_flag, mock_create_ensemble, mock_load_ensemble):
        """Test successful loading of default configurations."""
        # Setup mocks
        mock_load_ensemble.return_value = EnsembleConfig(
            models=[LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key_env_var=None,
                base_url=None,
                temperature=0.1,
                top_p=1.0,
                max_tokens=None,
                stop_sequences=None,
                extra_params=None
            )]
        )
        mock_create_ensemble.return_value = EnsembleConfig(
            models=[LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key_env_var=None,
                base_url=None,
                temperature=0.1,
                top_p=1.0,
                max_tokens=None,
                stop_sequences=None,
                extra_params=None
            )]
        )
        mock_create_red_flag.return_value = RedFlagConfig(rules=[], enabled=True)
        
        engine = MDAPEngine()
        
        # Verify configurations were loaded
        assert engine.default_ensemble_config is not None
        assert len(engine.default_ensemble_config.models) == 1
        assert engine.default_red_flag_config is not None
        assert engine.red_flag_engine.config == engine.default_red_flag_config

    @patch('src.mdapflow_mcp.mdap_engine.load_ensemble_config')
    @patch('src.mdapflow_mcp.mdap_engine.create_default_ensemble_config')
    @patch('src.mdapflow_mcp.mdap_engine.create_default_red_flag_config')
    def test_load_default_configs_exception_fallback(self, mock_create_red_flag, mock_create_ensemble, mock_load_ensemble):
        """Test fallback to created defaults when loading fails."""
        # Setup mocks to raise exception
        mock_load_ensemble.side_effect = Exception("Config file not found")
        mock_create_ensemble.return_value = EnsembleConfig(
            models=[LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key_env_var=None,
                base_url=None,
                temperature=0.1,
                top_p=1.0,
                max_tokens=None,
                stop_sequences=None,
                extra_params=None
            )]
        )
        mock_create_red_flag.return_value = RedFlagConfig(rules=[], enabled=True)
        
        engine = MDAPEngine()
        
        # Should use fallback defaults
        assert engine.default_ensemble_config is not None
        assert engine.default_red_flag_config is not None


class TestConfigurationPreparation:
    """Test configuration preparation methods."""

    def test_prepare_ensemble_config_provided(self):
        """Test using provided ensemble config."""
        engine = MDAPEngine()
        
        provided_config = EnsembleConfig(
            models=[LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key_env_var=None,
                base_url=None,
                temperature=0.1,
                top_p=1.0,
                max_tokens=None,
                stop_sequences=None,
                extra_params=None
            )]
        )
        
        mdap_input = MDAPInput(
            prompt="Test prompt",
            role_name="test",
            ensemble_config=provided_config,
            voting_k=3,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=False,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        result = engine._prepare_ensemble_config(mdap_input)
        assert result == provided_config

    def test_prepare_ensemble_config_default(self):
        """Test using default ensemble config."""
        engine = MDAPEngine()
        
        default_config = EnsembleConfig(
            models=[LLMConfig(
                provider="openai",
                model="gpt-4o",
                api_key_env_var=None,
                base_url=None,
                temperature=0.1,
                top_p=1.0,
                max_tokens=None,
                stop_sequences=None,
                extra_params=None
            )]
        )
        engine.default_ensemble_config = default_config
        
        mdap_input = MDAPInput(
            prompt="Test prompt",
            role_name="test",
            ensemble_config=None,
            voting_k=3,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=False,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        result = engine._prepare_ensemble_config(mdap_input)
        assert result == default_config

    def test_prepare_red_flag_config_provided(self):
        """Test using provided red flag config."""
        engine = MDAPEngine()
        
        provided_config = RedFlagConfig(
            rules=[RedFlagRule(type="keyword", value="error", message="Error keyword")],
            enabled=True
        )
        
        mdap_input = MDAPInput(
            prompt="Test prompt",
            role_name="test",
            ensemble_config=None,
            voting_k=3,
            red_flag_config=provided_config,
            output_parser_schema=None,
            fast_path_enabled=False,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        result = engine._prepare_red_flag_config(mdap_input)
        assert result == provided_config

    def test_prepare_red_flag_config_default(self):
        """Test using default red flag config."""
        engine = MDAPEngine()
        
        default_config = RedFlagConfig(
            rules=[RedFlagRule(type="keyword", value="error", message="Error keyword")],
            enabled=True
        )
        engine.default_red_flag_config = default_config
        
        mdap_input = MDAPInput(
            prompt="Test prompt",
            role_name="test",
            ensemble_config=None,
            voting_k=3,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=False,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        result = engine._prepare_red_flag_config(mdap_input)
        assert result == default_config


class TestResponseNormalization:
    """Test response normalization for voting."""

    def test_normalize_string_response(self):
        """Test normalization of string responses."""
        engine = MDAPEngine()
        
        result = engine._normalize_response_for_voting("  Hello World  ")
        assert result == "Hello World"

    def test_normalize_dict_response(self):
        """Test normalization of dictionary responses."""
        engine = MDAPEngine()
        
        response_dict = {"key1": "value1", "key2": "value2"}
        result = engine._normalize_response_for_voting(response_dict)
        
        # Should be JSON with sorted keys
        assert "key1" in result and "key2" in result
        assert ":" in result and "," in result

    def test_normalize_other_response(self):
        """Test normalization of other response types."""
        engine = MDAPEngine()
        
        result = engine._normalize_response_for_voting(123)
        assert result == "123"

    def test_normalize_dict_with_non_serializable(self):
        """Test handling of non-serializable dictionary values."""
        engine = MDAPEngine()
        
        # Create a response with a lambda function (non-serializable)
        response_dict = {"key1": "value1", "key2": lambda x: x}
        result = engine._normalize_response_for_voting(response_dict)
        
        # Should fallback to str() representation
        assert "key1" in result


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    def test_calculate_confidence_zero_rounds(self):
        """Test confidence calculation with zero rounds."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="test",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        metrics = MDAPMetrics(
            total_llm_calls=0,
            voting_rounds=0,
            red_flags_hit={},
            valid_responses_per_round=[],
            winning_response_votes=0,
            time_taken_ms=0,
            estimated_llm_cost_usd=0.0,
        )
        
        confidence = engine._calculate_confidence_score(parsed_response, metrics)
        assert confidence == 0.0

    def test_calculate_confidence_zero_valid_responses(self):
        """Test confidence calculation with zero valid responses."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="test",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        metrics = MDAPMetrics(
            total_llm_calls=0,
            voting_rounds=1,
            red_flags_hit={},
            valid_responses_per_round=[0],  # No valid responses
            winning_response_votes=0,
            time_taken_ms=100,
            estimated_llm_cost_usd=0.0,
        )
        
        confidence = engine._calculate_confidence_score(parsed_response, metrics)
        assert confidence == 0.0

    def test_calculate_confidence_high_vote_share(self):
        """Test confidence calculation with high vote share."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="test",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        metrics = MDAPMetrics(
            total_llm_calls=5,
            voting_rounds=2,
            red_flags_hit={},
            valid_responses_per_round=[3, 2],  # 5 total valid responses
            winning_response_votes=4,  # 4/5 = 0.8 vote share
            time_taken_ms=200,
            estimated_llm_cost_usd=0.1,
        )
        
        confidence = engine._calculate_confidence_score(parsed_response, metrics)
        # Should be 0.8 (vote share) + 0.02 (2 rounds * 0.01) + 0.0 (not fast path)
        expected = 0.8 + 0.02
        assert confidence == expected

    def test_calculate_confidence_with_fast_path_bonus(self):
        """Test confidence calculation with fast path bonus (capped at 1.0)."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="test",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        metrics = MDAPMetrics(
            total_llm_calls=2,
            voting_rounds=1,  # Single round = fast path
            red_flags_hit={},
            valid_responses_per_round=[2],
            winning_response_votes=2,
            time_taken_ms=50,
            estimated_llm_cost_usd=0.05,
        )
        
        confidence = engine._calculate_confidence_score(parsed_response, metrics)
        # Implementation: 1.0 (100% vote share) + 0.05 (fast path bonus) = 1.05, but capped at 1.0
        expected = 1.0  # Capped at maximum
        assert confidence == expected

    def test_calculate_confidence_maximum_cap(self):
        """Test that confidence score is capped at 1.0."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="test",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        metrics = MDAPMetrics(
            total_llm_calls=10,
            voting_rounds=20,  # Many rounds would create excessive bonus
            red_flags_hit={},
            valid_responses_per_round=[5, 5],
            winning_response_votes=5,
            time_taken_ms=1000,
            estimated_llm_cost_usd=1.0,
        )
        
        confidence = engine._calculate_confidence_score(parsed_response, metrics)
        # Should be capped at 1.0 even with bonuses
        assert confidence <= 1.0


class TestFastPathExecution:
    """Test fast-path execution logic."""

    @pytest.mark.asyncio
    async def test_execute_with_fast_path_no_responses(self):
        """Test fast-path when no valid responses are generated."""
        engine = MDAPEngine()
        
        # Mock ensemble manager to return empty responses
        mock_model = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        
        with patch.object(engine.ensemble_manager, 'select_models_for_round', return_value=[]), \
             patch.object(engine.voting_mechanism, 'run_voting', return_value=(
                 ParsedResponse(
                     raw_response=LLMResponse(
                         response="fallback response",
                         llm_config=mock_model,
                         cost_estimate=0.01,
                         latency_ms=0,
                         tokens_used=None
                     ),
                     parsed_content="fallback response",
                     red_flags_hit=[],
                     is_valid=True,
                     parse_error=None
                 ),
                 MDAPMetrics(
                     total_llm_calls=0,
                     voting_rounds=1,
                     red_flags_hit={},
                     valid_responses_per_round=[0],
                     winning_response_votes=0,
                     time_taken_ms=50,
                     estimated_llm_cost_usd=0.0,
                 )
             )):
            
            result_response, metrics = await engine._execute_with_fast_path(
                "test prompt",
                EnsembleConfig(models=[mock_model]),  # Non-empty config
                MDAPInput(
                    prompt="test",
                    role_name="test",
                    ensemble_config=None,
                    voting_k=3,
                    red_flag_config=None,
                    output_parser_schema=None,
                    fast_path_enabled=False,
                    client_request_id=None,
                    client_sub_step_id=None
                )
            )
            
            # Should fallback to normal voting
            assert result_response is not None
            assert metrics is not None
            assert metrics.total_llm_calls == 0

    @pytest.mark.asyncio
    async def test_execute_with_fast_path_triggered(self):
        """Test fast-path when it's triggered successfully."""
        engine = MDAPEngine()
        
        # Mock ensemble manager and components
        mock_model = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        
        with patch.object(engine.ensemble_manager, 'select_models_for_round', return_value=[mock_model]), \
             patch.object(engine.ensemble_manager, 'dispatch_ensemble_calls', return_value=[
                 LLMResponse(
                     response="fast response",
                     llm_config=mock_model,
                     cost_estimate=0.05,
                     latency_ms=0,
                     tokens_used=None
                 )
             ]), \
             patch.object(engine.red_flag_engine, 'apply_rules', return_value=[]), \
             patch.object(engine.output_parser, 'parse_output', return_value=("fast response", None)), \
             patch.object(engine.fast_path_controller, 'check_fast_path') as mock_fast_path:
            
            # Mock fast path to return a winner
            mock_winner = ParsedResponse(
                raw_response=LLMResponse(
                    response="fast response",
                    llm_config=mock_model,
                    cost_estimate=0.05,
                    latency_ms=0,
                    tokens_used=None
                ),
                parsed_content="fast response",
                red_flags_hit=[],
                is_valid=True,
                parse_error=None
            )
            mock_fast_path.return_value = mock_winner
            
            result_response, metrics = await engine._execute_with_fast_path(
                "test prompt",
                EnsembleConfig(models=[mock_model]),
                MDAPInput(
                    prompt="test",
                    role_name="test",
                    ensemble_config=None,
                    voting_k=3,
                    red_flag_config=None,
                    output_parser_schema=None,
                    fast_path_enabled=True,
                    client_request_id=None,
                    client_sub_step_id=None
                )
            )
            
            assert result_response == mock_winner
            assert metrics.total_llm_calls == 1
            assert metrics.voting_rounds == 1
            assert metrics.time_taken_ms == 100  # Fast path estimated time

    @pytest.mark.asyncio
    async def test_execute_with_fast_path_not_triggered(self):
        """Test fast-path when it's not triggered and falls back to normal voting."""
        engine = MDAPEngine()
        
        mock_model = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        mock_winner = ParsedResponse(
            raw_response=LLMResponse(
                response="normal response",
                llm_config=mock_model,
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="normal response",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        with patch.object(engine.ensemble_manager, 'select_models_for_round', return_value=[mock_model]), \
             patch.object(engine.ensemble_manager, 'dispatch_ensemble_calls', return_value=[
                 LLMResponse(
                     response="test response",
                     llm_config=mock_model,
                     cost_estimate=0.05,
                     latency_ms=0,
                     tokens_used=None
                 )
             ]), \
             patch.object(engine.red_flag_engine, 'apply_rules', return_value=[]), \
             patch.object(engine.output_parser, 'parse_output', return_value=("test response", None)), \
             patch.object(engine.fast_path_controller, 'check_fast_path', return_value=None), \
             patch.object(engine.voting_mechanism, 'run_voting', return_value=(mock_winner, MDAPMetrics(
                 total_llm_calls=3,
                 voting_rounds=2,
                 red_flags_hit={},
                 valid_responses_per_round=[2, 1],
                 winning_response_votes=2,
                 time_taken_ms=200,
                 estimated_llm_cost_usd=0.15,
             ))):
            
            result_response, metrics = await engine._execute_with_fast_path(
                "test prompt",
                EnsembleConfig(models=[mock_model]),
                MDAPInput(
                    prompt="test",
                    role_name="test",
                    ensemble_config=None,
                    voting_k=3,
                    red_flag_config=None,
                    output_parser_schema=None,
                    fast_path_enabled=True,
                    client_request_id=None,
                    client_sub_step_id=None
                )
            )
            
            assert result_response == mock_winner
            assert metrics.total_llm_calls == 3


class TestMainExecution:
    """Test main execute_llm_role functionality."""

    @pytest.mark.asyncio
    async def test_execute_llm_role_success(self, mock_tracer):
        """Test successful execution of LLM role."""
        engine = MDAPEngine()
        
        mock_model = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        mock_winner = ParsedResponse(
            raw_response=LLMResponse(
                response="success response",
                llm_config=mock_model,
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="success response",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        with patch.object(engine.voting_mechanism, 'run_voting', return_value=(mock_winner, MDAPMetrics(
            total_llm_calls=3,
            voting_rounds=2,
            red_flags_hit={},
            valid_responses_per_round=[2, 1],
            winning_response_votes=2,
            time_taken_ms=200,
            estimated_llm_cost_usd=0.15,
        ))):
            
            mdap_input = MDAPInput(
                prompt="test prompt",
                role_name="test_role",
                ensemble_config=None,
                voting_k=2,
                red_flag_config=None,
                output_parser_schema=None,
                fast_path_enabled=False,
                client_request_id=None,
                client_sub_step_id=None
            )
            
            result = await engine.execute_llm_role(mdap_input)
            
            assert isinstance(result, MDAPOutput)
            assert result.final_response == "success response"
            assert result.confidence_score > 0.0
            assert result.error_message is None
            assert result.mdap_metrics.total_llm_calls == 3

    @pytest.mark.asyncio
    async def test_execute_llm_role_with_fast_path(self, mock_tracer):
        """Test execution with fast-path enabled."""
        engine = MDAPEngine()
        
        mock_model = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        mock_winner = ParsedResponse(
            raw_response=LLMResponse(
                response="fast response",
                llm_config=mock_model,
                cost_estimate=0.02,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="fast response",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        with patch.object(engine, '_execute_with_fast_path', return_value=(mock_winner, MDAPMetrics(
            total_llm_calls=1,
            voting_rounds=1,
            red_flags_hit={},
            valid_responses_per_round=[1],
            winning_response_votes=1,
            time_taken_ms=50,
            estimated_llm_cost_usd=0.02,
        ))):
            
            mdap_input = MDAPInput(
                prompt="test prompt",
                role_name="test_role",
                ensemble_config=None,
                voting_k=3,
                red_flag_config=None,
                output_parser_schema=None,
                fast_path_enabled=True,
                client_request_id=None,
                client_sub_step_id=None
            )
            
            result = await engine.execute_llm_role(mdap_input)
            
            assert result.final_response == "fast response"
            assert result.confidence_score > 0.0

    @pytest.mark.asyncio
    async def test_execute_llm_role_exception_handling(self, mock_tracer):
        """Test exception handling in execution."""
        engine = MDAPEngine()
        
        with patch.object(engine.voting_mechanism, 'run_voting', side_effect=Exception("Test error")):
            
            mdap_input = MDAPInput(
                prompt="test prompt",
                role_name="test_role",
                ensemble_config=None,
                voting_k=3,
                red_flag_config=None,
                output_parser_schema=None,
                fast_path_enabled=False,
                client_request_id=None,
                client_sub_step_id=None
            )
            
            result = await engine.execute_llm_role(mdap_input)
            
            assert isinstance(result, MDAPOutput)
            assert result.final_response == ""
            assert result.confidence_score == 0.0
            assert result.error_message is not None
            assert "Test error" in result.error_message
            assert result.mdap_metrics.total_llm_calls == 0
            assert result.mdap_metrics.voting_rounds == 0


class TestFinalResponsePreparation:
    """Test final response preparation."""

    def test_prepare_string_response(self):
        """Test preparation of string response."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test response",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="final response",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        result = engine._prepare_final_response(parsed_response)
        assert result == "final response"

    def test_prepare_dict_response(self):
        """Test preparation of dictionary response."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content={"key1": "value1", "key2": "value2"},
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        result = engine._prepare_final_response(parsed_response)
        # Should be JSON formatted
        assert "key1" in result and "key2" in result

    def test_prepare_other_response(self):
        """Test preparation of other response types."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        parsed_response = ParsedResponse(
            raw_response=LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.0,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="42",  # Must be string or dict, not int
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        result = engine._prepare_final_response(parsed_response)
        assert result == "42"


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_estimate_cost_empty_list(self):
        """Test cost estimation with empty list."""
        engine = MDAPEngine()
        
        result = engine._estimate_cost([])
        assert result == 0.0

    def test_estimate_cost_single_response(self):
        """Test cost estimation with single response."""
        engine = MDAPEngine()
        
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        responses = [
            LLMResponse(
                response="test",
                llm_config=llm_config,
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            )
        ]
        
        result = engine._estimate_cost(responses)
        assert result == 0.05

    def test_estimate_cost_multiple_responses(self):
        """Test cost estimation with multiple responses."""
        engine = MDAPEngine()
        
        llm_config1 = LLMConfig(
            provider="openai",
            model="gpt-4o",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        llm_config2 = LLMConfig(
            provider="anthropic",
            model="claude-3",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        llm_config3 = LLMConfig(
            provider="openai",
            model="gpt-3.5",
            api_key_env_var=None,
            base_url=None,
            temperature=0.1,
            top_p=1.0,
            max_tokens=None,
            stop_sequences=None,
            extra_params=None
        )
        
        responses = [
            LLMResponse(
                response="test1",
                llm_config=llm_config1,
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            LLMResponse(
                response="test2",
                llm_config=llm_config2,
                cost_estimate=0.08,
                latency_ms=0,
                tokens_used=None
            ),
            LLMResponse(
                response="test3",
                llm_config=llm_config3,
                cost_estimate=0.02,
                latency_ms=0,
                tokens_used=None
            ),
        ]
        
        result = engine._estimate_cost(responses)
        assert result == 0.15