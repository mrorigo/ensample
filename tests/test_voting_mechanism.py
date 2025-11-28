"""Comprehensive tests for VotingMechanism - core MDAP voting algorithm."""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List

from src.mdapflow_mcp.voting_mechanism import VotingMechanism
from src.mdapflow_mcp.models import (
    EnsembleConfig,
    LLMConfig,
    LLMResponse,
    MDAPInput,
    MDAPMetrics,
    ParsedResponse,
)
from src.mdapflow_mcp.ensemble_manager import EnsembleManager
from src.mdapflow_mcp.red_flagging_engine import RedFlaggingEngine
from src.mdapflow_mcp.output_parser import OutputParser
from src.mdapflow_mcp.exceptions import VotingConvergenceError


class TestVotingMechanismInitialization:
    """Test VotingMechanism initialization and dependency injection."""

    def test_initialization(self):
        """Test proper initialization with required dependencies."""
        ensemble_manager = MagicMock(spec=EnsembleManager)
        red_flag_engine = MagicMock(spec=RedFlaggingEngine)
        output_parser = MagicMock(spec=OutputParser)
        
        voting_mechanism = VotingMechanism(
            ensemble_manager=ensemble_manager,
            red_flag_engine=red_flag_engine,
            output_parser=output_parser
        )
        
        assert voting_mechanism.ensemble_manager == ensemble_manager
        assert voting_mechanism.red_flag_engine == red_flag_engine
        assert voting_mechanism.output_parser == output_parser


class TestResponseNormalization:
    """Test response normalization for voting comparison."""

    def test_normalize_string_response(self):
        """Test normalization of string responses."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        result = voting_mechanism._normalize_response_for_voting("  Hello World  ")
        assert result == "Hello World"

    def test_normalize_dict_response(self):
        """Test normalization of dictionary responses."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_dict = {"key1": "value1", "key2": "value2"}
        result = voting_mechanism._normalize_response_for_voting(response_dict)
        
        # Should be JSON with sorted keys
        assert "key1" in result and "key2" in result
        assert ":" in result and "," in result

    def test_normalize_dict_with_sorting(self):
        """Test that dictionary responses are sorted consistently."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        # Test that same content produces same key regardless of order
        dict1 = {"z": 1, "a": 2}
        dict2 = {"a": 2, "z": 1}
        
        result1 = voting_mechanism._normalize_response_for_voting(dict1)
        result2 = voting_mechanism._normalize_response_for_voting(dict2)
        
        assert result1 == result2

    def test_normalize_other_response(self):
        """Test normalization of other response types."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        result = voting_mechanism._normalize_response_for_voting(123)
        assert result == "123"

    def test_normalize_dict_with_non_serializable(self):
        """Test handling of non-serializable dictionary values."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        # Create a response with a lambda function (non-serializable)
        response_dict = {"key1": "value1", "key2": lambda x: x}
        result = voting_mechanism._normalize_response_for_voting(response_dict)
        
        # Should fallback to str() representation
        assert "key1" in result


class TestWinnerFinding:
    """Test winner identification logic."""

    def test_find_winner_single_response(self):
        """Test finding winner with single response type."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response1 = ParsedResponse(
            raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
            parsed_content="test1",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        response_votes = {
            "test1": [response1]
        }
        
        winner_response, winner_votes = voting_mechanism._find_winner(response_votes)
        
        assert winner_response == response1
        assert winner_votes == 1

    def test_find_winner_multiple_responses(self):
        """Test finding winner with multiple response types."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response1 = ParsedResponse(
            raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
            parsed_content="winner",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        response2 = ParsedResponse(
            raw_response=LLMResponse(response="test2", llm_config=LLMConfig(provider="anthropic", model="claude-3"), cost_estimate=0.06),
            parsed_content="loser",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        response3 = ParsedResponse(
            raw_response=LLMResponse(response="test3", llm_config=LLMConfig(provider="openai", model="gpt-3.5"), cost_estimate=0.03),
            parsed_content="winner",  # Same as response1
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        response_votes = {
            "winner": [response1, response3],  # 2 votes
            "loser": [response2]               # 1 vote
        }
        
        winner_response, winner_votes = voting_mechanism._find_winner(response_votes)
        
        assert winner_response == response1  # First occurrence of winner
        assert winner_votes == 2

    def test_find_winner_empty_responses(self):
        """Test finding winner with no responses."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_votes = {}
        
        winner_response, winner_votes = voting_mechanism._find_winner(response_votes)
        
        assert winner_response is None
        assert winner_votes == 0


class TestConvergenceChecking:
    """Test convergence detection logic."""

    def test_convergence_greedy_mode(self):
        """Test convergence in greedy mode (k=0)."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_votes = {
            "response1": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
                    parsed_content="response1",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ]
        }
        
        result = voting_mechanism._has_achieved_convergence(1, response_votes, k=0)
        
        assert result is True  # Any valid response wins in greedy mode

    def test_convergence_majority_mode(self):
        """Test convergence in majority mode (k=1)."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_votes = {
            "winner": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                ),
                ParsedResponse(
                    raw_response=LLMResponse(response="test2", llm_config=LLMConfig(provider="anthropic", model="claude-3"), cost_estimate=0.06),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ],
            "loser": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test3", llm_config=LLMConfig(provider="openai", model="gpt-3.5"), cost_estimate=0.03),
                    parsed_content="loser",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ]
        }
        
        result = voting_mechanism._has_achieved_convergence(2, response_votes, k=1)
        
        assert result is True  # 2 > 3/2 = 1.5

    def test_convergence_majority_not_achieved(self):
        """Test convergence when majority not achieved."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_votes = {
            "response1": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
                    parsed_content="response1",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ],
            "response2": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test2", llm_config=LLMConfig(provider="anthropic", model="claude-3"), cost_estimate=0.06),
                    parsed_content="response2",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ]
        }
        
        result = voting_mechanism._has_achieved_convergence(1, response_votes, k=1)
        
        assert result is False  # 1 == 2/2 = 1, not >

    def test_convergence_k_advantage_mode(self):
        """Test convergence in k-advantage mode (k>1)."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_votes = {
            "winner": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                ),
                ParsedResponse(
                    raw_response=LLMResponse(response="test2", llm_config=LLMConfig(provider="anthropic", model="claude-3"), cost_estimate=0.06),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                ),
                ParsedResponse(
                    raw_response=LLMResponse(response="test3", llm_config=LLMConfig(provider="openai", model="gpt-3.5"), cost_estimate=0.03),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ],
            "loser": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test4", llm_config=LLMConfig(provider="openai", model="gpt-4"), cost_estimate=0.04),
                    parsed_content="loser",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ]
        }
        
        result = voting_mechanism._has_achieved_convergence(3, response_votes, k=2)
        
        assert result is True  # 3 >= 1 + 2

    def test_convergence_k_advantage_not_achieved(self):
        """Test convergence when k-advantage not achieved."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        response_votes = {
            "winner": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test1", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                ),
                ParsedResponse(
                    raw_response=LLMResponse(response="test2", llm_config=LLMConfig(provider="anthropic", model="claude-3"), cost_estimate=0.06),
                    parsed_content="winner",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ],
            "loser": [
                ParsedResponse(
                    raw_response=LLMResponse(response="test3", llm_config=LLMConfig(provider="openai", model="gpt-3.5"), cost_estimate=0.03),
                    parsed_content="loser",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ]
        }
        
        result = voting_mechanism._has_achieved_convergence(2, response_votes, k=2)
        
        assert result is False  # 2 < 1 + 2


class TestCostEstimation:
    """Test cost estimation functionality."""

    def test_estimate_cost_empty_list(self):
        """Test cost estimation with empty list."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        result = voting_mechanism._estimate_cost([])
        assert result == 0.0

    def test_estimate_cost_single_response(self):
        """Test cost estimation with single response."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        llm_config = LLMConfig(provider="openai", model="gpt-4o")
        responses = [
            LLMResponse(response="test", llm_config=llm_config, cost_estimate=0.05)
        ]
        
        result = voting_mechanism._estimate_cost(responses)
        assert result == 0.05

    def test_estimate_cost_multiple_responses(self):
        """Test cost estimation with multiple responses."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        llm_config1 = LLMConfig(provider="openai", model="gpt-4o")
        llm_config2 = LLMConfig(provider="anthropic", model="claude-3")
        llm_config3 = LLMConfig(provider="openai", model="gpt-3.5")
        
        responses = [
            LLMResponse(response="test1", llm_config=llm_config1, cost_estimate=0.05),
            LLMResponse(response="test2", llm_config=llm_config2, cost_estimate=0.08),
            LLMResponse(response="test3", llm_config=llm_config3, cost_estimate=0.02),
        ]
        
        result = voting_mechanism._estimate_cost(responses)
        assert result == 0.15


class TestRunVoting:
    """Test the main voting process."""

    @pytest.mark.asyncio
    async def test_run_voting_convergence_first_round(self):
        """Test voting converges in first round."""
        ensemble_manager = MagicMock()
        red_flag_engine = MagicMock()
        output_parser = MagicMock()
        
        # Mock ensemble manager - make dispatch_ensemble_calls async
        ensemble_manager.select_models_for_round.return_value = [LLMConfig(provider="openai", model="gpt-4o")]
        ensemble_manager.dispatch_ensemble_calls = AsyncMock(return_value=[
            LLMResponse(response="winner", llm_config=LLMConfig(provider="openai", model="gpt-4o"), cost_estimate=0.05)
        ])
        
        # Mock red flag engine
        red_flag_engine.apply_rules.return_value = []  # No red flags
        
        # Mock output parser - make it synchronous (NOT async)
        output_parser.parse_output.return_value = ("winner", None)  # Successful parse
        
        voting_mechanism = VotingMechanism(ensemble_manager, red_flag_engine, output_parser)
        
        mdap_input = MDAPInput(
            prompt="test prompt",
            role_name="test_role",
            voting_k=0  # Greedy mode
        )
        
        result_response, metrics = await voting_mechanism.run_voting(
            "test prompt",
            EnsembleConfig(models=[LLMConfig(provider="openai", model="gpt-4o")]),
            mdap_input
        )
        
        assert result_response is not None
        assert result_response.parsed_content == "winner"
        assert metrics.total_llm_calls == 1
        assert metrics.voting_rounds == 1
        assert metrics.winning_response_votes == 1