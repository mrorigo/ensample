"""Comprehensive tests for FastPathController - performance optimization component."""

from src.mdapflow_mcp.fast_path_controller import FastPathController
from src.mdapflow_mcp.models import (
    MDAPInput,
    LLMConfig,
    ParsedResponse,
    LLMResponse
)

class TestFastPathControllerInitialization:
    """Test FastPathController initialization and configuration."""

    def test_initialization_default_threshold(self):
        """Test initialization with default early termination threshold."""
        controller = FastPathController()
        
        assert controller._early_termination_threshold == 0.8
        assert isinstance(controller, FastPathController)

    def test_initialization_custom_threshold(self):
        """Test initialization with custom early termination threshold."""
        controller = FastPathController()
        controller._early_termination_threshold = 0.9
        
        assert controller._early_termination_threshold == 0.9


class TestFastPathCheck:
    """Test main fast-path check functionality."""

    def test_check_fast_path_disabled(self):
        """Test fast-path when disabled in input."""
        controller = FastPathController()
        
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
        
        current_responses = {
            "response1": [
                ParsedResponse(
                    raw_response=LLMResponse(
                        response="test1",
                        llm_config=LLMConfig(
                            provider="openai",
                            model="gpt-4o",
                            api_key_env_var=None,
                            base_url=None,
                            temperature=0.1,
                            top_p=1.0,
                            max_tokens=None,
                            stop_sequences=None,
                            extra_params=None
                        ),
                        cost_estimate=0.05,
                        latency_ms=0,
                        tokens_used=None
                    ),
                    parsed_content="test1",
                    red_flags_hit=[],
                    is_valid=True,
                    parse_error=None
                )
            ]
        }
        
        result = controller.check_fast_path(mdap_input, current_responses, 1)
        
        assert result is None

    def test_check_fast_path_empty_responses(self):
        """Test fast-path with empty responses."""
        controller = FastPathController()
        
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
        
        result = controller.check_fast_path(mdap_input, {}, 1)
        
        assert result is None

    def test_check_fast_path_greedy_mode(self):
        """Test fast-path for k=0 (greedy mode)."""
        controller = FastPathController()
        
        mdap_input = MDAPInput(
            prompt="test prompt",
            role_name="test_role",
            ensemble_config=None,
            voting_k=0,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=True,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        response1 = ParsedResponse(
            raw_response=LLMResponse(
                response="response1",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-4o",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response1",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        current_responses = {
            "response1": [response1]
        }
        
        result = controller.check_fast_path(mdap_input, current_responses, 1)
        
        assert result == response1

    def test_check_fast_path_majority_mode(self):
        """Test fast-path for k=1 (majority mode)."""
        controller = FastPathController()
        
        mdap_input = MDAPInput(
            prompt="test prompt",
            role_name="test_role",
            ensemble_config=None,
            voting_k=1,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=True,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        response1 = ParsedResponse(
            raw_response=LLMResponse(
                response="response1",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-4o",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response1",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        response2 = ParsedResponse(
            raw_response=LLMResponse(
                response="response2",
                llm_config=LLMConfig(
                    provider="anthropic",
                    model="claude-3",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.06,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response2",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        response1_dup = ParsedResponse(
            raw_response=LLMResponse(
                response="response1_dup",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-3.5",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.03,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response1",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        current_responses = {
            "response1": [response1, response1_dup],  # 2 votes
            "response2": [response2]  # 1 vote
        }
        
        result = controller.check_fast_path(mdap_input, current_responses, 1)
        
        assert result == response1  # Should return first response1 (majority)

    def test_check_fast_path_k_advantage_mode(self):
        """Test fast-path for k>1 (k-advantage mode)."""
        controller = FastPathController()
        
        mdap_input = MDAPInput(
            prompt="test prompt",
            role_name="test_role",
            ensemble_config=None,
            voting_k=2,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=True,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        response1 = ParsedResponse(
            raw_response=LLMResponse(
                response="response1",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-4o",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response1",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        response2 = ParsedResponse(
            raw_response=LLMResponse(
                response="response2",
                llm_config=LLMConfig(
                    provider="anthropic",
                    model="claude-3",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.06,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response2",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        # Need 3 votes for response1 vs 1 vote for response2 to have k=2 advantage
        current_responses = {
            "response1": [response1, response1, response1],  # 3 votes
            "response2": [response2]  # 1 vote (advantage = 3-1 = 2)
        }
        
        result = controller.check_fast_path(mdap_input, current_responses, 1)
        
        assert result == response1  # Should return first response1 (k=2 advantage)

    def test_check_fast_path_strong_consensus(self):
        """Test fast-path with strong consensus (80%+ agreement)."""
        controller = FastPathController()
        
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
        
        responses = []
        for i in range(5):  # 5 identical responses
            responses.append(ParsedResponse(
                raw_response=LLMResponse(
                    response=f"response{i}",
                    llm_config=LLMConfig(
                        provider="openai",
                        model="gpt-4o",
                        api_key_env_var=None,
                        base_url=None,
                        temperature=0.1,
                        top_p=1.0,
                        max_tokens=None,
                        stop_sequences=None,
                        extra_params=None
                    ),
                    cost_estimate=0.05,
                    latency_ms=0,
                    tokens_used=None
                ),
                parsed_content="dominant_response",
                red_flags_hit=[],
                is_valid=True,
                parse_error=None
            ))
        
        current_responses = {
            "dominant_response": responses  # All 5 votes for same response
        }
        
        result = controller.check_fast_path(mdap_input, current_responses, 1)
        
        assert result == responses[0]  # Should trigger on strong consensus

    def test_check_fast_path_no_consensus(self):
        """Test fast-path when no consensus is reached."""
        controller = FastPathController()
        
        mdap_input = MDAPInput(
            prompt="test prompt",
            role_name="test_role",
            ensemble_config=None,
            voting_k=1,
            red_flag_config=None,
            output_parser_schema=None,
            fast_path_enabled=True,
            client_request_id=None,
            client_sub_step_id=None
        )
        
        response1 = ParsedResponse(
            raw_response=LLMResponse(
                response="response1",
                llm_config=LLMConfig(
                    provider="openai",
                    model="gpt-4o",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.05,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response1",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        response2 = ParsedResponse(
            raw_response=LLMResponse(
                response="response2",
                llm_config=LLMConfig(
                    provider="anthropic",
                    model="claude-3",
                    api_key_env_var=None,
                    base_url=None,
                    temperature=0.1,
                    top_p=1.0,
                    max_tokens=None,
                    stop_sequences=None,
                    extra_params=None
                ),
                cost_estimate=0.06,
                latency_ms=0,
                tokens_used=None
            ),
            parsed_content="response2",
            red_flags_hit=[],
            is_valid=True,
            parse_error=None
        )
        
        current_responses = {
            "response1": [response1],  # 1 vote each
            "response2": [response2]   # 1 vote each
        }
        
        result = controller.check_fast_path(mdap_input, current_responses, 1)
        
        assert result is None  # No majority reached