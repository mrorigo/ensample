"""MDAP Engine - main orchestration for MDAP execution."""

from __future__ import annotations

import time

from .config import (
    Settings,
    create_default_ensemble_config,
    create_default_red_flag_config,
    load_ensemble_config,
    load_red_flag_config,
)
from .ensemble_manager import EnsembleManager
from .fast_path_controller import FastPathController
from .llm_provider import LLMProviderInterface
from .metrics import metrics_collector
from .models import (
    EnsembleConfig,
    LLMResponse,
    MDAPInput,
    MDAPMetrics,
    MDAPOutput,
    ParsedResponse,
    RedFlagConfig,
)
from .observability import LOGGER, TRACER, _with_trace
from .output_parser import OutputParser
from .red_flagging_engine import RedFlaggingEngine
from .voting_mechanism import VotingMechanism


class MDAPEngine:
    """Main MDAP execution engine."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.provider_interface = LLMProviderInterface()
        self.ensemble_manager = EnsembleManager(self.provider_interface)
        self.red_flag_engine = RedFlaggingEngine()
        self.output_parser = OutputParser()
        self.voting_mechanism = VotingMechanism(
            self.ensemble_manager,
            self.red_flag_engine,
            self.output_parser,
        )
        self.fast_path_controller = FastPathController()

        # Load default configurations
        self._load_default_configs()

    def _load_default_configs(self) -> None:
        """Load default ensemble and red flag configurations."""
        try:
            self.default_ensemble_config = load_ensemble_config() or create_default_ensemble_config()
            self.default_red_flag_config = load_red_flag_config() or create_default_red_flag_config()

            # Update red flag engine with default config
            self.red_flag_engine.config = self.default_red_flag_config

            LOGGER.info("Default configurations loaded", extra={
                "ensemble_models": len(self.default_ensemble_config.models),
                "red_flag_rules": len(self.default_red_flag_config.rules)
            })

        except Exception as e:
            LOGGER.warning(f"Failed to load default configurations: {e}")
            # Create fallback defaults
            self.default_ensemble_config = create_default_ensemble_config()
            self.default_red_flag_config = create_default_red_flag_config()

    async def execute_llm_role(
        self,
        mdap_input: MDAPInput,
    ) -> MDAPOutput:
        """Execute MDAP process for a given role and prompt.

        Args:
            mdap_input: Complete MDAP input configuration

        Returns:
            MDAPOutput with final response and metrics
        """
        start_time = time.perf_counter()
        metrics_collector.record_execution_start()

        # Prepare tracing context
        with TRACER.start_as_current_span("mdap_execute_llm_role") as span:
            # Set span attributes
            span.set_attribute("mdapflow.role_name", mdap_input.role_name)
            span.set_attribute("mdapflow.voting_k", mdap_input.voting_k)
            span.set_attribute("mdapflow.fast_path_enabled", mdap_input.fast_path_enabled)
            if mdap_input.client_request_id:
                span.set_attribute("mdapflow.client_request_id", mdap_input.client_request_id)
            if mdap_input.client_sub_step_id:
                span.set_attribute("mdapflow.client_sub_step_id", mdap_input.client_sub_step_id)

            # Set up logging context
            extra = _with_trace({
                "role_name": mdap_input.role_name,
                "voting_k": mdap_input.voting_k,
                "fast_path": mdap_input.fast_path_enabled,
            })
            if mdap_input.client_request_id:
                extra["client_request_id"] = mdap_input.client_request_id
            if mdap_input.client_sub_step_id:
                extra["client_sub_step_id"] = mdap_input.client_sub_step_id

            LOGGER.info("Starting MDAP execution", extra=extra)

            try:
                # Validate and prepare configurations
                ensemble_config = self._prepare_ensemble_config(mdap_input)
                red_flag_config = self._prepare_red_flag_config(mdap_input)

                # Update engines with configurations
                self.red_flag_engine.config = red_flag_config

                # Execute MDAP voting process
                if mdap_input.fast_path_enabled:
                    result_response, metrics = await self._execute_with_fast_path(
                        mdap_input.prompt,
                        ensemble_config,
                        mdap_input,
                    )
                else:
                    result_response, metrics = await self.voting_mechanism.run_voting(
                        mdap_input.prompt,
                        ensemble_config,
                        mdap_input,
                        max_rounds=self.settings.MDAP_MAX_VOTING_ROUNDS,
                    )

                # Calculate final confidence score
                confidence_score = self._calculate_confidence_score(result_response, metrics)

                # Prepare final response content
                final_response = self._prepare_final_response(result_response)

                # Update span with results
                span.set_attribute("mdapflow.total_rounds", metrics.voting_rounds)
                span.set_attribute("mdapflow.total_llm_calls", metrics.total_llm_calls)
                span.set_attribute("mdapflow.winning_votes", metrics.winning_response_votes)
                span.set_attribute("mdapflow.confidence_score", confidence_score)

                output = MDAPOutput(
                    final_response=final_response,
                    confidence_score=confidence_score,
                    mdap_metrics=metrics,
                    error_message=None,
                )

                metrics_collector.record_execution_success(
                    latency_ms=int((time.perf_counter() - start_time) * 1000),
                    voting_rounds=metrics.voting_rounds,
                    llm_calls=metrics.total_llm_calls,
                    cost_usd=metrics.estimated_llm_cost_usd,
                    total_tokens=0,
                    role_name=mdap_input.role_name,
                )

                LOGGER.info("MDAP execution completed successfully", extra=_with_trace({
                    "role_name": mdap_input.role_name,
                    "confidence_score": confidence_score,
                    "total_rounds": metrics.voting_rounds,
                    "total_llm_calls": metrics.total_llm_calls,
                }))

                return output

            except Exception as e:
                error_msg = f"MDAP execution failed: {e}"
                LOGGER.error(error_msg, extra=_with_trace({
                    "role_name": mdap_input.role_name,
                    "error": str(e)
                }))

                # Update span with error
                span.record_exception(e)
                span.set_status("ERROR", error_msg)

                metrics_collector.record_execution_failure(
                    latency_ms=int((time.perf_counter() - start_time) * 1000)
                )

                # Return error response
                time_taken_ms = int((time.perf_counter() - start_time) * 1000)

                error_metrics = MDAPMetrics(
                    total_llm_calls=0,
                    voting_rounds=0,
                    red_flags_hit={},
                    valid_responses_per_round=[],
                    winning_response_votes=0,
                    time_taken_ms=time_taken_ms,
                    estimated_llm_cost_usd=0.0,
                )

                return MDAPOutput(
                    final_response="",
                    confidence_score=0.0,
                    mdap_metrics=error_metrics,
                    error_message=error_msg,
                )

    def _prepare_ensemble_config(self, mdap_input: MDAPInput) -> EnsembleConfig:
        """Prepare ensemble configuration for execution."""
        if mdap_input.ensemble_config:
            return mdap_input.ensemble_config
        else:
            return self.default_ensemble_config

    def _prepare_red_flag_config(self, mdap_input: MDAPInput) -> RedFlagConfig:
        """Prepare red flag configuration for execution."""
        if mdap_input.red_flag_config:
            return mdap_input.red_flag_config
        else:
            return self.default_red_flag_config

    async def _execute_with_fast_path(
        self,
        prompt: str,
        ensemble_config: EnsembleConfig,
        mdap_input: MDAPInput,
    ) -> tuple[ParsedResponse, MDAPMetrics]:
        """Execute MDAP with fast-path optimization."""
        # For fast-path, we can potentially use a smaller ensemble
        # or reduce the number of voting rounds

        # Start with a smaller initial round
        initial_models = self.ensemble_manager.select_models_for_round(
            ensemble_config,
            min(2, len(ensemble_config.models))  # Start with at most 2 models
        )

        if initial_models:
            initial_responses = await self.ensemble_manager.dispatch_ensemble_calls(
                prompt,
                EnsembleConfig(models=initial_models),
                num_calls_per_model=1
            )

            # Process responses and check for fast-path conditions
            current_responses = {}
            for response in initial_responses:
                red_flags = self.red_flag_engine.apply_rules(
                    response.response,
                    mdap_input.output_parser_schema
                )

                if not red_flags:
                    parsed_content, parse_error = self.output_parser.parse_output(
                        response.response,
                        mdap_input.output_parser_schema
                    )

                    if not parse_error:
                        parsed_response = ParsedResponse(
                            raw_response=response,
                            parsed_content=parsed_content,
                            red_flags_hit=red_flags,
                            is_valid=True,
                            parse_error=parse_error
                        )

                        response_key = self._normalize_response_for_voting(parsed_content)
                        if response_key not in current_responses:
                            current_responses[response_key] = []
                        current_responses[response_key].append(parsed_response)

            # Check fast-path conditions
            fast_path_winner = self.fast_path_controller.check_fast_path(
                mdap_input,
                current_responses,
                current_round=1,
            )

            if fast_path_winner:
                # Fast-path triggered, create metrics for this result
                time_taken_ms = 100  # Estimated fast-path time
                metrics = MDAPMetrics(
                    total_llm_calls=len(initial_responses),
                    voting_rounds=1,
                    red_flags_hit={},  # Would be populated in real implementation
                    valid_responses_per_round=[len(initial_responses)],
                    winning_response_votes=1,
                    time_taken_ms=time_taken_ms,
                    estimated_llm_cost_usd=self._estimate_cost(initial_responses),
                )
                return fast_path_winner, metrics

        # Fast-path not triggered, continue with normal voting
        return await self.voting_mechanism.run_voting(
            prompt,
            ensemble_config,
            mdap_input,
            max_rounds=self.settings.MDAP_MAX_VOTING_ROUNDS,
        )

    def _normalize_response_for_voting(self, parsed_content) -> str:
        """Normalize parsed content for voting comparison."""
        if isinstance(parsed_content, str):
            return parsed_content.strip()
        elif isinstance(parsed_content, dict):
            import json
            try:
                return json.dumps(parsed_content, sort_keys=True, separators=(',', ':'))
            except (TypeError, ValueError):
                return str(parsed_content)
        else:
            return str(parsed_content)

    def _calculate_confidence_score(
        self,
        result_response: ParsedResponse,
        metrics: MDAPMetrics,
    ) -> float:
        """Calculate confidence score based on voting results."""
        if metrics.voting_rounds == 0:
            return 0.0

        # Base confidence on vote share
        total_valid_responses = sum(metrics.valid_responses_per_round)
        if total_valid_responses == 0:
            return 0.0

        vote_confidence = metrics.winning_response_votes / total_valid_responses

        # Adjust based on number of rounds (more rounds = higher confidence)
        round_bonus = min(0.1, metrics.voting_rounds * 0.01)

        # Adjust based on fast-path usage
        fast_path_bonus = 0.05 if metrics.voting_rounds == 1 else 0.0

        confidence = vote_confidence + round_bonus + fast_path_bonus
        return min(1.0, confidence)

    def _prepare_final_response(self, result_response: ParsedResponse) -> str:
        """Prepare the final response content."""
        if isinstance(result_response.parsed_content, str):
            return result_response.parsed_content
        elif isinstance(result_response.parsed_content, dict):
            import json
            try:
                return json.dumps(result_response.parsed_content, indent=2)
            except (TypeError, ValueError):
                return str(result_response.parsed_content)
        else:
            return str(result_response.parsed_content)

    def _estimate_cost(self, llm_responses: list[LLMResponse]) -> float:
        """Estimate total cost of LLM calls."""
        return sum(response.cost_estimate for response in llm_responses)
