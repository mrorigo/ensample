"""Voting Mechanism implementing 'first-to-ahead-by-k' algorithm."""

from __future__ import annotations

import time

from .ensemble_manager import EnsembleManager
from .exceptions import VotingConvergenceError
from .models import EnsembleConfig, LLMResponse, MDAPInput, MDAPMetrics, ParsedResponse
from .observability import LOGGER
from .output_parser import OutputParser
from .red_flagging_engine import RedFlaggingEngine


class VotingMechanism:
    """Implements the 'first-to-ahead-by-k' voting algorithm."""

    def __init__(
        self,
        ensemble_manager: EnsembleManager,
        red_flag_engine: RedFlaggingEngine,
        output_parser: OutputParser,
    ) -> None:
        self.ensemble_manager = ensemble_manager
        self.red_flag_engine = red_flag_engine
        self.output_parser = output_parser

    async def run_voting(
        self,
        prompt: str,
        ensemble_config: EnsembleConfig,
        mdap_input: MDAPInput,
        max_rounds: int = 20,
    ) -> tuple[ParsedResponse | None, MDAPMetrics]:
        """Run the MDAP voting process until convergence.

        Args:
            prompt: The prompt to send to LLMs
            ensemble_config: Configuration of models to use
            mdap_input: MDAP input parameters
            max_rounds: Maximum number of voting rounds

        Returns:
            Tuple of (winning_response, metrics)

        Raises:
            VotingConvergenceError: If voting doesn't converge within max rounds
        """
        start_time = time.perf_counter()
        voting_rounds = 0
        total_llm_calls = 0
        red_flags_hit: dict[str, int] = {}
        valid_responses_per_round: list[int] = []

        # Track responses and their vote counts
        response_votes: dict[str, list[ParsedResponse]] = {}

        LOGGER.info("Starting MDAP voting process", extra={
            "voting_k": mdap_input.voting_k,
            "fast_path_enabled": mdap_input.fast_path_enabled,
            "max_rounds": max_rounds
        })

        while voting_rounds < max_rounds:
            voting_rounds += 1

            # Select models for this round
            models_for_round = self.ensemble_manager.select_models_for_round(
                ensemble_config,
                len(ensemble_config.models)
            )

            LOGGER.info(f"Voting round {voting_rounds}", extra={
                "models_selected": len(models_for_round)
            })

            # Make LLM calls for this round
            llm_responses = await self.ensemble_manager.dispatch_ensemble_calls(
                prompt,
                EnsembleConfig(models=models_for_round),
                num_calls_per_model=1
            )

            total_llm_calls += len(llm_responses)
            round_valid_count = 0

            # Process each response
            for llm_response in llm_responses:
                # Apply red-flagging
                red_flags = self.red_flag_engine.apply_rules(
                    llm_response.response,
                    mdap_input.output_parser_schema
                )

                # Track red flags
                for flag in red_flags:
                    red_flags_hit[flag] = red_flags_hit.get(flag, 0) + 1

                # Skip responses that hit red flags
                if red_flags:
                    LOGGER.debug("Response filtered by red flags", extra={
                        "red_flags": red_flags,
                        "response_preview": llm_response.response[:100]
                    })
                    continue

                # Parse output
                parsed_content, parse_error = self.output_parser.parse_output(
                    llm_response.response,
                    mdap_input.output_parser_schema
                )

                if parse_error:
                    LOGGER.debug("Response failed parsing", extra={
                        "parse_error": parse_error,
                        "response_preview": llm_response.response[:100]
                    })
                    continue

                # Create parsed response - handle None case
                if parsed_content is None:
                    LOGGER.debug("Parsed content is None, skipping response")
                    continue

                parsed_response = ParsedResponse(
                    raw_response=llm_response,
                    parsed_content=parsed_content,
                    red_flags_hit=red_flags,
                    is_valid=True,
                    parse_error=parse_error
                )

                # Convert parsed content to string for voting comparison
                response_key = self._normalize_response_for_voting(parsed_content)

                if response_key not in response_votes:
                    response_votes[response_key] = []

                response_votes[response_key].append(parsed_response)
                round_valid_count += 1

            valid_responses_per_round.append(round_valid_count)

            LOGGER.info(f"Round {voting_rounds} completed", extra={
                "valid_responses": round_valid_count,
                "unique_responses": len(response_votes),
                "total_valid_responses": sum(len(responses) for responses in response_votes.values())
            })

            # Check for convergence
            if response_votes:
                winner_response, winner_votes = self._find_winner(response_votes)

                # Check if we've achieved k-vote advantage
                if self._has_achieved_convergence(winner_votes, response_votes, mdap_input.voting_k):
                    time_taken_ms = int((time.perf_counter() - start_time) * 1000)

                    metrics = MDAPMetrics(
                        total_llm_calls=total_llm_calls,
                        voting_rounds=voting_rounds,
                        red_flags_hit=red_flags_hit,
                        valid_responses_per_round=valid_responses_per_round,
                        winning_response_votes=winner_votes,
                        time_taken_ms=time_taken_ms,
                        estimated_llm_cost_usd=self._estimate_cost(llm_responses)
                    )

                    LOGGER.info("Voting converged", extra={
                        "winning_votes": winner_votes,
                        "total_rounds": voting_rounds,
                        "time_taken_ms": time_taken_ms
                    })

                    return winner_response, metrics

        # If we reach here, voting didn't converge
        time_taken_ms = int((time.perf_counter() - start_time) * 1000)

        # Select best effort response (most voted)
        if response_votes:
            winner_response, winner_votes = self._find_winner(response_votes)
        else:
            # No valid responses at all
            raise VotingConvergenceError(
                f"No valid responses received after {max_rounds} rounds"
            )

        metrics = MDAPMetrics(
            total_llm_calls=total_llm_calls,
            voting_rounds=voting_rounds,
            red_flags_hit=red_flags_hit,
            valid_responses_per_round=valid_responses_per_round,
            winning_response_votes=winner_votes,
            time_taken_ms=time_taken_ms,
            estimated_llm_cost_usd=self._estimate_cost([])
        )

        LOGGER.warning("Voting timed out, returning best effort", extra={
            "winning_votes": winner_votes,
            "max_rounds": max_rounds
        })

        return winner_response, metrics

    def _normalize_response_for_voting(self, parsed_content) -> str:
        """Convert parsed content to string for voting comparison."""
        if isinstance(parsed_content, str):
            return parsed_content.strip()
        elif isinstance(parsed_content, dict):
            # For structured responses, use a canonical string representation
            import json
            try:
                return json.dumps(parsed_content, sort_keys=True, separators=(',', ':'))
            except (TypeError, ValueError):
                return str(parsed_content)
        else:
            return str(parsed_content)

    def _find_winner(
        self,
        response_votes: dict[str, list[ParsedResponse]]
    ) -> tuple[ParsedResponse | None, int]:
        """Find the response with the most votes."""
        max_votes = 0
        winner_response = None

        for _response_key, responses in response_votes.items():
            vote_count = len(responses)
            if vote_count > max_votes:
                max_votes = vote_count
                winner_response = responses[0]  # Take first occurrence as winner

        return winner_response, max_votes

    def _has_achieved_convergence(
        self,
        winner_votes: int,
        response_votes: dict[str, list[ParsedResponse]],
        k: int
    ) -> bool:
        """Check if we've achieved k-vote advantage convergence."""
        if k == 0:
            # Greedy mode: any valid response wins
            return winner_votes > 0

        if k == 1:
            # Simple majority
            total_votes = sum(len(responses) for responses in response_votes.values())
            return winner_votes > total_votes / 2

        # First-to-ahead-by-k: winner must have k more votes than any other
        second_highest = 0
        for _response_key, responses in response_votes.items():
            if len(responses) != winner_votes:
                second_highest = max(second_highest, len(responses))

        return winner_votes >= second_highest + k

    def _estimate_cost(self, llm_responses: list[LLMResponse]) -> float:
        """Estimate total cost of LLM calls."""
        return sum(response.cost_estimate for response in llm_responses)
