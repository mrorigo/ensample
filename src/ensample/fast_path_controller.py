"""Fast-Path Controller for early termination of MDAP execution."""

from __future__ import annotations

from .models import MDAPInput, ParsedResponse
from .observability import LOGGER


class FastPathController:
    """Controls early termination of MDAP voting when confidence is sufficient."""

    def __init__(self) -> None:
        self._early_termination_threshold = 0.8  # 80% consensus threshold

    def check_fast_path(
        self,
        mdap_input: MDAPInput,
        current_responses: dict[str, list[ParsedResponse]],
        current_round: int,
    ) -> ParsedResponse | None:
        """Check if fast-path termination criteria are met.

        Args:
            mdap_input: MDAP input parameters
            current_responses: Current response votes
            current_round: Current voting round number

        Returns:
            Winning response if fast-path criteria met, None otherwise
        """
        if not mdap_input.fast_path_enabled:
            return None

        if not current_responses:
            return None

        LOGGER.debug("Checking fast-path criteria", extra={
            "voting_k": mdap_input.voting_k,
            "current_round": current_round,
            "response_count": len(current_responses)
        })

        # Fast-path logic based on voting_k value
        if mdap_input.voting_k == 0:
            return self._check_greedy_fast_path(current_responses)
        elif mdap_input.voting_k == 1:
            return self._check_majority_fast_path(current_responses)
        else:
            return self._check_k_advantage_fast_path(current_responses, mdap_input.voting_k)

    def _check_greedy_fast_path(
        self,
        current_responses: dict[str, list[ParsedResponse]]
    ) -> ParsedResponse | None:
        """Fast-path for k=0 (greedy mode): return first valid response."""
        # In greedy mode with fast-path, we return the first valid response
        for _response_key, responses in current_responses.items():
            if responses:
                LOGGER.info("Fast-path triggered (greedy mode)", extra={
                    "response_preview": responses[0].parsed_content[:100] if isinstance(responses[0].parsed_content, str) else str(responses[0].parsed_content)[:100]
                })
                return responses[0]

        return None

    def _check_majority_fast_path(
        self,
        current_responses: dict[str, list[ParsedResponse]]
    ) -> ParsedResponse | None:
        """Fast-path for k=1 (simple majority)."""
        if len(current_responses) == 1:
            # Only one unique response type, return it
            response_list = list(current_responses.values())[0]
            LOGGER.info("Fast-path triggered (unanimous response)", extra={
                "response_count": len(response_list)
            })
            return response_list[0]

        # Check if any response has majority
        total_responses = sum(len(responses) for responses in current_responses.values())

        for _response_key, responses in current_responses.items():
            if len(responses) > total_responses / 2:
                LOGGER.info("Fast-path triggered (majority reached)", extra={
                    "response_votes": len(responses),
                    "total_responses": total_responses
                })
                return responses[0]

        return None

    def _check_k_advantage_fast_path(
        self,
        current_responses: dict[str, list[ParsedResponse]],
        k: int
    ) -> ParsedResponse | None:
        """Fast-path for k>1 (k-advantage mode)."""
        if len(current_responses) <= 1:
            # Only one response type, return it
            response_list = list(current_responses.values())[0]
            LOGGER.info("Fast-path triggered (single response type)", extra={
                "response_count": len(response_list)
            })
            return response_list[0]

        # Sort responses by vote count
        sorted_responses = sorted(
            current_responses.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        if len(sorted_responses) < 2:
            return None

        winner_votes = len(sorted_responses[0][1])
        second_place_votes = len(sorted_responses[1][1])

        # Check for early k-advantage
        if winner_votes >= second_place_votes + k:
            LOGGER.info("Fast-path triggered (k-advantage early)", extra={
                "winner_votes": winner_votes,
                "second_place_votes": second_place_votes,
                "k_advantage": k
            })
            return sorted_responses[0][1][0]

        # Check for strong consensus (80%+ agreement)
        total_votes = sum(len(responses) for responses in current_responses.values())
        consensus_ratio = winner_votes / total_votes if total_votes > 0 else 0

        if consensus_ratio >= self._early_termination_threshold and total_votes >= 3:
            LOGGER.info("Fast-path triggered (strong consensus)", extra={
                "consensus_ratio": consensus_ratio,
                "winner_votes": winner_votes,
                "total_votes": total_votes
            })
            return sorted_responses[0][1][0]

        return None

    def calculate_fast_path_confidence(
        self,
        winner_response: ParsedResponse,
        total_responses: int,
        is_fast_path: bool,
    ) -> float:
        """Calculate confidence score for fast-path results.

        Args:
            winner_response: The winning response
            total_responses: Total valid responses received
            is_fast_path: Whether this was a fast-path termination

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not is_fast_path:
            # Regular voting confidence based on vote share
            winner_votes = len([r for r in winner_response.raw_response.model_dump() if r])  # This is a simplification
            return min(1.0, winner_votes / max(1, total_responses))

        # For fast-path, return high confidence due to early termination
        if total_responses == 1:
            return 1.0  # Single response gets full confidence
        else:
            # High confidence but not perfect due to limited sampling
            return 0.95

    def should_continue_voting(
        self,
        mdap_input: MDAPInput,
        current_round: int,
        max_rounds: int,
        current_responses: dict[str, list[ParsedResponse]],
    ) -> bool:
        """Determine if voting should continue.

        Args:
            mdap_input: MDAP input parameters
            current_round: Current round number
            max_rounds: Maximum rounds allowed
            current_responses: Current response state

        Returns:
            True if voting should continue, False otherwise
        """
        # Stop if we've reached max rounds
        if current_round >= max_rounds:
            return False

        # Stop if fast-path criteria are met
        if self.check_fast_path(mdap_input, current_responses, current_round):
            return False

        # Continue voting
        return True
