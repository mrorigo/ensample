"""Ensemble Manager for parallel LLM calls."""

from __future__ import annotations

import asyncio
import random

from .llm_provider import LLMProviderInterface
from .models import EnsembleConfig, LLMConfig, LLMResponse
from .observability import LOGGER


class EnsembleManager:
    """Manages ensemble of LLM models for parallel execution."""

    def __init__(self, provider_interface: LLMProviderInterface) -> None:
        self.provider_interface = provider_interface
        self._call_history: list[LLMConfig] = []

    async def dispatch_ensemble_calls(
        self,
        prompt: str,
        ensemble_config: EnsembleConfig,
        num_calls_per_model: int = 1,
    ) -> list[LLMResponse]:
        """Dispatch parallel LLM calls across the ensemble.

        Args:
            prompt: The prompt to send to all models
            ensemble_config: Configuration of models to use
            num_calls_per_model: Number of calls to make per model

        Returns:
            List of LLM responses from all calls
        """
        LOGGER.info("Dispatching ensemble calls", extra={
            "num_models": len(ensemble_config.models),
            "num_calls_per_model": num_calls_per_model,
            "total_calls": len(ensemble_config.models) * num_calls_per_model
        })

        # Create task list for parallel execution
        tasks = []
        call_configs = []

        for model_config in ensemble_config.models:
            for _ in range(num_calls_per_model):
                client = self.provider_interface.get_client(model_config)
                task = asyncio.create_task(client.generate(prompt))
                tasks.append(task)
                call_configs.append(model_config)

        # Execute all calls in parallel
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                LOGGER.error("Ensemble call failed", extra={
                    "model": call_configs[i].model,
                    "provider": call_configs[i].provider,
                    "error": str(response)
                })
                continue

            valid_responses.append(response)
            self._call_history.append(call_configs[i])

        LOGGER.info("Ensemble calls completed", extra={
            "successful_calls": len(valid_responses),
            "failed_calls": len(responses) - len(valid_responses)
        })

        return valid_responses

    def select_models_for_round(
        self,
        ensemble_config: EnsembleConfig,
        max_calls: int,
    ) -> list[LLMConfig]:
        """Select models for a voting round, ensuring diversity.

        Args:
            ensemble_config: Full ensemble configuration
            max_calls: Maximum number of calls to make

        Returns:
            List of model configurations to use in this round
        """
        if len(ensemble_config.models) <= max_calls:
            return ensemble_config.models

        # Prioritize models that haven't been called recently
        recent_models = set(self._call_history[-max_calls:])
        available_models = [
            config for config in ensemble_config.models
            if config not in recent_models
        ]

        if len(available_models) >= max_calls:
            # We have enough unused models
            selected = random.sample(available_models, max_calls)
        else:
            # Use all available unused models and fill rest with random selection
            selected = available_models.copy()
            remaining_needed = max_calls - len(selected)
            remaining_models = [
                config for config in ensemble_config.models
                if config not in selected
            ]
            selected.extend(random.sample(remaining_models, remaining_needed))

        return selected

    def get_ensemble_stats(self) -> dict:
        """Get statistics about ensemble usage."""
        from collections import Counter

        model_counts = Counter(
            f"{config.provider}/{config.model}"
            for config in self._call_history
        )

        return {
            "total_calls": len(self._call_history),
            "unique_models_used": len(model_counts),
            "model_usage": dict(model_counts),
        }
