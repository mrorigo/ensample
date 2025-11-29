"""LLM Provider Interface using LiteLLM for wide provider coverage."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from litellm import acompletion

from .exceptions import LLMProviderError
from .models import LLMConfig, LLMResponse, TokenUsage
from .observability import (
    LOGGER, 
    log_llm_call_start, 
    log_llm_call_success, 
    log_llm_call_failure,
    create_llm_span,
    add_token_span_attributes
)
from .pricing_service import get_pricing_service


class BaseLLMClient:
    """Abstract base class for LLM clients."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError


class LiteLLMClient(BaseLLMClient):
    """LiteLLM-based client for multiple LLM providers."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._api_key = self._get_api_key()
        self._pricing_service = None  # Will be initialized when needed

    def _get_api_key(self) -> str | None:
        """Get API key from environment variable."""
        if self.config.api_key_env_var:
            return os.getenv(self.config.api_key_env_var)

        # Fallback to default environment variables
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "together": "TOGETHER_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "cohere": "COHERE_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

        default_var = env_var_map.get(self.config.provider.lower())
        return os.getenv(default_var) if default_var else None

    def _build_litellm_params(self, prompt: str) -> dict[str, Any]:
        """Build parameters for LiteLLM API call."""
        params = {
            "model": f"{self.config.provider}/{self.config.model}",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "stop": self.config.stop_sequences,
        }

        # Add API key if available
        if self._api_key:
            params["api_key"] = self._api_key

        # Add custom base URL if provided
        if self.config.base_url:
            params["base_url"] = self.config.base_url

        # Add any extra parameters
        if self.config.extra_params:
            params.update(self.config.extra_params)

        return params

    def _extract_token_usage(self, response: Any) -> TokenUsage | None:
        """Extract comprehensive token usage from LiteLLM response following official standards."""
        try:
            # Handle different usage object formats from various providers
            usage = getattr(response, 'usage', None)
            if not usage:
                return None

            # Initialize with defaults
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None

            # Common patterns across providers
            if hasattr(usage, 'prompt_tokens'):
                prompt_tokens = usage.prompt_tokens
            elif hasattr(usage, 'input_tokens'):
                prompt_tokens = usage.input_tokens
            elif hasattr(usage, 'context_tokens'):
                prompt_tokens = usage.context_tokens

            if hasattr(usage, 'completion_tokens'):
                completion_tokens = usage.completion_tokens
            elif hasattr(usage, 'output_tokens'):
                completion_tokens = usage.output_tokens
            elif hasattr(usage, 'generated_tokens'):
                completion_tokens = usage.generated_tokens

            if hasattr(usage, 'total_tokens'):
                total_tokens = usage.total_tokens
            elif prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            elif prompt_tokens is not None:
                total_tokens = prompt_tokens
            elif completion_tokens is not None:
                total_tokens = completion_tokens

            # Only return if we found at least some token information
            if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
                # Create TokenUsage object
                return TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated=False
                )

            return None

        except Exception as e:
            LOGGER.warning("Failed to extract token usage", extra={
                "provider": self.config.provider,
                "model": self.config.model,
                "error": str(e)
            })
            return None

    def _estimate_tokens(self, text: str, model: str) -> int:
        """Estimate tokens for text using model-specific logic."""
        # Simple estimation - in production, use provider-specific tokenizers
        # This is a fallback when actual token counts aren't available
        words = len(text.split())
        # Rough estimate: 1 token ≈ 0.75 words for most models
        return max(1, int(words * 1.3))

    def _extract_response_content(self, response: Any) -> tuple[str, TokenUsage | None]:
        """Extract content and detailed token usage from LiteLLM response."""
        try:
            # Handle different response formats from various providers
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                    content = choice.message.content or ""
                elif hasattr(choice, 'text'):
                    content = choice.text or ""
                else:
                    content = str(choice)
            elif hasattr(response, 'text') and response.text is not None:
                content = response.text
            elif hasattr(response, 'content') and response.content is not None:
                content = response.content
            else:
                # Convert to string as last resort
                content = str(response) if response else ""

            # Extract comprehensive token usage
            token_usage = self._extract_token_usage(response)

            # If no token usage data available, estimate
            if token_usage is None:
                prompt_tokens = self._estimate_tokens(
                    response.get('prompt', '') if hasattr(response, 'get') else '',
                    f"{self.config.provider}/{self.config.model}"
                )
                completion_tokens = self._estimate_tokens(content, f"{self.config.provider}/{self.config.model}")

                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    estimated=True
                )

            return content, token_usage

        except Exception as e:
            LOGGER.warning("Failed to extract response content", extra={
                "provider": self.config.provider,
                "model": self.config.model,
                "error": str(e)
            })
            # Fallback content extraction
            content = str(response) if response else ""
            # Create estimated token usage as fallback
            token_usage = TokenUsage(
                prompt_tokens=self._estimate_tokens('', f"{self.config.provider}/{self.config.model}"),
                completion_tokens=self._estimate_tokens(content, f"{self.config.provider}/{self.config.model}"),
                total_tokens=0,
                estimated=True
            )
            return content, token_usage

    async def _estimate_cost_with_source(self, model: str, token_usage: TokenUsage | None) -> tuple[float, str]:
        """Estimate cost using dynamic pricing service with caching, return cost and source."""
        if token_usage is None:
            return 0.0, "none"

        try:
            # Initialize pricing service if not already done
            if self._pricing_service is None:
                self._pricing_service = await get_pricing_service()

            # Extract provider and model from model string
            if "/" in model:
                provider, model_name = model.split("/", 1)
            else:
                provider = "unknown"
                model_name = model

            # Get dynamic pricing information
            pricing_info = await self._pricing_service.get_pricing(provider, model_name)

            # Calculate cost using dynamic pricing
            input_cost = (token_usage.prompt_tokens or 0) * pricing_info.input_rate
            output_cost = (token_usage.completion_tokens or 0) * pricing_info.output_rate

            total_cost = input_cost + output_cost

            LOGGER.debug("Dynamic cost calculation", extra={
                "model": model,
                "input_tokens": token_usage.prompt_tokens,
                "output_tokens": token_usage.completion_tokens,
                "input_rate": pricing_info.input_rate,
                "output_rate": pricing_info.output_rate,
                "total_cost": total_cost,
                "source": pricing_info.source
            })

            return total_cost, pricing_info.source

        except Exception as e:
            LOGGER.warning("Dynamic pricing calculation failed, using fallback", extra={
                "model": model,
                "error": str(e)
            })

            # Fallback to static calculation
            return self._fallback_cost_calculation_internal(model, token_usage)

    def _fallback_cost_calculation_internal(self, model: str, token_usage: TokenUsage | None) -> tuple[float, str]:
        """Internal fallback cost calculation using static pricing."""
        if token_usage is None:
            return 0.0, "fallback"

        # Static fallback pricing (matches the original implementation)
        static_pricing = {
            "openai/gpt-4o": {"input": 0.000005, "output": 0.000015},
            "openai/gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
            "openai/gpt-3.5-turbo": {"input": 0.0000005, "output": 0.0000015},
            "anthropic/claude-3-opus-20240229": {"input": 0.000015, "output": 0.000075},
            "anthropic/claude-3-sonnet-20240229": {"input": 0.000003, "output": 0.000015},
            "anthropic/claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125},
            "openrouter/mistralai/mixtral-8x7b-instruct-v0.1": {"input": 0.0000007, "output": 0.0000007},
            "openrouter/meta-llama/llama-2-70b-chat": {"input": 0.0000007, "output": 0.0000007},
            "togethercomputer/redpajama-instruct-3b-v1": {"input": 0.0000002, "output": 0.0000002},
            "cohere/command": {"input": 0.0000005, "output": 0.0000015},
            "google/palm-2-chat-bison": {"input": 0.0000005, "output": 0.0000005},
        }

        model_pricing = static_pricing.get(model.lower(), {"input": 0.000001, "output": 0.000001})

        # Calculate cost using static rates
        input_cost = (token_usage.prompt_tokens or 0) * model_pricing["input"]
        output_cost = (token_usage.completion_tokens or 0) * model_pricing["output"]

        return input_cost + output_cost, "static"

    def _fallback_cost_calculation(self, model: str, token_usage: TokenUsage | None) -> float:
        """Fallback cost calculation - maintains backward compatibility for tests."""
        cost, _ = self._fallback_cost_calculation_internal(model, token_usage)
        return cost

    async def _estimate_cost(self, model: str, token_usage: TokenUsage | None) -> float:
        """Estimate cost - maintains backward compatibility for tests."""
        cost, _ = await self._estimate_cost_with_source(model, token_usage)
        return cost

    async def generate(self, prompt: str) -> LLMResponse:
        """Generate response using LiteLLM with comprehensive token tracking and observability."""
        start_time = time.perf_counter()
        execution_id = f"{self.config.provider}-{self.config.model}-{int(start_time * 1000)}"

        # Estimate initial prompt tokens for observability
        estimated_prompt_tokens = self._estimate_tokens(prompt, f"{self.config.provider}/{self.config.model}")

        # Log LLM call start with observability
        log_llm_call_start(
            provider=self.config.provider,
            model=self.config.model,
            prompt_tokens=estimated_prompt_tokens,
            prompt_length=len(prompt),
            execution_id=execution_id
        )

        # Create OpenTelemetry span for observability
        span, _ = create_llm_span(
            operation="generate",
            provider=self.config.provider,
            model=self.config.model,
            prompt_tokens=estimated_prompt_tokens,
            prompt_length=len(prompt),
            execution_id=execution_id
        )

        try:
            # Prepare LiteLLM parameters
            litellm_params = self._build_litellm_params(prompt)

            # Make the API call using LiteLLM
            response = await acompletion(**litellm_params)

            # Extract response content and detailed token usage
            content, token_usage = self._extract_response_content(response)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Calculate cost using dynamic pricing service (with source tracking for observability)
            cost_estimate, pricing_source = await self._estimate_cost_with_source(
                f"{self.config.provider}/{self.config.model}",
                token_usage
            )

            # Add token and cost attributes to OpenTelemetry span
            if token_usage:
                add_token_span_attributes(
                    span,
                    provider=self.config.provider,
                    model=self.config.model,
                    prompt_tokens=token_usage.prompt_tokens,
                    completion_tokens=token_usage.completion_tokens,
                    total_tokens=token_usage.total_tokens,
                    cost_usd=cost_estimate,
                    pricing_source=pricing_source,
                    prompt_length=len(prompt)
                )

            # Log successful LLM call with comprehensive token and cost data
            log_llm_call_success(
                provider=self.config.provider,
                model=self.config.model,
                completion_tokens=token_usage.completion_tokens if token_usage else None,
                total_tokens=token_usage.total_tokens if token_usage else None,
                cost_usd=cost_estimate,
                latency_ms=latency_ms,
                pricing_source=pricing_source,
                execution_id=execution_id
            )

            return LLMResponse(
                response=content or "",
                llm_config=self.config,
                cost_estimate=cost_estimate,
                latency_ms=latency_ms,
                tokens_used=token_usage,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            
            # Log failed LLM call
            log_llm_call_failure(
                provider=self.config.provider,
                model=self.config.model,
                error_message=str(e),
                error_type=type(e).__name__,
                latency_ms=latency_ms,
                execution_id=execution_id
            )
            
            LOGGER.error("LiteLLM API call failed", extra={
                "provider": self.config.provider,
                "model": self.config.model,
                "error": str(e),
                "execution_id": execution_id,
                "latency_ms": latency_ms
            })
            raise LLMProviderError(f"LiteLLM API call failed: {e}") from e
        finally:
            if span:
                span.end()


class LLMProviderInterface:
    """Interface for managing multiple LLM providers."""

    def __init__(self) -> None:
        self._clients: dict[str, BaseLLMClient] = {}

    def get_client(self, config: LLMConfig) -> BaseLLMClient:
        """Get or create an LLM client for the given configuration."""
        client_key = f"{config.provider}:{config.model}"

        if client_key not in self._clients:
            client = LiteLLMClient(config)
            self._clients[client_key] = client

        return self._clients[client_key]

    async def generate_parallel(
        self,
        prompt: str,
        configs: list[LLMConfig]
    ) -> list[LLMResponse]:
        """Generate responses from multiple LLMs in parallel."""
        tasks = []
        for config in configs:
            client = self.get_client(config)
            task = asyncio.create_task(client.generate(prompt))
            tasks.append(task)

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to LLMResponse objects
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                LOGGER.error("LLM call failed", extra={
                    "config": configs[i].model_dump(),
                    "error": str(response)
                })
                continue
            valid_responses.append(response)

        return valid_responses

    async def estimate_cost(self, configs: list[LLMConfig], prompt: str) -> float:
        """Estimate the cost for making calls with the given configurations using dynamic pricing."""
        # Rough token estimation for prompt
        estimated_prompt_tokens = len(prompt.split()) * 1.3

        total_cost = 0.0
        pricing_service = await get_pricing_service()

        for config in configs:
            # Extract provider and model
            model_key = f"{config.provider}/{config.model}"

            # Get dynamic pricing
            try:
                pricing_info = await pricing_service.get_pricing(config.provider, config.model)

                # Estimate cost using dynamic rates
                estimated_completion_tokens = 100  # Rough estimate for completion
                input_cost = estimated_prompt_tokens * pricing_info.input_rate
                output_cost = estimated_completion_tokens * pricing_info.output_rate

                total_cost += input_cost + output_cost

                LOGGER.debug("Cost estimation using dynamic pricing", extra={
                    "model": model_key,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "source": pricing_info.source
                })

            except Exception as e:
                LOGGER.warning("Dynamic pricing estimation failed, using fallback", extra={
                    "model": model_key,
                    "error": str(e)
                })

                # Fallback to static calculation
                temp_client = LiteLLMClient(config)
                estimated_token_usage = TokenUsage(
                    prompt_tokens=int(estimated_prompt_tokens),
                    completion_tokens=100,
                    total_tokens=int(estimated_prompt_tokens) + 100,
                    estimated=True
                )
                fallback_cost, _ = temp_client._fallback_cost_calculation_internal(model_key, estimated_token_usage)
                total_cost += fallback_cost

        return total_cost
