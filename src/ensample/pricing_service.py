"""Dynamic pricing service for LLM providers with caching and real-time retrieval."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, NamedTuple

import httpx

from .observability import LOGGER


class PricingInfo(NamedTuple):
    """Pricing information for a specific model."""
    input_rate: float  # Cost per input token
    output_rate: float  # Cost per output token
    currency: str = "USD"
    last_updated: float | None = None
    source: str = "static"  # "static", "provider_api", "third_party", "fallback"


class PricingService:
    """Service for dynamic pricing retrieval with caching."""

    def __init__(self) -> None:
        self._cache: dict[str, PricingInfo] = {}
        self._cache_ttl: dict[str, float] = {}
        self._default_ttl = 3600.0  # 1 hour cache TTL
        self._lock = asyncio.Lock()
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Initialize with static fallback pricing
        self._initialize_static_pricing()

    def _initialize_static_pricing(self) -> None:
        """Initialize static pricing as fallback."""
        static_pricing = {
            # OpenAI / closed models
            "openai/gpt-5.1":            PricingInfo(1.25e-6, 10.00e-6),
            "openai/gpt-5.1-codex":      PricingInfo(1.25e-6, 10.00e-6),
            "openai/gpt-5.1-codex-mini": PricingInfo(0.25e-6, 2.00e-6),
            "openai/gpt-4o":             PricingInfo(2.50e-6, 10.00e-6),
            "openai/gpt-4o-mini":        PricingInfo(0.15e-6, 0.60e-6),
            "openai/gpt-3.5-turbo":      PricingInfo(0.50e-6, 1.50e-6),

            # Google Gemini
            "google/gemini-2.5-pro":  PricingInfo(1.25e-6, 10.00e-6),
            "google/gemini-2.5-flash": PricingInfo(0.30e-6, 2.5e-6),
            # (optionally add flash / cheaper variants if you know names)

            # Anthropic Claude
            "anthropic/claude-opus-4.5": PricingInfo(5.00e-6, 25.00e-6),
            "anthropic/claude-opus-4":  PricingInfo(15.00e-6, 75.00e-6),
            "anthropic/claude-sonnet-4": PricingInfo(3.00e-6, 15.00e-6),
            "anthropic/claude-sonnet-4.5": PricingInfo(3.00e-6, 15.00e-6),
            "anthropic/claude-haiku-4.5": PricingInfo(1.00e-6, 5.00e-6),

            # DeepSeek
            "deepseek/deepseek-v3.1-terminus": PricingInfo(0.216e-6, 0.80e-6),

            # Mistral / open-source-ish
            "mistralai/mixtral-8x22b-instruct": PricingInfo(0.65e-6, 0.65e-6),
            "mistralai/mistral-7b-instruct":    PricingInfo(0.13e-6, 0.13e-6),
        }

        for model_key, pricing in static_pricing.items():
            self._cache[model_key] = pricing
            self._cache_ttl[model_key] = float('inf')  # Static pricing doesn't expire

    async def get_pricing(self, provider: str, model: str) -> PricingInfo:
        """Get pricing information for a model, with caching and fallbacks."""
        model_key = f"{provider}/{model}".lower()

        async with self._lock:
            # Check cache first
            if model_key in self._cache:
                if time.time() < self._cache_ttl.get(model_key, 0):
                    LOGGER.debug("Using cached pricing", extra={"model": model_key})
                    return self._cache[model_key]
                else:
                    LOGGER.debug("Pricing cache expired, refreshing", extra={"model": model_key})

            # Try to fetch fresh pricing
            try:
                pricing = await self._fetch_dynamic_pricing(provider, model)
                if pricing:
                    # Cache successful result
                    self._cache[model_key] = pricing
                    self._cache_ttl[model_key] = time.time() + self._default_ttl
                    LOGGER.info("Successfully fetched dynamic pricing", extra={
                        "model": model_key,
                        "source": pricing.source,
                        "input_rate": pricing.input_rate,
                        "output_rate": pricing.output_rate
                    })
                    return pricing
            except Exception as e:
                LOGGER.warning("Failed to fetch dynamic pricing", extra={
                    "model": model_key,
                    "error": str(e)
                })

            # Fallback to static pricing
            if model_key in self._cache:
                fallback_pricing = self._cache[model_key]
                # Update source to indicate fallback
                fallback_pricing = fallback_pricing._replace(source="fallback")
                self._cache[model_key] = fallback_pricing
                LOGGER.warning("Using fallback pricing", extra={"model": model_key})
                return fallback_pricing

            # Default fallback for unknown models
            default_pricing = PricingInfo(0.000001, 0.000001, source="default_fallback")
            self._cache[model_key] = default_pricing
            self._cache_ttl[model_key] = time.time() + self._default_ttl
            LOGGER.warning("Using default fallback pricing", extra={"model": model_key})
            return default_pricing

    async def _fetch_dynamic_pricing(self, provider: str, model: str) -> PricingInfo | None:
        """Fetch dynamic pricing from various sources."""
        # Try OpenRouter pricing API first (most comprehensive)
        if provider.lower() in ["openrouter", "openai", "anthropic", "together", "cohere"]:
            try:
                return await self._fetch_openrouter_pricing(provider, model)
            except Exception as e:
                LOGGER.debug("OpenRouter pricing fetch failed", extra={"error": str(e)})

        return None

    async def _fetch_openrouter_pricing(self, provider: str, model: str) -> PricingInfo | None:
        """Fetch pricing from OpenRouter's pricing API."""
        try:
            # OpenRouter provides pricing for many models through their API
            response = await self._http_client.get("https://openrouter.ai/api/v1/models")
            response.raise_for_status()
            
            models_data = response.json()
            
            # Find the specific model
            model_id = f"{provider}/{model}"
            for model_info in models_data.get("data", []):
                if model_info.get("id") == model_id:
                    pricing_data = model_info.get("pricing", {})
                    input_rate = float(pricing_data.get("prompt", "0"))
                    output_rate = float(pricing_data.get("completion", "0"))
                    
                    if input_rate > 0 or output_rate > 0:
                        return PricingInfo(input_rate, output_rate, source="openrouter_api")
            
            LOGGER.debug("Model not found in OpenRouter pricing", extra={"model": model_id})
            return None
            
        except Exception as e:
            LOGGER.warning("Failed to fetch OpenRouter pricing", extra={"error": str(e)})
            return None

    async def clear_cache(self) -> None:
        """Clear the pricing cache."""
        async with self._lock:
            self._cache.clear()
            self._cache_ttl.clear()
            LOGGER.info("Pricing cache cleared")

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for model_key, ttl in self._cache_ttl.items()
                if time.time() >= ttl
            )
            
            sources = {}
            for pricing in self._cache.values():
                sources[pricing.source] = sources.get(pricing.source, 0) + 1
            
            return {
                "total_entries": total_entries,
                "expired_entries": expired_entries,
                "valid_entries": total_entries - expired_entries,
                "sources": sources,
                "cache_ttl_seconds": self._default_ttl,
            }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http_client.aclose()


# Global pricing service instance
_pricing_service: PricingService | None = None


async def get_pricing_service() -> PricingService:
    """Get the global pricing service instance."""
    global _pricing_service
    if _pricing_service is None:
        _pricing_service = PricingService()
    return _pricing_service


async def close_pricing_service() -> None:
    """Close the global pricing service."""
    global _pricing_service
    if _pricing_service:
        await _pricing_service.close()
        _pricing_service = None