"""Configuration management for MDAPFlow-MCP."""

from __future__ import annotations

import json
import os

from pydantic_settings import BaseSettings

from .models import EnsembleConfig, LLMConfig, RedFlagConfig, RedFlagRule


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # MDAP Configuration
    MDAP_DEFAULT_ENSEMBLE_CONFIG_PATH: str | None = None
    MDAP_DEFAULT_RED_FLAG_CONFIG_PATH: str | None = None
    MDAP_DEFAULT_VOTING_K: int = 3
    MDAP_MAX_CONCURRENT_LLM_CALLS: int = 10
    MDAP_MAX_VOTING_ROUNDS: int = 20

    # Logging
    MDAP_LOG_LEVEL: str = "INFO"

    # OpenTelemetry
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None

    # LLM Provider API Keys
    LLM_PROVIDER_OPENAI_API_KEY: str | None = None
    LLM_PROVIDER_ANTHROPIC_API_KEY: str | None = None
    LLM_PROVIDER_OPENROUTER_API_KEY: str | None = None
    LLM_PROVIDER_CUSTOM_BASE_URL: str | None = None
    LLM_PROVIDER_DEFAULT_MAX_TOKENS: int = 2048

    class Config:
        env_file = ".env"
        case_sensitive = True


def load_ensemble_config(path: str | None = None) -> EnsembleConfig | None:
    """Load ensemble configuration from JSON file."""
    if not path:
        path = os.environ.get("MDAP_DEFAULT_ENSEMBLE_CONFIG_PATH")

    if not path:
        return None

    try:
        with open(path) as f:
            config_data = json.load(f)
        return EnsembleConfig.model_validate(config_data)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Failed to load ensemble config from {path}: {e}") from e


def load_red_flag_config(path: str | None = None) -> RedFlagConfig | None:
    """Load red flag configuration from JSON file."""
    if not path:
        path = os.environ.get("MDAP_DEFAULT_RED_FLAG_CONFIG_PATH")

    if not path:
        return None

    try:
        with open(path) as f:
            config_data = json.load(f)
        return RedFlagConfig.model_validate(config_data)
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Failed to load red flag config from {path}: {e}") from e


def create_default_ensemble_config() -> EnsembleConfig:
    """Create a default ensemble configuration with common models."""
    return EnsembleConfig(
        models=[
            LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
            ),
            LLMConfig(
                provider="anthropic",
                model="claude-3-haiku-20240307",
            ),
        ]
    )


def create_default_red_flag_config() -> RedFlagConfig:
    """Create a default red flag configuration."""
    return RedFlagConfig(
        enabled=True,
        rules=[
            # Flag responses that indicate refusal or inability
            RedFlagRule(
                type="keyword",
                value="i cannot|i can't|i don't know|i'm sorry|i cannot help",
                message="LLM refused to provide response"
            ),
            # Flag very short responses (likely errors)
            RedFlagRule(
                type="length_exceeds",
                value="10",
                message="Response too short, likely an error"
            ),
        ],
    )
