"""Basic tests for MDAP models."""

from __future__ import annotations

from src.mdapflow_mcp.models import LLMConfig, EnsembleConfig, RedFlagConfig, MDAPInput


def test_llm_config_creation():
    """Test LLMConfig creation with minimal required fields."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
    )
    assert config.provider == "openai"
    assert config.model == "gpt-4o-mini"
    assert config.temperature == 0.1  # Default value


def test_ensemble_config_creation():
    """Test EnsembleConfig creation."""
    models = [
        LLMConfig(provider="openai", model="gpt-4o-mini"),
        LLMConfig(provider="anthropic", model="claude-3-haiku"),
    ]
    config = EnsembleConfig(models=models)
    assert len(config.models) == 2


def test_red_flag_config_creation():
    """Test RedFlagConfig creation."""
    from src.mdapflow_mcp.models import RedFlagRule

    rules = [
        RedFlagRule(
            type="keyword",
            value="error",
            message="Test error rule"
        )
    ]
    config = RedFlagConfig(rules=rules)
    assert len(config.rules) == 1
    assert config.enabled is True


def test_mdap_input_creation():
    """Test MDAPInput creation."""
    mdap_input = MDAPInput(
        prompt="Test prompt",
        role_name="test_role"
    )
    assert mdap_input.prompt == "Test prompt"
    assert mdap_input.role_name == "test_role"
    assert mdap_input.voting_k == 3  # Default value