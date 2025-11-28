"""Red-Flagging Engine for filtering unreliable LLM outputs."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from .models import RedFlagConfig, RedFlagRule
from .observability import LOGGER


class RedFlaggingEngine:
    """Engine for applying red-flag rules to LLM outputs."""

    def __init__(self, config: RedFlagConfig | None = None) -> None:
        self.config = config or RedFlagConfig()

    def apply_rules(
        self,
        response_text: str,
        output_parser_schema: dict[str, Any] | None = None
    ) -> list[str]:
        """Apply red-flag rules to a response.

        Args:
            response_text: The raw LLM response text
            output_parser_schema: Optional JSON schema for parsing validation

        Returns:
            List of red flag types that were hit
        """
        if not self.config.enabled:
            return []

        hit_flags = []

        for rule in self.config.rules:
            if self._check_rule(response_text, rule, output_parser_schema):
                hit_flags.append(rule.type)
                LOGGER.debug("Red flag triggered", extra={
                    "rule_type": rule.type,
                    "message": rule.message,
                    "response_preview": response_text[:100]
                })

        return hit_flags

    def _check_rule(
        self,
        response_text: str,
        rule: RedFlagRule,
        output_parser_schema: dict[str, Any] | None
    ) -> bool:
        """Check if a specific rule is triggered."""
        if rule.type == "keyword":
            return self._check_keyword_rule(response_text, rule)
        elif rule.type == "regex":
            return self._check_regex_rule(response_text, rule)
        elif rule.type == "length_exceeds":
            return self._check_length_rule(response_text, rule)
        elif rule.type == "json_parse_error":
            return self._check_json_parse_rule(response_text, rule, output_parser_schema)
        else:
            LOGGER.warning(f"Unknown red flag rule type: {rule.type}")
            return False

    def _check_keyword_rule(self, response_text: str, rule: RedFlagRule) -> bool:
        """Check keyword-based red flag rule."""
        if not rule.value:
            return False

        # Case-insensitive keyword matching
        keywords = [kw.strip().lower() for kw in rule.value.split("|")]
        response_lower = response_text.lower()

        return any(keyword in response_lower for keyword in keywords)

    def _check_regex_rule(self, response_text: str, rule: RedFlagRule) -> bool:
        """Check regex-based red flag rule."""
        if not rule.value:
            return False

        try:
            pattern = re.compile(rule.value, re.IGNORECASE | re.MULTILINE)
            return bool(pattern.search(response_text))
        except re.error as e:
            LOGGER.error(f"Invalid regex pattern in red flag rule: {e}")
            return False

    def _check_length_rule(self, response_text: str, rule: RedFlagRule) -> bool:
        """Check length-based red flag rule."""
        if not rule.value:
            return False

        try:
            # Parse threshold (e.g., "10" tokens, "100" chars)
            threshold = int(rule.value)
            # Simple token count by whitespace splitting
            token_count = len(response_text.split())
            return token_count < threshold
        except ValueError:
            LOGGER.error(f"Invalid length threshold in red flag rule: {rule.value}")
            return False

    def _check_json_parse_rule(
        self,
        response_text: str,
        rule: RedFlagRule,
        output_parser_schema: dict[str, Any] | None
    ) -> bool:
        """Check JSON parsing red flag rule."""
        if not output_parser_schema:
            return False

        try:
            # Try to parse as JSON first
            parsed = json.loads(response_text)

            # Validate against schema if provided
            if output_parser_schema:
                jsonschema.validate(parsed, output_parser_schema)

            return False  # No error, rule not triggered

        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            LOGGER.debug("JSON parsing failed", extra={
                "error": str(e),
                "response_preview": response_text[:200]
            })
            return True

    def create_default_config() -> RedFlagConfig:
        """Create a default red flag configuration."""
        return RedFlagConfig(
            enabled=True,
            rules=[
                # Flag responses that indicate refusal
                RedFlagRule(
                    type="keyword",
                    value="i cannot|i can't|i don't know|i'm sorry|i cannot help|i am unable",
                    message="LLM refused to provide response"
                ),
                # Flag very short responses (likely errors)
                RedFlagRule(
                    type="length_exceeds",
                    value="5",
                    message="Response too short, likely an error"
                ),
                # Flag responses with common error indicators
                RedFlagRule(
                    type="regex",
                    value="error|exception|failed|timeout|connection.*refused",
                    message="Response contains error indicators"
                ),
            ]
        )
