"""Comprehensive tests for RedFlaggingEngine - LLM output filtering component."""

import pytest
import json
import jsonschema
from unittest.mock import MagicMock

from src.ensample.red_flagging_engine import RedFlaggingEngine
from src.ensample.models import RedFlagConfig, RedFlagRule


class TestRedFlaggingEngineInitialization:
    """Test RedFlaggingEngine initialization and configuration."""

    def test_initialization_default_config(self):
        """Test initialization with default configuration."""
        engine = RedFlaggingEngine()
        
        assert engine.config.enabled is True
        # RedFlaggingEngine initializes with empty config, not create_default_config()
        assert len(engine.config.rules) == 0
        assert isinstance(engine, RedFlaggingEngine)

    def test_initialization_custom_config(self):
        """Test initialization with custom configuration."""
        custom_rules = [
            RedFlagRule(type="keyword", value="test", message="Test keyword")
        ]
        custom_config = RedFlagConfig(enabled=False, rules=custom_rules)
        
        engine = RedFlaggingEngine(custom_config)
        
        assert engine.config.enabled is False
        assert len(engine.config.rules) == 1
        assert engine.config.rules[0].type == "keyword"

    def test_initialization_none_config(self):
        """Test initialization with None config."""
        engine = RedFlaggingEngine(None)
        
        assert engine.config.enabled is True  # Default config
        assert isinstance(engine.config, RedFlagConfig)
        assert len(engine.config.rules) == 0  # Default config has no rules


class TestApplyRulesBasic:
    """Test basic apply_rules functionality."""

    def test_apply_rules_disabled_config(self):
        """Test apply_rules when config is disabled."""
        engine = RedFlaggingEngine()
        engine.config.enabled = False
        
        result = engine.apply_rules("test response")
        
        assert result == []

    def test_apply_rules_no_rules(self):
        """Test apply_rules with empty rules list."""
        config = RedFlagConfig(enabled=True, rules=[])
        engine = RedFlaggingEngine(config)
        
        result = engine.apply_rules("test response")
        
        assert result == []

    def test_apply_rules_no_hits(self):
        """Test apply_rules when no rules are triggered."""
        rules = [
            RedFlagRule(type="keyword", value="forbidden", message="Test rule")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        result = engine.apply_rules("allowed response")
        
        assert result == []

    def test_apply_rules_single_hit(self):
        """Test apply_rules when one rule is triggered."""
        rules = [
            RedFlagRule(type="keyword", value="forbidden", message="Test rule")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        result = engine.apply_rules("this contains forbidden word")
        
        assert result == ["keyword"]

    def test_apply_rules_multiple_hits(self):
        """Test apply_rules when multiple rules are triggered."""
        rules = [
            RedFlagRule(type="keyword", value="forbidden", message="Test rule 1"),
            RedFlagRule(type="regex", value="error", message="Test rule 2")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        result = engine.apply_rules("this contains forbidden and error")
        
        assert "keyword" in result
        assert "regex" in result
        assert len(result) == 2

    def test_apply_rules_with_output_schema(self):
        """Test apply_rules with output parser schema."""
        rules = [
            RedFlagRule(type="json_parse_error", value=None, message="JSON rule")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        schema = {"type": "object"}
        result = engine.apply_rules("invalid json", schema)
        
        assert result == ["json_parse_error"]


class TestKeywordRule:
    """Test keyword-based red flag rules."""

    def test_keyword_rule_single_word(self):
        """Test keyword rule with single word."""
        rule = RedFlagRule(type="keyword", value="forbidden", message="Test")
        engine = RedFlaggingEngine()
        
        # Test case where keyword is present
        result = engine._check_keyword_rule("this is forbidden", rule)
        assert result is True
        
        # Test case where keyword is absent
        result = engine._check_keyword_rule("this is allowed", rule)
        assert result is False

    def test_keyword_rule_case_insensitive(self):
        """Test keyword rule is case-insensitive."""
        rule = RedFlagRule(type="keyword", value="FORBIDDEN", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_keyword_rule("this is FORBIDDEN", rule)
        assert result is True
        
        result = engine._check_keyword_rule("this is forbidden", rule)
        assert result is True
        
        result = engine._check_keyword_rule("this is Forbidden", rule)
        assert result is True

    def test_keyword_rule_pipe_separated(self):
        """Test keyword rule with pipe-separated keywords."""
        rule = RedFlagRule(type="keyword", value="word1|word2|word3", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_keyword_rule("this contains word1", rule)
        assert result is True
        
        result = engine._check_keyword_rule("this contains word2", rule)
        assert result is True
        
        result = engine._check_keyword_rule("this contains word3", rule)
        assert result is True
        
        result = engine._check_keyword_rule("this contains word4", rule)
        assert result is False

    def test_keyword_rule_empty_value(self):
        """Test keyword rule with empty value."""
        rule = RedFlagRule(type="keyword", value="", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_keyword_rule("any text", rule)
        assert result is False

    def test_keyword_rule_none_value(self):
        """Test keyword rule with None value."""
        rule = RedFlagRule(type="keyword", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_keyword_rule("any text", rule)
        assert result is False

    def test_keyword_rule_whitespace_handling(self):
        """Test keyword rule handles whitespace in value."""
        rule = RedFlagRule(type="keyword", value=" word1 | word2 ", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_keyword_rule("this contains word1", rule)
        assert result is True
        
        result = engine._check_keyword_rule("this contains word2", rule)
        assert result is True

    def test_keyword_rule_substring_matching(self):
        """Test keyword rule does substring matching."""
        rule = RedFlagRule(type="keyword", value="can", message="Test")
        engine = RedFlaggingEngine()
        
        # Should match "cannot" as it contains "can"
        result = engine._check_keyword_rule("I cannot help", rule)
        assert result is True


class TestRegexRule:
    """Test regex-based red flag rules."""

    def test_regex_rule_simple_pattern(self):
        """Test regex rule with simple pattern."""
        rule = RedFlagRule(type="regex", value="error", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("this contains error", rule)
        assert result is True
        
        result = engine._check_regex_rule("no match here", rule)
        assert result is False

    def test_regex_rule_case_insensitive(self):
        """Test regex rule is case-insensitive."""
        rule = RedFlagRule(type="regex", value="ERROR", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("this contains ERROR", rule)
        assert result is True
        
        result = engine._check_regex_rule("this contains error", rule)
        assert result is True

    def test_regex_rule_multiline(self):
        """Test regex rule with multiline pattern."""
        rule = RedFlagRule(type="regex", value="line2", message="Test")
        engine = RedFlaggingEngine()
        
        text = "line1\nline2\nline3"
        result = engine._check_regex_rule(text, rule)
        assert result is True

    def test_regex_rule_complex_pattern(self):
        """Test regex rule with complex pattern."""
        rule = RedFlagRule(type="regex", value=r"error|exception|failed", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("this contains error", rule)
        assert result is True
        
        result = engine._check_regex_rule("this contains exception", rule)
        assert result is True
        
        result = engine._check_regex_rule("this contains failed", rule)
        assert result is True
        
        result = engine._check_regex_rule("this contains success", rule)
        assert result is False

    def test_regex_rule_empty_value(self):
        """Test regex rule with empty value."""
        rule = RedFlagRule(type="regex", value="", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("any text", rule)
        assert result is False

    def test_regex_rule_none_value(self):
        """Test regex rule with None value."""
        rule = RedFlagRule(type="regex", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("any text", rule)
        assert result is False

    def test_regex_rule_invalid_pattern(self):
        """Test regex rule with invalid pattern (should handle gracefully)."""
        rule = RedFlagRule(type="regex", value="[invalid", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("any text", rule)
        assert result is False  # Should return False on error

    def test_regex_rule_anchored_patterns(self):
        """Test regex rule with anchored patterns."""
        rule = RedFlagRule(type="regex", value="^ERROR", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_regex_rule("ERROR: something went wrong", rule)
        assert result is True
        
        result = engine._check_regex_rule("no ERROR here", rule)
        assert result is False


class TestLengthRule:
    """Test length-based red flag rules."""

    def test_length_rule_below_threshold(self):
        """Test length rule when response is below threshold."""
        rule = RedFlagRule(type="length_exceeds", value="10", message="Test")
        engine = RedFlaggingEngine()
        
        # 3 tokens, should be flagged (less than 10)
        result = engine._check_length_rule("short text", rule)
        assert result is True

    def test_length_rule_above_threshold(self):
        """Test length rule when response is above threshold."""
        rule = RedFlagRule(type="length_exceeds", value="5", message="Test")
        engine = RedFlaggingEngine()
        
        # 10 tokens, should not be flagged (greater than or equal to 5)
        result = engine._check_length_rule("this is a longer response with more words", rule)
        assert result is False

    def test_length_rule_exact_threshold(self):
        """Test length rule when response is exactly at threshold."""
        rule = RedFlagRule(type="length_exceeds", value="5", message="Test")
        engine = RedFlaggingEngine()
        
        # Exactly 5 tokens, should not be flagged
        result = engine._check_length_rule("one two three four five", rule)
        assert result is False

    def test_length_rule_empty_response(self):
        """Test length rule with empty response."""
        rule = RedFlagRule(type="length_exceeds", value="5", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_length_rule("", rule)
        assert result is True  # 0 tokens < 5

    def test_length_rule_whitespace_handling(self):
        """Test length rule with various whitespace."""
        rule = RedFlagRule(type="length_exceeds", value="3", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_length_rule("  word1   word2   word3  ", rule)
        assert result is False  # Still 3 words

    def test_length_rule_invalid_threshold(self):
        """Test length rule with invalid threshold value."""
        rule = RedFlagRule(type="length_exceeds", value="invalid", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_length_rule("any text", rule)
        assert result is False  # Should return False on invalid threshold

    def test_length_rule_none_value(self):
        """Test length rule with None value."""
        rule = RedFlagRule(type="length_exceeds", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_length_rule("any text", rule)
        assert result is False

    def test_length_rule_zero_threshold(self):
        """Test length rule with zero threshold."""
        rule = RedFlagRule(type="length_exceeds", value="0", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_length_rule("any text", rule)
        assert result is False  # Most text has more than 0 tokens


class TestJSONParseRule:
    """Test JSON parsing red flag rules."""

    def test_json_parse_rule_valid_json(self):
        """Test JSON parse rule with valid JSON."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        valid_json = '{"key": "value"}'
        schema = {"type": "object"}
        
        result = engine._check_json_parse_rule(valid_json, rule, schema)
        assert result is False  # No error, rule not triggered

    def test_json_parse_rule_invalid_json(self):
        """Test JSON parse rule with invalid JSON."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        invalid_json = '{"key": invalid}'
        schema = {"type": "object"}
        
        result = engine._check_json_parse_rule(invalid_json, rule, schema)
        assert result is True  # JSON parse error, rule triggered

    def test_json_parse_rule_schema_validation_failure(self):
        """Test JSON parse rule with schema validation failure."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        invalid_schema_json = '{"key": "value"}'  # Valid JSON but wrong schema
        schema = {"type": "string"}  # Expecting string, got object
        
        result = engine._check_json_parse_rule(invalid_schema_json, rule, schema)
        assert result is True  # Schema validation error, rule triggered

    def test_json_parse_rule_no_schema(self):
        """Test JSON parse rule when no schema is provided."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        invalid_json = '{"key": invalid}'
        
        result = engine._check_json_parse_rule(invalid_json, rule, None)
        assert result is False  # No schema, rule not applicable

    def test_json_parse_rule_empty_text(self):
        """Test JSON parse rule with empty text."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        schema = {"type": "object"}
        
        result = engine._check_json_parse_rule("", rule, schema)
        assert result is True  # Empty string can't be parsed as JSON

    def test_json_parse_rule_plain_text(self):
        """Test JSON parse rule with plain text."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        plain_text = "This is just plain text"
        schema = {"type": "object"}
        
        result = engine._check_json_parse_rule(plain_text, rule, schema)
        assert result is True  # Can't parse as JSON

    def test_json_parse_rule_complex_nested_json(self):
        """Test JSON parse rule with complex nested JSON."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        complex_json = '{"nested": {"array": [1, 2, 3], "string": "value"}}'
        schema = {"type": "object"}
        
        result = engine._check_json_parse_rule(complex_json, rule, schema)
        assert result is False  # Valid JSON, valid schema

    def test_json_parse_rule_invalid_schema_json(self):
        """Test JSON parse rule with schema validation failure on nested structure."""
        rule = RedFlagRule(type="json_parse_error", value=None, message="Test")
        engine = RedFlaggingEngine()
        
        json_with_numbers = '{"value": 123}'
        schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        
        result = engine._check_json_parse_rule(json_with_numbers, rule, schema)
        assert result is True  # Schema validation fails (number vs string)


class TestCreateDefaultConfig:
    """Test create_default_config method."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        config = RedFlaggingEngine.create_default_config()
        
        assert isinstance(config, RedFlagConfig)
        assert config.enabled is True
        assert len(config.rules) == 3
        
        # Check rule types
        rule_types = [rule.type for rule in config.rules]
        assert "keyword" in rule_types
        assert "length_exceeds" in rule_types
        assert "regex" in rule_types

    def test_create_default_config_keyword_rule(self):
        """Test default configuration keyword rule."""
        config = RedFlaggingEngine.create_default_config()
        
        keyword_rule = next(rule for rule in config.rules if rule.type == "keyword")
        
        assert "cannot" in keyword_rule.value
        assert "don't" in keyword_rule.value
        assert "sorry" in keyword_rule.value
        assert keyword_rule.message == "LLM refused to provide response"

    def test_create_default_config_length_rule(self):
        """Test default configuration length rule."""
        config = RedFlaggingEngine.create_default_config()
        
        length_rule = next(rule for rule in config.rules if rule.type == "length_exceeds")
        
        assert length_rule.value == "5"
        assert length_rule.message == "Response too short, likely an error"

    def test_create_default_config_regex_rule(self):
        """Test default configuration regex rule."""
        config = RedFlaggingEngine.create_default_config()
        
        regex_rule = next(rule for rule in config.rules if rule.type == "regex")
        
        assert "error" in regex_rule.value
        assert "exception" in regex_rule.value
        assert "failed" in regex_rule.value
        assert regex_rule.message == "Response contains error indicators"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_apply_rules_with_unicode_text(self):
        """Test apply_rules with unicode characters."""
        rules = [
            RedFlagRule(type="keyword", value="test", message="Test")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        result = engine.apply_rules("this is test with unicode: éñüðø")
        assert result == ["keyword"]

    def test_apply_rules_with_very_long_response(self):
        """Test apply_rules with very long response."""
        rules = [
            RedFlagRule(type="keyword", value="test", message="Test")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        long_response = "test " * 10000  # Very long response
        result = engine.apply_rules(long_response)
        assert result == ["keyword"]

    def test_apply_rules_with_newlines_and_special_chars(self):
        """Test apply_rules with various special characters."""
        rules = [
            RedFlagRule(type="regex", value="special", message="Test")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        result = engine.apply_rules("this has\nspecial\tchars!@#$%")
        assert result == ["regex"]

    def test_apply_rules_with_default_config(self):
        """Test apply_rules with default configuration."""
        engine = RedFlaggingEngine(RedFlaggingEngine.create_default_config())
        
        # Test with refusal text
        result = engine.apply_rules("I cannot help with that request")
        assert "keyword" in result
        
        # Test with error text
        result = engine.apply_rules("error occurred during processing")
        assert "regex" in result

    def test_token_counting_edge_cases(self):
        """Test length rule token counting edge cases."""
        engine = RedFlaggingEngine()
        
        # Test with various whitespace combinations
        rule = RedFlagRule(type="length_exceeds", value="3", message="Test")
        
        # Multiple spaces
        result = engine._check_length_rule("word1    word2   word3", rule)
        assert result is False
        
        # Tabs and newlines
        result = engine._check_length_rule("word1\tword2\nword3", rule)
        assert result is False
        
        # Only punctuation - note: punctuation is counted as tokens
        result = engine._check_length_rule("!@# $%^ &*()", rule)
        assert result is False  # These are still counted as tokens by split()

    def test_multiple_red_flag_types_in_sequence(self):
        """Test applying multiple different red flag rules in sequence."""
        rules = [
            RedFlagRule(type="keyword", value="bad", message="Keyword rule"),
            RedFlagRule(type="regex", value="error", message="Regex rule"),
            RedFlagRule(type="length_exceeds", value="10", message="Length rule")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        # This should trigger all three rules
        result = engine.apply_rules("bad error short")
        assert len(result) == 3
        assert "keyword" in result
        assert "regex" in result
        assert "length_exceeds" in result

    def test_empty_responses_handling(self):
        """Test handling of empty and whitespace-only responses."""
        rules = [
            RedFlagRule(type="length_exceeds", value="1", message="Length rule")
        ]
        config = RedFlagConfig(enabled=True, rules=rules)
        engine = RedFlaggingEngine(config)
        
        # Empty string
        result = engine.apply_rules("")
        assert "length_exceeds" in result
        
        # Whitespace only
        result = engine.apply_rules("   \t\n   ")
        assert "length_exceeds" in result

    def test_mixed_case_keyword_matching(self):
        """Test keyword rule with mixed case patterns."""
        rule = RedFlagRule(type="keyword", value="HELLO|goodbye", message="Test")
        engine = RedFlaggingEngine()
        
        result = engine._check_keyword_rule("Hello world", rule)
        assert result is True
        
        result = engine._check_keyword_rule("Goodbye cruel world", rule)
        assert result is True
        
        result = engine._check_keyword_rule("HELLO and GOODBYE", rule)
        assert result is True
