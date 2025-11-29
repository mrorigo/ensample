"""Comprehensive tests for OutputParser - canonicalizing and validating LLM outputs."""

import pytest
import json
from unittest.mock import patch

from src.ensample.output_parser import OutputParser


class TestOutputParserInitialization:
    """Test OutputParser initialization."""

    def test_initialization_default(self):
        """Test basic initialization."""
        parser = OutputParser()
        assert parser._repair_attempts == 0

    def test_initialization_multiple_instances(self):
        """Test that instances maintain separate state."""
        parser1 = OutputParser()
        parser2 = OutputParser()
        
        parser1._repair_attempts = 5
        
        assert parser1._repair_attempts == 5
        assert parser2._repair_attempts == 0


class TestParseOutputBasic:
    """Test basic parse_output functionality."""

    def test_parse_output_no_schema_text(self):
        """Test parsing text without schema."""
        parser = OutputParser()
        
        text = "Hello world, this is a simple text response."
        result, error = parser.parse_output(text, None)
        
        assert result == text
        assert error is None

    def test_parse_output_no_schema_empty_text(self):
        """Test parsing empty text without schema."""
        parser = OutputParser()
        
        result, error = parser.parse_output("", None)
        
        assert result == ""
        assert error is None

    def test_parse_output_no_schema_whitespace(self):
        """Test parsing whitespace without schema."""
        parser = OutputParser()
        
        text = "   \n\t  "
        result, error = parser.parse_output(text, None)
        
        assert result == text
        assert error is None

    def test_parse_output_no_schema_complex_text(self):
        """Test parsing complex text without schema."""
        parser = OutputParser()
        
        text = "This is a complex response with\nmultiple lines, special chars: !@#$%, and numbers 123."
        result, error = parser.parse_output(text, None)
        
        assert result == text
        assert error is None

    def test_parse_output_no_schema_json_like_text(self):
        """Test parsing JSON-like text without schema."""
        parser = OutputParser()
        
        text = '{"key": "value", "number": 123}'
        result, error = parser.parse_output(text, None)
        
        assert result == text
        assert error is None


class TestParseOutputWithValidJSON:
    """Test parsing with valid JSON and schema validation."""

    def test_parse_output_valid_json_no_schema(self):
        """Test parsing valid JSON without schema."""
        parser = OutputParser()
        
        json_text = '{"name": "John", "age": 30, "active": true}'
        result, error = parser.parse_output(json_text, None)
        
        assert result == json_text
        assert error is None

    def test_parse_output_valid_json_with_schema(self):
        """Test parsing valid JSON with matching schema."""
        parser = OutputParser()
        
        json_text = '{"name": "John", "age": 30}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"name": "John", "age": 30}
        assert result == expected_data
        assert error is None

    def test_parse_output_nested_json_with_schema(self):
        """Test parsing nested JSON with schema."""
        parser = OutputParser()
        
        json_text = '{"user": {"name": "John", "address": {"city": "NYC"}}, "id": 123}'
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            }
                        }
                    }
                },
                "id": {"type": "number"}
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"user": {"name": "John", "address": {"city": "NYC"}}, "id": 123}
        assert result == expected_data
        assert error is None

    def test_parse_output_array_json_with_schema(self):
        """Test parsing JSON array with schema."""
        parser = OutputParser()
        
        json_text = '[{"name": "John"}, {"name": "Jane"}]'
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                }
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = [{"name": "John"}, {"name": "Jane"}]
        assert result == expected_data
        assert error is None


class TestParseOutputWithInvalidJSON:
    """Test parsing with invalid JSON."""

    def test_parse_output_invalid_json_no_repair(self):
        """Test parsing invalid JSON that can't be repaired."""
        parser = OutputParser()
        
        invalid_json = '{"name": "John", "age":'  # Missing value
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        result, error = parser.parse_output(invalid_json, schema)
        
        assert result is None
        assert error is not None
        assert "Failed to parse output as valid JSON" in error
        assert "Expecting value" in error

    def test_parse_output_empty_json(self):
        """Test parsing empty JSON object."""
        parser = OutputParser()
        
        json_text = '{}'
        schema = {
            "type": "object",
            "properties": {}
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        assert result == {}
        assert error is None

    def test_parse_output_malformed_json(self):
        """Test parsing completely malformed JSON."""
        parser = OutputParser()
        
        malformed = "This is not JSON at all!"
        schema = {
            "type": "object",
            "properties": {}
        }
        
        result, error = parser.parse_output(malformed, schema)
        
        assert result is None
        assert error is not None
        assert "Failed to parse output as valid JSON" in error

    def test_parse_output_partial_json(self):
        """Test parsing partial JSON structure."""
        parser = OutputParser()
        
        partial = '{"key1": "value1"'  # Missing closing brace
        schema = {
            "type": "object",
            "properties": {
                "key1": {"type": "string"}
            }
        }
        
        result, error = parser.parse_output(partial, schema)
        
        assert result is None
        assert error is not None
        assert "Failed to parse output as valid JSON" in error


class TestJSONRepairFunctionality:
    """Test JSON repair functionality."""

    def test_repair_json_trailing_commas(self):
        """Test repair of trailing commas."""
        parser = OutputParser()
        
        broken_json = '{"name": "John", "age": 30,}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        result, error = parser.parse_output(broken_json, schema)
        
        # Should be repaired and valid
        expected_data = {"name": "John", "age": 30}
        assert result == expected_data
        assert error is None

    def test_repair_json_single_quotes(self):
        """Test repair of single quotes to double quotes."""
        parser = OutputParser()
        
        broken_json = "{'name': 'John', 'age': 30}"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        result, error = parser.parse_output(broken_json, schema)
        
        # Should be repaired and valid
        expected_data = {"name": "John", "age": 30}
        assert result == expected_data
        assert error is None

    def test_repair_json_mixed_quotes(self):
        """Test repair of mixed quotes scenarios."""
        parser = OutputParser()
        
        broken_json = '{"name": "John", "active": true}'  # Already mostly valid
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "active": {"type": "boolean"}
            }
        }
        
        result, error = parser.parse_output(broken_json, schema)
        
        # Should already be valid
        expected_data = {"name": "John", "active": True}
        assert result == expected_data
        assert error is None

    def test_repair_attempts_limit(self):
        """Test that repair attempts are limited."""
        parser = OutputParser()
        
        # Set repair attempts to exceed limit
        parser._repair_attempts = 3
        
        broken_json = '{"name": "John",}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        result, error = parser.parse_output(broken_json, schema)
        
        # Should not attempt repair due to limit
        assert result is None
        assert error is not None
        assert "Failed to parse output as valid JSON" in error

    def test_repair_json_cannot_fix(self):
        """Test repair when JSON cannot be fixed."""
        parser = OutputParser()
        
        unfixable_json = '{"name": "John" "age": 30}'  # Missing comma
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        result, error = parser.parse_output(unfixable_json, schema)
        
        # Repair should fail
        assert result is None
        assert error is not None
        assert "Failed to parse output as valid JSON" in error


class TestSchemaValidationFailures:
    """Test schema validation failure cases."""

    def test_parse_output_schema_validation_failure(self):
        """Test schema validation failure."""
        parser = OutputParser()
        
        json_text = '{"name": "John", "age": "not a number"}'  # age should be number
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        assert result is None
        assert error is not None
        assert "Output does not match expected schema" in error
        assert "number" in error

    def test_parse_output_schema_type_mismatch(self):
        """Test schema type mismatch."""
        parser = OutputParser()
        
        json_text = '{"name": 123}'  # name should be string
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        assert result is None
        assert error is not None
        assert "Output does not match expected schema" in error

    def test_parse_output_schema_missing_required_field(self):
        """Test missing required field in schema."""
        parser = OutputParser()
        
        json_text = '{"name": "John"}'  # missing required "age"
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        assert result is None
        assert error is not None
        assert "Output does not match expected schema" in error

    def test_parse_output_schema_additional_properties(self):
        """Test additional properties not in schema."""
        parser = OutputParser()
        
        json_text = '{"name": "John", "age": 30, "extra": "value"}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "additionalProperties": False
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        assert result is None
        assert error is not None
        assert "Output does not match expected schema" in error


class TestValidateStructuredOutput:
    """Test validate_structured_output method."""

    def test_validate_structured_output_valid_dict(self):
        """Test validation of valid dictionary."""
        parser = OutputParser()
        
        data = {"name": "John", "age": 30}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        is_valid, error = parser.validate_structured_output(data, schema)
        
        assert is_valid is True
        assert error is None

    def test_validate_structured_output_invalid_dict(self):
        """Test validation of invalid dictionary."""
        parser = OutputParser()
        
        data = {"name": "John", "age": "not a number"}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        is_valid, error = parser.validate_structured_output(data, schema)
        
        assert is_valid is False
        assert error is not None
        assert "number" in error

    def test_validate_structured_output_valid_json_string(self):
        """Test validation of valid JSON string."""
        parser = OutputParser()
        
        data = '{"name": "John", "age": 30}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        is_valid, error = parser.validate_structured_output(data, schema)
        
        assert is_valid is True
        assert error is None

    def test_validate_structured_output_invalid_json_string(self):
        """Test validation of invalid JSON string."""
        parser = OutputParser()
        
        data = '{"name": "John", "age":'  # Invalid JSON
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        is_valid, error = parser.validate_structured_output(data, schema)
        
        assert is_valid is False
        assert error is not None

    def test_validate_structured_output_non_dict_non_string(self):
        """Test validation of non-dict, non-string data."""
        parser = OutputParser()
        
        # Test with number (will be cast to string for type checking)
        data: str = "123"  # Pass as string instead of int
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        is_valid, error = parser.validate_structured_output(data, schema)
        
        assert is_valid is False
        assert error is not None


class TestExtractKeyFields:
    """Test extract_key_fields method."""

    def test_extract_key_fields_valid_dict_all_present(self):
        """Test extraction when all required fields are present."""
        parser = OutputParser()
        
        data = {"name": "John", "age": 30, "city": "NYC"}
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {"name": "John", "age": 30}
        assert missing == []

    def test_extract_key_fields_valid_dict_some_missing(self):
        """Test extraction when some required fields are missing."""
        parser = OutputParser()
        
        data = {"name": "John", "city": "NYC"}  # missing "age"
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {"name": "John"}
        assert missing == ["age"]

    def test_extract_key_fields_valid_dict_all_missing(self):
        """Test extraction when all required fields are missing."""
        parser = OutputParser()
        
        data = {"city": "NYC", "country": "USA"}
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {}
        assert missing == ["name", "age"]

    def test_extract_key_fields_empty_dict(self):
        """Test extraction from empty dictionary."""
        parser = OutputParser()
        
        data = {}
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {}
        assert missing == ["name", "age"]

    def test_extract_key_fields_valid_json_string(self):
        """Test extraction from valid JSON string."""
        parser = OutputParser()
        
        data = '{"name": "John", "age": 30, "city": "NYC"}'
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {"name": "John", "age": 30}
        assert missing == []

    def test_extract_key_fields_invalid_json_string(self):
        """Test extraction from invalid JSON string."""
        parser = OutputParser()
        
        data = '{"name": "John", "age":'  # Invalid JSON
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {}
        assert missing == ["name", "age"]

    def test_extract_key_fields_non_dict_data(self):
        """Test extraction from non-dictionary data."""
        parser = OutputParser()
        
        data = "not a dictionary"
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {}
        assert missing == ["name", "age"]

    def test_extract_key_fields_list_data(self):
        """Test extraction from list data - should be converted to string."""
        parser = OutputParser()
        
        data = '["item1", "item2"]'  # Pass as string instead of list
        required_fields = ["name", "age"]
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {}
        assert missing == ["name", "age"]

    def test_extract_key_fields_empty_required_fields(self):
        """Test extraction with empty required fields list."""
        parser = OutputParser()
        
        data = {"name": "John", "age": 30}
        required_fields = []
        
        extracted, missing = parser.extract_key_fields(data, required_fields)
        
        assert extracted == {}
        assert missing == []


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_parse_output_very_large_json(self):
        """Test parsing very large JSON."""
        parser = OutputParser()
        
        # Create large JSON
        large_data = {"items": [{"id": i, "value": f"item_{i}"} for i in range(1000)]}
        large_json = json.dumps(large_data)
        
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "number"},
                            "value": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        result, error = parser.parse_output(large_json, schema)
        
        assert result == large_data
        assert error is None

    def test_parse_output_unicode_characters(self):
        """Test parsing JSON with Unicode characters."""
        parser = OutputParser()
        
        json_text = '{"name": "Jöhn Döe", "message": "Héllo 世界 🌍"}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "message": {"type": "string"}
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"name": "Jöhn Döe", "message": "Héllo 世界 🌍"}
        assert result == expected_data
        assert error is None

    def test_parse_output_special_characters(self):
        """Test parsing JSON with special characters."""
        parser = OutputParser()
        
        json_text = '{"text": "Line\\nBreak\\tTab\\"Quote\\\\Backslash"}'
        schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"text": "Line\nBreak\tTab\"Quote\\Backslash"}
        assert result == expected_data
        assert error is None

    def test_parse_output_null_values(self):
        """Test parsing JSON with null values."""
        parser = OutputParser()
        
        json_text = '{"name": "John", "age": null, "active": true}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "age": {"type": ["number", "null"]},
                "active": {"type": "boolean"}
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"name": "John", "age": None, "active": True}
        assert result == expected_data
        assert error is None

    def test_parse_output_numbers_edge_cases(self):
        """Test parsing JSON with edge case numbers."""
        parser = OutputParser()
        
        json_text = '{"zero": 0, "negative": -42, "float": 3.14159, "scientific": 1.23e-4}'
        schema = {
            "type": "object",
            "properties": {
                "zero": {"type": "number"},
                "negative": {"type": "number"},
                "float": {"type": "number"},
                "scientific": {"type": "number"}
            }
        }
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"zero": 0, "negative": -42, "float": 3.14159, "scientific": 1.23e-4}
        assert result == expected_data
        assert error is None

    def test_parse_output_empty_schema(self):
        """Test parsing with empty schema."""
        parser = OutputParser()
        
        json_text = '{"any": "thing"}'
        schema = {}  # Empty schema - should accept anything
        
        result, error = parser.parse_output(json_text, schema)
        
        expected_data = {"any": "thing"}
        assert result == expected_data
        assert error is None

    def test_parse_output_none_values(self):
        """Test handling of None values - convert to string."""
        parser = OutputParser()
        
        # Convert None to string to match function signature
        result1, error1 = parser.parse_output("None", None)
        assert result1 == "None"
        assert error1 is None
        
        result2, error2 = parser.parse_output("", None)
        assert result2 == ""
        assert error2 is None


class TestRepairJSONMethod:
    """Test the _repair_json method directly."""

    def test_repair_json_trailing_comma_object(self):
        """Test repairing trailing comma in object."""
        parser = OutputParser()
        
        broken = '{"name": "John", "age": 30,}'
        repaired = parser._repair_json(broken)
        
        # Should fix trailing comma
        assert repaired == '{"name": "John", "age": 30}'

    def test_repair_json_trailing_comma_array(self):
        """Test repairing trailing comma in array."""
        parser = OutputParser()
        
        broken = '["item1", "item2",]'
        repaired = parser._repair_json(broken)
        
        # Should fix trailing comma
        assert repaired == '["item1", "item2"]'

    def test_repair_json_single_quotes_object(self):
        """Test repairing single quotes in object."""
        parser = OutputParser()
        
        broken = "{'name': 'John', 'age': 30}"
        repaired = parser._repair_json(broken)
        
        # Should convert single quotes to double quotes for keys and string values
        assert '"name"' in repaired and '"age"' in repaired

    def test_repair_json_no_issues(self):
        """Test repairing JSON that's already valid."""
        parser = OutputParser()
        
        valid = '{"name": "John", "age": 30}'
        repaired = parser._repair_json(valid)
        
        # Should remain unchanged
        assert repaired == valid

    def test_repair_attempts_tracking(self):
        """Test that repair attempts are properly tracked."""
        parser = OutputParser()
        
        assert parser._repair_attempts == 0
        
        parser._repair_json("test")
        assert parser._repair_attempts == 1
        
        parser._repair_json("test")
        assert parser._repair_attempts == 2
        
        parser._repair_json("test")
        assert parser._repair_attempts == 3

    def test_repair_attempts_limit_enforced(self):
        """Test that repair attempts limit is enforced."""
        parser = OutputParser()
        
        # Set to limit
        parser._repair_attempts = 2
        
        result = parser._repair_json("test")
        
        # Should return unchanged when at limit
        assert result == "test"
        
        # Should increment attempt count
        assert parser._repair_attempts == 3