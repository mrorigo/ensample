"""Output Parser for canonicalizing LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from .observability import LOGGER


class OutputParser:
    """Parser for canonicalizing and validating LLM outputs."""

    def __init__(self) -> None:
        self._repair_attempts = 0

    def parse_output(
        self,
        response_text: str,
        output_parser_schema: dict[str, Any] | None = None,
    ) -> tuple[str | dict[str, Any] | None, str | None]:
        """Parse and validate LLM output.

        Args:
            response_text: Raw response text from LLM
            output_parser_schema: Optional JSON schema for validation

        Returns:
            Tuple of (parsed_content, error_message)
            If successful, parsed_content is the parsed data and error_message is None
            If failed, parsed_content is None and error_message contains the error
        """
        if output_parser_schema is None:
            # No schema provided, return text as-is
            return response_text, None

        try:
            # First, try to parse as JSON
            parsed_data = json.loads(response_text)

            # Validate against schema if provided
            jsonschema.validate(parsed_data, output_parser_schema)

            LOGGER.debug("Output parsing successful", extra={
                "schema_provided": True,
                "response_length": len(response_text)
            })

            return parsed_data, None

        except json.JSONDecodeError as e:
            LOGGER.debug("JSON parsing failed, attempting repair", extra={
                "error": str(e),
                "response_preview": response_text[:200]
            })

            # Try to repair common JSON issues
            repaired_text = self._repair_json(response_text)
            if repaired_text != response_text:
                try:
                    repaired_data = json.loads(repaired_text)
                    jsonschema.validate(repaired_data, output_parser_schema)

                    LOGGER.info("JSON repair successful", extra={
                        "original_length": len(response_text),
                        "repaired_length": len(repaired_text)
                    })

                    return repaired_data, None

                except (json.JSONDecodeError, jsonschema.ValidationError) as repair_error:
                    LOGGER.warning("JSON repair failed", extra={
                        "repair_error": str(repair_error)
                    })

            return None, f"Failed to parse output as valid JSON: {e}"

        except jsonschema.ValidationError as e:
            LOGGER.debug("JSON schema validation failed", extra={
                "error": str(e),
                "response_preview": response_text[:200]
            })

            return None, f"Output does not match expected schema: {e}"

    def _repair_json(self, json_text: str) -> str:
        """Attempt to repair common JSON formatting issues."""
        if self._repair_attempts > 2:  # Limit repair attempts
            return json_text

        self._repair_attempts += 1

        # Common repairs - order matters for complex patterns
        repairs = [
            # Fix trailing commas in objects and arrays first
            (r',(\s*[}\]])', r'\1'),
            # Fix unquoted property names (keys)
            (r'\{([^"\']\w+)\s*:', r'{"\1":'),
            # Fix single quotes to double quotes for keys
            (r"'([^']*)':", r'"\1":'),
            # Fix single quotes to double quotes for string values (but not already quoted)
            (r':\s*\'([^\']*)\'', r': "\1"'),
            # Fix unquoted boolean values (before general identifier repair)
            (r':\s*\b(true)\b', r': true'),
            (r':\s*\b(false)\b', r': false'),
            (r':\s*\b(null)\b', r': null'),
            # Fix unquoted string values - only for simple identifiers
            (r':\s*\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?!\s*[}\],])', r': "\1"'),
        ]

        repaired = json_text
        for pattern, replacement in repairs:
            repaired = re.sub(pattern, replacement, repaired)

        return repaired

    def validate_structured_output(
        self,
        data: str | dict[str, Any],
        expected_schema: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Validate structured output against a schema.

        Args:
            data: The data to validate
            expected_schema: JSON schema to validate against

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if isinstance(data, str):
                # Try to parse string as JSON first
                data = json.loads(data)

            jsonschema.validate(data, expected_schema)
            return True, None

        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return False, str(e)

    def extract_key_fields(
        self,
        data: str | dict[str, Any],
        required_fields: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        """Extract required fields from structured data.

        Args:
            data: The data to extract from
            required_fields: List of required field names

        Returns:
            Tuple of (extracted_fields, missing_fields)
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return {}, required_fields

        if not isinstance(data, dict):
            return {}, required_fields

        extracted = {}
        missing = []

        for field in required_fields:
            if field in data:
                extracted[field] = data[field]
            else:
                missing.append(field)

        return extracted, missing
