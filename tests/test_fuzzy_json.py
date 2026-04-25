import unittest
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from claude_to_openai_forwarder.translators.tool_prompt import (
    _extract_json_fuzzy,
    _extract_fields_regex,
    _parse_simple_object,
    _fix_json_control_chars,
    parse_all_tool_calls,
    strip_control_text_tags,
)


class TestExtractJsonFuzzy(unittest.TestCase):
    """Tests for _extract_json_fuzzy function"""

    def test_valid_json(self):
        """Test extraction of valid JSON"""
        text = '{"type": "tool_use", "name": "Read", "id": "call_1", "input": {"file_path": "test.txt"}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_use")
        self.assertEqual(result["name"], "Read")
        self.assertEqual(result["id"], "call_1")

    def test_valid_json_with_text_prefix(self):
        """Test extraction when JSON is preceded by text"""
        text = 'Some explanatory text {"type": "tool_use", "name": "Edit", "input": {"file_path": "a.txt"}} followed by more text'
        # Find the actual position of "{"
        import re

        match = re.search(r'\{\s*"type"', text)
        self.assertIsNotNone(match)
        result = _extract_json_fuzzy(text, match.start())
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Edit")

    def test_missing_id(self):
        """Test extraction when id is missing - should still extract but without id (caller adds it)"""
        text = (
            '{"type": "tool_use", "name": "Read", "input": {"file_path": "test.txt"}}'
        )
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Read")
        # Note: ID is NOT added by this function - that's done by the caller

    def test_no_json_found(self):
        """Test when no JSON is found"""
        text = "This is just plain text without any JSON"
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNone(result)


class TestExtractFieldsRegex(unittest.TestCase):
    """Tests for _extract_fields_regex function"""

    def test_basic_extraction(self):
        """Test basic field extraction"""
        text = (
            '{"type": "tool_use", "name": "Edit", "id": "call_5", "input": {"a": "b"}}'
        )
        result = _extract_fields_regex(text, 0, len(text))
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Edit")

    def test_missing_type(self):
        """Test when type is missing"""
        text = '{"name": "Edit", "input": {}}'
        result = _extract_fields_regex(text, 0, len(text))
        self.assertIsNone(result)

    def test_missing_name(self):
        """Test when name is missing"""
        text = '{"type": "tool_use", "input": {}}'
        result = _extract_fields_regex(text, 0, len(text))
        self.assertIsNone(result)

    def test_non_tool_use_type(self):
        """Test when type is not tool_use"""
        text = '{"type": "text", "content": "hello"}'
        result = _extract_fields_regex(text, 0, len(text))
        self.assertIsNone(result)


class TestParseSimpleObject(unittest.TestCase):
    """Tests for _parse_simple_object function"""

    def test_simple_object(self):
        """Test parsing simple object"""
        obj_str = '{"file_path": "test.txt", "limit": 10}'
        result = _parse_simple_object(obj_str)
        self.assertEqual(result["file_path"], "test.txt")
        self.assertEqual(result["limit"], 10)

    def test_boolean_values(self):
        """Test parsing boolean values"""
        obj_str = '{"enabled": true, "disabled": false}'
        result = _parse_simple_object(obj_str)
        self.assertEqual(result["enabled"], True)
        self.assertEqual(result["disabled"], False)

    def test_null_value(self):
        """Test parsing null value"""
        obj_str = '{"value": null}'
        result = _parse_simple_object(obj_str)
        self.assertIsNone(result["value"])

    def test_empty_object(self):
        """Test parsing empty object"""
        obj_str = "{}"
        result = _parse_simple_object(obj_str)
        self.assertEqual(result, {})


class TestFixJsonControlChars(unittest.TestCase):
    """Tests for _fix_json_control_chars function"""

    def test_newline_in_string(self):
        """Test fixing literal newline in string"""
        json_str = '{"text": "line1\nline2"}'
        fixed = _fix_json_control_chars(json_str)
        result = json.loads(fixed)
        self.assertEqual(result["text"], "line1\nline2")

    def test_tab_in_string(self):
        """Test fixing literal tab in string"""
        json_str = '{"text": "col1\tcol2"}'
        fixed = _fix_json_control_chars(json_str)
        result = json.loads(fixed)
        self.assertEqual(result["text"], "col1\tcol2")

    def test_valid_json_unchanged(self):
        """Test that valid JSON is unchanged"""
        json_str = '{"type": "tool_use", "name": "Test"}'
        fixed = _fix_json_control_chars(json_str)
        result = json.loads(fixed)
        self.assertEqual(result["name"], "Test")


class TestStripControlTextTags(unittest.TestCase):
    def test_strips_result_content_wrappers(self):
        text = "Before\n<result content trimmed for brevity>hello</result>\nAfter"
        result = strip_control_text_tags(text)
        self.assertNotIn("<result content", result)
        self.assertNotIn("</result>", result)
        self.assertIn("hello", result)

    def test_strips_thinking_wrappers(self):
        text = "Before<thinking>internal reasoning</thinking>After"
        result = strip_control_text_tags(text)
        self.assertNotIn("<thinking>", result)
        self.assertNotIn("</thinking>", result)
        self.assertIn("internal reasoning", result)


class TestIntegration(unittest.TestCase):
    """Integration tests for the fuzzy extraction pipeline"""

    def test_real_world_malformed_case(self):
        """Test the actual case from the logs that was failing"""
        text = '{"type": "tool_use", "id": "call_29", "name": "AskUserQuestion", "input": {"questions": [{"question": "What additional information would you like to include in the README.md file next?", "header": "Additional Info", "options": [{"label": "Release notes", "description": "Include release notes or changelogs"}, {"label": "Related projects", "description": "Describe link to related projects or tools"}, {"label"] = "Customization guides", "description = "Provide detailed guides on customizing the project"}, {"label"] = "Other", "description": "Provide custom text to be added to README.md"}], "multiSelect": false}]}'

        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["type"], "tool_use")
        self.assertEqual(result["name"], "AskUserQuestion")

    def test_complex_nested_input(self):
        """Test with complex nested input structure"""
        text = '{"type": "tool_use", "name": "Edit", "input": {"replace_all": false, "file_path": "/test/README.md", "old_string": "## Usage", "new_string": "## Usage\\n\\n- Item 1\\n- Item 2"}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Edit")
        self.assertEqual(result["input"]["replace_all"], False)

    def test_text_with_multiple_tool_calls(self):
        """Test extraction when text contains multiple potential JSON objects"""
        text = 'Some text {"type": "tool_use", "name": "Read", "input": {"file_path": "a.txt"}} more text {"type": "tool_use", "name": "Edit", "input": {"file_path": "b.txt"}}'

        import re

        pattern = r'\{\s*"type"\s*:\s*"tool_use"'
        for match in re.finditer(pattern, text):
            result = _extract_json_fuzzy(text, match.start())
            if result:
                self.assertEqual(result["type"], "tool_use")

    def test_function_style_agent_call_with_json_like_object(self):
        text = """Certainly! I'll proceed.\n\nAgent({description: "Security review", subagent_type: "Explore", prompt: "Check auth", run_in_background: true})"""

        result = parse_all_tool_calls(text)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool_call"]["name"], "Agent")
        self.assertEqual(result[0]["tool_call"]["input"]["description"], "Security review")
        self.assertEqual(result[0]["tool_call"]["input"]["subagent_type"], "Explore")
        self.assertEqual(result[0]["tool_call"]["input"]["run_in_background"], True)


class TestEdgeCases(unittest.TestCase):
    """Additional edge case tests"""

    def test_empty_input(self):
        """Test with empty string"""
        result = _extract_json_fuzzy("", 0)
        self.assertIsNone(result)

    def test_whitespace_before_json(self):
        """Test JSON with leading whitespace"""
        text = '   {"type": "tool_use", "name": "Test", "input": {}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test")

    def test_json_with_numbers(self):
        """Test JSON with numeric values in input"""
        text = '{"type": "tool_use", "name": "Read", "input": {"offset": 100, "limit": 50}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["input"]["offset"], 100)
        self.assertEqual(result["input"]["limit"], 50)

    def test_json_with_arrays(self):
        """Test JSON with array values"""
        text = '{"type": "tool_use", "name": "MultiEdit", "input": {"edits": [{"a": 1}, {"b": 2}]}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "MultiEdit")

    def test_escaped_quotes_in_string(self):
        """Test JSON with escaped quotes"""
        text = '{"type": "tool_use", "name": "Edit", "input": {"old_string": "He said \\"hello\\"", "new_string": "Hi"}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertIn("old_string", result["input"])

    def test_unicode_characters(self):
        """Test JSON with unicode characters"""
        text = (
            '{"type": "tool_use", "name": "Test", "input": {"text": "Hello 世界 🌍"}}'
        )
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertIn("text", result["input"])

    def test_very_large_json(self):
        """Test with large JSON input"""
        large_input = {"data": "x" * 10000}
        text = json.dumps({"type": "tool_use", "name": "Test", "input": large_input})
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["input"]["data"]), 10000)

    def test_unclosed_braces(self):
        """Test with unclosed JSON braces - regex fallback extracts what it can"""
        text = '{"type": "tool_use", "name": "Test", "input": {'
        result = _extract_json_fuzzy(text, 0)
        # Fuzzy extraction uses regex fallback to extract partial data
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test")

    def test_only_opening_braces(self):
        """Test with only opening braces - regex fallback extracts what it can"""
        text = '{"type": "tool_use", "name": "Test"'
        result = _extract_json_fuzzy(text, 0)
        # Fuzzy extraction uses regex fallback to extract partial data
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test")

    def test_deeply_nested_json(self):
        """Test with deeply nested JSON structure"""
        nested = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        text = json.dumps({"type": "tool_use", "name": "Test", "input": nested})
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["input"]["a"]["b"]["c"]["d"]["e"], "value")

    def test_json_with_special_chars_in_keys(self):
        """Test JSON with special characters in keys"""
        text = '{"type": "tool_use", "name": "Test", "input": {"file_path_123": "test.txt", "limit-num": 10}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        # Note: _parse_simple_object only matches \w+ for keys, so this may not fully parse
        # but it should not crash

    def test_json_with_negative_numbers(self):
        """Test JSON with negative numbers"""
        text = '{"type": "tool_use", "name": "Test", "input": {"offset": -1, "count": -10}}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)

    def test_json_with_float_values(self):
        """Test JSON with float values"""
        text = (
            '{"type": "tool_use", "name": "Test", "input": {"ratio": 0.5, "pi": 3.14}}'
        )
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)

    def test_multiple_json_objects_not_tool_use(self):
        """Test extraction skips non-tool_use JSON"""
        text = '{"type": "text", "content": "hello"}{"type": "tool_use", "name": "Read", "input": {}}'

        import re

        pattern = r'\{\s*"type"\s*:\s*"tool_use"'
        match = re.search(pattern, text)
        result = _extract_json_fuzzy(text, match.start())
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Read")

    def test_tool_use_in_middle_of_text(self):
        """Test tool_use JSON embedded in middle of text"""
        text = 'Before text {"type": "tool_use", "name": "Edit", "input": {"x": 1}} After text'
        import re

        match = re.search(r'\{\s*"type"\s*:\s*"tool_use"', text)
        result = _extract_json_fuzzy(text, match.start())
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Edit")

    def test_empty_json_object(self):
        """Test with empty JSON object"""
        text = "{}"
        result = _extract_json_fuzzy(text, 0)
        # Empty JSON is valid but has no type field - caller will filter it out
        self.assertEqual(result, {})

    def test_tool_use_minimal(self):
        """Test minimal tool_use with just type and name"""
        text = '{"type":"tool_use","name":"Test"}'
        result = _extract_json_fuzzy(text, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Test")


class TestParseSimpleObjectEdgeCases(unittest.TestCase):
    """Edge case tests for _parse_simple_object"""

    def test_nested_braces_in_string(self):
        """Test when braces appear inside string values"""
        obj_str = '{"text": "a { b } c"}'
        result = _parse_simple_object(obj_str)
        self.assertEqual(result.get("text"), "a { b } c")

    def test_commas_in_string(self):
        """Test when commas appear inside string values"""
        obj_str = '{"text": "a, b, c"}'
        result = _parse_simple_object(obj_str)
        self.assertEqual(result.get("text"), "a, b, c")

    def test_quotes_in_string(self):
        """Test when quotes appear inside string values"""
        obj_str = '{"text": "a \\" b"}'
        result = _parse_simple_object(obj_str)
        # May not handle escaped quotes, but shouldn't crash

    def test_brace_imbalance(self):
        """Test with text containing tool_use JSON after other content"""
        obj_str = """
No files were found containing the pattern "preview_patch.*error" in the codebase.

Since I couldn't find any specific configurations or error messages related to `preview_patch`, I'll use the `AskUserQuestion` tool again to ask the user if they have any additional information about the issue or if they can provide more context.

{"type": "tool_use", "id": "ask_user_again", "name": "AskUserQuestion", "input": {"questions": [{"question": "Can you provide more context or details about the issue with preview_patch?", "header": "Additional context", "options": [{"label": "Claude Code configuration issue", "description": "There might be a configuration issue with Claude Code"}, {"label": "MCP tool implementation issue", "description": "There might be an issue with the MCP tool implementation"}, {"label": "Other", "description": "Other reasons or additional context"}], "multiSelect": false}]}
"""
        result = parse_all_tool_calls(obj_str)
        # Should find 1 tool call
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["tool_call"]["name"], "AskUserQuestion")


class TestFixControlCharsEdgeCases(unittest.TestCase):
    """Edge case tests for _fix_json_control_chars"""

    def test_carriage_return_in_string(self):
        """Test fixing carriage return in string"""
        json_str = '{"text": "line1\rline2"}'
        fixed = _fix_json_control_chars(json_str)
        result = json.loads(fixed)
        self.assertIn("text", result)

    def test_mixed_control_chars(self):
        """Test fixing mixed control characters"""
        json_str = '{"text": "a\nb\tc\rd"}'
        fixed = _fix_json_control_chars(json_str)
        result = json.loads(fixed)
        self.assertEqual(result["text"], "a\nb\tc\rd")

    def test_all_control_chars(self):
        """Test with various control characters"""
        json_str = '{"text": "a\x00b\x01c\x02d"}'
        fixed = _fix_json_control_chars(json_str)
        result = json.loads(fixed)
        self.assertIn("text", result)


if __name__ == "__main__":
    unittest.main()
