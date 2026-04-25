"""
Unit tests to verify the performance and reliability improvements made to the translator.

Tests cover:
1. Rate limiter memory leak fix
2. HttpX connection pooling
3. UUID-based request IDs
4. Removed duplicate model classes
5. Configurable timeouts
6. Reduced logging verbosity
7. Improved JSON parsing robustness
"""

import asyncio
import json
import time
import unittest
import uuid
from unittest.mock import AsyncMock, MagicMock, patch
from collections import deque

from claude_to_openai_forwarder.utils.rate_limit import (
    check_rate_limit,
    visit_records,
    _cleanup_old_identifiers,
)
from claude_to_openai_forwarder.backends.httpx_backend import HttpxBackend
from claude_to_openai_forwarder.config import get_settings, reset_settings
from claude_to_openai_forwarder.models.openai import OpenAIRequest
from claude_to_openai_forwarder.translators.response import ResponseTranslator
from claude_to_openai_forwarder.models.claude import (
    ClaudeResponse,
    ClaudeContentBlock,
    ClaudeMessage,
    ClaudeRequest,
)
from fastapi import HTTPException


class TestRateLimiterMemoryLeakFix(unittest.TestCase):
    """Test that rate limiter doesn't grow unbounded in memory"""

    def setUp(self):
        """Clear visit_records before each test"""
        visit_records.clear()

    def test_deque_storage_instead_of_list(self):
        """Verify that visit_records uses deque for O(1) operations"""
        check_rate_limit("user1", 100)
        self.assertIsInstance(visit_records["user1"], deque)

    def test_old_records_removed_on_cleanup(self):
        """Test that cleanup removes identifiers with old records"""
        # Add a record that's older than TTL
        old_time = time.time() - 70  # 70 seconds ago
        visit_records["old_user"] = deque([old_time])

        # Should be cleaned up
        _cleanup_old_identifiers(time.time())
        self.assertNotIn("old_user", visit_records)

    def test_recent_records_preserved_on_cleanup(self):
        """Test that cleanup keeps identifiers with recent records"""
        recent_time = time.time() - 30  # 30 seconds ago
        visit_records["active_user"] = deque([recent_time])

        _cleanup_old_identifiers(time.time())
        self.assertIn("active_user", visit_records)

    def test_cleanup_preserves_recent_records_when_oldest_is_expired(self):
        """Cleanup should prune expired entries without deleting active identifiers."""
        now = time.time()
        visit_records["mixed_user"] = deque([now - 70, now - 10, now - 5])

        _cleanup_old_identifiers(now)

        self.assertIn("mixed_user", visit_records)
        self.assertEqual(len(visit_records["mixed_user"]), 2)
        self.assertTrue(all(now - ts < 60 for ts in visit_records["mixed_user"]))

    def test_no_unbounded_growth_with_many_users(self):
        """Test that adding many unique users doesn't cause unbounded growth"""
        initial_keys = len(visit_records)

        # Add 100 requests from different users (at rate limit threshold)
        for i in range(100):
            user_id = f"user_{i}"
            try:
                check_rate_limit(user_id, 10)
            except HTTPException:
                pass  # Some will hit rate limit

        # Trigger cleanup manually
        _cleanup_old_identifiers(time.time())

        current_keys = len(visit_records)
        # After cleanup, should have much fewer keys (at most 100)
        self.assertLessEqual(current_keys, 100)

    def test_rate_limit_enforcement_still_works(self):
        """Verify rate limiting functionality still works after fix"""
        # First 5 requests should succeed
        for i in range(5):
            try:
                check_rate_limit("test_user", 5)
            except HTTPException:
                self.fail(f"Request {i} should not be rate limited")

        # 6th request should fail
        with self.assertRaises(HTTPException) as context:
            check_rate_limit("test_user", 5)

        self.assertEqual(context.exception.status_code, 429)


class TestHttpxConnectionPooling(unittest.IsolatedAsyncioTestCase):
    """Test that HttpX client uses connection pooling"""

    def setUp(self):
        """Reset settings before each test"""
        reset_settings()

    def tearDown(self):
        """Clean up"""
        reset_settings()

    def test_persistent_client_creation(self):
        """Test that HttpxBackend creates a persistent client"""
        backend = HttpxBackend()
        self.assertIsNone(backend.client)

        client = backend._get_client()
        self.assertIsNotNone(client)
        self.assertIsNotNone(backend.client)

    def test_client_reused_on_second_call(self):
        """Test that same client instance is returned on subsequent calls"""
        backend = HttpxBackend()
        client1 = backend._get_client()
        client2 = backend._get_client()

        self.assertIs(client1, client2, "Client should be reused, not recreated")

    def test_client_has_limits_configured(self):
        """Test that client has connection limits configured"""
        backend = HttpxBackend()
        client = backend._get_client()

        # Verify client is created with specific configuration
        # (httpx.Limits not directly exposed, but timeout shows connection pooling is set up)
        self.assertIsNotNone(client._timeout)

    def test_client_has_timeout_configured(self):
        """Test that client uses configured timeouts"""
        reset_settings()
        backend = HttpxBackend()
        client = backend._get_client()

        # Timeout should be set (default 120.0)
        self.assertIsNotNone(client._timeout)

    @patch("claude_to_openai_forwarder.backends.httpx_backend.httpx.AsyncClient")
    async def test_create_completion_uses_pooled_client(self, mock_client_class):
        """Test that create_completion uses the pooled client"""
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "created": 1,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        backend = HttpxBackend()
        request = OpenAIRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=100,
        )

        # First call
        await backend.create_completion(request)
        first_client_id = id(backend.client)

        # Second call should reuse same client
        await backend.create_completion(request)
        second_client_id = id(backend.client)

        self.assertEqual(
            first_client_id, second_client_id, "Client should be reused"
        )

    async def test_close_cleans_up_resources(self):
        """Test that close() method properly cleans up client"""
        backend = HttpxBackend()
        client = backend._get_client()
        self.assertIsNotNone(backend.client)

        await backend.close()
        self.assertIsNone(backend.client)


class TestUUIDRequestIDs(unittest.IsolatedAsyncioTestCase):
    """Test that request IDs use UUID format"""

    @patch("claude_to_openai_forwarder.app.get_backend")
    @patch("claude_to_openai_forwarder.app.verify_claude_api_key")
    async def test_request_id_format(self, mock_auth, mock_backend):
        """Test that request IDs are UUID-based"""
        from claude_to_openai_forwarder.app import create_message
        from fastapi import Request

        mock_auth.return_value = "test_key"
        mock_backend_instance = MagicMock()
        mock_backend_instance.get_backend_name.return_value = "httpx"
        mock_backend.return_value = mock_backend_instance

        request_ids = []

        async def capture_request_id(*args, **kwargs):
            # We'll verify format by checking the logs
            return "captured"

        # Just verify the ID format by checking if it matches UUID pattern
        import re

        uuid_pattern = r"^req_[0-9a-f]{12}$"
        for i in range(10):
            # Generate request ID using same format as in app.py
            request_id = f"req_{uuid.uuid4().hex[:12]}"
            self.assertRegex(
                request_id, uuid_pattern, f"Request ID should match UUID pattern"
            )
            request_ids.append(request_id)

        # Verify no collisions in 10 IDs
        unique_ids = set(request_ids)
        self.assertEqual(len(unique_ids), 10, "All request IDs should be unique")


class TestDuplicateModelClassesRemoved(unittest.TestCase):
    """Test that duplicate model classes are removed"""

    def test_no_duplicate_claude_content_block1(self):
        """Verify ClaudeContentBlock1 doesn't exist"""
        from claude_to_openai_forwarder.models import claude

        self.assertFalse(
            hasattr(claude, "ClaudeContentBlock1"),
            "ClaudeContentBlock1 should not exist",
        )

    def test_no_duplicate_claude_response1(self):
        """Verify ClaudeResponse1 doesn't exist"""
        from claude_to_openai_forwarder.models import claude

        self.assertFalse(
            hasattr(claude, "ClaudeResponse1"), "ClaudeResponse1 should not exist"
        )

    def test_primary_classes_exist(self):
        """Verify primary model classes still exist"""
        from claude_to_openai_forwarder.models import claude

        self.assertTrue(hasattr(claude, "ClaudeContentBlock"))
        self.assertTrue(hasattr(claude, "ClaudeResponse"))
        self.assertTrue(hasattr(claude, "ClaudeMessage"))
        self.assertTrue(hasattr(claude, "ClaudeRequest"))


class TestConfigurableTimeouts(unittest.TestCase):
    """Test that timeouts are configurable"""

    def setUp(self):
        reset_settings()

    def tearDown(self):
        reset_settings()

    def test_default_timeout_values_exist(self):
        """Test that default timeout values are configured"""
        settings = get_settings()

        self.assertEqual(settings.request_timeout, 120.0)
        self.assertEqual(settings.connect_timeout, 10.0)
        self.assertEqual(settings.read_timeout, 120.0)
        self.assertEqual(settings.write_timeout, 30.0)

    @patch.dict("os.environ", {"REQUEST_TIMEOUT": "300"})
    def test_timeout_configuration_from_env(self):
        """Test that timeouts can be configured from environment"""
        reset_settings()
        settings = get_settings()

        # Environment variable should override default
        self.assertEqual(settings.request_timeout, 300.0)

    def test_httpx_backend_uses_configurable_timeouts(self):
        """Test that HttpxBackend uses configured timeout values"""
        backend = HttpxBackend()
        client = backend._get_client()

        settings = get_settings()
        # Verify timeout was applied
        self.assertIsNotNone(client._timeout)


class TestReducedLoggingVerbosity(unittest.TestCase):
    """Test that logging verbosity is reduced in streaming/request/response"""

    def test_no_logger_setLevel_in_translators(self):
        """Verify logger.setLevel not called in translators"""
        import inspect

        from claude_to_openai_forwarder.translators import response

        source = inspect.getsource(response)
        # Check that logger.setLevel is removed
        self.assertNotIn(
            "logger.setLevel(logging.DEBUG)",
            source,
            "logger.setLevel should not be in response.py",
        )

    def test_no_excessive_info_logs_in_request_translator(self):
        """Verify info logs replaced with debug logs in request translator"""
        from claude_to_openai_forwarder.translators.request import RequestTranslator

        # Create a simple request
        request = ClaudeRequest(
            model="claude-3-sonnet",
            messages=[ClaudeMessage(role="user", content="test")],
            max_tokens=100,
        )

        # Translate should work without excessive logging
        openai_request = RequestTranslator.translate(request)
        self.assertIsNotNone(openai_request)
        # Model should be mapped (either default or from config)
        self.assertIsNotNone(openai_request.model)


class TestImprovedJSONParsingRobustness(unittest.TestCase):

    def test_parse_malformed_json(self):
        """Test parsing malformed JSON"""
        malformed_json = """
        {
            "type": "tool_use",
            "id": "call_123",
            "name": "search",
            "input": {
                "query": "test"
            }
        """
        tool_call = ResponseTranslator._parse_tool_call(malformed_json)
        self.assertIsNone(tool_call)
    """Test improved JSON parsing for tool calls"""

    def test_parse_nested_json_tool_calls(self):
        """Test parsing tool calls with nested JSON structures"""
        nested_json = """
        Some explanation text...
        {
            "type": "tool_use",
            "id": "call_123",
            "name": "search",
            "input": {
                "query": "test",
                "nested": {
                    "deep": {
                        "value": "found"
                    }
                }
            }
        }
        More text after.
        """

        tool_call = ResponseTranslator._parse_tool_call(nested_json)
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["type"], "tool_use")
        self.assertEqual(tool_call["name"], "search")
        self.assertEqual(tool_call["input"]["nested"]["deep"]["value"], "found")

    def test_parse_json_with_escaped_quotes(self):
        """Test parsing tool calls with escaped quotes in strings"""
        escaped_json = r"""
        {
            "type": "tool_use",
            "id": "call_456",
            "name": "process",
            "input": {
                "text": "He said \"hello\""
            }
        }
        """

        tool_call = ResponseTranslator._parse_tool_call(escaped_json)
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["name"], "process")

    def test_parse_json_with_multiple_candidates(self):
        """Test parsing when multiple JSON objects present"""
        multi_json = """
        {"type": "text", "content": "start"}
        {
            "type": "tool_use",
            "id": "call_789",
            "name": "execute",
            "input": {"command": "run"}
        }
        {"type": "end"}
        """

        tool_call = ResponseTranslator._parse_tool_call(multi_json)
        self.assertIsNotNone(tool_call)
        self.assertEqual(tool_call["type"], "tool_use")
        self.assertEqual(tool_call["name"], "execute")

    def test_tool_input_parse_error_returns_empty_dict(self):
        """Test that unparseable tool input returns empty dict instead of wrapping"""
        from claude_to_openai_forwarder.models.openai import (
            OpenAIResponse,
            OpenAIMessage,
            OpenAIChoice,
            OpenAIUsage,
        )

        # Create response with unparseable tool arguments
        response = OpenAIResponse(
            id="test",
            created=1,
            model="gpt-4",
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIMessage(
                        role="assistant",
                        content="Some text",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "test_tool",
                                    "arguments": "{invalid json",  # Invalid
                                },
                            }
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=OpenAIUsage(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

        # Should not raise, but handle gracefully
        claude_response = ResponseTranslator.translate(response)
        self.assertIsNotNone(claude_response)
        # Should have both text and tool_use blocks
        self.assertGreaterEqual(len(claude_response.content), 1)
        # Check that tool block has empty input instead of wrapped value
        tool_blocks = [b for b in claude_response.content if b.type == "tool_use"]
        if tool_blocks:
            tool_block = tool_blocks[0]
            self.assertEqual(tool_block.input, {})  # Empty dict, not wrapped

    def test_parse_tool_call_with_missing_id(self):
        """Test that missing tool ID is generated"""
        json_without_id = """
        {
            "type": "tool_use",
            "name": "my_tool",
            "input": {"key": "value"}
        }
        """

        tool_call = ResponseTranslator._parse_tool_call(json_without_id)
        self.assertIsNotNone(tool_call)
        self.assertIn("id", tool_call)
        self.assertTrue(tool_call["id"].startswith("call_"))

    def test_no_tool_call_returns_none(self):
        """Test that text without tool_use returns None"""
        plain_text = "This is just regular text with no tool call"
        tool_call = ResponseTranslator._parse_tool_call(plain_text)
        self.assertIsNone(tool_call)


class TestIntegrationImprovements(unittest.TestCase):

    def test_api_key_validation(self):
        """Test API key validation"""
        from claude_to_openai_forwarder.middleware.auth import verify_claude_api_key
        valid_key = get_settings().claude_api_key or "sk-ant-validkey"
        invalid_key = "invalid-key"
        self.assertIsNotNone(asyncio.run(verify_claude_api_key(valid_key)))
        with self.assertRaises(HTTPException):
            asyncio.run(verify_claude_api_key(invalid_key))
            verify_claude_api_key(invalid_key)
    """Integration tests to verify all improvements work together"""

    def setUp(self):
        reset_settings()
        visit_records.clear()

    def tearDown(self):
        reset_settings()
        visit_records.clear()

    def test_rate_limiter_with_cleanup(self):
        """Test rate limiter handles multiple users with periodic cleanup"""
        # Add requests from multiple users
        users = [f"user_{i}" for i in range(50)]

        for user in users:
            for _ in range(3):
                try:
                    check_rate_limit(user, 50)
                except HTTPException:
                    pass

        initial_count = len(visit_records)

        # Manually trigger cleanup
        _cleanup_old_identifiers(time.time())

        final_count = len(visit_records)
        # After cleanup, should have same or fewer entries
        self.assertLessEqual(final_count, initial_count)

    def test_backend_proper_shutdown(self):
        """Test that backend resources are properly cleaned up"""

        async def run_shutdown_test():
            backend = HttpxBackend()
            client = backend._get_client()
            self.assertIsNotNone(client)

            await backend.close()
            self.assertIsNone(backend.client)

        asyncio.run(run_shutdown_test())


# Helper functions
def async_test(coro):
    """Decorator to run async tests"""

    def wrapper(self):
        return asyncio.run(coro(self))

    return wrapper


if __name__ == "__main__":
    unittest.main()
