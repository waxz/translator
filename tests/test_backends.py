import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from claude_to_openai_forwarder.app import app
from claude_to_openai_forwarder.middleware.auth import verify_claude_api_key
from claude_to_openai_forwarder.models.claude import ClaudeRequest
from claude_to_openai_forwarder.models.openai import OpenAIRequest
from claude_to_openai_forwarder.translators.request import RequestTranslator
from claude_to_openai_forwarder.translators.streaming import StreamingTranslator
from claude_to_openai_forwarder.utils.exceptions import OpenAIAPIError
from fastapi import HTTPException


def make_request(**overrides) -> OpenAIRequest:
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "stream": False,
    }
    payload.update(overrides)
    return OpenAIRequest.model_validate(payload)


class FakeHTTPXResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text or json.dumps(self._json_data)

    def json(self):
        return self._json_data


class FakeHTTPXStreamResponse:
    def __init__(self, status_code=200, chunks=None, error_body=b""):
        self.status_code = status_code
        self._chunks = chunks or []
        self._error_body = error_body

    async def aread(self):
        return self._error_body

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAsyncClient:
    def __init__(self, *, post_response=None, stream_response=None, post_spy=None):
        self._post_response = post_response
        self._stream_response = stream_response
        self._post_spy = post_spy

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        if self._post_spy is not None:
            self._post_spy["args"] = args
            self._post_spy["kwargs"] = kwargs
        return self._post_response

    def stream(self, *args, **kwargs):
        return FakeStreamContext(self._stream_response)


class BackendFactoryTests(unittest.TestCase):
    def tearDown(self):
        import claude_to_openai_forwarder.backends

        claude_to_openai_forwarder.backends._backend_cache = None

    def test_get_backend_returns_httpx_backend(self):
        import claude_to_openai_forwarder.backends

        sentinel = object()
        with patch("claude_to_openai_forwarder.backends.get_settings", return_value=SimpleNamespace(backend_type="httpx")), patch(
            "claude_to_openai_forwarder.backends.HttpxBackend", return_value=sentinel
        ):
            backend = claude_to_openai_forwarder.backends.get_backend()

        self.assertIs(backend, sentinel)

    def test_get_backend_returns_litellm_backend(self):
        import claude_to_openai_forwarder.backends

        sentinel = object()
        with patch("claude_to_openai_forwarder.backends.get_settings", return_value=SimpleNamespace(backend_type="litellm")), patch(
            "claude_to_openai_forwarder.backends.litellm_backend.LiteLLMBackend", return_value=sentinel
        ):
            backend = claude_to_openai_forwarder.backends.get_backend()

        self.assertIs(backend, sentinel)


class AuthMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_claude_api_key_accepts_exact_configured_key(self):
        with patch(
            "claude_to_openai_forwarder.middleware.auth.get_settings",
            return_value=SimpleNamespace(claude_api_key="sk-ant-secret"),
        ):
            result = await verify_claude_api_key("sk-ant-secret")

        self.assertEqual(result, "sk-ant-secret")

    async def test_verify_claude_api_key_rejects_wrong_configured_key(self):
        with patch(
            "claude_to_openai_forwarder.middleware.auth.get_settings",
            return_value=SimpleNamespace(claude_api_key="sk-ant-secret"),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await verify_claude_api_key("sk-ant-wrong")

        self.assertEqual(ctx.exception.status_code, 401)

    async def test_verify_claude_api_key_allows_missing_key_when_unconfigured(self):
        with patch(
            "claude_to_openai_forwarder.middleware.auth.get_settings",
            return_value=SimpleNamespace(claude_api_key=None),
        ):
            result = await verify_claude_api_key(None)

        self.assertEqual(result, "test-key")


class HttpxBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = SimpleNamespace(
            openai_base_url="https://api.example.com/v1",
            openai_api_key="test-key",
        )
        settings_patcher = patch("claude_to_openai_forwarder.backends.httpx_backend.get_settings", return_value=settings)
        self.addCleanup(settings_patcher.stop)
        settings_patcher.start()

        from claude_to_openai_forwarder.backends.httpx_backend import HttpxBackend

        self.backend = HttpxBackend()

    async def test_create_completion_posts_chat_completion_request(self):
        post_spy = {}
        response = FakeHTTPXResponse(
            json_data={
                "id": "chatcmpl-1",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        )

        with patch(
            "claude_to_openai_forwarder.backends.httpx_backend.httpx.AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(
                post_response=response,
                post_spy=post_spy,
            ),
        ):
            result = await self.backend.create_completion(make_request())

        self.assertEqual(result.id, "chatcmpl-1")
        self.assertEqual(post_spy["args"][0], "https://api.example.com/v1/chat/completions")
        self.assertEqual(post_spy["kwargs"]["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(post_spy["kwargs"]["json"]["model"], "gpt-4o-mini")

    async def test_create_completion_raises_openai_api_error_on_non_200(self):
        response = FakeHTTPXResponse(
            status_code=429,
            json_data={"error": {"message": "rate limited"}},
        )

        with patch(
            "claude_to_openai_forwarder.backends.httpx_backend.httpx.AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(post_response=response),
        ):
            with self.assertRaises(OpenAIAPIError) as ctx:
                await self.backend.create_completion(make_request())

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.message, "rate limited")

    async def test_create_completion_stream_yields_upstream_chunks(self):
        chunks = [b"data: first\n\n", b"data: second\n\n"]
        stream_response = FakeHTTPXStreamResponse(status_code=200, chunks=chunks)

        with patch(
            "claude_to_openai_forwarder.backends.httpx_backend.httpx.AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(stream_response=stream_response),
        ):
            result = [
                chunk
                async for chunk in self.backend.create_completion_stream(
                    make_request(stream=True)
                )
            ]

        self.assertEqual(result, chunks)

    async def test_create_completion_stream_raises_openai_api_error_on_non_200(self):
        error_body = b'{"error":{"message":"bad request"}}'
        stream_response = FakeHTTPXStreamResponse(status_code=400, error_body=error_body)

        with patch(
            "claude_to_openai_forwarder.backends.httpx_backend.httpx.AsyncClient",
            side_effect=lambda timeout: FakeAsyncClient(stream_response=stream_response),
        ):
            with self.assertRaises(OpenAIAPIError) as ctx:
                result = [
                    chunk
                    async for chunk in self.backend.create_completion_stream(
                        make_request(stream=True)
                    )
                ]
                self.assertEqual(result, [])

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.message, "bad request")


class LiteLLMBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        settings = SimpleNamespace(
            openai_api_key="nvidia-key",
            openai_base_url="https://integrate.api.nvidia.com/v1",
            model_provider=None,
        )
        settings_patcher = patch("claude_to_openai_forwarder.backends.litellm_backend.get_settings", return_value=settings)
        self.addCleanup(settings_patcher.stop)
        settings_patcher.start()

        from claude_to_openai_forwarder.backends.litellm_backend import LiteLLMBackend

        self.backend = LiteLLMBackend()

    def test_convert_to_litellm_format_embeds_tools_for_nvidia_nim(self):
        request = make_request(
            model="meta/llama-4-maverick-17b-128e-instruct",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "Bash",
                        "description": "Execute bash",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        converted = self.backend._convert_to_litellm_format(request)

        self.assertEqual(
            converted["model"],
            "nvidia_nim/meta/llama-4-maverick-17b-128e-instruct",
        )
        self.assertEqual(converted["api_base"], "https://integrate.api.nvidia.com/v1")
        self.assertNotIn("tools", converted)
        self.assertEqual(converted["messages"][0]["role"], "system")
        self.assertIn("Available Tools", converted["messages"][0]["content"])

    def test_convert_to_litellm_format_uses_configured_provider(self):
        self.backend.settings.model_provider = "nvidia_nim"
        request = make_request(model="some/other-model")

        converted = self.backend._convert_to_litellm_format(request)

        self.assertEqual(converted["model"], "nvidia_nim/some/other-model")
        self.assertEqual(converted["api_base"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(converted["api_key"], "nvidia-key")

    async def test_create_completion_converts_litellm_response_to_openai_response(self):
        litellm_response = SimpleNamespace(
            id="chatcmpl-litellm",
            created=123,
            model="meta/llama-4-maverick-17b-128e-instruct",
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="stop",
                    message=SimpleNamespace(role="assistant", content="hello", tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

        with patch(
            "claude_to_openai_forwarder.backends.litellm_backend.litellm.acompletion",
            new=AsyncMock(return_value=litellm_response),
        ) as completion_mock:
            result = await self.backend.create_completion(make_request())

        self.assertEqual(result.id, "chatcmpl-litellm")
        self.assertEqual(result.choices[0].message.content, "hello")
        self.assertEqual(result.usage.total_tokens, 6)
        completion_mock.assert_awaited_once()

    async def test_create_completion_normalizes_object_tool_calls(self):
        litellm_response = SimpleNamespace(
            id="chatcmpl-litellm-tool",
            created=123,
            model="meta/llama-4-maverick-17b-128e-instruct",
            choices=[
                SimpleNamespace(
                    index=0,
                    finish_reason="tool_calls",
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                type="function",
                                function=SimpleNamespace(
                                    name="Glob",
                                    arguments='{"pattern":"./app/**"}',
                                ),
                            )
                        ],
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2, total_tokens=6),
        )

        with patch(
            "claude_to_openai_forwarder.backends.litellm_backend.litellm.acompletion",
            new=AsyncMock(return_value=litellm_response),
        ):
            result = await self.backend.create_completion(make_request())

        self.assertEqual(result.choices[0].finish_reason, "tool_calls")
        self.assertEqual(result.choices[0].message.tool_calls[0]["id"], "call_1")
        self.assertEqual(
            result.choices[0].message.tool_calls[0]["function"]["name"], "Glob"
        )

    async def test_create_completion_stream_emits_sse_chunks(self):
        async def fake_stream():
            yield SimpleNamespace(
                id="chunk-1",
                created=10,
                model="meta/llama-4-maverick-17b-128e-instruct",
                choices=[
                    SimpleNamespace(
                        delta={"content": "hel"},
                        finish_reason=None,
                    )
                ],
                usage=None,
            )
            yield SimpleNamespace(
                id="chunk-2",
                created=11,
                model="meta/llama-4-maverick-17b-128e-instruct",
                choices=[
                    SimpleNamespace(
                        delta={"content": "lo"},
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5),
            )

        with patch(
            "claude_to_openai_forwarder.backends.litellm_backend.litellm.acompletion",
            new=AsyncMock(return_value=fake_stream()),
        ):
            chunks = [
                chunk
                async for chunk in self.backend.create_completion_stream(
                    make_request(model="meta/llama-4-maverick-17b-128e-instruct", stream=True)
                )
            ]

        decoded = [chunk.decode("utf-8") for chunk in chunks]
        self.assertEqual(len(decoded), 2)
        self.assertIn('"content": "hel"', decoded[0])
        self.assertIn('"finish_reason": "stop"', decoded[1])
        self.assertIn('"total_tokens": 5', decoded[1])

    async def test_create_completion_stream_normalizes_object_delta_tool_calls(self):
        async def fake_stream():
            yield SimpleNamespace(
                id="chunk-1",
                created=10,
                model="meta/llama-4-maverick-17b-128e-instruct",
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call_1",
                                    function=SimpleNamespace(
                                        name="Glob",
                                        arguments='{"pattern":"./app/**"}',
                                    ),
                                )
                            ]
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )

        with patch(
            "claude_to_openai_forwarder.backends.litellm_backend.litellm.acompletion",
            new=AsyncMock(return_value=fake_stream()),
        ):
            chunks = [
                chunk
                async for chunk in self.backend.create_completion_stream(
                    make_request(model="meta/llama-4-maverick-17b-128e-instruct", stream=True)
                )
            ]

        decoded = chunks[0].decode("utf-8")
        self.assertIn('"tool_calls"', decoded)
        self.assertIn('"id": "call_1"', decoded)
        self.assertIn('"name": "Glob"', decoded)


class StreamingTranslatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_translate_stream_converts_json_text_into_tool_use_events(self):
        async def openai_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"{\\"type\\":\\"tool_use\\",'
                b'\\"name\\":\\"Glob\\",\\"input\\":{\\"pattern\\":\\"./app/**\\"}}"},'
                b'"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":4}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        events = [event async for event in StreamingTranslator.translate_stream(openai_stream())]
        joined = "".join(events)

        self.assertIn('"type": "tool_use"', joined)
        self.assertIn('"name": "Glob"', joined)
        self.assertIn('"stop_reason": "tool_use"', joined)
        self.assertNotIn('"type": "text_delta"', joined)

    async def test_translate_stream_preserves_normal_text_events(self):
        async def openai_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"choices":[{"delta":{"content":" world"},"finish_reason":"stop"}],'
                b'"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        events = [event async for event in StreamingTranslator.translate_stream(openai_stream())]
        joined = "".join(events)

        self.assertIn('"type": "text_delta"', joined)
        self.assertIn('"text": "hello"', joined)
        self.assertIn('"text": " world"', joined)
        self.assertIn('"stop_reason": "end_turn"', joined)

    async def test_translate_stream_converts_trailing_embedded_tool_use_after_text(self):
        async def openai_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"It seems that the Glob tool is not able to '
                b'find any files in ./app. Let me use Bash instead. "},"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"choices":[{"delta":{"content":"{\\"type\\":\\"tool_use\\",'
                b'\\"name\\":\\"Bash\\",\\"input\\":{\\"command\\":\\"ls -R ./app\\"}}"},'
                b'"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        events = [event async for event in StreamingTranslator.translate_stream(openai_stream())]
        joined = "".join(events)

        self.assertIn('"type": "tool_use"', joined)
        self.assertIn('"name": "Bash"', joined)
        self.assertIn('"stop_reason": "tool_use"', joined)
        self.assertNotIn('\\"type\\":\\"tool_use\\"', joined)

    async def test_translate_stream_converts_embedded_tool_use_for_arbitrary_tool_name(self):
        async def openai_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"I need to call a custom tool. "},'
                b'"finish_reason":null}]}\n\n'
            )
            yield (
                b'data: {"choices":[{"delta":{"content":"{\\"type\\":\\"tool_use\\",'
                b'\\"name\\":\\"CustomTool\\",\\"input\\":{\\"path\\":\\"./app\\",'
                b'\\"options\\":{\\"recursive\\":true,\\"limit\\":5}}}"},'
                b'"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        events = [event async for event in StreamingTranslator.translate_stream(openai_stream())]
        joined = "".join(events)

        self.assertIn('"type": "tool_use"', joined)
        self.assertIn('"name": "CustomTool"', joined)
        self.assertIn('\\"recursive\\": true', joined)
        self.assertIn('\\"limit\\": 5', joined)
        self.assertIn('"stop_reason": "tool_use"', joined)

    async def test_translate_stream_handles_sse_json_split_across_chunks(self):
        async def openai_stream():
            yield b'data: {"choices":[{"delta":{"content":"hello '
            yield b'world"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
            yield b"data: [DONE]\n\n"

        events = [event async for event in StreamingTranslator.translate_stream(openai_stream())]
        joined = "".join(events)

        self.assertIn('"type": "text_delta"', joined)
        self.assertIn('"text": "hello world"', joined)
        self.assertIn('"stop_reason": "end_turn"', joined)


class RequestTranslatorTests(unittest.TestCase):
    def test_translate_uses_prompt_tool_mode_for_tool_history(self):
        claude_request = ClaudeRequest.model_validate(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "tools": [
                    {
                        "name": "Glob",
                        "description": "List files matching a glob pattern",
                        "input_schema": {
                            "type": "object",
                            "properties": {"pattern": {"type": "string"}},
                            "required": ["pattern"],
                        },
                    }
                ],
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Glob",
                                "input": {"pattern": "./app/**"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "./app/main.py\n./app/config.py",
                            }
                        ],
                    },
                ],
            }
        )

        with patch(
            "claude_to_openai_forwarder.translators.request.get_settings",
            return_value=SimpleNamespace(
                default_openai_model="meta/llama-4-maverick-17b-128e-instruct",
                claude_model_map={},
                force_tool_in_prompt=True,
            ),
        ):
            translated = RequestTranslator.translate(claude_request)

        self.assertIsNone(translated.tools)
        self.assertIsNone(translated.tool_choice)
        self.assertEqual(translated.messages[0].role, "system")
        self.assertIn("Available Tools", translated.messages[0].content)
        self.assertEqual(translated.messages[1].role, "assistant")
        self.assertIn('"type": "tool_use"', translated.messages[1].content)
        self.assertIn('"name": "Glob"', translated.messages[1].content)
        self.assertEqual(translated.messages[2].role, "user")
        self.assertIn("Tool result received.", translated.messages[2].content)
        self.assertIn("tool_use_id: toolu_1", translated.messages[2].content)
        self.assertIn("./app/main.py", translated.messages[2].content)

    def test_translate_uses_claude_model_map_from_settings(self):
        claude_request = ClaudeRequest.model_validate(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "hello"}],
            }
        )

        with patch(
            "claude_to_openai_forwarder.translators.request.get_settings",
            return_value=SimpleNamespace(
                default_openai_model="gpt-4o-mini",
                claude_model_map={
                    "claude-3-5-sonnet-20241022": "meta/llama-4-maverick-17b-128e-instruct"
                },
                force_tool_in_prompt=False,
            ),
        ):
            translated = RequestTranslator.translate(claude_request)

        self.assertEqual(
            translated.model, "meta/llama-4-maverick-17b-128e-instruct"
        )

    def test_translate_prompt_tool_mode_is_generic_for_non_nvidia_provider(self):
        claude_request = ClaudeRequest.model_validate(
            {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 128,
                "tools": [
                    {
                        "name": "CustomTool",
                        "description": "Run a custom action",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "messages": [
                    {"role": "user", "content": "use the custom tool"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_custom_1",
                                "name": "CustomTool",
                                "input": {"path": "./app"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_custom_1",
                                "content": "ok",
                            }
                        ],
                    },
                ],
            }
        )

        with patch(
            "claude_to_openai_forwarder.translators.request.get_settings",
            return_value=SimpleNamespace(
                default_openai_model="gpt-oss:120b-cloud",
                claude_model_map={
                    "claude-3-5-sonnet-20241022": "gpt-oss:120b-cloud"
                },
                force_tool_in_prompt=True,
            ),
        ):
            translated = RequestTranslator.translate(claude_request)

        self.assertEqual(translated.model, "gpt-oss:120b-cloud")
        self.assertIsNone(translated.tools)
        self.assertEqual(translated.messages[0].role, "system")
        self.assertIn("CustomTool", translated.messages[0].content)
        self.assertEqual(translated.messages[1].role, "user")
        self.assertEqual(translated.messages[2].role, "assistant")
        self.assertIn('"type": "tool_use"', translated.messages[2].content)
        self.assertEqual(translated.messages[3].role, "user")
        self.assertIn("Tool result received.", translated.messages[3].content)


    async def test_count_tokens_endpoint(self):
        """Verify the /v1/messages/count_tokens endpoint returns token usage."""
        # Minimal payload with two messages
        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
        }
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/v1/messages/count_tokens",
                headers={"x-api-key": "sk-ant-test123"},
                json=payload,
            )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Should contain a usage object with input and output token counts
        self.assertIn("usage", data)
        self.assertIn("input_tokens", data["usage"])
        self.assertIn("output_tokens", data["usage"])
        # For a token‑count request output_tokens should be 0 (no generation)
        self.assertEqual(data["usage"]["output_tokens"], 0)

    def setUp(self):
        self.settings = SimpleNamespace(
            default_openai_model="gpt-4o-mini",
            openai_base_url="https://api.example.com/v1",
            claude_api_key=None,
        )

    async def test_messages_endpoint_returns_claude_text_response(self):
        fake_backend = MagicMock()
        fake_backend.get_backend_name.return_value = "httpx"
        fake_backend.create_completion = AsyncMock(
            return_value=SimpleNamespace(
                id="chatcmpl-text",
                created=1,
                model="gpt-4o-mini",
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant",
                            content="Files listed.",
                            tool_calls=None,
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
            )
        )

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "list files in ./app"}],
            "stream": False,
        }

        transport = httpx.ASGITransport(app=app)
        with patch("claude_to_openai_forwarder.app.get_backend", return_value=fake_backend), patch(
            "claude_to_openai_forwarder.app.get_settings", return_value=self.settings
        ), patch(
            "claude_to_openai_forwarder.middleware.auth.get_settings", return_value=self.settings
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/messages",
                    headers={"x-api-key": "sk-ant-test123"},
                    json=payload,
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["stop_reason"], "end_turn")
        self.assertEqual(data["content"][0]["type"], "text")
        self.assertEqual(data["content"][0]["text"], "Files listed.")
        fake_backend.create_completion.assert_awaited_once()

    async def test_messages_endpoint_returns_claude_tool_use_response(self):
        fake_backend = MagicMock()
        fake_backend.get_backend_name.return_value = "litellm"
        fake_backend.create_completion = AsyncMock(
            return_value=SimpleNamespace(
                id="chatcmpl-tool",
                created=1,
                model="meta/llama-4-maverick-17b-128e-instruct",
                choices=[
                    SimpleNamespace(
                        index=0,
                        finish_reason="stop",
                        message=SimpleNamespace(
                            role="assistant",
                            content='{"type":"tool_use","name":"Glob","input":{"pattern":"./app/**"}}',
                            tool_calls=None,
                        ),
                    )
                ],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10),
            )
        )

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "list files in ./app"}],
            "tools": [
                {
                    "name": "Glob",
                    "description": "List files matching a glob pattern",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern"}
                        },
                        "required": ["pattern"],
                    },
                }
            ],
            "stream": False,
        }

        transport = httpx.ASGITransport(app=app)
        with patch("claude_to_openai_forwarder.app.get_backend", return_value=fake_backend), patch(
            "claude_to_openai_forwarder.app.get_settings", return_value=self.settings
        ), patch(
            "claude_to_openai_forwarder.middleware.auth.get_settings", return_value=self.settings
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/v1/messages",
                    headers={"x-api-key": "sk-ant-test123"},
                    json=payload,
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["stop_reason"], "tool_use")
        self.assertEqual(data["content"][0]["type"], "tool_use")
        self.assertEqual(data["content"][0]["name"], "Glob")
        self.assertEqual(data["content"][0]["input"], {"pattern": "./app/**"})

    async def test_messages_endpoint_streaming_returns_tool_use_sse_for_claude_code(self):
        async def fake_stream():
            yield (
                b'data: {"choices":[{"delta":{"content":"{\\"type\\":\\"tool_use\\",'
                b'\\"name\\":\\"Glob\\",\\"input\\":{\\"pattern\\":\\"./app/**\\"}}"},'
                b'"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n'
            )
            yield b"data: [DONE]\n\n"

        fake_backend = MagicMock()
        fake_backend.get_backend_name.return_value = "litellm"
        fake_backend.create_completion_stream = MagicMock(return_value=fake_stream())

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 128,
            "messages": [{"role": "user", "content": "list files in ./app"}],
            "tools": [
                {
                    "name": "Glob",
                    "description": "List files matching a glob pattern",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Glob pattern"}
                        },
                        "required": ["pattern"],
                    },
                }
            ],
            "stream": True,
        }

        transport = httpx.ASGITransport(app=app)
        with patch("claude_to_openai_forwarder.app.get_backend", return_value=fake_backend), patch(
            "claude_to_openai_forwarder.app.get_settings", return_value=self.settings
        ), patch(
            "claude_to_openai_forwarder.middleware.auth.get_settings", return_value=self.settings
        ):
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                async with client.stream(
                    "POST",
                    "/v1/messages",
                    headers={"x-api-key": "sk-ant-test123"},
                    json=payload,
                ) as response:
                    body = await response.aread()
                    body = body.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: content_block_start", body)
        self.assertIn('"type": "tool_use"', body)
        self.assertIn('"name": "Glob"', body)
        self.assertIn('"stop_reason": "tool_use"', body)
        self.assertNotIn('"type": "text_delta"', body)


if __name__ == "__main__":
    unittest.main()
