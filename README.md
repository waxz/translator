# Claude-to-OpenAI Forwarder

A FastAPI service that accepts Claude Messages API requests and forwards them to OpenAI-compatible chat completions backends.

It is designed as a drop-in bridge for tools such as Claude Code when the upstream model is served by OpenAI, NVIDIA NIM, Ollama, or another OpenAI-compatible endpoint.

## Design

The project is organized around a small translation pipeline:

- `app.py`: FastAPI entrypoint and Claude-compatible HTTP surface
- `translators/request.py`: Claude request -> OpenAI chat completions request
- `backends/`: upstream transport layer
- `translators/response.py`: OpenAI response -> Claude message response
- `translators/streaming.py`: OpenAI SSE stream -> Claude SSE stream
- `translators/tool_prompt.py`: tool-call parsing and prompt-embedded tool helpers

The proxy supports two backend modes:

- `httpx`
  Best when the upstream already exposes a stable OpenAI-compatible `/chat/completions` API.

- `litellm`
  Best when provider-specific routing or LiteLLM normalization is needed.

## Features

- Claude Messages API compatible `/v1/messages`
- Claude-compatible streaming responses over SSE
- Non-streaming and streaming tool-call translation
- Support for native OpenAI `tool_calls`
- Support for prompt-embedded tool calling for providers that return tool calls inside text
- Configurable model mapping from Claude model names to upstream model names
- Optional inbound API-key enforcement for Claude clients
- Local per-key rate limiting
- Persistent `httpx` connection pooling with configurable timeouts
- Claude-compatible error responses
- `/v1/messages/count_tokens` endpoint

## Tool-Calling Behavior

The proxy handles several tool-call shapes that appear in practice:

- native OpenAI `tool_calls`
- embedded JSON such as `{"type":"tool_use", ...}`
- function-style calls such as `Agent({...})`

The response sanitization layer also strips control/meta wrappers that should not leak into user-visible assistant text, including:

- `<thinking>...</thinking>`
- `<result ...>...</result>`
- `<analysis>...</analysis>`
- `<commentary>...</commentary>`
- `<final>...</final>`

## Configuration

Configuration is loaded from environment variables or `.env`.

Core settings:

| Variable | Description | Default |
| --- | --- | --- |
| `BACKEND_TYPE` | `httpx` or `litellm` | `httpx` |
| `OPENAI_API_KEY` | Upstream API key | required |
| `OPENAI_BASE_URL` | Upstream OpenAI-compatible base URL | `https://api.openai.com/v1` |
| `MODEL_PROVIDER` | LiteLLM provider hint | unset |
| `CLAUDE_API_KEY` | Inbound client auth key | unset |
| `DEFAULT_OPENAI_MODEL` | Default upstream model | `gpt-4o-mini` |
| `CLAUDE_MODEL_MAP` | Claude-to-upstream model map | `{}` |
| `FORCE_TOOL_IN_PROMPT` | Use prompt-embedded tool calling | `false` |
| `FORCE_CONTENT_FLAT` | Flatten message content for upstream compatibility | `false` |
| `RATE_LIMIT_RPM` | Local requests per minute per inbound key | `40` |
| `HOST` | Bind host | `0.0.0.0` |
| `PORT` | Bind port | `8000` |
| `LOG_LEVEL` | Server log level | `INFO` |

HTTP client settings:

| Variable | Description | Default |
| --- | --- | --- |
| `REQUEST_TIMEOUT` | Total request timeout | `120.0` |
| `CONNECT_TIMEOUT` | Connect timeout | `10.0` |
| `READ_TIMEOUT` | Read timeout | `120.0` |
| `WRITE_TIMEOUT` | Write timeout | `30.0` |
| `MAX_CONNECTIONS` | Max pooled connections | `100` |
| `MAX_KEEPALIVE_CONNECTIONS` | Max keepalive connections | `20` |
| `KEEPALIVE_EXPIRY` | Keepalive expiry seconds | `30.0` |

Example `.env`:

```env
BACKEND_TYPE=httpx
CLAUDE_API_KEY=sk-ant-local-forwarder-key
OPENAI_API_KEY=your-upstream-key
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_PROVIDER=nvidia_nim
DEFAULT_OPENAI_MODEL=meta/llama-4-maverick-17b-128e-instruct
FORCE_TOOL_IN_PROMPT=true
RATE_LIMIT_RPM=40
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
```

## Running

Install:

```bash
uv pip install -r requirements.txt
```

Run with Uvicorn:

```bash
uvicorn src.claude_to_openai_forwarder.app:app --host 0.0.0.0 --port 8000
```

Or run the packaged entrypoint:

```bash
claude-to-openai-forwarder
```

## Usage

Send a Claude-style request:

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-local-forwarder-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Count tokens:

```bash
curl http://127.0.0.1:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-local-forwarder-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "Hello"}
    ]
  }'
```

## Claude Code Integration

Example Claude Code environment settings:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_API_KEY": "sk-ant-local-forwarder-key",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-3-5-sonnet-20241022"
  }
}
```

## Recent Improvements

The current code in `src/` includes the following hardening and compatibility fixes:

- local rate limiting is enabled in the request path
- rate-limit cleanup preserves active identifiers while pruning expired timestamps
- `httpx` backend error parsing handles nested errors, flat objects, and JSON string payloads
- streaming translation no longer leaks recognized tool-call text into normal assistant text
- parser support for function-style tool calls such as `Agent({...})`
- control/meta wrappers such as `<thinking>` and `<result ...>` are stripped from visible assistant text
- persistent `httpx` client pooling with configurable timeout and connection settings

## Verification

Basic syntax check:

```bash
python -m compileall src/claude_to_openai_forwarder
```

Focused regression tests:

```bash
PYTHONPATH=src python -m unittest tests.test_backends.HttpxBackendTests
PYTHONPATH=src python -m unittest tests.test_backends.AppRateLimitTests.test_messages_endpoint_enforces_local_rate_limit
PYTHONPATH=src python -m unittest tests.test_backends.StreamingTranslatorTests.test_translate_stream_converts_function_style_agent_call_after_text
PYTHONPATH=src python -m unittest tests.test_backends.StreamingTranslatorTests.test_translate_stream_strips_thinking_wrappers_from_text
PYTHONPATH=src python -m unittest tests.test_improvements.TestRateLimiterMemoryLeakFix
```

## Notes

- `MODEL_PROVIDER` is mainly relevant for the `litellm` backend.
- For providers such as NVIDIA NIM, `FORCE_TOOL_IN_PROMPT=true` may be required depending on model behavior.
- The inbound `CLAUDE_API_KEY` is optional, but it should be set in any non-local deployment.
