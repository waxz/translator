# FastAPI Claude-to-OpenAI Forwarder

A FastAPI service that translates Claude Messages API requests to OpenAI-compatible chat completions endpoints.

## Features and Functionality

The FastAPI service accepts Claude Messages API-style requests and translates them to OpenAI-compatible chat completions endpoints. Key features include:

* Claude-to-OpenAI request translation
* OpenAI-to-Claude response translation
* Streaming SSE translation
* Claude Code-style tool calling
* Support for two backend modes: `httpx` and `litellm`

## Project Structure

The project is organized into the following components:

* `app/main.py`: The FastAPI application entry point, responsible for handling incoming requests and routing them to the appropriate backend.
* `app/translators/`: A module containing the logic for translating Claude requests to OpenAI format and vice versa.
* `app/backends/`: A module providing the backend implementations for interacting with OpenAI-compatible APIs, including `httpx` and `litellm`.
* `tests/test_backends.py`: A test suite for verifying the correctness of the backend implementations and translation logic.
* `requirements.txt`: A file listing the Python dependencies required by the project.

- [`app/main.py`](./app/main.py): FastAPI entrypoint
- [`app/translators/`](./app/translators): request, response, and streaming translation
- [`app/backends/`](./app/backends): backend abstraction and implementations
- [`tests/test_backends.py`](./tests/test_backends.py): backend and translator tests
- [`requirements.txt`](./requirements.txt): Python dependencies

## Backend Modes

The project supports two backend modes:

* **`httpx`**: Use when the upstream exposes a compatible OpenAI-style `/chat/completions` API.
  * Characteristics: direct HTTP calls, minimal provider-specific behavior, ignores `MODEL_PROVIDER`
  * Example:
    ```env
    BACKEND_TYPE=httpx
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://api.openai.com/v1
    DEFAULT_OPENAI_MODEL=gpt-4o-mini
    FORCE_TOOL_IN_PROMPT=false
    ```
* **`litellm`**: Use when provider-specific routing or compatibility behavior matters.
  * Characteristics: provider-aware routing, model/provider normalization, uses `MODEL_PROVIDER`
  * Example (for NVIDIA NIM):
    ```env
    BACKEND_TYPE=litellm
    OPENAI_API_KEY=...
    OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
    MODEL_PROVIDER=nvidia_nim
    DEFAULT_OPENAI_MODEL=meta/llama-4-maverick-17b-128e-instruct
    FORCE_TOOL_IN_PROMPT=true
    ```

## `MODEL_PROVIDER`

`MODEL_PROVIDER` is a LiteLLM-only setting.

Rules:

- `BACKEND_TYPE=httpx`: do not rely on `MODEL_PROVIDER`
- `BACKEND_TYPE=litellm` with OpenAI: `MODEL_PROVIDER=openai` or unset
- `BACKEND_TYPE=litellm` with NVIDIA NIM: `MODEL_PROVIDER=nvidia_nim`
- any other LiteLLM provider: set `MODEL_PROVIDER` to the provider name LiteLLM expects

In this repo, explicit `MODEL_PROVIDER` is preferred over relying on model-name prefix guessing.

## `FORCE_TOOL_IN_PROMPT`

`FORCE_TOOL_IN_PROMPT` controls whether tool definitions and tool history are translated into prompt text instead of native OpenAI `tools`, `tool_calls`, and `tool` messages.

Set it per provider/tool support:

- OpenAI via `httpx`
  - usually `FORCE_TOOL_IN_PROMPT=false`
- OpenAI via `litellm`
  - usually `FORCE_TOOL_IN_PROMPT=false`
- NVIDIA NIM via `litellm`
  - use `FORCE_TOOL_IN_PROMPT=true`
- NVIDIA NIM via `httpx`
  - use `true` if native tool calling is unreliable for the selected model
  - use `false` if the upstream behaves correctly with native OpenAI tool calls
- any provider without reliable native tool-calling support
  - use `FORCE_TOOL_IN_PROMPT=true`

Practical rule:

- if the provider supports native OpenAI `tools` and `tool_calls` end to end, use `false`
- if the provider emits tool use as text, or fails native multi-turn tool-calling flows, use `true`

## OpenAI API vs NVIDIA NIM

In this project, the practical difference is:

- OpenAI API is the native request/response target for the proxy
- NVIDIA NIM exposes an OpenAI-compatible HTTP surface, so raw `httpx` calls can still work
- LiteLLM still needs to know the real provider for NVIDIA NIM, which is why `MODEL_PROVIDER=nvidia_nim` matters
- some NVIDIA model flows are more reliable with prompt-embedded tool instructions than native tool calling, which is why this repo currently uses `FORCE_TOOL_IN_PROMPT=true` for the NVIDIA + LiteLLM example

## Tool Calling Behavior

The proxy is designed to preserve Claude-compatible tool calling semantics.

Important requirements:

- Claude tool calls must be returned as content blocks with `type="tool_use"`
- Claude tool-calling responses must end with `stop_reason="tool_use"`

Current verified behavior:

- non-streaming tool calls work on both backends
- streaming tool calls work on both backends
- mixed streaming output that contains prose followed by embedded JSON `tool_use` is translated into proper Claude tool-use events
- tool handling is generic and not limited to specific tool names such as `Bash` or `Glob`

## Verified Findings

During backend verification, these problems were found and resolved:

- LiteLLM provider routing was too dependent on model-prefix guessing
- LiteLLM responses could contain object-shaped tool calls and stream deltas that needed normalization
- some models emitted explanatory text followed by embedded JSON `tool_use`, which initially surfaced as plain text instead of Claude tool calls

Current verified state:

- `httpx` backend works
- `litellm` backend works
- backend tests pass with the current changes

## Installation

Requirements:

- Python 3.11+
- `uv` recommended

Install from source:

```bash
source ~/.bashrc
UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
cp .env.example .env
```

Install from wheel:

```bash
python -m build
uv pip install dist/claude_to_openai_forwarder-0.1.0-py3-none-any.whl
```

## Running The API

From source:

```bash
source ~/.bashrc
UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn app.main:app --reload
```

From installed wheel:

```bash
claude_to_openai_forwarder
```

Or specify a different port:

```bash
claude_to_openai_forwarder --port 8002
```

Default endpoint:

```text
http://127.0.0.1:8000/v1/messages
```

## Configuration

The project is configured using the following environment variables:

| Variable | Description | Default Value |
| --- | --- | --- |
| `BACKEND_TYPE` | The backend to use (either `httpx` or `litellm`) | `httpx` |
| `OPENAI_API_KEY` | The OpenAI API key | None |
| `OPENAI_BASE_URL` | The base URL for the OpenAI API | `https://api.openai.com/v1` |
| `CLAUDE_API_KEY` | Optional inbound API key required from clients via `x-api-key` | None |
| `MODEL_PROVIDER` | The model provider (used with `litellm` backend) | None |
| `DEFAULT_OPENAI_MODEL` | The default OpenAI model to use | `gpt-4o-mini` |
| `FORCE_TOOL_IN_PROMPT` | Whether to embed tool definitions in the prompt | `false` |
| `HOST` | The host to bind the API to | `0.0.0.0` |
| `PORT` | The port to bind the API to | `8000` |
| `LOG_LEVEL` | The logging level | `INFO` |

The backend reads configuration from `.env`.

Key variables:

- `BACKEND_TYPE`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `CLAUDE_API_KEY`
- `MODEL_PROVIDER`
- `DEFAULT_OPENAI_MODEL`
- `FORCE_TOOL_IN_PROMPT`
- `HOST`
- `PORT`
- `LOG_LEVEL`

See [`.env.example`](./.env.example) for a working example.

### Inbound Auth

Clients call this proxy with the `x-api-key` header.

Behavior:

- if `CLAUDE_API_KEY` is set, the header must exactly match that value
- if `CLAUDE_API_KEY` is unset, the proxy falls back to permissive local/test auth behavior

Example:

```bash
curl -X POST http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-test123" \
  -d '{"model":"claude-3-5-sonnet-20241022","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}'
```

### Claude Client Settings

If you want a Claude client that reads `~/.claude/settings.json` to use this proxy, configure the `env.ANTHROPIC_*` values to point at the local service.

Example `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_API_KEY": "sk-ant-test123",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-3-5-sonnet-20241022",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-3-5-sonnet-20241022",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-3-5-sonnet-20241022"
  }
}
```

Notes:

- `ANTHROPIC_BASE_URL` should point to the proxy root URL, not `/v1/messages`
- `ANTHROPIC_API_KEY` should match the proxy's `CLAUDE_API_KEY`
- the Claude model names in the client config should stay Claude-style names
- the proxy translates those Claude model names using `CLAUDE_MODEL_MAP` or `DEFAULT_OPENAI_MODEL`

## Verification

Recommended verification commands in this workspace:

```bash
source ~/.bashrc
UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall app
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest tests.test_backends
```

Optional manual check:

```bash
./test_with_tools.sh
```
