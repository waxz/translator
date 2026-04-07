# FastAPI Claude-to-OpenAI Forwarder 🚀

An open-source HTTP forwarder that translates Claude Messages API requests to OpenAI-compatible chat completion backends. Run any model—from NVIDIA NIM, Ollama, or OpenAI—as a drop-in replacement for Claude.

## Features

- ✅ Drop-in replacement for Claude Messages API
- ✅ Tool calling support (Claude Code compatible)
- ✅ Streaming with Server-Sent Events (SSE)
- ✅ Multiple backends: OpenAI, NVIDIA NIM, Ollama, LiteLLM
- ✅ Token counting endpoint
- ✅ Flexible backend modes: `httpx` and `litellm`

## Quick Start

### Installation

```bash
uv pip install -r requirements.txt
cp .env.example .env
```

### Running the Server

### Installation Troubleshooting

* Issue: uv pip install fails
  Solution: Check your network connection and try again.
* Issue: Missing dependencies
  Solution: Run `uv pip install -r requirements.txt` to ensure all dependencies are installed.

```bash
uvicorn src.claude_to_openai_forwarder.app:app --port 8000
```

Or as a standalone command:

```bash
claude_to_openai_forwarder --port 8000
```

## Configuration

### Environment Variables

The following environment variables can be set in a `.env` file:

```env
BACKEND_TYPE=httpx
CLAUDE_API_KEY=REDACTED
RATE_LIMIT_RPM=40
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
OPENAI_API_KEY=REDACTED
OPENAI_API_KEY_LIST=REDACTED;REDACTED
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_PROVIDER=nvidia_nim
DEFAULT_OPENAI_MODEL=meta/llama-4-maverick-17b-128e-instruct
CLAUDE_MODEL_MAP={...}
FORCE_TOOL_IN_PROMPT=true
FORCE_CONTENT_FLAT=true
```

##

Set these environment variables (or use `.env`):

| Variable | Description | Default |
|----------|-------------|--------|
| `BACKEND_TYPE` | `httpx` or `litellm` | `httpx` |
| `OPENAI_API_KEY` | Your API key for the backend | Required |
| `OPENAI_BASE_URL` | Backend API URL | `https://api.openai.com/v1` |
| `MODEL_PROVIDER` | LiteLLM provider name | Optional |
| `DEFAULT_OPENAI_MODEL` | Default model to use | `gpt-4o-mini` |
| `CLAUDE_API_KEY` | Auth key for inbound clients | Optional |
| `FORCE_TOOL_IN_PROMPT` | Embed tools in prompt text | `false` |
| `HOST` | Host to bind | `0.0.0.0` |
| `PORT` | Port to bind | `8000` |

## Backend Modes

## Advanced Configuration Examples

* Example 1: Configuring multiple backend providers
* Example 2: Customizing model settings for specific use cases



### `httpx` Mode

Best for providers with a standard OpenAI-compatible API.

```env
BACKEND_TYPE=httpx
OPENAI_BASE_URL=https://api.openai.com/v1
DEFAULT_OPENAI_MODEL=gpt-4o-mini
FORCE_TOOL_IN_PROMPT=false
```

### `litellm` Mode

Best for provider-specific routing, model normalization, or when you need LiteLLM's compatibility layer.

```env
BACKEND_TYPE=litellm
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_PROVIDER=nvidia_nim
DEFAULT_OPENAI_MODEL=meta/llama-4-maverick-17b-128e-instruct
FORCE_TOOL_IN_PROMPT=true
```

## Usage Examples

## Troubleshooting

### Common Error Messages

* Error 404: Model not found. Check the MODEL_PROVIDER and DEFAULT_OPENAI_MODEL environment variables.
* Error 401: Authentication failed. Verify the OPENAI_API_KEY and CLAUDE_API_KEY environment variables.

### Send a Message

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-local" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Count Tokens

```bash
curl http://127.0.0.1:8000/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-local" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### List Available Models (NVIDIA NIM)

```bash
curl https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer nvapi-<your_api_key>" | jq -r "."
```
working models:
- meta/llama-4-maverick-17b-128e-instruct ok
- meta/llama-4-scout-17b-16e-instruct Not Found
- deepseek-ai/deepseek-r1 end of life
- meta/llama-3.3-70b-instruct ok
- qwen/qwen2.5-coder-32b-instruct ContextWindow too small

## API Documentation

The API endpoints are documented in the [OpenAPI specification](src/claude_to_openai_forwarder/app.py).

## Integration with Claude Code

To use this forwarder with the Claude Code CLI, add these settings to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8000",
    "ANTHROPIC_API_KEY": "sk-local",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-3-5-sonnet-20241022"
  }
}
```

## Verification

## Release Notes

* v1.0: Initial release with support for OpenAI and NVIDIA NIM backends
* v1.1: Added support for LiteLLM backend and improved error handling



```bash
uv run python -m compileall app
uv run python -m unittest tests.test_backends
```

## Future Plans

We plan to add support for more backend providers and improve error handling in future releases.

## FAQs

* Q: How do I configure the backend provider?
A: See the Configuration section for details.
* Q: What models are supported?
A: Check the MODEL_PROVIDER documentation for supported models.

## Support

For support, please open an issue on our [GitHub repository](https://github.com/anthropics/claude-code/issues) or contact us at <support@anthropic.com>.

## Code Style Guidelines

We follow standard Python PEP 8 guidelines for code style. For more details, see [PEP 8](https://peps.python.org/pep-0008/).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

