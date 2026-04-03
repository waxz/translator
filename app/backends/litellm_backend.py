import litellm
from typing import AsyncIterator, Dict, Any, Optional
from app.config import get_settings
from app.models.openai import OpenAIRequest, OpenAIResponse
from app.backends.base import BaseBackend
from app.translators.request import RequestTranslator
import logging
import json
import uuid
from types import SimpleNamespace

logger = logging.getLogger(__name__)


class LiteLLMBackend(BaseBackend):
    """LiteLLM backend for unified API access"""

    # Fallback model prefix mappings for LiteLLM when MODEL_PROVIDER is unset.
    MODEL_PROVIDER_MAP = {
        "meta/llama-4-maverick": "nvidia_nim",
        "meta/llama-4-scout": "nvidia_nim",
    }

    # Providers that don't support native tool calling
    NO_TOOL_CALL_PROVIDERS = {"nvidia_nim"}

    def __init__(self):
        self.settings = get_settings()
        self._setup_litellm()

    def _setup_litellm(self):
        """Setup LiteLLM with API key"""
        litellm.api_key = self.settings.openai_api_key
        # Disable detailed LiteLLM logging
        litellm.suppress_debug_info = True

    def get_backend_name(self) -> str:
        return "litellm"

    def _get_provider(self, model: str) -> Optional[str]:
        """Get the provider for a model"""
        if self.settings.model_provider:
            return self.settings.model_provider.strip()

        for prefix, provider in self.MODEL_PROVIDER_MAP.items():
            if model.startswith(prefix):
                return provider
        return None

    def _get_litellm_model(self, model: str) -> str:
        """Get LiteLLM model string with provider prefix"""
        provider = self._get_provider(model)
        if provider:
            return f"{provider}/{model}"
        return model

    def _provider_supports_tools(self, model: str) -> bool:
        """Check if the provider supports native tool calling"""
        provider = self._get_provider(model)
        if provider in self.NO_TOOL_CALL_PROVIDERS:
            return False
        return True

    def _convert_to_litellm_format(self, request: OpenAIRequest) -> Dict[str, Any]:
        """Convert OpenAI request to LiteLLM format"""
        litellm_model = self._get_litellm_model(request.model)
        provider = self._get_provider(request.model)

        litellm_request = {
            "model": litellm_model,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }

        # Always pass through the configured OpenAI-compatible endpoint for explicit providers.
        if provider:
            litellm_request["api_base"] = self.settings.openai_base_url
            litellm_request["api_key"] = self.settings.openai_api_key

        # Handle tools - embed in system prompt for providers without native tool support
        if request.tools and not self._provider_supports_tools(request.model):
            logger.info(
                f"Provider doesn't support native tools, embedding in system prompt"
            )
            litellm_request = self._embed_tools_in_system(
                litellm_request, request.tools
            )
        else:
            if request.tools:
                litellm_request["tools"] = request.tools

        if request.top_p:
            litellm_request["top_p"] = request.top_p

        if request.stop:
            litellm_request["stop"] = request.stop

        if request.tool_choice:
            litellm_request["tool_choice"] = request.tool_choice

        return litellm_request

    def _embed_tools_in_system(
        self, litellm_request: Dict[str, Any], tools: list
    ) -> Dict[str, Any]:
        """Embed tools in system message for providers without native tool support"""
        # Convert tools to prompt format
        tool_prompt = RequestTranslator._tools_to_prompt(tools)

        messages = litellm_request["messages"]
        system_content = tool_prompt

        # Find existing system message or create one
        if messages and messages[0].get("role") == "system":
            # Append to existing system message
            messages[0]["content"] = (
                f"{messages[0].get('content', '')}\n\n{tool_prompt}"
            )
        else:
            # Prepend system message
            messages.insert(0, {"role": "system", "content": tool_prompt})

        litellm_request["messages"] = messages
        return litellm_request

    async def create_completion(self, request: OpenAIRequest) -> OpenAIResponse:
        """Make a non-streaming completion request via LiteLLM"""
        litellm_request = self._convert_to_litellm_format(request)

        logger.info(f"Sending request via LiteLLM:")
        logger.info(f"  Model: {request.model}")

        try:
            response = await litellm.acompletion(**litellm_request)

            # Convert LiteLLM response to OpenAI format
            return self._convert_from_litellm(response)
        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            raise

    def _convert_from_litellm(self, response) -> OpenAIResponse:
        """Convert LiteLLM response to OpenAI format"""
        # Extract from LiteLLM ModelResponse
        choices = []
        for choice in response.choices:
            from app.models.openai import OpenAIMessage

            msg = choice.message

            content = msg.content if hasattr(msg, "content") else ""
            tool_calls = (
                [self._to_plain_data(tool_call) for tool_call in msg.tool_calls]
                if hasattr(msg, "tool_calls") and msg.tool_calls
                else None
            )

            openai_msg = OpenAIMessage(
                role=msg.role, content=content, tool_calls=tool_calls
            )

            from app.models.openai import OpenAIChoice

            choices.append(
                OpenAIChoice(
                    index=choice.index,
                    message=openai_msg,
                    finish_reason=choice.finish_reason,
                )
            )

        usage = response.usage
        from app.models.openai import OpenAIUsage

        openai_usage = OpenAIUsage(
            prompt_tokens=usage.prompt_tokens if hasattr(usage, "prompt_tokens") else 0,
            completion_tokens=usage.completion_tokens
            if hasattr(usage, "completion_tokens")
            else 0,
            total_tokens=usage.total_tokens if hasattr(usage, "total_tokens") else 0,
        )

        return OpenAIResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=choices,
            usage=openai_usage,
        )

    async def create_completion_stream(
        self, request: OpenAIRequest
    ) -> AsyncIterator[bytes]:
        """Make a streaming completion request via LiteLLM"""
        litellm_request = self._convert_to_litellm_format(request)
        litellm_request["stream"] = True

        logger.info(f"Sending streaming request via LiteLLM:")
        logger.info(f"  Model: {request.model}")

        try:
            response = await litellm.acompletion(**litellm_request)

            # Convert streaming response to bytes
            async for chunk in response:
                chunk_data = self._chunk_to_bytes(chunk)
                if chunk_data:
                    yield chunk_data
        except Exception as e:
            logger.error(f"LiteLLM streaming error: {e}")
            raise

    def _chunk_to_bytes(self, chunk) -> Optional[bytes]:
        """Convert LiteLLM chunk to SSE bytes"""
        try:
            # Get the delta from the chunk
            if hasattr(chunk, "choices") and chunk.choices:
                choice = chunk.choices[0]
                delta = (
                    self._to_plain_data(choice.delta)
                    if hasattr(choice, "delta")
                    else {}
                )

                # Build SSE data
                content = delta.get("content", "") if isinstance(delta, dict) else ""
                tool_calls = (
                    delta.get("tool_calls", []) if isinstance(delta, dict) else []
                )

                # Create the response structure
                response_dict = {
                    "id": getattr(chunk, "id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
                    "object": "chat.completion.chunk",
                    "created": getattr(chunk, "created", 0),
                    "model": chunk.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": content, "tool_calls": tool_calls}
                            if content or tool_calls
                            else {},
                            "finish_reason": getattr(choice, "finish_reason", None),
                        }
                    ],
                }

                # Add usage if available
                if hasattr(chunk, "usage") and chunk.usage:
                    response_dict["usage"] = {
                        "prompt_tokens": chunk.usage.prompt_tokens
                        if hasattr(chunk.usage, "prompt_tokens")
                        else 0,
                        "completion_tokens": chunk.usage.completion_tokens
                        if hasattr(chunk.usage, "completion_tokens")
                        else 0,
                        "total_tokens": chunk.usage.total_tokens
                        if hasattr(chunk.usage, "total_tokens")
                        else 0,
                    }

                return f"data: {json.dumps(response_dict)}\n\n".encode("utf-8")
        except Exception as e:
            logger.warning(f"Error converting chunk: {e}")

        return None

    def _to_plain_data(self, value: Any) -> Any:
        """Convert LiteLLM objects into plain JSON-serializable structures."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, list):
            return [self._to_plain_data(item) for item in value]

        if isinstance(value, dict):
            return {key: self._to_plain_data(item) for key, item in value.items()}

        if hasattr(value, "model_dump"):
            return self._to_plain_data(value.model_dump(exclude_none=True))

        if hasattr(value, "dict") and callable(value.dict):
            return self._to_plain_data(value.dict())

        if isinstance(value, SimpleNamespace):
            return self._to_plain_data(vars(value))

        if hasattr(value, "__dict__"):
            return self._to_plain_data(
                {
                    key: item
                    for key, item in vars(value).items()
                    if not key.startswith("_")
                }
            )

        return value
