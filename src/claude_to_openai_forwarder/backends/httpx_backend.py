import httpx
from typing import AsyncIterator, Optional
from claude_to_openai_forwarder.config import get_settings
from claude_to_openai_forwarder.models.openai import OpenAIRequest, OpenAIResponse
from claude_to_openai_forwarder.backends.base import BaseBackend
from claude_to_openai_forwarder.utils.exceptions import OpenAIAPIError
from claude_to_openai_forwarder.translators.content_process import flatten_content
import logging
import json

logger = logging.getLogger(__name__)


class HttpxBackend(BaseBackend):
    """Direct httpx client backend with connection pooling"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.openai_base_url
        self.api_key = self.settings.openai_api_key
        self.client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent async client with connection pooling"""
        if self.client is None:
            # Configure timeouts
            timeout = httpx.Timeout(
                timeout=self.settings.request_timeout,
                connect=self.settings.connect_timeout,
                read=self.settings.read_timeout,
                write=self.settings.write_timeout
            )
            
            # Configure connection limits for optimal performance
            limits = httpx.Limits(
                max_connections=self.settings.max_connections,
                max_keepalive_connections=self.settings.max_keepalive_connections,
                keepalive_expiry=self.settings.keepalive_expiry
            )
            self.client = httpx.AsyncClient(
                timeout=timeout,
                limits=limits,
                http2=False  # Set to True if backend supports HTTP/2
            )
            logger.info(f"Created persistent HttpX client with connection pooling (timeout={self.settings.request_timeout}s)")
        return self.client

    async def close(self):
        """Close the persistent client connection pool"""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Closed HttpX client connection pool")

    def get_backend_name(self) -> str:
        return "httpx"

    @staticmethod
    def _extract_error_message(error_payload, fallback_text: str) -> str:
        if isinstance(error_payload, dict):
            nested_error = error_payload.get("error")
            if isinstance(nested_error, dict):
                message = nested_error.get("message")
                if isinstance(message, str) and message:
                    return message
            elif isinstance(nested_error, str) and nested_error:
                return nested_error

            for key in ("message", "title", "detail"):
                value = error_payload.get(key)
                if isinstance(value, str) and value:
                    return value

            return fallback_text

        if isinstance(error_payload, str) and error_payload:
            return error_payload

        return fallback_text

    async def create_completion(self, request: OpenAIRequest) -> OpenAIResponse:
        """Make a non-streaming completion request"""
        request_json = request.model_dump(exclude_none=True)

        logger.debug(f"API call: model={request_json.get('model')}, messages={len(request_json.get('messages', []))}")
        messages = request_json.get('messages', [])
        if self.settings.force_content_flat:
            for m in messages:
                m["content"] = flatten_content(m.get("content"))

        client = self._get_client()
        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_json,
        )

        if response.status_code != 200:
            logger.error(f"API Error Response: {response.text}")
            try:
                error_data = response.json()
                logger.error(f"Parsed error: {json.dumps(error_data, indent=2)}")
                message = self._extract_error_message(error_data, response.text)
            except Exception:
                message = response.text
            raise OpenAIAPIError(response.status_code, message)

        return OpenAIResponse(**response.json())

    async def create_completion_stream(
        self, request: OpenAIRequest
    ) -> AsyncIterator[bytes]:
        """Make a streaming completion request"""
        request_json = request.model_dump(exclude_none=True)

        logger.debug(f"Streaming call: model={request_json.get('model')}")

        client = self._get_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=request_json,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                decoded_error = error_text.decode()
                logger.error(f"API Error: {decoded_error}")
                try:
                    error_data = json.loads(error_text)
                    message = self._extract_error_message(error_data, decoded_error)
                except json.JSONDecodeError:
                    message = decoded_error
                raise OpenAIAPIError(response.status_code, message)

            async for chunk in response.aiter_bytes():
                yield chunk
