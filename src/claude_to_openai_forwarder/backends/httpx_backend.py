import httpx
from typing import AsyncIterator
from claude_to_openai_forwarder.config import get_settings
from claude_to_openai_forwarder.models.openai import OpenAIRequest, OpenAIResponse
from claude_to_openai_forwarder.backends.base import BaseBackend
from claude_to_openai_forwarder.utils.exceptions import OpenAIAPIError
import logging
import json

logger = logging.getLogger(__name__)


class HttpxBackend(BaseBackend):
    """Direct httpx client backend"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.openai_base_url
        self.api_key = self.settings.openai_api_key

    def get_backend_name(self) -> str:
        return "httpx"

    async def create_completion(self, request: OpenAIRequest) -> OpenAIResponse:
        """Make a non-streaming completion request"""
        request_json = request.model_dump(exclude_none=True)

        logger.info(f"Sending to OpenAI API via httpx:")
        logger.info(f"  Model: {request_json.get('model')}")
        logger.info(f"  Messages: {len(request_json.get('messages', []))}")

        async with httpx.AsyncClient(timeout=120.0) as client:
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
                    message = error_data.get("error", {}).get("message", response.text)
                except:
                    message = response.text
                raise OpenAIAPIError(response.status_code, message)

            return OpenAIResponse(**response.json())

    async def create_completion_stream(
        self, request: OpenAIRequest
    ) -> AsyncIterator[bytes]:
        """Make a streaming completion request"""
        request_json = request.model_dump(exclude_none=True)

        logger.info(f"Sending streaming request via httpx:")
        logger.info(f"  Model: {request_json.get('model')}")

        async with httpx.AsyncClient(timeout=120.0) as client:
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
                    logger.error(f"API Error: {error_text.decode()}")
                    try:
                        error_data = json.loads(error_text)
                        message = error_data.get("error", {}).get(
                            "message", error_text.decode()
                        )
                    except json.JSONDecodeError:
                        message = error_text.decode()
                    raise OpenAIAPIError(response.status_code, message)

                async for chunk in response.aiter_bytes():
                    yield chunk
