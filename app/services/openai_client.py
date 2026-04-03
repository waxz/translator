import httpx
from typing import AsyncIterator
from app.config import get_settings
from app.models.openai import OpenAIRequest, OpenAIResponse
import logging
import json

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for making requests to OpenAI API"""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.openai_base_url
        self.api_key = self.settings.openai_api_key

    async def create_completion(self, request: OpenAIRequest) -> OpenAIResponse:
        """Make a non-streaming completion request"""
        request_json = request.model_dump(exclude_none=True)

        # Log the full request
        logger.info(f"Sending to OpenAI API:")
        logger.info(f"  Model: {request_json.get('model')}")
        logger.info(f"  Messages: {len(request_json.get('messages', []))}")
        if request_json.get("tools"):
            logger.info(f"  Tools: {len(request_json.get('tools', []))}")
            for tool in request_json.get("tools", []):
                logger.info(f"    - {tool.get('function', {}).get('name', 'unknown')}")

        logger.debug(f"Full request: {json.dumps(request_json, indent=2)}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_json,
            )

            logger.info(f"OpenAI API response status: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"API Error Response: {response.text}")
                try:
                    error_data = response.json()
                    logger.error(f"Parsed error: {json.dumps(error_data, indent=2)}")
                except:
                    pass
                response.raise_for_status()

            response_data = response.json()
            logger.debug(f"Full response: {json.dumps(response_data, indent=2)}")

            # Log response details
            if "choices" in response_data and response_data["choices"]:
                choice = response_data["choices"][0]
                logger.info(f"Response finish_reason: {choice.get('finish_reason')}")
                if choice.get("message", {}).get("tool_calls"):
                    logger.info(f"Tool calls: {len(choice['message']['tool_calls'])}")

            return OpenAIResponse(**response_data)

    async def create_completion_stream(
        self, request: OpenAIRequest
    ) -> AsyncIterator[bytes]:
        """Make a streaming completion request"""
        request_json = request.model_dump(exclude_none=True)

        # Log the full request
        logger.info(f"Sending streaming request to OpenAI API:")
        logger.info(f"  Model: {request_json.get('model')}")
        logger.info(f"  Messages: {len(request_json.get('messages', []))}")
        if request_json.get("tools"):
            logger.info(f"  Tools: {len(request_json.get('tools', []))}")
            for tool in request_json.get("tools", []):
                logger.info(f"    - {tool.get('function', {}).get('name', 'unknown')}")

        logger.debug(f"Full request: {json.dumps(request_json, indent=2)}")

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
                    response.raise_for_status()

                async for chunk in response.aiter_bytes():
                    yield chunk
