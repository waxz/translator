from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional
from claude_to_openai_forwarder.models.claude import ClaudeRequest
from claude_to_openai_forwarder.models.openai import OpenAIRequest, OpenAIResponse


class BaseBackend(ABC):
    """Base class for API backends"""

    @abstractmethod
    async def create_completion(self, request: OpenAIRequest) -> OpenAIResponse:
        """Make a non-streaming completion request"""
        pass

    @abstractmethod
    async def create_completion_stream(
        self, request: OpenAIRequest
    ) -> AsyncIterator[bytes]:
        """Make a streaming completion request"""
        pass

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the backend name"""
        pass
