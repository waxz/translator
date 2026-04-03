from claude_to_openai_forwarder.backends.base import BaseBackend
from claude_to_openai_forwarder.backends.httpx_backend import HttpxBackend
from claude_to_openai_forwarder.config import get_settings


_backend_cache = None


def get_backend() -> BaseBackend:
    """Get the configured backend"""
    global _backend_cache

    if _backend_cache is not None:
        return _backend_cache

    settings = get_settings()
    backend_type = settings.backend_type.lower() if settings.backend_type else "httpx"

    if backend_type == "litellm":
        from claude_to_openai_forwarder.backends.litellm_backend import LiteLLMBackend

        _backend_cache = LiteLLMBackend()
    else:
        _backend_cache = HttpxBackend()

    return _backend_cache


def get_backend_name() -> str:
    """Get the name of the current backend"""
    return get_backend().get_backend_name()
