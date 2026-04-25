from functools import lru_cache
from typing import Dict

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )

    # Backend Configuration
    backend_type: str = "httpx"  # "httpx" or "litellm"

    rate_limit_rpm: int = 40

    # API Configuration
    openai_api_key: str
    openai_api_key_list: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    model_provider: str | None = None  # e.g. "openai", "nvidia_nim"
    claude_api_key: str | None = None

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # Model Configuration
    default_openai_model: str = "gpt-4o-mini"
    claude_model_map: Dict[str, str] = Field(default_factory=dict)

    @field_validator('claude_model_map')
    def validate_model_map(cls, v):
        if not isinstance(v, dict):
            raise ValueError('claude_model_map must be a dictionary')
        return v or {}

    # Tool handling
    force_tool_in_prompt: bool = False
    force_content_flat: bool = False

    # Timeout Configuration (in seconds)
    request_timeout: float = 120.0  # Total timeout for entire request
    connect_timeout: float = 10.0  # Timeout for connection establishment
    read_timeout: float = 120.0  # Timeout for reading response body
    write_timeout: float = 30.0  # Timeout for writing request body

    # Connection Pool Configuration
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_expiry: float = 30.0

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
