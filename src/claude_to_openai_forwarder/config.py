# config.py

from functools import lru_cache
from typing import Dict

from pydantic import Field
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

    # Tool handling
    force_tool_in_prompt: bool = False
    force_content_flat: bool = False


_settings_cache: Settings | None = None


def get_settings() -> Settings:
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = Settings()
    return _settings_cache


def reset_settings() -> None:
    global _settings_cache
    _settings_cache = None
