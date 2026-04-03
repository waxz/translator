from fastapi import Header, HTTPException, status
from typing import Optional

from app.config import get_settings


async def verify_claude_api_key(x_api_key: Optional[str] = Header(None)):
    """
    Verify Claude API key.

    If CLAUDE_API_KEY is configured, require an exact match.
    Otherwise, fall back to permissive format checks for local/test use.
    """
    settings = get_settings()

    if settings.claude_api_key:
        if x_api_key != settings.claude_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )
        return x_api_key

    # Make it optional only when no explicit Claude API key is configured.
    if not x_api_key:
        return "test-key"

    # Basic format validation
    if not x_api_key.startswith("sk-ant-"):
        # Allow test keys
        if x_api_key.startswith("sk-"):
            return x_api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    return x_api_key
