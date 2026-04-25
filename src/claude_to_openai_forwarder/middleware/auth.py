from fastapi import Header, HTTPException, status
from typing import Optional
import re
import logging
from claude_to_openai_forwarder.config import get_settings
logger = logging.getLogger(__name__)

def rotate_by_key_string(current_key, api_list_str):
    """
    Finds the current_key in the list to determine its ID, 
    then rotates to the next key.
    """
    # Convert semicolon-separated string to a Python list
    if not api_list_str:
        raise ValueError(f"api_list_str is not valid :{api_list_str}")
    keys = api_list_str.split(";")
    if len(keys) == 0:
        raise ValueError(f"len(keys) == 0")
    try:
        # 1. Search for the current key to find its ID (index)
        current_id = keys.index(current_key)
    except ValueError:
        # 2. If not found, default to -1 so rotation starts at 0
        current_id = -1
        
    # 3. Rotate to next index (round for list length using modulo)
    next_id = (current_id + 1) % len(keys)
    
    return keys[next_id]


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
    if not re.match(r'^sk-ant-[a-zA-Z0-9_-]{20,}$', x_api_key):
        # Allow test keys
        if re.match(r"^sk-[a-zA-Z0-9_-]{20,}$", x_api_key):
            return x_api_key

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    return x_api_key
