from typing import List, Union, Dict, Any, Optional

def flatten_content(content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    """
    Converts OpenAI-style content (string or list of blocks) 
    into a single string to avoid 400 Bad Request errors.
    """
    if not content:
        return ""
    
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            # Extract 'text' from content blocks like {"type": "text", "text": "..."}
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            # Handle raw strings inside the list
            elif isinstance(item, str):
                parts.append(item)
        
        return "\n".join(parts)
    
    return str(content)