from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Union, Dict, Any

class ClaudeContentBlock(BaseModel):
    """Content block for messages or system prompts"""
    type: str
    text: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None
    # For tool use
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None
    
    class Config:
        # Exclude None values when serializing
        exclude_none = True


class ClaudeMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Dict[str, Any]]]


class ClaudeRequest(BaseModel):
    model: str
    messages:  Optional[List[ClaudeMessage]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Tools - be very permissive with the schema
    tools: Optional[List[Dict[str, Any]]] = None
    
    # Additional fields that might be sent
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"  # Allow extra fields

class ClaudeUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ClaudeResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: List[ClaudeContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: ClaudeUsage
    
    class Config:
        exclude_none = True


class ClaudeStreamEvent(BaseModel):
    type: str
    index: Optional[int] = None
    delta: Optional[Dict[str, Any]] = None
    content_block: Optional[Dict[str, Any]] = None
    message: Optional[Dict[str, Any]] = None
    usage: Optional[ClaudeUsage] = None