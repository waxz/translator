import time
import json
from typing import List, Optional
from claude_to_openai_forwarder.models.openai import OpenAIResponse, OpenAIChoice, OpenAIMessage
from claude_to_openai_forwarder.models.claude import ClaudeResponse, ClaudeContentBlock, ClaudeUsage
import logging

logger = logging.getLogger(__name__)


class ResponseTranslator:
    """Translates OpenAI API responses to Claude format"""

    @classmethod
    def translate(cls, openai_resp: OpenAIResponse) -> ClaudeResponse:
        """Convert OpenAI response to Claude format"""

        # Get the first choice (Claude doesn't support multiple choices)
        choice = openai_resp.choices[0]

        # Convert content and tool calls
        content_blocks = cls._convert_content(choice.message)

        # Map stop reason - pass content_blocks to check for tool_use
        stop_reason = cls._map_stop_reason(choice.finish_reason, content_blocks)

        logger.info(
            f"Translating OpenAI response: finish_reason={choice.finish_reason}, "
            f"tool_calls={bool(choice.message.tool_calls)}, content_blocks={len(content_blocks)}"
        )

        return ClaudeResponse(
            id=openai_resp.id,
            type="message",
            role="assistant",
            content=content_blocks,
            model=cls._map_model_name(openai_resp.model),
            stop_reason=stop_reason,
            usage=ClaudeUsage(
                input_tokens=openai_resp.usage.prompt_tokens,
                output_tokens=openai_resp.usage.completion_tokens,
            ),
        )

    @classmethod
    def _convert_content(cls, message: OpenAIMessage) -> List[ClaudeContentBlock]:
        """Convert OpenAI message to Claude content blocks"""
        content_blocks = []

        # Handle text content first
        if message.content:
            content_text = message.content
            if isinstance(content_text, list):
                # Extract text from content array
                text_parts = []
                for item in content_text:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content_text = "\n".join(text_parts)

            if content_text and isinstance(content_text, str):
                # Check if content looks like a JSON tool_use
                tool_use_block = cls._parse_tool_use_from_text(content_text)
                if tool_use_block:
                    content_blocks.append(tool_use_block)
                else:
                    content_blocks.append(
                        ClaudeContentBlock(type="text", text=content_text)
                    )

        # Handle tool calls - CRITICAL for Claude Code
        if message.tool_calls:
            logger.info(f"Converting {len(message.tool_calls)} tool calls")
            for tool_call in message.tool_calls:
                # Parse arguments if they're a string
                tool_input = tool_call.get("function", {}).get("arguments", {})
                if isinstance(tool_input, str):
                    try:
                        tool_input = json.loads(tool_input)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {tool_input}")
                        tool_input = {"raw": tool_input}

                tool_block = ClaudeContentBlock(
                    type="tool_use",
                    id=tool_call.get("id", f"toolu_{int(time.time() * 1000)}"),
                    name=tool_call.get("function", {}).get("name", ""),
                    input=tool_input,
                )
                content_blocks.append(tool_block)
                logger.info(
                    f"Added tool_use block: {tool_block.name} with id {tool_block.id}"
                )

        # Default to empty text if no content
        if not content_blocks:
            content_blocks.append(ClaudeContentBlock(type="text", text=""))

        return content_blocks

    @classmethod
    def _parse_tool_use_from_text(cls, text: str) -> Optional[ClaudeContentBlock]:
        """Parse tool_use JSON from text content"""
        text = text.strip()

        # Try to find JSON object in text
        if text.startswith("{") and "type" in text and "tool_use" in text:
            try:
                data = json.loads(text)
                if data.get("type") == "tool_use" and "name" in data:
                    return ClaudeContentBlock(
                        type="tool_use",
                        id=data.get("id", f"toolu_{int(time.time() * 1000)}"),
                        name=data.get("name", ""),
                        input=data.get("input", {}),
                    )
            except json.JSONDecodeError:
                pass

        # Try to extract JSON from anywhere in text
        try:
            # Look for JSON-like structure
            start = text.find("{")
            if start != -1:
                # Find matching closing brace
                depth = 0
                end = start
                for i, char in enumerate(text[start:], start):
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break

                json_str = text[start:end]
                data = json.loads(json_str)
                if data.get("type") == "tool_use" and "name" in data:
                    return ClaudeContentBlock(
                        type="tool_use",
                        id=data.get("id", f"toolu_{int(time.time() * 1000)}"),
                        name=data.get("name", ""),
                        input=data.get("input", {}),
                    )
        except (json.JSONDecodeError, ValueError):
            pass

        return None

    @classmethod
    def _map_stop_reason(
        cls, finish_reason: Optional[str], content_blocks: list = None
    ) -> Optional[str]:
        """Map OpenAI finish_reason to Claude stop_reason"""
        # If content has tool_use blocks, always return tool_use stop_reason
        if content_blocks:
            has_tool_use = False
            for block in content_blocks:
                # Handle both dict and Pydantic model
                if hasattr(block, "type") and block.type == "tool_use":
                    has_tool_use = True
                    break
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    has_tool_use = True
                    break
            if has_tool_use:
                logger.info(
                    f"Content has tool_use blocks, setting stop_reason to tool_use"
                )
                return "tool_use"

        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "stop_sequence",
        }
        result = mapping.get(finish_reason, "end_turn")
        logger.info(f"Mapped stop_reason: {finish_reason} -> {result}")
        return result

    @classmethod
    def _map_model_name(cls, openai_model: str) -> str:
        """Map OpenAI model back to Claude model name"""
        reverse_map = {
            "meta/llama-4-maverick-17b-128e-instruct": "claude-3-5-sonnet-20241022",
            "gpt-4-turbo-preview": "claude-3-5-sonnet-20241022",
            "gpt-4-turbo": "claude-3-5-sonnet-20241022",
            "gpt-4": "claude-3-opus-20240229",
            "gpt-3.5-turbo": "claude-3-haiku-20240307",
        }
        return reverse_map.get(openai_model, "claude-3-5-sonnet-20241022")
