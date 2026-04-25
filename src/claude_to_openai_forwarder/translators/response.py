import time
import json
from typing import List, Dict, Any, Optional
from claude_to_openai_forwarder.config import get_settings

from claude_to_openai_forwarder.models.openai import (
    OpenAIResponse,
    OpenAIChoice,
    OpenAIMessage,
)
from claude_to_openai_forwarder.models.claude import (
    ClaudeResponse,
    ClaudeContentBlock,
    ClaudeUsage,
)
import logging
import re
from json_repair import repair_json
from claude_to_openai_forwarder.translators.tool_prompt import (
    tools_to_prompt,
    parse_all_tool_calls,
    parse_tool_call,
    strip_control_text_tags,
)

logger = logging.getLogger(__name__)


class ResponseTranslator:
    """Translates OpenAI API responses to Claude format"""

    @classmethod
    def translate(cls, openai_resp: OpenAIResponse) -> ClaudeResponse:
        """Convert OpenAI response to Claude format"""

        settings = get_settings()

        # Get the first choice (Claude doesn't support multiple choices)
        choice = openai_resp.choices[0]
        message = choice.message
        if settings.force_tool_in_prompt:
            # Extract tool call from text (for NVIDIA NIM)
            content_blocks = cls._parse_text_content(message.content)
        else:
            # Convert content and tool calls
            content_blocks = cls._convert_content(message)

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

        content_text = cls._extract_text_content(message.content)
        if content_text:
            prefix_text, embedded_tool_use = cls._split_embedded_tool_use(content_text)
            if prefix_text:
                content_blocks.append(ClaudeContentBlock(type="text", text=prefix_text))
            if embedded_tool_use:
                content_blocks.append(embedded_tool_use)

        # Handle tool calls - FIXED: Access as dictionary
        if message.tool_calls:
            logger.debug(f"Converting {len(message.tool_calls)} tool calls")
            for tool_call in message.tool_calls:
                # tool_call is a Dict[str, Any]
                tool_id = tool_call.get("id", f"toolu_{int(time.time() * 1000)}")
                function_data = tool_call.get("function", {})
                tool_name = function_data.get("name", "")
                tool_args = function_data.get("arguments", {})

                # Parse arguments if they're a string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                        logger.debug(f"Parsed tool arguments for {tool_name}")
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse tool arguments for '{tool_name}': {e}. Raw: {tool_args[:100]}"
                        )
                        # Return empty dict instead of wrapping - upstream should handle invalid schema
                        tool_args = {}

                tool_block = ClaudeContentBlock(
                    type="tool_use",
                    id=tool_id,
                    name=tool_name,
                    input=tool_args,
                )
                content_blocks.append(tool_block)
                logger.debug(
                    f"Added tool_use: {tool_block.name} (id: {tool_block.id[:8]}...)"
                )

        # Default to empty text if no content
        if not content_blocks:
            content_blocks.append(ClaudeContentBlock(type="text", text=""))

        return content_blocks

    @staticmethod
    def _extract_text_content(content: Optional[Any]) -> str:
        """Normalize OpenAI message content into a plain text string."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(part for part in text_parts if part)

        return str(content)

    @classmethod
    def _split_embedded_tool_use(
        cls, text: str
    ) -> tuple[str, Optional[ClaudeContentBlock]]:
        """Split leading text from an embedded tool_use block when present."""
        if not text:
            return "", None

        sanitized = strip_control_text_tags(text)
        tool_matches = parse_all_tool_calls(sanitized)
        if not tool_matches:
            return sanitized, None

        first_match = tool_matches[0]
        tool_call = first_match["tool_call"]
        prefix = sanitized[: first_match["start"]].strip()

        return (
            prefix,
            ClaudeContentBlock(
                type="tool_use",
                id=tool_call.get("id", f"toolu_{int(time.time() * 1000)}"),
                name=tool_call.get("name", ""),
                input=tool_call.get("input", {}),
            ),
        )

    @classmethod
    def _parse_tool_call(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call JSON from response text.
        Handles nested JSON structures properly.
        """
        # Find all potential JSON objects with type: tool_use
        for match in re.finditer(r'\{', text):
            start = match.start()
            # Try to extract balanced JSON from this position
            try:
                bracket_count = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] == '{':
                        bracket_count += 1
                    elif text[i] == '}':
                        bracket_count -= 1
                    if bracket_count == 0:
                        end = i + 1
                        break
                
                if bracket_count != 0:
                    continue  # Unbalanced brackets
                
                potential_json = text[start:end]
                data = json.loads(potential_json)
                
                # Check if this is a valid tool_use block
                if data.get("type") == "tool_use" and data.get("name"):
                    if "id" not in data:
                        data["id"] = f"call_{int(time.time() * 1000)}"
                    logger.debug(f"Extracted tool_use from text: {data.get('name')}")
                    return data
            except (json.JSONDecodeError, ValueError):
                continue  # Try next JSON object
        
        return None

    @classmethod
    def _parse_text_content(cls, content: str) -> List[ClaudeContentBlock]:
        """Parse tool calls from text content (for NVIDIA NIM)"""
        logger.debug(
            f"_parse_text_content: content length = {len(content) if content else 0}"
        )
        if not content:
            return [ClaudeContentBlock(type="text", text="")]

        content = strip_control_text_tags(content)

        # Extract all tool calls with positions
        tool_matches = parse_all_tool_calls(content)
        logger.debug(f"Found {len(tool_matches) if tool_matches else 0} tool calls in text")

        if not tool_matches:
            logger.debug("No tool calls found, returning text content")
            return [ClaudeContentBlock(type="text", text=content)]

        blocks = []
        last_end = 0

        for match in tool_matches:
            # Text before tool call
            if match["start"] > last_end:
                text_before = content[last_end : match["start"]].strip()
                if text_before:
                    blocks.append(ClaudeContentBlock(type="text", text=text_before))

            # Tool call
            tool_call = match["tool_call"]
            logger.debug(
                f"Processing tool: {tool_call.get('name')}"
            )
            blocks.append(
                ClaudeContentBlock(
                    type="tool_use",
                    id=tool_call["id"],
                    name=tool_call["name"],
                    input=tool_call.get("input", {}),
                )
            )

            last_end = match["end"]

        # Text after last tool call
        if last_end < len(content):
            text_after = content[last_end:].strip()
            if text_after:
                blocks.append(ClaudeContentBlock(type="text", text=text_after))

        return blocks if blocks else [ClaudeContentBlock(type="text", text="")]

    @classmethod
    def _parse_tool_use_from_text(cls, text: str) -> Optional[ClaudeContentBlock]:
        """Parse tool_use JSON from text content"""
        text = strip_control_text_tags(text).strip()

        # Search for tool_use JSON anywhere in text
        tool_call = cls._parse_tool_call(text)
        if tool_call:
            return ClaudeContentBlock(
                type="tool_use",
                id=tool_call.get("id", f"toolu_{int(time.time() * 1000)}"),
                name=tool_call.get("name", ""),
                input=tool_call.get("input", {}),
            )

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
