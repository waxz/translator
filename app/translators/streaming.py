import json
import time
from typing import AsyncIterator, Optional, Dict, Any, List, Tuple
from app.models.claude import ClaudeStreamEvent, ClaudeUsage
from app.translators.response import ResponseTranslator
import logging

logger = logging.getLogger(__name__)


class StreamingTranslator:
    """Translates OpenAI streaming responses to Claude SSE format"""

    @classmethod
    async def translate_stream(
        cls, openai_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str]:
        """
        Convert OpenAI SSE stream to Claude SSE format

        Claude streaming events:
        1. message_start
        2. content_block_start (text or tool_use)
        3. content_block_delta (multiple)
        4. content_block_stop
        5. message_delta (includes stop_reason and usage)
        6. message_stop
        """

        message_id = f"msg_{int(time.time() * 1000)}"
        content_blocks: List[Dict[str, Any]] = []
        current_block_index = -1
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        buffered_text = ""
        buffered_text_segments: List[str] = []

        total_input_tokens = 0
        total_output_tokens = 0
        finish_reason = None

        # Send message_start event
        yield cls._format_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        # Process OpenAI stream
        async for chunk in openai_stream:
            if not chunk:
                continue

            # Parse SSE data
            lines = chunk.decode("utf-8").strip().split("\n")
            for line in lines:
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove 'data: ' prefix

                    if data_str == "[DONE]":
                        continue

                    try:
                        data = json.loads(data_str)

                        # Extract delta
                        if data.get("choices"):
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})

                            # Handle text content delta
                            if "content" in delta and delta["content"]:
                                buffered_text += delta["content"]
                                buffered_text_segments.append(delta["content"])

                            # Handle tool calls
                            if "tool_calls" in delta and delta["tool_calls"]:
                                (
                                    buffered_events,
                                    current_block_index,
                                    buffered_text,
                                    buffered_text_segments,
                                ) = cls._flush_buffered_text(
                                    content_blocks=content_blocks,
                                    current_block_index=current_block_index,
                                    buffered_text=buffered_text,
                                    buffered_text_segments=buffered_text_segments,
                                )
                                for event in buffered_events:
                                    yield event

                                for tool_call_delta in delta["tool_calls"]:
                                    tc_index = tool_call_delta.get("index", 0)

                                    # Initialize tool call buffer
                                    if tc_index not in tool_calls_buffer:
                                        tool_calls_buffer[tc_index] = {
                                            "id": tool_call_delta.get(
                                                "id",
                                                f"toolu_{int(time.time() * 1000)}_{tc_index}",
                                            ),
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        }

                                        # Close previous text block if exists
                                        # Start new tool_use block
                                        current_block_index = len(content_blocks)

                                        tool_name = tool_call_delta.get(
                                            "function", {}
                                        ).get("name", "")
                                        if tool_name:
                                            tool_calls_buffer[tc_index]["function"][
                                                "name"
                                            ] = tool_name

                                        logger.info(
                                            f"Starting tool_use block: index={current_block_index}, name={tool_name}"
                                        )

                                        yield cls._format_sse(
                                            "content_block_start",
                                            {
                                                "type": "content_block_start",
                                                "index": current_block_index,
                                                "content_block": {
                                                    "type": "tool_use",
                                                    "id": tool_calls_buffer[tc_index][
                                                        "id"
                                                    ],
                                                    "name": tool_name,
                                                },
                                            },
                                        )

                                        content_blocks.append(
                                            {
                                                "type": "tool_use",
                                                "id": tool_calls_buffer[tc_index]["id"],
                                                "name": tool_name,
                                                "input": {},
                                            }
                                        )
                                        tool_calls_buffer[tc_index]["content_block_index"] = current_block_index

                                    # Update tool call
                                    if "function" in tool_call_delta:
                                        func = tool_call_delta["function"]
                                        content_block_index = tool_calls_buffer[
                                            tc_index
                                        ]["content_block_index"]
                                        if "name" in func:
                                            tool_calls_buffer[tc_index]["function"][
                                                "name"
                                            ] = func["name"]
                                            content_blocks[content_block_index][
                                                "name"
                                            ] = func["name"]

                                        if "arguments" in func:
                                            tool_calls_buffer[tc_index]["function"][
                                                "arguments"
                                            ] += func["arguments"]

                                            # Send input_json_delta
                                            yield cls._format_sse(
                                                "content_block_delta",
                                                {
                                                    "type": "content_block_delta",
                                                    "index": content_block_index,
                                                    "delta": {
                                                        "type": "input_json_delta",
                                                        "partial_json": func[
                                                            "arguments"
                                                        ],
                                                    },
                                                },
                                            )

                            # Track finish reason
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]

                        # Get usage if available
                        if "usage" in data:
                            usage = data["usage"]
                            total_input_tokens = usage.get("prompt_tokens", 0)
                            total_output_tokens = usage.get("completion_tokens", 0)

                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse SSE data: {e}")
                        continue

        (
            buffered_events,
            current_block_index,
            buffered_text,
            buffered_text_segments,
        ) = cls._flush_buffered_text(
            content_blocks=content_blocks,
            current_block_index=current_block_index,
            buffered_text=buffered_text,
            buffered_text_segments=buffered_text_segments,
        )
        for event in buffered_events:
            yield event

        # Parse tool call arguments
        for tc_index, tc_data in tool_calls_buffer.items():
            args_str = tc_data["function"]["arguments"]
            try:
                args = json.loads(args_str) if args_str else {}
                # Find corresponding content block
                for block in content_blocks:
                    if (
                        block.get("type") == "tool_use"
                        and block.get("id") == tc_data["id"]
                    ):
                        block["input"] = args
                        break
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments: {args_str}")

        # Stop final native tool_use block if still open.
        if (
            current_block_index >= 0
            and content_blocks
            and content_blocks[current_block_index].get("type") == "tool_use"
            and tool_calls_buffer
        ):
            yield cls._format_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": current_block_index},
            )

        # Map finish reason
        stop_reason = cls._map_stop_reason(finish_reason, content_blocks)

        logger.info(
            f"Stream complete: finish_reason={finish_reason}, stop_reason={stop_reason}, "
            f"blocks={len(content_blocks)}, tool_calls={len(tool_calls_buffer)}"
        )

        # Send message_delta with usage and stop_reason
        yield cls._format_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": total_output_tokens},
            },
        )

        # Send message_stop
        yield cls._format_sse("message_stop", {"type": "message_stop"})

    @staticmethod
    def _map_stop_reason(
        finish_reason: Optional[str], content_blocks: list = None
    ) -> str:
        """Map OpenAI finish_reason to Claude stop_reason"""
        # If content has tool_use blocks, return tool_use
        if content_blocks:
            has_tool_use = False
            for block in content_blocks:
                if hasattr(block, 'type') and block.type == "tool_use":
                    has_tool_use = True
                    break
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    has_tool_use = True
                    break
            if has_tool_use:
                return "tool_use"
        
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "function_call": "tool_use",
            "content_filter": "stop_sequence",
        }
        return mapping.get(finish_reason, "end_turn")

    @staticmethod
    def _parse_tool_use_from_text(text: str):
        return ResponseTranslator._parse_tool_use_from_text(text)

    @staticmethod
    def _looks_like_json_tool_use(text: str) -> bool:
        stripped = text.lstrip()
        if not stripped:
            return True
        if stripped.startswith("{"):
            return True
        if '{"type"' in stripped or '"tool_use"' in stripped:
            return True
        return False

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        """Format data as Server-Sent Event"""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    @classmethod
    def _flush_buffered_text(
        cls,
        content_blocks: List[Dict[str, Any]],
        current_block_index: int,
        buffered_text: str,
        buffered_text_segments: List[str],
    ) -> Tuple[List[str], int, str, List[str]]:
        if not buffered_text:
            return [], current_block_index, buffered_text, buffered_text_segments

        prefix_text, parsed_tool_use = cls._split_embedded_tool_use(buffered_text)
        events: List[str] = []

        if prefix_text:
            current_block_index = len(content_blocks)
            content_blocks.append({"type": "text", "text": prefix_text})
            yield_text_segments = (
                buffered_text_segments
                if parsed_tool_use is None and buffered_text_segments
                else [prefix_text]
            )
            events.append(
                cls._format_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": current_block_index,
                    "content_block": {"type": "text", "text": ""},
                },
            ))
            for segment in yield_text_segments:
                events.append(
                    cls._format_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": current_block_index,
                        "delta": {"type": "text_delta", "text": segment},
                    },
                ))
            events.append(
                cls._format_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": current_block_index},
            ))

        if parsed_tool_use:
            current_block_index = len(content_blocks)
            content_blocks.append(
                {
                    "type": "tool_use",
                    "id": parsed_tool_use.id,
                    "name": parsed_tool_use.name,
                    "input": parsed_tool_use.input,
                }
            )
            events.append(
                cls._format_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": current_block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": parsed_tool_use.id,
                        "name": parsed_tool_use.name,
                    },
                },
            ))
            events.append(
                cls._format_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": current_block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(parsed_tool_use.input or {}),
                    },
                },
            ))
            events.append(
                cls._format_sse(
                "content_block_stop",
                {"type": "content_block_stop", "index": current_block_index},
            ))
            logger.info("Converted buffered text to tool_use: %s", parsed_tool_use.name)

        return events, current_block_index, "", []

    @classmethod
    def _split_embedded_tool_use(cls, text: str):
        stripped = text.strip()
        for start in range(len(stripped)):
            if stripped[start] != "{":
                continue
            depth = 0
            end = None
            for idx in range(start, len(stripped)):
                char = stripped[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end is None:
                continue

            candidate = stripped[start:end]
            parsed = cls._parse_tool_use_from_text(candidate)
            if parsed:
                prefix = stripped[:start].rstrip()
                return prefix, parsed

        return stripped, None
