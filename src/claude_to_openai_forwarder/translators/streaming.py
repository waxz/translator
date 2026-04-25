import json
import re
import time
from typing import AsyncIterator, Optional, Dict, Any, List, Tuple
from claude_to_openai_forwarder.models.claude import ClaudeStreamEvent, ClaudeUsage
from claude_to_openai_forwarder.translators.response import ResponseTranslator
from claude_to_openai_forwarder.translators.tool_prompt import (
    parse_all_tool_calls,
    strip_control_text_tags,
)
from claude_to_openai_forwarder.models.claude import ClaudeContentBlock
from claude_to_openai_forwarder.config import get_settings

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
        settings = get_settings()
        force_tool_in_prompt = settings.force_tool_in_prompt

        message_id = f"msg_{int(time.time() * 1000)}"
        content_blocks: List[Dict[str, Any]] = []
        current_block_index = -1
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        buffered_text = ""
        buffered_text_segments: List[str] = []
        sse_buffer = ""

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
                    "model": settings.default_openai_model,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

        # Process OpenAI stream
        async for chunk in openai_stream:
            if not chunk:
                continue

            sse_buffer += chunk.decode("utf-8")

            while True:
                event_parts = re.split(r"\r?\n\r?\n", sse_buffer, maxsplit=1)
                if len(event_parts) < 2:
                    break

                raw_event, sse_buffer = event_parts
                data_str = cls._extract_sse_data(raw_event)
                if data_str is None:
                    continue

                if data_str == "[DONE]":
                    continue

                try:
                    data = json.loads(data_str)

                    # Extract delta
                    if data.get("choices"):
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})

                        if "content" in delta and delta["content"]:
                            content = delta["content"]
                            buffered_text += content
                            buffered_text_segments.append(content)

                            # Note: We don't try to parse tool calls incrementally during streaming
                            # because the JSON is incomplete until the stream finishes.
                            # Tool extraction happens in _flush_buffered_text when stream ends.

                        # Handle native tool calls (OpenAI format)
                        if "tool_calls" in delta and delta["tool_calls"]:
                            (
                                buffered_events,
                                current_block_index,
                                buffered_text,
                                buffered_text_segments,
                            ) = cls._handle_buffered_text(
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

                                    current_block_index = len(content_blocks)

                                    tool_name = tool_call_delta.get("function", {}).get(
                                        "name", ""
                                    )
                                    if tool_name:
                                        tool_calls_buffer[tc_index]["function"][
                                            "name"
                                        ] = tool_name

                                    logger.debug(
                                        f"Starting tool_use block: index={current_block_index}, name={tool_name}"
                                    )

                                    yield cls._format_sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": current_block_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": tool_calls_buffer[tc_index]["id"],
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
                                    tool_calls_buffer[tc_index][
                                        "content_block_index"
                                    ] = current_block_index

                                # Update tool call
                                if "function" in tool_call_delta:
                                    func = tool_call_delta["function"]
                                    content_block_index = tool_calls_buffer[tc_index][
                                        "content_block_index"
                                    ]
                                    if "name" in func:
                                        tool_calls_buffer[tc_index]["function"][
                                            "name"
                                        ] = func["name"]
                                        content_blocks[content_block_index]["name"] = (
                                            func["name"]
                                        )

                                    if "arguments" in func:
                                        tool_calls_buffer[tc_index]["function"][
                                            "arguments"
                                        ] += func["arguments"]

                                        yield cls._format_sse(
                                            "content_block_delta",
                                            {
                                                "type": "content_block_delta",
                                                "index": content_block_index,
                                                "delta": {
                                                    "type": "input_json_delta",
                                                    "partial_json": func["arguments"],
                                                },
                                            },
                                        )

                        # Track finish reason
                        if choice.get("finish_reason"):
                            old_finish_reason = finish_reason
                            finish_reason = choice["finish_reason"]
                            # When finish_reason is set and we have buffered text,
                            # flush immediately to extract any embedded tool calls
                            # This is ONLY done for NIM provider (force_tool_in_prompt=True)
                            # OpenAI native format sends tool calls separately
                            if buffered_text and old_finish_reason is None:
                                logger.debug(
                                    f"Stream finishing: finish_reason={finish_reason}, buffered_text_len={len(buffered_text)}, force_tool_in_prompt={force_tool_in_prompt}"
                                )

                                (
                                    buffered_events,
                                    current_block_index,
                                    buffered_text,
                                    buffered_text_segments,
                                ) = cls._handle_buffered_text(
                                    content_blocks=content_blocks,
                                    current_block_index=current_block_index,
                                    buffered_text=buffered_text,
                                    buffered_text_segments=buffered_text_segments,
                                    force_tool_in_prompt=force_tool_in_prompt,
                                )
                                for event in buffered_events:
                                    yield event

                        # Get usage if available
                        if "usage" in data:
                            usage = data["usage"]
                            if usage and isinstance(usage, dict):
                                total_input_tokens = usage.get("prompt_tokens", 0)
                                total_output_tokens = usage.get("completion_tokens", 0)

                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse SSE data: {e}. Data preview: {data_str[:200]}")
                    continue

        # Process any trailing buffer content
        if sse_buffer.strip():
            data_str = cls._extract_sse_data(sse_buffer)
            if data_str and data_str != "[DONE]":
                try:
                    data = json.loads(data_str)
                    if data.get("choices"):
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        if "content" in delta and delta["content"]:
                            buffered_text += delta["content"]
                            buffered_text_segments.append(delta["content"])
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
                    if "usage" in data:
                        usage = data["usage"]
                        total_input_tokens = usage.get("prompt_tokens", 0)
                        total_output_tokens = usage.get("completion_tokens", 0)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse trailing SSE data: {e}")

        # Final flush at end of stream
        if buffered_text:
            (
                buffered_events,
                current_block_index,
                buffered_text,
                buffered_text_segments,
            ) = cls._handle_buffered_text(
                content_blocks=content_blocks,
                current_block_index=current_block_index,
                buffered_text=buffered_text,
                buffered_text_segments=buffered_text_segments,
                force_tool_in_prompt=force_tool_in_prompt,
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

        logger.debug(
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
                if hasattr(block, "type") and block.type == "tool_use":
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

    @staticmethod
    def _extract_sse_data(raw_event: str) -> Optional[str]:
        data_fragments: List[str] = []
        saw_data = False

        for raw_line in raw_event.splitlines():
            line = raw_line.rstrip('\r')
            if line.startswith("data: "):
                data_fragments.append(line[6:])
                saw_data = True
                continue
            if line.startswith("data:"):
                data_fragments.append(line[5:].lstrip())
                saw_data = True
                continue
            if not line:
                continue

            # Some providers split JSON across physical lines inside a single SSE
            # event. Preserve those fragments once a data field has started.
            if saw_data and ":" not in line:
                data_fragments.append(line)
                continue

            if saw_data and line[:1] in {"{", "}", "[", "]", '"', ",", " "}:
                data_fragments.append(line)

        if not data_fragments:
            return None

        return "".join(data_fragments)

    @classmethod
    def _handle_buffered_text(
        cls,
        content_blocks: List[Dict[str, Any]],
        current_block_index: int,
        buffered_text: str,
        buffered_text_segments: List[str],
        force_tool_in_prompt: bool = False,
    ) -> Tuple[List[str], int, str, List[str]]:
        """Handle buffered text, extracting any embedded tool calls."""
        if not buffered_text:
            return [], current_block_index, buffered_text, buffered_text_segments

        if force_tool_in_prompt:
            prefix_text, parsed_tool_use = cls._split_embedded_tool_use(buffered_text)
        else:
            prefix_text = buffered_text
            parsed_tool_use = None
        prefix_text = strip_control_text_tags(prefix_text)
        events: List[str] = []

        if prefix_text:
            current_block_index = len(content_blocks)
            content_blocks.append({"type": "text", "text": prefix_text})
            if parsed_tool_use:
                text_segments = [prefix_text]
            else:
                sanitized_text = strip_control_text_tags(buffered_text)
                if sanitized_text != buffered_text:
                    text_segments = [sanitized_text]
                else:
                    text_segments = buffered_text_segments
            events.append(
                cls._format_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": current_block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            for segment in text_segments:
                events.append(
                    cls._format_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": current_block_index,
                            "delta": {"type": "text_delta", "text": segment},
                        },
                    )
                )
            events.append(
                cls._format_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": current_block_index},
                )
            )

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
                )
            )
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
                )
            )
            events.append(
                cls._format_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": current_block_index},
                )
            )

        return events, current_block_index, "", []

    @classmethod
    def _split_embedded_tool_use(cls, text: str):
        from claude_to_openai_forwarder.translators.tool_prompt import (
            parse_all_tool_calls,
        )

        tool_matches = parse_all_tool_calls(text)
        if not tool_matches:
            return text, None

        first_match = tool_matches[0]
        tool_call = first_match["tool_call"]
        prefix = text[: first_match["start"]].rstrip()

        parsed = ClaudeContentBlock(
            type="tool_use",
            id=tool_call["id"],
            name=tool_call["name"],
            input=tool_call.get("input", {}),
        )

        return prefix, parsed

    @classmethod
    def _extract_complete_tool_calls(cls, text: str) -> List[Dict[str, Any]]:
        from claude_to_openai_forwarder.translators.tool_prompt import (
            parse_all_tool_calls,
        )

        blocks = []
        last_end = 0

        tool_matches = parse_all_tool_calls(text)
        if tool_matches:
            for match in tool_matches:
                start_pos = match["start"]
                end_pos = match["end"]
                tool_call = match["tool_call"]

                if start_pos > last_end:
                    text_before = text[last_end:start_pos].rstrip()
                    if text_before:
                        blocks.append(
                            {"type": "text", "text": text_before, "end_pos": start_pos}
                        )

                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", f"toolu_{int(time.time() * 1000)}"),
                        "name": tool_call["name"],
                        "input": tool_call.get("input", {}),
                        "end_pos": end_pos,
                    }
                )

                last_end = end_pos

        return blocks
