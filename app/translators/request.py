import logging
import json
from typing import Any, Dict, List, Optional, Union

from app.config import get_settings
from app.models.claude import ClaudeMessage, ClaudeRequest
from app.models.openai import OpenAIMessage, OpenAIRequest

logger = logging.getLogger(__name__)


class RequestTranslator:
    """Translate Claude-style requests into OpenAI chat completions requests."""

    @classmethod
    def translate(
        cls, claude_req: ClaudeRequest, default_model: Optional[str] = None
    ) -> OpenAIRequest:
        settings = get_settings()
        if default_model is None:
            default_model = settings.default_openai_model

        use_prompt_tool_mode = settings.force_tool_in_prompt

        messages = cls._convert_messages(
            claude_req.messages,
            claude_req.system,
            claude_req.tools,
            use_prompt_tool_mode=use_prompt_tool_mode,
        )
        model = settings.claude_model_map.get(claude_req.model, default_model)

        logger.info(
            "Translating Claude model %s to upstream model %s", claude_req.model, model
        )

        return OpenAIRequest(
            model=model,
            messages=messages,
            max_tokens=claude_req.max_tokens,
            temperature=claude_req.temperature,
            top_p=claude_req.top_p if claude_req.top_p is not None else 1.0,
            stream=claude_req.stream,
            stop=claude_req.stop_sequences,
            tools=None if use_prompt_tool_mode else cls._convert_tools(claude_req.tools),
            tool_choice=None
            if use_prompt_tool_mode
            else cls._convert_tool_choice(claude_req.tool_choice),
        )

    @classmethod
    def _convert_messages(
        cls,
        claude_messages: List[ClaudeMessage],
        system: Optional[Union[str, List[Dict[str, Any]]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        use_prompt_tool_mode: bool = False,
    ) -> List[OpenAIMessage]:
        settings = get_settings()
        openai_messages: List[OpenAIMessage] = []
        system_parts: List[str] = []

        if system:
            system_content = cls._extract_system_content(system)
            if system_content:
                system_parts.append(system_content)

        if tools and settings.force_tool_in_prompt:
            tool_prompt = cls._tools_to_prompt(tools)
            if tool_prompt:
                system_parts.append(tool_prompt)
                logger.info("Embedded %s tools into the system prompt", len(tools))

        if system_parts:
            openai_messages.append(
                OpenAIMessage(
                    role="system",
                    content="\n\n".join(system_parts),
                )
            )

        for msg in claude_messages:
            if msg.role == "assistant":
                openai_messages.extend(
                    cls._convert_assistant_message(
                        msg, use_prompt_tool_mode=use_prompt_tool_mode
                    )
                )
            else:
                openai_messages.extend(
                    cls._convert_user_message(
                        msg, use_prompt_tool_mode=use_prompt_tool_mode
                    )
                )

        return openai_messages

    @classmethod
    def _convert_assistant_message(
        cls, message: ClaudeMessage, use_prompt_tool_mode: bool = False
    ) -> List[OpenAIMessage]:
        if not isinstance(message.content, list):
            return [
                OpenAIMessage(
                    role="assistant", content=cls._convert_content(message.content)
                )
            ]

        if use_prompt_tool_mode:
            prompt_content = cls._assistant_blocks_to_prompt_content(message.content)
            return [OpenAIMessage(role="assistant", content=prompt_content)]

        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in message.content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "text" and block.get("text"):
                text_parts.append(block["text"])
                continue

            if block_type != "tool_use":
                continue

            tool_calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )

        content = "\n\n".join(text_parts) if text_parts else None
        if not content and not tool_calls:
            content = ""

        return [
            OpenAIMessage(
                role="assistant",
                content=content,
                tool_calls=tool_calls or None,
            )
        ]

    @classmethod
    def _convert_user_message(
        cls, message: ClaudeMessage, use_prompt_tool_mode: bool = False
    ) -> List[OpenAIMessage]:
        if not isinstance(message.content, list):
            return [OpenAIMessage(role="user", content=cls._convert_content(message.content))]

        openai_messages: List[OpenAIMessage] = []
        buffered_content: List[Dict[str, Any]] = []

        for block in message.content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "tool_result":
                if buffered_content:
                    openai_messages.append(
                        OpenAIMessage(
                            role="user",
                            content=cls._convert_content(buffered_content),
                        )
                    )
                    buffered_content = []

                if use_prompt_tool_mode:
                    openai_messages.append(
                        OpenAIMessage(
                            role="user",
                            content=cls._tool_result_to_prompt_text(block),
                        )
                    )
                    continue

                openai_messages.append(
                    OpenAIMessage(
                        role="tool",
                        tool_call_id=block.get("tool_use_id", ""),
                        content=cls._stringify_tool_result_content(block.get("content", "")),
                    )
                )
                continue

            buffered_content.append(block)

        if buffered_content:
            openai_messages.append(
                OpenAIMessage(
                    role="user",
                    content=cls._convert_content(buffered_content),
                )
            )

        return openai_messages

    @classmethod
    def _assistant_blocks_to_prompt_content(
        cls, content_blocks: List[Dict[str, Any]]
    ) -> str:
        parts: List[str] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "text" and block.get("text"):
                parts.append(block["text"])
                continue

            if block_type != "tool_use":
                continue

            parts.append(
                json.dumps(
                    {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }
                )
            )

        return "\n\n".join(part for part in parts if part)

    @classmethod
    def _stringify_tool_result_content(
        cls, content: Union[str, List[Dict[str, Any]], Any]
    ) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_parts.append(json.dumps(item))
            return "\n".join(part for part in text_parts if part)

        if content is None:
            return ""

        if isinstance(content, (dict, list)):
            return json.dumps(content)

        return str(content)

    @classmethod
    def _tool_result_to_prompt_text(cls, block: Dict[str, Any]) -> str:
        tool_use_id = block.get("tool_use_id", "")
        content = cls._stringify_tool_result_content(block.get("content", ""))
        return (
            "Tool result received.\n"
            f"tool_use_id: {tool_use_id}\n"
            f"{content}"
        ).strip()

    @classmethod
    def _extract_system_content(cls, system: Union[str, List[Dict[str, Any]]]) -> str:
        if isinstance(system, str):
            return system

        if not isinstance(system, list):
            return ""

        text_parts = []
        for block in system:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and "text" in block:
                text_parts.append(block["text"])
            elif "text" in block:
                text_parts.append(block["text"])

        result = "\n\n".join(text_parts)
        logger.info(
            "Extracted %s system blocks into %s chars", len(system), len(result)
        )
        return result

    @classmethod
    def _convert_content(
        cls, content: Union[str, List[Dict[str, Any]], Any]
    ) -> Union[str, List[Dict[str, Any]]]:
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content) if content else ""

        converted: List[Dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")
            if block_type == "text":
                converted.append({"type": "text", "text": block.get("text", "")})
                continue

            if block_type == "image":
                image_data = block.get("source", {})
                media_type = image_data.get("media_type", "image/png")
                data = image_data.get("data", "")
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    }
                )
                continue

            if block_type in {"tool_use", "tool_result"}:
                continue

            if "text" in block:
                converted.append({"type": "text", "text": block["text"]})

        if len(converted) == 1 and converted[0].get("type") == "text":
            return converted[0]["text"]
        if converted:
            return converted

        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n\n".join(text_parts) if text_parts else ""

    @classmethod
    def _tools_to_prompt(cls, tools: List[Dict[str, Any]]) -> str:
        if not tools:
            return ""

        tool_descriptions = []
        for tool in tools:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            schema = tool.get("input_schema", {})
            properties = schema.get("properties", {})
            required = set(schema.get("required", []))

            params = []
            for param_name, param_info in properties.items():
                param_type = param_info.get("type", "string")
                param_desc = param_info.get("description", "")
                requirement = "required" if param_name in required else "optional"
                params.append(
                    f"  - {param_name}: {param_type} ({requirement}) - {param_desc}"
                )

            params_str = "\n".join(params) if params else "  No parameters"
            tool_descriptions.append(
                f"### {name}\n{description}\nParameters:\n{params_str}"
            )

        prompt = (
            "# Available Tools\n\n"
            "You have access to the following tools. To use a tool, respond with a JSON object "
            "in this exact format:\n\n"
            '{"type": "tool_use", "name": "tool_name", "input": {"param1": "value1"}}\n\n'
            "Do not include any other text when calling a tool.\n\n"
            f"{chr(10).join(tool_descriptions)}\n\n"
            "Remember: when you want to use a tool, output only the JSON tool call."
        )
        return prompt.strip()

    @classmethod
    def _convert_tools(
        cls, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert Claude tools format to OpenAI tools format."""
        if not tools:
            return None

        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)

        logger.info(f"Converted {len(openai_tools)} tools to OpenAI format")
        return openai_tools

    @classmethod
    def _convert_tool_choice(
        cls, tool_choice: Optional[Union[str, Dict[str, Any]]]
    ) -> Optional[Union[str, Dict[str, Any]]]:
        if not tool_choice:
            return None

        if isinstance(tool_choice, str):
            return "required" if tool_choice == "any" else tool_choice

        choice_type = tool_choice.get("type")
        if choice_type == "tool" and tool_choice.get("name"):
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }

        if choice_type == "any":
            return "required"

        if choice_type in {"auto", "none", "required"}:
            return choice_type

        return None
