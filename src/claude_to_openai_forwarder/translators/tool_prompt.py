from typing import List, Dict, Any, Optional
import json
import re
import time
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def tools_to_prompt(tools: List[Dict[str, Any]]) -> str:
    """Convert tools to a simple prompt instruction"""
    if not tools:
        return ""

    tool_list = []
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})
        tool_list.append(f"- {name}: {desc}\n  Input: {json.dumps(schema)}")

    return f"""You have these tools available:

{chr(10).join(tool_list)}

To use a tool, respond with ONLY this JSON format (no markdown):
{{"type": "tool_use", "name": "tool_name", "id": "call_123", "input": {{"param": "value"}}}}

After using a tool, you'll receive:
Tool result for call_123:
<result content>
"""


def parse_all_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract ALL tool calls from text with their positions.
    Handles invalid control characters that some LLMs output.

    Returns:
        List of {tool_call: dict, start: int, end: int}
    """
    logger.debug(f"parse_all_tool_calls called with text length: {len(text)}")
    logger.debug(f"Text preview: {repr(text[:500]) if text else 'EMPTY'}...")

    results = []

    # First try alternative format: <tool_call>name{...} or <tool_call> name {...}
    # Pattern: <tool_call> followed by tool name (word chars + underscores), optional whitespace, then JSON
    # The JSON may contain newlines which are invalid - we try to parse and fix
    alt_pattern = r'<tool_call>\s*([\w_]+)\s*(\{[\s\S]*?\})'
    for match in re.finditer(alt_pattern, text):
        tool_name = match.group(1)
        json_str = match.group(2)
        start_pos = match.start()
        end_pos = match.end()
        logger.debug(f"Found alternative format tool_call: {tool_name} at {start_pos}")

        try:
            # Try to parse JSON, fixing control chars if needed
            try:
                input_data = json.loads(json_str)
            except json.JSONDecodeError:
                fixed_str = _fix_json_control_chars(json_str)
                input_data = json.loads(fixed_str)
                logger.debug(f"Fixed JSON for {tool_name}")

            results.append({
                "tool_call": {
                    "type": "tool_use",
                    "name": tool_name,
                    "id": f"call_{int(time.time() * 1000)}_{len(results)}",
                    "input": input_data
                },
                "start": start_pos,
                "end": end_pos
            })
            logger.info(f"Successfully parsed alternative tool_call: {tool_name}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse alternative format: {e}")
            continue

    # Find all potential tool_use JSON objects (standard format)
    pattern = r'\{\s*"type"\s*:\s*"tool_use"'

    for match in re.finditer(pattern, text):
        start_pos = match.start()
        logger.debug(f"Found potential tool_use at position {start_pos}")

        # Find matching closing brace
        depth = 0
        in_string = False
        escape = False

        for i in range(start_pos, len(text)):
            char = text[i]

            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        json_str = text[start_pos:end_pos]
                        logger.debug(f"Found JSON string: {json_str[:200]}...")

                        try:
                            data = json.loads(json_str)
                            if data.get("type") == "tool_use" and data.get("name"):
                                if "id" not in data:
                                    data["id"] = f"call_{int(time.time() * 1000)}"
                                results.append({
                                    "tool_call": data,
                                    "start": start_pos,
                                    "end": end_pos
                                })
                                logger.info(f"Successfully parsed tool_use: {data.get('name')} with id {data.get('id')}")
                            else:
                                logger.debug(f"JSON missing required fields: {data}")
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON decode error: {e}")
                            # Try to fix common issues: unescaped control characters
                            try:
                                fixed_str = _fix_json_control_chars(json_str)
                                logger.debug(f"Fixed JSON string length: {len(fixed_str)} (was {len(json_str)})")
                                data = json.loads(fixed_str)
                                if data.get("type") == "tool_use" and data.get("name"):
                                    if "id" not in data:
                                        data["id"] = f"call_{int(time.time() * 1000)}"
                                    results.append({
                                        "tool_call": data,
                                        "start": start_pos,
                                        "end": end_pos
                                    })
                                    logger.info(f"Successfully parsed tool_use after fixing: {data.get('name')}")
                                else:
                                    logger.debug(f"Fixed JSON missing required fields: {data}")
                            except (json.JSONDecodeError, ValueError) as e2:
                                logger.debug(f"Failed to parse even after fixing: {e2}")
                                # Log a preview of what we're trying to fix for debugging
                                if len(json_str) > 5000:
                                    logger.debug(f"Problematic JSON around error (4900-5100): {repr(json_str[4900:5100])}")
                        break

    logger.info(f"parse_all_tool_calls found {len(results)} tool calls (alt_format + standard)")
    return results


def _fix_json_control_chars(json_str: str) -> str:
    """
    Fix common JSON issues from LLM outputs:
    - Unescaped literal newlines (\x0a) and tabs (\x09) inside strings
    - Other control characters
    """
    # Replace literal newlines and tabs inside strings with escaped versions
    # This is a heuristic approach - we need to be careful to only fix inside strings
    result = []
    in_string = False
    escape = False

    for char in json_str:
        if escape:
            result.append(char)
            escape = False
            continue
        if char == '\\':
            result.append(char)
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string and ord(char) < 0x20:
            # Control character inside string - escape it
            if char == '\n':
                result.append('\\n')
            elif char == '\t':
                result.append('\\t')
            elif char == '\r':
                result.append('\\r')
            else:
                # Other control characters - replace with space
                result.append(' ')
        else:
            result.append(char)

    return ''.join(result)


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from response text"""
    logger.debug(f"parse_tool_call called with text length: {len(text)}")

    # Find JSON object with type: tool_use
    # This simple regex only works for simple JSON without nested braces
    match = re.search(r'\{[^{}]*"type"\s*:\s*"tool_use"[^{}]*\}', text, re.DOTALL)
    if not match:
        logger.debug("No simple tool_use JSON found, trying parse_all_tool_calls")
        # Try the more robust parser
        results = parse_all_tool_calls(text)
        if results:
            logger.info(f"parse_tool_call returning first result from parse_all_tool_calls: {results[0]['tool_call']}")
            return results[0]["tool_call"]
        logger.debug("No tool calls found")
        return None

    try:
        data = json.loads(match.group(0))
        if data.get("type") == "tool_use" and data.get("name"):
            # Add ID if missing
            if "id" not in data:
                data["id"] = f"call_{int(time.time() * 1000)}"
            logger.info(f"parse_tool_call found tool: {data.get('name')}")
            return data
    except json.JSONDecodeError as e:
        logger.debug(f"JSON decode error in parse_tool_call: {e}")

    logger.debug("parse_tool_call returning None")
    return None