from typing import List, Dict, Any, Optional
import json
import re
import time
import logging

logger = logging.getLogger(__name__)

# Try to import json_repair for robust JSON extraction
try:
    from json_repair import repair_json

    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logger.warning("json_repair not available, using basic parsing")

# Schema keys for fuzzy correction
TOOL_SCHEMA_KEYS = ["type", "id", "name", "input", "arguments", "parameters"]


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


def extract_with_json_repair(text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls using json_repair library for robust JSON repair.
    This handles malformed JSON that the standard parser can't handle.
    """
    if not JSON_REPAIR_AVAILABLE:
        return []

    from rapidfuzz import process, fuzz

    results = []

    # Find potential JSON blocks using regex
    # This regex finds { } blocks that contain "type" key
    pattern = r'\{"type"\s*:\s*"[^"]*"[^}]*\}'

    for match in re.finditer(pattern, text, re.DOTALL):
        start_pos = match.start()
        json_str = match.group(0)

        try:
            # Use json_repair to fix malformed JSON
            repaired_str = repair_json(json_str)
            data = json.loads(repaired_str)

            if not isinstance(data, dict):
                continue

            # Check if it's a valid tool_use
            if data.get("type") != "tool_use":
                continue

            tool_name = data.get("name")
            if not tool_name:
                continue

            # Find the end position
            end_pos = start_pos + len(json_str)

            # Fuzzy fix key typos
            fixed_data = {}
            for k, v in data.items():
                match_result, score, _ = process.extractOne(
                    k, TOOL_SCHEMA_KEYS, scorer=fuzz.WRatio
                )
                if score > 80:
                    fixed_data[match_result] = v
                else:
                    fixed_data[k] = v

            # Ensure id is present
            if "id" not in fixed_data:
                fixed_data["id"] = f"call_{int(time.time() * 1000)}"

            results.append(
                {
                    "tool_call": fixed_data,
                    "start": start_pos,
                    "end": end_pos,
                }
            )
            logger.info(f"Successfully parsed tool_use via json_repair: {tool_name}")

        except Exception as e:
            logger.debug(f"Failed to parse with json_repair: {e}")
            continue

    return results


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
    alt_pattern = r"<tool_call>\s*([\w_]+)\s*(\{[\s\S]*?\})"
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

            results.append(
                {
                    "tool_call": {
                        "type": "tool_use",
                        "name": tool_name,
                        "id": f"call_{int(time.time() * 1000)}_{len(results)}",
                        "input": input_data,
                    },
                    "start": start_pos,
                    "end": end_pos,
                }
            )
            logger.info(f"Successfully parsed alternative tool_call: {tool_name}")
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"Failed to parse alternative format: {e}")
            continue

    # Support Claude Code-style function calls such as Agent({...})
    # where the payload may be JSON-like rather than strict JSON.
    fn_pattern = r"(?<![\w<])([A-Z][\w]*)\s*\("
    for match in re.finditer(fn_pattern, text):
        tool_name = match.group(1)
        paren_start = match.end() - 1
        call_body = _extract_balanced_segment(text, paren_start, "(", ")")
        if not call_body:
            continue

        inner = call_body[1:-1].strip()
        if not inner.startswith("{"):
            continue

        input_data = _parse_tool_input_object(inner)
        if input_data is None:
            logger.debug(f"Failed to parse function-style tool call: {tool_name}")
            continue

        results.append(
            {
                "tool_call": {
                    "type": "tool_use",
                    "name": tool_name,
                    "id": f"call_{int(time.time() * 1000)}_{len(results)}",
                    "input": input_data,
                },
                "start": match.start(),
                "end": paren_start + len(call_body),
            }
        )
        logger.info(f"Successfully parsed function-style tool_call: {tool_name}")

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
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue

            if not in_string:
                if char == "{":
                    depth += 1
                elif char == "}":
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
                                results.append(
                                    {
                                        "tool_call": data,
                                        "start": start_pos,
                                        "end": end_pos,
                                    }
                                )
                                logger.info(
                                    f"Successfully parsed tool_use: {data.get('name')} with id {data.get('id')}"
                                )
                            else:
                                logger.debug(f"JSON missing required fields: {data}")
                        except json.JSONDecodeError as e:
                            logger.debug(f"JSON decode error: {e}")
                            # Try to fix common issues: unescaped control characters
                            try:
                                fixed_str = _fix_json_control_chars(json_str)
                                logger.debug(
                                    f"Fixed JSON string length: {len(fixed_str)} (was {len(json_str)})"
                                )
                                data = json.loads(fixed_str)
                                if data.get("type") == "tool_use" and data.get("name"):
                                    if "id" not in data:
                                        data["id"] = f"call_{int(time.time() * 1000)}"
                                    results.append(
                                        {
                                            "tool_call": data,
                                            "start": start_pos,
                                            "end": end_pos,
                                        }
                                    )
                                    logger.info(
                                        f"Successfully parsed tool_use after fixing: {data.get('name')}"
                                    )
                                else:
                                    logger.debug(
                                        f"Fixed JSON missing required fields: {data}"
                                    )
                            except (json.JSONDecodeError, ValueError) as e2:
                                logger.debug(f"Failed to parse even after fixing: {e2}")
                                # Try fuzzy extraction as last resort
                                fuzzy_data = _extract_json_fuzzy(text, start_pos)
                                if (
                                    fuzzy_data
                                    and fuzzy_data.get("type") == "tool_use"
                                    and fuzzy_data.get("name")
                                ):
                                    if "id" not in fuzzy_data:
                                        fuzzy_data["id"] = (
                                            f"call_{int(time.time() * 1000)}"
                                        )
                                    results.append(
                                        {
                                            "tool_call": fuzzy_data,
                                            "start": start_pos,
                                            "end": end_pos,
                                        }
                                    )
                                    logger.info(
                                        f"Successfully parsed tool_use via fuzzy extraction: {fuzzy_data.get('name')}"
                                    )
                                else:
                                    logger.debug(f"Fuzzy extraction also failed")
                        break

    # If no results found, try json_repair as fallback
    if not results and JSON_REPAIR_AVAILABLE:
        logger.debug("No results from basic parsing, trying json_repair...")
        results = extract_with_json_repair(text)

    results.sort(key=lambda item: item["start"])

    logger.info(
        f"parse_all_tool_calls found {len(results)} tool calls (alt_format + standard)"
    )
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
        if char == "\\":
            result.append(char)
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            continue

        if in_string and ord(char) < 0x20:
            # Control character inside string - escape it
            if char == "\n":
                result.append("\\n")
            elif char == "\t":
                result.append("\\t")
            elif char == "\r":
                result.append("\\r")
            else:
                # Other control characters - replace with space
                result.append(" ")
        else:
            result.append(char)

    return "".join(result)


CONTROL_TEXT_TAGS = (
    "analysis",
    "commentary",
    "final",
    "result",
    "thinking",
)


def strip_control_text_tags(text: str) -> str:
    if not text:
        return text

    for tag in CONTROL_TEXT_TAGS:
        text = re.sub(rf"<{tag}\b[^>]*>", "", text, flags=re.IGNORECASE)
        text = re.sub(rf"</{tag}>", "", text, flags=re.IGNORECASE)
    return text


def _extract_balanced_segment(
    text: str, start_pos: int, open_char: str, close_char: str
) -> Optional[str]:
    depth = 0
    in_string = False
    escape = False

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start_pos : i + 1]

    return None


def _parse_tool_input_object(obj_str: str) -> Optional[Dict[str, Any]]:
    try:
        parsed = json.loads(obj_str)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fixed = _fix_json_control_chars(obj_str)
    try:
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    if JSON_REPAIR_AVAILABLE:
        try:
            repaired = repair_json(obj_str)
            parsed = json.loads(repaired)
            if isinstance(parsed, dict):
                return parsed
        except Exception as e:
            logger.debug(f"json_repair failed for function-style tool call: {e}")

    return _parse_simple_object(obj_str) or None


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
            logger.info(
                f"parse_tool_call returning first result from parse_all_tool_calls: {results[0]['tool_call']}"
            )
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


def _extract_json_fuzzy(text: str, start_pos: int) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object using fuzzy parsing that tolerates common malformations.
    Returns None if extraction fails.
    """
    depth = 0
    in_string = False
    escape = False
    json_end = -1

    for i in range(start_pos, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue

        if not in_string:
            if char in "{[":
                depth += 1
            elif char in "}])":
                depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

    json_str = text[start_pos:json_end] if json_end > 0 else text[start_pos:]
    if not json_str.strip():
        return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    fixed = _fix_json_control_chars(json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    end_pos = json_end if json_end > 0 else len(text)
    return _extract_fields_regex(text, start_pos, end_pos)


def _extract_fields_regex(
    text: str, start_pos: int, end_pos: int
) -> Optional[Dict[str, Any]]:
    """
    Fallback: Extract tool_use fields using regex when JSON parsing fails.
    """
    json_substr = text[start_pos:end_pos]

    tool_type_match = re.search(r'"type"\s*:\s*"([^"]+)"', json_substr)
    if not tool_type_match or tool_type_match.group(1) != "tool_use":
        return None

    tool_name_match = re.search(r'"name"\s*:\s*"([^"]+)"', json_substr)
    if not tool_name_match:
        return None

    tool_id_match = re.search(r'"id"\s*:\s*"([^"]+)"', json_substr)
    tool_id = (
        tool_id_match.group(1) if tool_id_match else f"call_{int(time.time() * 1000)}"
    )

    tool_input = {}
    input_match = re.search(r'"input"\s*:\s*(\{.*?\})', json_substr, re.DOTALL)
    if input_match:
        try:
            tool_input = json.loads(input_match.group(1))
        except json.JSONDecodeError:
            tool_input = _parse_simple_object(input_match.group(1))

    return {
        "type": "tool_use",
        "id": tool_id,
        "name": tool_name_match.group(1),
        "input": tool_input,
    }


def _parse_simple_object(obj_str: str) -> Dict[str, Any]:
    """
    Parse a simple JSON object using regex for key-value pairs.
    """
    result = {}
    pairs = re.findall(r'"(\w+)"\s*:\s*("[^"]*"|[\d]+|true|false|null)', obj_str)
    for key, value in pairs:
        if value.startswith('"') and value.endswith('"'):
            result[key] = value[1:-1]
        elif value == "true":
            result[key] = True
        elif value == "false":
            result[key] = False
        elif value == "null":
            result[key] = None
        else:
            try:
                result[key] = int(value)
            except ValueError:
                pass
    return result
