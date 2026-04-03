from typing import List, Dict, Any
import json


def tools_to_prompt(tools: List[Dict[str, Any]]) -> str:
    """
    Convert tools to a prompt instruction for LLMs that don't support native tool calling
    """
    if not tools:
        return ""
    
    tool_descriptions = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})
        
        tool_desc = f"""
Tool: {name}
Description: {desc}
Parameters: {json.dumps(schema, indent=2)}
"""
        tool_descriptions.append(tool_desc)
    
    prompt = f"""
You have access to the following tools:

{chr(10).join(tool_descriptions)}

To use a tool, respond with a JSON object in this format:
{{"type": "tool_use", "name": "tool_name", "input": {{"param": "value"}}}}

Available tools: {', '.join(t.get('name', '') for t in tools)}
"""
    
    return prompt