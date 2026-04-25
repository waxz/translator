import json
import re
from json_repair import repair_json
from rapidfuzz import process, fuzz

# Define your standard Tool Call schema keys to fix LLM typos
EXPECTED_TOOL_KEYS = ["type", "id", "name", "input", "arguments", "parameters"]

def process_llm_tool_calls(text, schema_keys=EXPECTED_TOOL_KEYS):
    """
    Extracts, repairs, and fuzzy-corrects tool call JSON from raw LLM text.
    """
    extracted_objects = []
    
    # 1. Regex to find potential JSON blocks (handles multiple)
    # This finds everything from the first { to the last } in a block
    blocks = re.findall(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    
    for block in blocks:
        try:
            # 2. Structural Repair (Fixes missing brackets, single quotes, trailing text)
            repaired_str = repair_json(block)
            data = json.loads(repaired_str)
            
            # If it's a list of tool calls, process each; if dict, wrap in list
            items = data if isinstance(data, list) else [data]
            
            for item in items:
                if not isinstance(item, dict):
                    continue
                
                # 3. Semantic Fuzzy Fix (Fixes typos in keys like 'nam' -> 'name')
                fixed_item = {}
                for k, v in item.items():
                    # Find closest match in our expected schema
                    match, score, _ = process.extractOne(k, schema_keys, scorer=fuzz.WRatio)
                    
                    if score > 80: # 80% similarity threshold
                        fixed_item[match] = v
                    else:
                        fixed_item[k] = v
                
                extracted_objects.append(fixed_item)
                
        except Exception as e:
            print(f"Failed to recover a block: {e}")
            continue

    return extracted_objects

# --- TEST ---
obj_str = """
No files were found containing the pattern "preview_patch.*error" in the codebase.

Since I couldn't find any specific configurations or error messages related to `preview_patch`, I'll use the `AskUserQuestion` tool again to ask the user if they have any additional information about the issue or if they can provide more context.

{"type": "tool_use", "id": "ask_user_again", "name": "AskUserQuestion", "input": {"questions": [{"question": "Can you provide more context or details about the issue with preview_patch?", "header": "Additional context", "options": [{"label": "Claude Code configuration issue", "description": "There might be a configuration issue with Claude Code"}, {"label": "MCP tool implementation issue", "description": "There might be an issue with the MCP tool implementation"}, {"label": "Other", "description": "Other reasons or additional context"}], "multiSelect": false}]}
"""
tools = process_llm_tool_calls(obj_str)
print(json.dumps(tools, indent=2))
