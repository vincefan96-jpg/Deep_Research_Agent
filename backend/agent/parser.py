import json
import re
from dataclasses import dataclass


@dataclass
class ParsedOutput:
    thought: str
    action: str | None = None       # tool name
    action_params: dict | None = None
    is_final: bool = False
    final_answer: str | None = None
    parse_error: str | None = None


def parse(raw: str) -> ParsedOutput:
    thought = ""
    action = None
    action_params = None
    is_final = False
    final_answer = None
    parse_error = None

    # Extract THOUGHT
    thought_match = re.search(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION|FINAL_ANSWER)|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Check for FINAL_ANSWER
    if re.search(r"FINAL_ANSWER", raw, re.IGNORECASE):
        is_final = True
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", raw, re.DOTALL | re.IGNORECASE)
        if fa_match:
            final_answer = fa_match.group(1).strip()
        else:
            # Everything after FINAL_ANSWER marker
            idx = raw.upper().find("FINAL_ANSWER")
            final_answer = raw[idx + len("FINAL_ANSWER"):].strip(": \n")

    # Extract ACTION
    action_match = re.search(r"ACTION:\s*(.+?)(?=\n(?:THOUGHT|OBSERVATION|FINAL_ANSWER)|\Z)", raw, re.DOTALL | re.IGNORECASE)
    if action_match and not is_final:
        action_str = action_match.group(1).strip()

        # Action format: <tool_name>|<json_params>
        if "|" in action_str:
            parts = action_str.split("|", 1)
            action = parts[0].strip()
            try:
                action_params = json.loads(parts[1].strip())
            except json.JSONDecodeError:
                parse_error = f"Failed to parse action params JSON: {parts[1][:100]}"
                action_params = {}

    # If nothing was parsed, treat as parse failure
    if not thought and not action and not is_final:
        parse_error = "Could not parse THOUGHT, ACTION, or FINAL_ANSWER from LLM output."
        thought = raw[:500]

    return ParsedOutput(
        thought=thought,
        action=action,
        action_params=action_params,
        is_final=is_final,
        final_answer=final_answer,
        parse_error=parse_error,
    )
