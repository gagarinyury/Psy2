"""
Robust JSON parsing for LLM responses with fallback mechanisms.

Handles various response formats including reasoning_content, code blocks,
comments, and malformed JSON structures.
"""

import json
import re
from typing import Any, Dict, List

try:
    import json5
except ImportError:
    json5 = None


def extract_json_blocks(text: str) -> List[str]:
    """
    Returns JSON candidates from text: ```json...```, ```...```, and raw {...} by regex.
    Order: codefence json → codefence any → raw braces.
    """
    candidates = []

    # 1. Extract ```json code blocks first (highest priority)
    json_code_pattern = r"```json\s*\n(.*?)\n```"
    json_matches = re.findall(json_code_pattern, text, re.DOTALL | re.IGNORECASE)
    candidates.extend(json_matches)

    # 2. Extract any ``` code blocks (medium priority)
    general_code_pattern = r"```[^j\n]*\n(.*?)\n```"  # Not starting with 'j' to avoid json blocks
    general_matches = re.findall(general_code_pattern, text, re.DOTALL)
    candidates.extend(general_matches)

    # 3. Extract raw {...} with balanced braces (lowest priority)
    def find_balanced_braces(text: str) -> List[str]:
        results = []
        i = 0
        while i < len(text):
            if text[i] == "{":
                brace_count = 1
                start = i
                i += 1
                while i < len(text) and brace_count > 0:
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    results.append(text[start:i])
            else:
                i += 1
        return results

    raw_json_blocks = find_balanced_braces(text)
    candidates.extend(raw_json_blocks)

    return candidates


def parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Tries in order:
      1) json.loads on each candidate,
      2) json5.loads on each candidate,
      3) final regex extraction {...} with balanced braces.
    Normalizes keys and returns dict.
    Raises ValueError if nothing worked.
    """
    candidates = extract_json_blocks(text)

    if not candidates:
        raise ValueError("No JSON candidates found in text")

    # Try standard JSON parsing first
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # Try json5 parsing if available
    if json5:
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            try:
                result = json5.loads(candidate)
                if isinstance(result, dict):
                    return result
            except Exception:
                continue

    # Final attempt: try to clean and parse the first candidate
    if candidates:
        candidate = candidates[0].strip()
        # Remove common issues
        candidate = re.sub(r",\s*}", "}", candidate)  # Remove trailing commas
        candidate = re.sub(r",\s*]", "]", candidate)  # Remove trailing commas in arrays
        candidate = re.sub(r"//.*?$", "", candidate, flags=re.MULTILINE)  # Remove // comments

        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Failed to parse JSON from {len(candidates)} candidates")


def normalize_reason_payload(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes to schema:
      content_plan: list[str]
      style_directives: {tempo:str='medium', length:str='short'}
      state_updates: {trust_delta:float=0.0, fatigue_delta:float=0.0}
      telemetry: {chosen_ids:list[str]=[]}
    Ignores extra fields. Empty → ValueError.
    """
    if not d:
        raise ValueError("Empty dictionary provided")

    result = {}

    # Normalize content_plan
    content_plan = d.get("content_plan", [])
    if isinstance(content_plan, list):
        result["content_plan"] = [str(item) for item in content_plan]
    elif isinstance(content_plan, str):
        result["content_plan"] = [content_plan]
    else:
        result["content_plan"] = ["I'm feeling a bit confused right now"]

    # Normalize style_directives
    style_directives = d.get("style_directives", {})
    if not isinstance(style_directives, dict):
        style_directives = {}

    result["style_directives"] = {
        "tempo": str(style_directives.get("tempo", "medium")),
        "length": str(style_directives.get("length", "short")),
    }

    # Normalize state_updates
    state_updates = d.get("state_updates", {})
    if not isinstance(state_updates, dict):
        state_updates = {}

    try:
        trust_delta = float(state_updates.get("trust_delta", 0.0))
    except (ValueError, TypeError):
        trust_delta = 0.0

    try:
        fatigue_delta = float(state_updates.get("fatigue_delta", 0.0))
    except (ValueError, TypeError):
        fatigue_delta = 0.0

    result["state_updates"] = {
        "trust_delta": trust_delta,
        "fatigue_delta": fatigue_delta,
    }

    # Normalize telemetry
    telemetry = d.get("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}

    chosen_ids = telemetry.get("chosen_ids", [])
    if not isinstance(chosen_ids, list):
        chosen_ids = []

    result["telemetry"] = {"chosen_ids": [str(item) for item in chosen_ids]}

    # Validate that we have at least something meaningful
    if (
        not result["content_plan"]
        and result["state_updates"]["trust_delta"] == 0.0
        and result["state_updates"]["fatigue_delta"] == 0.0
    ):
        raise ValueError("Normalized payload contains no meaningful data")

    return result
