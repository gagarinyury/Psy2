"""
LLM-based reasoning node using DeepSeek API.

Alternative to the stub reason node that uses actual LLM reasoning
to determine patient response strategy.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.llm.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)


def _load_reasoning_prompt() -> str:
    """Load the reasoning system prompt from file."""
    prompt_path = (
        Path(__file__).parent.parent.parent / "llm" / "prompts" / "reasoning.prompt.txt"
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load reasoning prompt: {e}")
        return "You are a therapeutic reasoning system. Analyze the input and return JSON with content_plan, style_directives, state_updates, and telemetry."


def _truncate_candidates(
    candidates: List[Dict[str, Any]], max_length: int = 500
) -> List[Dict[str, Any]]:
    """Truncate candidate text to avoid token limits."""
    truncated = []
    for candidate in candidates:
        truncated_candidate = candidate.copy()
        if "text" in candidate and len(candidate["text"]) > max_length:
            truncated_candidate["text"] = candidate["text"][:max_length] + "..."
        truncated.append(truncated_candidate)
    return truncated


def _create_fallback_response(case_truth: dict, session_state: dict) -> dict:
    """Create fallback response when LLM fails."""
    logger.warning("Using fallback reasoning response due to LLM failure")

    # Simple fallback logic - trust decreases slightly when LLM fails
    trust_delta = -0.1
    fatigue_delta = 0.05

    return {
        "content_plan": ["I'm feeling a bit confused right now"],
        "style_directives": {"tempo": "calm", "length": "short"},
        "state_updates": {"trust_delta": trust_delta, "fatigue_delta": fatigue_delta},
        "telemetry": {"chosen_ids": []},
    }


async def reason_llm(
    case_truth: dict,
    session_state: dict,
    candidates: List[Dict[str, Any]],
    policies: dict,
) -> dict:
    """
    LLM-based reasoning using DeepSeek API.

    Args:
        case_truth: Patient's diagnostic information and hidden facts
        session_state: Current session state (trust, fatigue, etc.)
        candidates: Retrieved knowledge fragments
        policies: Disclosure and risk policies

    Returns:
        dict: Reasoning result with content_plan, style_directives, state_updates, telemetry
    """
    try:
        # Load system prompt
        system_prompt = _load_reasoning_prompt()

        # Prepare input data with truncated candidates
        truncated_candidates = _truncate_candidates(candidates)

        input_data = {
            "case_truth": case_truth,
            "session_state": session_state,
            "candidates": truncated_candidates,
            "policies": policies,
        }

        # Create messages for API
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(input_data, ensure_ascii=False, indent=2),
            },
        ]

        logger.debug(
            "Sending reasoning request to DeepSeek",
            extra={"input_size": len(str(input_data))},
        )

        # Call DeepSeek API
        async with DeepSeekClient() as client:
            response = await client.reasoning(
                messages, temperature=0.3, max_tokens=1000
            )

        # Extract response content
        if not response.get("choices") or not response["choices"]:
            logger.error("Empty response from DeepSeek API")
            return _create_fallback_response(case_truth, session_state)

        message = response["choices"][0]["message"]
        # DeepSeek reasoner model uses 'reasoning_content', regular models use 'content'
        content = message.get("reasoning_content") or message.get("content", "")

        # Parse JSON response
        try:
            result = json.loads(content)

            # Validate and fix response structure
            # Add missing fields with defaults
            if "content_plan" not in result:
                result["content_plan"] = ["I'm feeling a bit confused right now"]
            if "style_directives" not in result:
                result["style_directives"] = {"tempo": "medium", "length": "short"}
            if "state_updates" not in result:
                result["state_updates"] = {"trust_delta": 0.0, "fatigue_delta": 0.0}
            if "telemetry" not in result:
                result["telemetry"] = {"chosen_ids": []}

            # Validate nested structures and fix types
            if not isinstance(result["content_plan"], list):
                result["content_plan"] = [str(result["content_plan"])]

            if not isinstance(result["style_directives"], dict):
                result["style_directives"] = {"tempo": "medium", "length": "short"}
            else:
                # Ensure required sub-fields exist
                if "tempo" not in result["style_directives"]:
                    result["style_directives"]["tempo"] = "medium"
                if "length" not in result["style_directives"]:
                    result["style_directives"]["length"] = "short"

            if not isinstance(result["state_updates"], dict):
                result["state_updates"] = {"trust_delta": 0.0, "fatigue_delta": 0.0}
            else:
                # Ensure required sub-fields exist
                if "trust_delta" not in result["state_updates"]:
                    result["state_updates"]["trust_delta"] = 0.0
                if "fatigue_delta" not in result["state_updates"]:
                    result["state_updates"]["fatigue_delta"] = 0.0

            if (
                not isinstance(result["telemetry"], dict)
                or "chosen_ids" not in result["telemetry"]
            ):
                result["telemetry"] = {"chosen_ids": []}

            logger.info(
                "DeepSeek reasoning successful",
                extra={
                    "content_plan_items": len(result["content_plan"]),
                    "chosen_fragments": len(result["telemetry"]["chosen_ids"]),
                },
            )

            return result

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON response from DeepSeek: {e}",
                extra={"content": content},
            )
            return _create_fallback_response(case_truth, session_state)

    except Exception as e:
        logger.error(f"DeepSeek reasoning failed: {e}")
        return _create_fallback_response(case_truth, session_state)
