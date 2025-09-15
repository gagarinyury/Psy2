"""
LLM-based reasoning node using DeepSeek API.

Alternative to the stub reason node that uses actual LLM reasoning
to determine patient response strategy.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.infra.tracing import get_tracer
from app.llm.deepseek_client import DeepSeekClient
from app.llm.json_parse import normalize_reason_payload, parse_llm_json
from app.llm.validate import validate_reason_payload

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


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
        return (
            "You are a therapeutic reasoning system. Analyze the input and return JSON "
            "with content_plan, style_directives, state_updates, and telemetry."
        )


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


def _create_fallback_response(
    case_truth: dict, session_state: dict, parse_error: bool = False
) -> dict:
    """Create fallback response when LLM fails."""
    logger.warning("Using fallback reasoning response due to LLM failure")

    # Simple fallback logic - trust decreases slightly when LLM fails
    trust_delta = -0.1
    fatigue_delta = 0.05

    telemetry = {"chosen_ids": []}
    if parse_error:
        telemetry["llm_parse_error"] = True

    return {
        "content_plan": ["I'm feeling a bit confused right now"],
        "style_directives": {"tempo": "calm", "length": "short"},
        "state_updates": {"trust_delta": trust_delta, "fatigue_delta": fatigue_delta},
        "telemetry": telemetry,
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
        with tracer.start_as_current_span("llm.reasoning") as span:
            span.set_attribute("llm.model", "deepseek-reasoning")
            span.set_attribute("llm.task", "reasoning")
            span.set_attribute("input.candidates_count", len(candidates))

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

        # Parse JSON response using robust parser
        try:
            parsed_data = parse_llm_json(content)
            result = normalize_reason_payload(parsed_data)

            # Validate and auto-repair the payload
            validated_result, validation_warnings = validate_reason_payload(
                result, candidates
            )

            # If content_plan is still empty after validation, use fallback
            if not validated_result.get("content_plan"):
                logger.warning(
                    "content_plan empty after validation, using fallback",
                    extra={"validation_warnings": validation_warnings},
                )
                fallback_result = _create_fallback_response(
                    case_truth, session_state, parse_error=False
                )
                fallback_result["telemetry"]["llm_validation_failed"] = True
                return fallback_result

            logger.info(
                "DeepSeek reasoning successful",
                extra={
                    "content_plan_items": len(validated_result["content_plan"]),
                    "chosen_fragments": len(
                        validated_result["telemetry"]["chosen_ids"]
                    ),
                    "validation_warnings_count": len(validation_warnings),
                },
            )

            return validated_result

        except (ValueError, Exception) as e:
            logger.error(
                f"Failed to parse response from DeepSeek: {e}",
                extra={
                    "content": content[:500] + "..." if len(content) > 500 else content
                },
            )
            return _create_fallback_response(
                case_truth, session_state, parse_error=True
            )

    except Exception as e:
        logger.error(f"DeepSeek reasoning failed: {e}")
        return _create_fallback_response(case_truth, session_state)
