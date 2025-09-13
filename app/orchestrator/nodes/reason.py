"""
Reason node for orchestrator pipeline.
Processes retrieved knowledge base fragments and creates execution plan.
"""

import logging

logger = logging.getLogger(__name__)


def reason(
    case_truth: dict, session_state: dict, candidates: list[dict], policies: dict
) -> dict:
    """
    Process retrieved candidates and create content/distortion execution plan.

    Args:
        case_truth: Truth about the case (not used in current implementation)
        session_state: Current session state with trust/fatigue levels
        candidates: List of retrieved KB fragments with id, type, text, metadata
        policies: Policy configuration including style_profile

    Returns:
        dict with keys:
            - content_plan: list[str] - selected text contents for response
            - distortions_plan: list[dict] - planned distortions (empty for now)
            - style_directives: dict - tempo/length from policies
            - state_updates: dict - trust_delta, fatigue_delta
            - telemetry: dict - counts, chosen_ids for monitoring
    """
    logger.debug(f"Processing {len(candidates)} candidates for reasoning")

    # Create content plan - take first 1-2 texts from candidates
    content_plan = []
    chosen_ids = []

    if candidates:
        # Take up to 2 candidates for content plan
        for candidate in candidates[:2]:
            if "text" in candidate:
                content_plan.append(candidate["text"])
                chosen_ids.append(candidate.get("id", "unknown"))

    # Calculate trust delta based on candidate availability
    trust_delta = 0.02 if candidates else -0.01

    # Extract style directives from policies
    style_directives = _extract_style_directives(policies)

    # State updates - trust based on success, no fatigue change for now
    state_updates = {
        "trust_delta": trust_delta,
        "fatigue_delta": 0.0,  # No fatigue impact in current implementation
    }

    # Telemetry for monitoring
    telemetry = {
        "candidates_count": len(candidates),
        "chosen_count": len(chosen_ids),
        "chosen_ids": chosen_ids,
        "content_plan_size": len(content_plan),
    }

    logger.debug(
        f"Reason complete: {len(content_plan)} content items, trust_delta={trust_delta}"
    )

    return {
        "content_plan": content_plan,
        "distortions_plan": [],  # Empty for now as per specification
        "style_directives": style_directives,
        "state_updates": state_updates,
        "telemetry": telemetry,
    }


def _extract_style_directives(policies: dict) -> dict:
    """
    Extract style directives from policies configuration.

    Args:
        policies: Policy configuration dict

    Returns:
        dict with tempo and length directives
    """
    # Default style directives
    default_style = {"tempo": "medium", "length": "short"}

    if not policies:
        return default_style

    # Try to extract from style_profile
    style_profile = policies.get("style_profile", {})
    if isinstance(style_profile, dict):
        return {
            "tempo": style_profile.get("tempo", default_style["tempo"]),
            "length": style_profile.get("length", default_style["length"]),
        }

    return default_style
