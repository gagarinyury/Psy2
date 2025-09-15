"""
Validation and auto-repair for reasoning payload.

Normalizes values, drops garbage, safely fixes fields before guard/generation.
"""

import math


def validate_reason_payload(
    payload: dict,
    candidates: list[dict],
) -> tuple[dict, list[str]]:
    """
    Returns (payload_norm, warnings).
    Does not raise exceptions.
    """
    if not isinstance(payload, dict):
        payload = {}

    if not isinstance(candidates, list):
        candidates = []

    result = payload.copy()
    warnings = []

    # Normalize content_plan
    result, plan_warnings = _normalize_content_plan(result, candidates)
    warnings.extend(plan_warnings)

    # Normalize style_directives
    result, style_warnings = _normalize_style_directives(result)
    warnings.extend(style_warnings)

    # Normalize state_updates
    result, state_warnings = _normalize_state_updates(result)
    warnings.extend(state_warnings)

    # Normalize telemetry
    result, telemetry_warnings = _normalize_telemetry(result, candidates)
    warnings.extend(telemetry_warnings)

    # Add validation warnings to telemetry
    if "telemetry" not in result:
        result["telemetry"] = {}
    if warnings:
        result["telemetry"]["validation_warnings"] = warnings

    return result, warnings


def _normalize_content_plan(payload: dict, candidates: list[dict]) -> tuple[dict, list[str]]:
    """Normalize content_plan: list of strings, trim, drop empty, max 2 elements."""
    warnings = []
    result = payload.copy()

    content_plan = payload.get("content_plan", [])

    # Ensure it's a list
    if not isinstance(content_plan, list):
        if isinstance(content_plan, str):
            content_plan = [content_plan]
        else:
            content_plan = []
            warnings.append("content_plan was not a list, converted to empty list")

    # Trim and filter empty strings, then limit to max 2
    normalized_plan = []
    for item in content_plan:
        if isinstance(item, str):
            trimmed = item.strip()
            if trimmed:
                normalized_plan.append(trimmed)
                if len(normalized_plan) >= 2:  # Stop when we have 2 valid items
                    break
        else:
            warnings.append(f"content_plan item was not string: {type(item)}")

    # If plan is empty, generate from candidates
    if not normalized_plan and candidates:
        for candidate in candidates[:2]:  # Use first 1-2 candidates
            if isinstance(candidate, dict) and "text" in candidate:
                text = candidate["text"]
                if isinstance(text, str) and text.strip():
                    # Take first 200 chars
                    snippet = text[:200].strip()
                    if snippet:
                        normalized_plan.append(snippet)

        if normalized_plan:
            warnings.append("content_plan was empty, generated from candidates")
            # Set flag for empty plan
            if "telemetry" not in result:
                result["telemetry"] = {}
            result["telemetry"]["llm_empty_plan"] = True

    result["content_plan"] = normalized_plan
    return result, warnings


def _normalize_style_directives(payload: dict) -> tuple[dict, list[str]]:
    """Normalize style_directives: tempo in {slow,medium,fast}, length in {short,medium,long}."""
    warnings = []
    result = payload.copy()

    style_directives = payload.get("style_directives", {})
    if not isinstance(style_directives, dict):
        style_directives = {}
        warnings.append("style_directives was not a dict, reset to empty")

    normalized_style = {}

    # Normalize tempo
    tempo = style_directives.get("tempo", "medium")
    if tempo not in {"slow", "medium", "fast"}:
        warnings.append(f"tempo '{tempo}' invalid, set to 'medium'")
        tempo = "medium"
    normalized_style["tempo"] = tempo

    # Normalize length
    length = style_directives.get("length", "short")
    if length not in {"short", "medium", "long"}:
        warnings.append(f"length '{length}' invalid, set to 'short'")
        length = "short"
    normalized_style["length"] = length

    result["style_directives"] = normalized_style
    return result, warnings


def _normalize_state_updates(payload: dict) -> tuple[dict, list[str]]:
    """
    Normalize state_updates:
    - trust_delta: clamp to [-0.2, 0.2], NaN -> 0.0
    - fatigue_delta: clamp to [0.0, 0.2], NaN -> 0.0
    """
    warnings = []
    result = payload.copy()

    state_updates = payload.get("state_updates", {})
    if not isinstance(state_updates, dict):
        state_updates = {}
        warnings.append("state_updates was not a dict, reset to empty")

    normalized_state = {}

    # Normalize trust_delta
    trust_delta = state_updates.get("trust_delta", 0.0)
    try:
        trust_delta = float(trust_delta)
        if math.isnan(trust_delta) or math.isinf(trust_delta):
            trust_delta = 0.0
            warnings.append("trust_delta was NaN/inf, set to 0.0")
    except (ValueError, TypeError):
        trust_delta = 0.0
        warnings.append("trust_delta was not numeric, set to 0.0")

    # Clamp to [-0.2, 0.2]
    original_trust = trust_delta
    trust_delta = max(-0.2, min(0.2, trust_delta))
    if trust_delta != original_trust:
        warnings.append(f"trust_delta {original_trust} clamped to {trust_delta}")

    normalized_state["trust_delta"] = trust_delta

    # Normalize fatigue_delta
    fatigue_delta = state_updates.get("fatigue_delta", 0.0)
    try:
        fatigue_delta = float(fatigue_delta)
        if math.isnan(fatigue_delta) or math.isinf(fatigue_delta):
            fatigue_delta = 0.0
            warnings.append("fatigue_delta was NaN/inf, set to 0.0")
    except (ValueError, TypeError):
        fatigue_delta = 0.0
        warnings.append("fatigue_delta was not numeric, set to 0.0")

    # Clamp to [0.0, 0.2]
    original_fatigue = fatigue_delta
    fatigue_delta = max(0.0, min(0.2, fatigue_delta))
    if fatigue_delta != original_fatigue:
        warnings.append(f"fatigue_delta {original_fatigue} clamped to {fatigue_delta}")

    normalized_state["fatigue_delta"] = fatigue_delta

    result["state_updates"] = normalized_state
    return result, warnings


def _normalize_telemetry(payload: dict, candidates: list[dict]) -> tuple[dict, list[str]]:
    """
    Normalize telemetry.chosen_ids:
    - Keep only IDs from current candidates
    - Deduplicate
    - If empty but candidates exist, substitute their IDs
    """
    warnings = []
    result = payload.copy()

    telemetry = payload.get("telemetry", {})
    if not isinstance(telemetry, dict):
        telemetry = {}
        warnings.append("telemetry was not a dict, reset to empty")

    normalized_telemetry = telemetry.copy()

    # Get valid candidate IDs
    valid_ids = set()
    for candidate in candidates:
        if isinstance(candidate, dict) and "id" in candidate:
            candidate_id = candidate["id"]
            if isinstance(candidate_id, str):
                valid_ids.add(candidate_id)

    # Normalize chosen_ids
    chosen_ids = telemetry.get("chosen_ids", [])
    if not isinstance(chosen_ids, list):
        chosen_ids = []
        warnings.append("chosen_ids was not a list, reset to empty")

    # Filter and deduplicate
    normalized_ids = []
    seen = set()
    for chosen_id in chosen_ids:
        if isinstance(chosen_id, str) and chosen_id in valid_ids:
            if chosen_id not in seen:
                normalized_ids.append(chosen_id)
                seen.add(chosen_id)
        else:
            warnings.append(f"chosen_id '{chosen_id}' not in valid candidates, removed")

    # If empty but candidates exist, substitute their IDs
    if not normalized_ids and valid_ids:
        normalized_ids = list(valid_ids)
        warnings.append("chosen_ids was empty, substituted candidate IDs")

    normalized_telemetry["chosen_ids"] = normalized_ids
    result["telemetry"] = normalized_telemetry

    return result, warnings
