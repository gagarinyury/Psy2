"""
Guard node for orchestrator pipeline.
Filters and modifies content based on risk flags to ensure safe therapeutic responses.
"""

import logging
import copy

logger = logging.getLogger(__name__)


def guard(reason_output: dict, policies: dict, risk_flags: list[str]) -> dict:
    """
    Apply risk-based content filtering and modification to reason output.

    Args:
        reason_output: Result from reason node with content_plan, style_directives, etc.
        policies: Policy configuration dict (RiskProtocol)
        risk_flags: List of risk indicator strings (e.g., ["suicide_ideation"])

    Returns:
        dict with keys:
            - safe_output: dict - Modified or unchanged reason_output
            - risk_status: str - "none" or "acute"
    """
    logger.debug(f"Processing guard with {len(risk_flags)} risk flags")

    # Deep copy to avoid modifying original reason_output
    safe_output = copy.deepcopy(reason_output) if reason_output else {}

    # Determine risk status and apply filtering
    if risk_flags:
        risk_status = "acute"

        # Replace content plan with risk protocol message
        safe_output["content_plan"] = ["[Риск-триггер: обращение к протоколу]"]

        # Override tempo to calm for risk situations
        if "style_directives" not in safe_output:
            safe_output["style_directives"] = {}
        safe_output["style_directives"]["tempo"] = "calm"

        logger.warning(f"Risk detected: {risk_flags} - content filtered")
    else:
        risk_status = "none"
        # No modifications needed - safe_output remains as reason_output
        logger.debug("No risk detected - content passes through unchanged")

    return {
        "safe_output": safe_output,
        "risk_status": risk_status,
    }
