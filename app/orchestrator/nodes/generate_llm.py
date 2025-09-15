"""
LLM-based generation node using DeepSeek API.

Alternative to stub generation that uses actual LLM to create
natural patient responses based on content plan and style.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from app.infra.tracing import get_tracer
from app.llm.deepseek_client import DeepSeekClient

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _load_generation_prompt() -> str:
    """Load the generation system prompt from file."""
    prompt_path = (
        Path(__file__).parent.parent.parent
        / "llm"
        / "prompts"
        / "generation.prompt.txt"
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Failed to load generation prompt: {e}")
        return (
            "You are a digital patient. Generate a natural response based on the "
            "content plan and style directives provided."
        )


def _create_fallback_response(content_plan: List[str]) -> str:
    """Create fallback response when LLM fails."""
    logger.warning("Using fallback generation response due to LLM failure")

    if not content_plan:
        return "I'm not sure how to respond right now."

    # Simple fallback - just join the content points
    return " ".join(content_plan[:2])  # Use first 2 points max


async def generate_llm(
    content_plan: List[str],
    style_directives: Dict[str, str],
    patient_context: str = None,
) -> str:
    """
    LLM-based patient response generation using DeepSeek API.

    Args:
        content_plan: List of factual points to communicate
        style_directives: tempo and length specifications
        patient_context: Optional patient background context

    Returns:
        str: Natural patient response
    """
    try:
        # Load system prompt
        system_prompt = _load_generation_prompt()

        # Prepare input data
        input_data = {
            "content_plan": content_plan,
            "style_directives": style_directives,
            "patient_context": patient_context or "General therapy patient",
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
            "Sending generation request to DeepSeek",
            extra={"content_plan_items": len(content_plan), "style": style_directives},
        )

        # Call DeepSeek API
        with tracer.start_as_current_span("llm.generation") as span:
            span.set_attribute("llm.model", "deepseek-generation")
            span.set_attribute("llm.task", "generation")
            span.set_attribute("input.content_plan_items", len(content_plan))

            async with DeepSeekClient() as client:
                response = await client.generate(messages, temperature=0.7, max_tokens=200)

        # Extract response content
        if not response.get("choices") or not response["choices"]:
            logger.error("Empty response from DeepSeek API")
            return _create_fallback_response(content_plan)

        content = response["choices"][0]["message"]["content"].strip()

        if not content:
            logger.error("Empty content from DeepSeek generation")
            return _create_fallback_response(content_plan)

        # Clean up the response - remove quotes if it's wrapped in quotes
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1]

        # Validate length constraints
        sentences = content.split(".")
        sentence_count = len([s for s in sentences if s.strip()])

        # Respect style directive length
        length_style = style_directives.get("length", "medium")
        if length_style == "short" and sentence_count > 1:
            # Take first sentence only
            first_sentence = sentences[0].strip()
            if first_sentence:
                content = (
                    first_sentence + "."
                    if not first_sentence.endswith(".")
                    else first_sentence
                )
        elif length_style == "long" and sentence_count > 3:
            # Take first 3 sentences
            valid_sentences = [s.strip() for s in sentences[:3] if s.strip()]
            content = ". ".join(valid_sentences)
            if not content.endswith("."):
                content += "."

        logger.info(
            "DeepSeek generation successful",
            extra={"response_length": len(content), "style": style_directives},
        )

        return content

    except Exception as e:
        logger.error(f"DeepSeek generation failed: {e}")
        return _create_fallback_response(content_plan)
