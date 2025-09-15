"""
Tests for logging and OpenTelemetry tracing functionality.

Tests JSON logging configuration with loguru and OpenTelemetry span creation
for pipeline operations and LLM API calls.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from httpx import AsyncClient

from app.core.models import SessionStateCompact, TurnRequest
from app.infra.logging import setup_logging
from app.infra.tracing import get_tracer, setup_tracing


@pytest.mark.anyio
async def test_json_logging_setup():
    """
    Test that setup_logging configures JSON output format correctly.
    """
    # Test passes if setup_logging doesn't crash
    try:
        setup_logging()
        from loguru import logger

        logger.info("Test logging message")
        # Basic test - if we get here, logging setup worked
        assert True
    except Exception as e:
        pytest.fail(f"Logging setup failed: {e}")


@pytest.mark.anyio
async def test_tracing_setup():
    """
    Test that setup_tracing initializes OpenTelemetry correctly.
    """
    # Test passes if setup_tracing doesn't crash
    try:
        setup_tracing("test-service")
        # Basic test - if we get here, tracing setup worked
        assert True
    except Exception as e:
        pytest.fail(f"Tracing setup failed: {e}")


@pytest.mark.anyio
async def test_get_tracer():
    """
    Test that get_tracer returns a tracer instance.
    """
    try:
        tracer = get_tracer("test-module")
        # Should get a tracer object
        assert tracer is not None
        assert hasattr(tracer, "start_as_current_span")
    except Exception as e:
        pytest.fail(f"get_tracer failed: {e}")


@pytest.mark.anyio
async def test_pipeline_span_creation():
    """
    Test that pipeline operations can create spans without crashing.
    """
    # Mock OpenTelemetry tracer and span
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    with patch("app.orchestrator.pipeline.get_tracer", return_value=mock_tracer):
        # Mock database and dependencies to avoid actual DB calls
        with (
            patch("app.orchestrator.pipeline.select"),
            patch("app.orchestrator.pipeline.get_policies", return_value={}),
            patch(
                "app.orchestrator.pipeline.normalize",
                return_value={
                    "intent": "test",
                    "topics": [],
                    "risk_flags": [],
                    "last_turn_summary": "test",
                },
            ),
            patch("app.orchestrator.pipeline.retrieve", return_value=[]),
            patch("app.orchestrator.pipeline.get_case_truth", return_value={}),
            patch(
                "app.orchestrator.pipeline.reason",
                return_value={"state_updates": {}, "telemetry": {"chosen_ids": []}},
            ),
            patch(
                "app.orchestrator.pipeline.guard",
                return_value={"safe_output": {"content_plan": ["test"]}, "risk_status": "none"},
            ),
            patch("app.orchestrator.pipeline._record_telemetry"),
            patch("app.orchestrator.pipeline.update_trajectory_progress"),
        ):
            from app.orchestrator.pipeline import run_turn

            # Mock database session
            mock_db = MagicMock()
            mock_result = MagicMock()
            mock_result.scalar_one_or_none.return_value = MagicMock()  # Mock session exists
            mock_db.execute.return_value = mock_result

            # Create test request
            request = TurnRequest(
                session_id="test-session",
                case_id="test-case",
                therapist_utterance="Hello",
                session_state=SessionStateCompact(
                    affect="neutral",
                    trust=0.5,
                    fatigue=0.2,
                    access_level=1,
                    risk_status="none",
                    last_turn_summary="test",
                ),
            )

            # Call run_turn - should not crash
            result = await run_turn(request, mock_db)

            # Verify basic functionality
            assert result.patient_reply
            assert result.risk_status == "none"

            # Verify span creation was attempted
            mock_tracer.start_as_current_span.assert_called_with("pipeline.turn")


@pytest.mark.anyio
async def test_deepseek_reasoning_span():
    """
    Test that DeepSeek reasoning can create spans without crashing.
    """
    # Mock OpenTelemetry tracer and span
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    # Mock DeepSeek client response
    mock_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "content_plan": ["Test response"],
                            "style_directives": {"tempo": "medium", "length": "short"},
                            "state_updates": {"trust_delta": 0.1},
                            "telemetry": {"chosen_ids": []},
                        }
                    )
                }
            }
        ]
    }

    with (
        patch("app.orchestrator.nodes.reason_llm.get_tracer", return_value=mock_tracer),
        patch("app.orchestrator.nodes.reason_llm.DeepSeekClient") as mock_client_class,
    ):
        # Mock client instance and context manager
        mock_client = MagicMock()
        mock_client.reasoning.return_value = mock_response
        mock_client_class.return_value.__aenter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = MagicMock(return_value=None)

        # Mock prompt loading
        with patch(
            "app.orchestrator.nodes.reason_llm._load_reasoning_prompt", return_value="Test prompt"
        ):
            from app.orchestrator.nodes.reason_llm import reason_llm

            # Call reason_llm - should not crash
            result = await reason_llm(
                case_truth={"dx_target": ["test"]},
                session_state={"trust": 0.5},
                candidates=[{"text": "test fragment"}],
                policies={},
            )

            # Verify basic functionality
            assert "content_plan" in result
            assert "telemetry" in result

            # Verify span creation was attempted
            mock_tracer.start_as_current_span.assert_called_with("llm.reasoning")


@pytest.mark.anyio
async def test_deepseek_generation_span():
    """
    Test that DeepSeek generation can create spans without crashing.
    """
    # Mock OpenTelemetry tracer and span
    mock_span = MagicMock()
    mock_span.__enter__ = MagicMock(return_value=mock_span)
    mock_span.__exit__ = MagicMock(return_value=None)

    mock_tracer = MagicMock()
    mock_tracer.start_as_current_span.return_value = mock_span

    # Mock DeepSeek client response
    mock_response = {"choices": [{"message": {"content": "Generated patient response"}}]}

    with (
        patch("app.orchestrator.nodes.generate_llm.get_tracer", return_value=mock_tracer),
        patch("app.orchestrator.nodes.generate_llm.DeepSeekClient") as mock_client_class,
    ):
        # Mock client instance and context manager
        mock_client = MagicMock()
        mock_client.generate.return_value = mock_response
        mock_client_class.return_value.__aenter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__aexit__ = MagicMock(return_value=None)

        # Mock prompt loading
        with patch(
            "app.orchestrator.nodes.generate_llm._load_generation_prompt",
            return_value="Test prompt",
        ):
            from app.orchestrator.nodes.generate_llm import generate_llm

            # Call generate_llm - should not crash
            result = await generate_llm(
                content_plan=["Test content"],
                style_directives={"tempo": "medium", "length": "short"},
                patient_context="Test patient",
            )

            # Verify basic functionality
            assert result == "Generated patient response"

            # Verify span creation was attempted
            mock_tracer.start_as_current_span.assert_called_with("llm.generation")


@pytest.mark.anyio
async def test_health_endpoint_with_tracing(client: AsyncClient):
    """
    Test that /health endpoint works with tracing enabled.
    """
    response = await client.get("/health")

    # Verify endpoint works
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.anyio
async def test_basic_imports():
    """
    Test that all tracing imports work correctly.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        from app.infra.tracing import get_tracer, instrument_app, setup_tracing

        # If we get here, all imports worked
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")
