"""
OpenTelemetry tracing setup for RAG Patient API.

Provides tracing for FastAPI endpoints, HTTP clients, and custom spans
for pipeline operations and LLM calls.
"""

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.core.settings import settings

logger = logging.getLogger(__name__)


def setup_tracing(service_name: str = "rag-patient") -> None:
    """
    Setup OpenTelemetry tracing with OTLP gRPC exporter or console fallback.

    Configures:
    - TracerProvider with service name resource
    - BatchSpanProcessor for async span export
    - OTLP gRPC exporter if OTEL_EXPORTER_OTLP_ENDPOINT is set
    - ConsoleSpanExporter as fallback for dev/test
    - FastAPI instrumentation for automatic /turn tracing
    - HTTPX instrumentation for DeepSeek API calls

    Args:
        service_name: Service identifier for traces
    """
    # Configure resource with service name
    resource = Resource(attributes={
        SERVICE_NAME: service_name
    })

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(tracer_provider)

    # Choose exporter based on configuration
    if settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        # Use OTLP gRPC exporter for production
        exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True  # Use insecure for local development, configure TLS for production
        )
        logger.info(f"OpenTelemetry: Using OTLP exporter to {settings.OTEL_EXPORTER_OTLP_ENDPOINT}")
    else:
        # Use console exporter for development/testing
        exporter = ConsoleSpanExporter()
        logger.info("OpenTelemetry: Using console exporter")

    # Add batch span processor for async processing
    span_processor = BatchSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)

    # Instrument HTTP clients for automatic DeepSeek API tracing
    HTTPXClientInstrumentor().instrument()
    logger.info("OpenTelemetry: HTTPX instrumentation enabled")


def instrument_app(app) -> None:
    """
    Instrument FastAPI application for automatic endpoint tracing.

    Args:
        app: FastAPI application instance
    """
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry: FastAPI instrumentation enabled")


def get_tracer(name: str) -> trace.Tracer:
    """
    Get a tracer instance for creating custom spans.

    Args:
        name: Tracer name (typically module name)

    Returns:
        Tracer instance for span creation
    """
    return trace.get_tracer(name)