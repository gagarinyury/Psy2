import time
from typing import Callable
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


# Metrics
REQUEST_COUNT = Counter(
    "http_requests_total", 
    "Total HTTP requests", 
    ["method", "endpoint", "status_code"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds", 
    "HTTP request duration", 
    ["method", "endpoint"]
)

CASE_OPERATIONS = Counter(
    "case_operations_total", 
    "Total case operations", 
    ["operation"]
)

SESSION_OPERATIONS = Counter(
    "session_operations_total", 
    "Total session operations", 
    ["operation"]
)

TURN_OPERATIONS = Counter(
    "turn_operations_total", 
    "Total turn operations", 
    ["operation"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Get endpoint path template
        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)
        
        # Record metrics
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        
        return response


def get_metrics() -> Response:
    """Get Prometheus metrics"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)