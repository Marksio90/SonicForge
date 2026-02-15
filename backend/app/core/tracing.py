"""Distributed Tracing with OpenTelemetry and Jaeger.

Provides:
- End-to-end request tracing
- Performance bottleneck identification
- Service dependency mapping
- Error tracking across services
"""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import structlog

from ..core.config import get_settings

settings = get_settings()
logger = structlog.get_logger(__name__)


def setup_tracing():
    """Initialize distributed tracing."""
    # Create resource
    resource = Resource.create({
        "service.name": "sonicforge-api",
        "service.version": settings.app_version,
        "deployment.environment": settings.environment.value,
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure Jaeger exporter
    jaeger_host = getattr(settings, 'jaeger_host', 'localhost')
    jaeger_port = getattr(settings, 'jaeger_port', 6831)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name=jaeger_host,
        agent_port=jaeger_port,
    )
    
    # Add span processor
    processor = BatchSpanProcessor(jaeger_exporter)
    provider.add_span_processor(processor)
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Auto-instrument libraries
    HTTPXClientInstrumentor().instrument()
    RedisInstrumentor().instrument()
    
    logger.info(
        "tracing_initialized",
        jaeger_host=jaeger_host,
        jaeger_port=jaeger_port,
    )


def instrument_fastapi(app):
    """Instrument FastAPI application."""
    FastAPIInstrumentor.instrument_app(app)
    logger.info("fastapi_instrumented")


def get_tracer(name: str):
    """Get tracer instance."""
    return trace.get_tracer(name)


class TracingMiddleware:
    """Custom tracing middleware for additional context."""
    
    def __init__(self, tracer_name: str = "sonicforge"):
        self.tracer = get_tracer(tracer_name)
    
    def trace_function(self, name: str):
        """Decorator to trace a function.
        
        Usage:
            @tracing.trace_function("generate_track")
            async def generate_track(concept):
                ...
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(name) as span:
                    # Add custom attributes
                    span.set_attribute("function.name", func.__name__)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))
                        raise
            return wrapper
        return decorator


# Global tracing instance
tracing = TracingMiddleware()
