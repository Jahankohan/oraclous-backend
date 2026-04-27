"""
OpenTelemetry instrumentation for Oraclous Knowledge Graph Builder.

Opt-in via env var: OTEL_ENABLED=true
Exporter endpoint: OTEL_EXPORTER_OTLP_ENDPOINT (default: http://jaeger:4317)
Protocol: OTEL_EXPORTER_OTLP_PROTOCOL ("grpc" or "http/protobuf")

Compatible with Jaeger (via OTLP) and Grafana Tempo.
"""

from __future__ import annotations

import json

from neo4j.exceptions import ServiceUnavailable
from opentelemetry import metrics, trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Module-level providers — set once at startup
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def _build_resource() -> Resource:
    return Resource.create(
        {
            SERVICE_NAME: settings.OTEL_SERVICE_NAME,
            SERVICE_VERSION: settings.OTEL_SERVICE_VERSION,
            "deployment.environment": "production",
        }
    )


def _build_span_exporter():
    """Return the appropriate span exporter based on protocol setting."""
    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    protocol = settings.OTEL_EXPORTER_OTLP_PROTOCOL

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=endpoint, insecure=True)
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        return OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")


def _build_metric_exporter():
    """Return the appropriate metric exporter based on protocol setting."""
    endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT
    protocol = settings.OTEL_EXPORTER_OTLP_PROTOCOL

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        return OTLPMetricExporter(endpoint=endpoint, insecure=True)
    else:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )

        return OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics")


def setup_telemetry() -> None:
    """
    Initialize OpenTelemetry tracing and metrics.

    No-op if OTEL_ENABLED=false (default).
    Call once at application startup (lifespan).
    """
    global _tracer_provider, _meter_provider

    if not settings.OTEL_ENABLED:
        logger.info("OpenTelemetry disabled (OTEL_ENABLED=false)")
        return

    resource = _build_resource()

    # ── Tracing ────────────────────────────────────────────────────────────
    try:
        span_exporter = _build_span_exporter()
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(_tracer_provider)
        set_global_textmap(TraceContextTextMapPropagator())
        logger.info(
            "OTel tracing initialised",
            extra={"endpoint": settings.OTEL_EXPORTER_OTLP_ENDPOINT},
        )
    except Exception as exc:
        logger.warning(f"OTel tracing setup failed — falling back to console: {exc}")
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_tracer_provider)

    # ── Metrics ────────────────────────────────────────────────────────────
    try:
        metric_exporter = _build_metric_exporter()
        reader = PeriodicExportingMetricReader(
            metric_exporter, export_interval_millis=30_000
        )
        _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(_meter_provider)
        logger.info("OTel metrics initialised")
    except Exception as exc:
        logger.warning(f"OTel metrics setup failed — falling back to console: {exc}")
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        _meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(_meter_provider)

    _register_custom_metrics()


def _register_custom_metrics() -> None:
    """Register Oraclous-specific application metrics."""
    meter = metrics.get_meter(settings.OTEL_SERVICE_NAME, settings.OTEL_SERVICE_VERSION)

    # Ingestion metrics
    meter.create_counter(
        name="oraclous.ingestion.jobs.total",
        description="Total ingestion jobs by status",
        unit="1",
    )
    meter.create_histogram(
        name="oraclous.ingestion.duration.seconds",
        description="Ingestion job duration in seconds",
        unit="s",
    )
    meter.create_counter(
        name="oraclous.ingestion.entities.extracted.total",
        description="Total entities extracted per graph",
        unit="1",
    )

    # Chat metrics
    meter.create_counter(
        name="oraclous.chat.queries.total",
        description="Total chat queries by retriever type",
        unit="1",
    )
    meter.create_histogram(
        name="oraclous.chat.response.duration.seconds",
        description="Chat response duration including LLM call",
        unit="s",
    )
    meter.create_counter(
        name="oraclous.chat.llm.tokens.total",
        description="Total LLM tokens consumed",
        unit="1",
    )

    # Graph metrics
    meter.create_up_down_counter(
        name="oraclous.graph.nodes.total",
        description="Current node count per graph",
        unit="1",
    )
    meter.create_up_down_counter(
        name="oraclous.graph.edges.total",
        description="Current edge count per graph",
        unit="1",
    )

    # ── Redis pool metrics ─────────────────────────────────────────────────────
    # Observable gauges read the live Redis connection pool state on each
    # metric collection cycle (every 30 s via PeriodicExportingMetricReader).
    # Import is deferred to avoid circular imports at module load time.

    def _redis_active_connections_callback(options):  # noqa: ANN001
        try:
            from app.core.dependencies import redis_client  # type: ignore[import]

            pool = getattr(redis_client, "connection_pool", None)
            if pool is None:
                return []
            active = getattr(pool, "_created_connections", 0) - len(
                getattr(pool, "_available_connections", [])
            )
            from opentelemetry.sdk.metrics.export import NumberDataPoint  # noqa: F401
            from opentelemetry.metrics import Observation

            return [Observation(active)]
        except Exception:
            from opentelemetry.metrics import Observation

            return [Observation(0)]

    def _redis_max_connections_callback(options):  # noqa: ANN001
        try:
            from app.core.dependencies import redis_client  # type: ignore[import]

            pool = getattr(redis_client, "connection_pool", None)
            if pool is None:
                return []
            max_conn = getattr(pool, "max_connections", 0) or 0
            from opentelemetry.metrics import Observation

            return [Observation(max_conn)]
        except Exception:
            from opentelemetry.metrics import Observation

            return [Observation(0)]

    meter.create_observable_gauge(
        name="redis.pool.active_connections",
        description="Current number of active Redis connections in the pool",
        unit="1",
        callbacks=[_redis_active_connections_callback],
    )
    meter.create_observable_gauge(
        name="redis.pool.max_connections",
        description="Maximum allowed Redis connections in the pool",
        unit="1",
        callbacks=[_redis_max_connections_callback],
    )

    # ── Celery queue depth metric ──────────────────────────────────────────────
    # Sampled via Celery inspect on each collection cycle.

    def _celery_queue_depth_callback(options):  # noqa: ANN001
        try:
            from app.services.background_jobs import celery_app  # type: ignore[import]
            from opentelemetry.metrics import Observation

            inspect = celery_app.control.inspect(timeout=1)
            active = inspect.active() or {}
            reserved = inspect.reserved() or {}
            total = sum(len(v) for v in active.values()) + sum(
                len(v) for v in reserved.values()
            )
            return [Observation(total)]
        except Exception:
            from opentelemetry.metrics import Observation

            return [Observation(0)]

    meter.create_observable_gauge(
        name="celery.queue.depth",
        description="Current number of active + reserved Celery tasks across all workers",
        unit="1",
        callbacks=[_celery_queue_depth_callback],
    )


def shutdown_telemetry() -> None:
    """Flush and shut down providers gracefully. Call in app lifespan shutdown."""
    if _tracer_provider:
        _tracer_provider.shutdown()
    if _meter_provider:
        _meter_provider.shutdown()
    logger.info("OpenTelemetry shutdown complete")


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer. Works whether OTel is enabled or not (returns noop if disabled)."""
    return trace.get_tracer(name, settings.OTEL_SERVICE_VERSION)


def get_meter(name: str) -> metrics.Meter:
    """Return a named meter. Works whether OTel is enabled or not (returns noop if disabled)."""
    return metrics.get_meter(name, settings.OTEL_SERVICE_VERSION)


class Neo4jDegradationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that converts neo4j.exceptions.ServiceUnavailable into a
    structured HTTP 503 response with a Retry-After header and KGB-5001 error code.

    This prevents unhandled ServiceUnavailable exceptions from propagating as
    generic 500 errors to API clients.
    """

    _RETRY_AFTER_SECONDS = 30

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except ServiceUnavailable as exc:
            logger.critical(f"Neo4j unavailable (middleware caught): {exc}")
            from app.core.errors import KGBError

            code, message = KGBError.NEO4J_UNAVAILABLE
            body = json.dumps(
                {
                    "error_code": code,
                    "message": message,
                    "retry_after": self._RETRY_AFTER_SECONDS,
                }
            )
            return Response(
                content=body,
                status_code=503,
                headers={
                    "Content-Type": "application/json",
                    "Retry-After": str(self._RETRY_AFTER_SECONDS),
                },
            )


def attach_degradation_middleware(app) -> None:
    """
    Attach the Neo4j degradation middleware to a FastAPI/Starlette app.

    Must be called AFTER the app is fully configured so the middleware wraps all routes.
    This is always attached regardless of OTEL_ENABLED because it provides
    graceful error handling independent of observability.
    """
    app.add_middleware(Neo4jDegradationMiddleware)
    logger.info("Neo4j degradation middleware attached")


def instrument_fastapi(app) -> None:
    """
    Attach FastAPI auto-instrumentation and degradation middleware.

    Records:
    - HTTP request spans (endpoint, method, status_code)
    - Request latency histogram
    - 4xx / 5xx error spans with exception events

    Also attaches Neo4jDegradationMiddleware unconditionally (OTEL_ENABLED
    controls only the tracing/metrics instrumentation, not error handling).
    """
    # Always attach degradation middleware — it is not an OTEL concern.
    attach_degradation_middleware(app)

    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI OTel instrumentation attached")
    except Exception as exc:
        logger.warning(f"FastAPI instrumentation failed: {exc}")


def instrument_celery() -> None:
    """
    Attach Celery auto-instrumentation.

    Records:
    - Task span per Celery task (task name, queue, retries)
    - Task duration histogram
    - Failure spans with exception events

    No-op if OTEL_ENABLED=false.
    Call this once in the Celery app startup (after celery app creation).
    """
    if not settings.OTEL_ENABLED:
        return
    try:
        from opentelemetry.instrumentation.celery import CeleryInstrumentor

        CeleryInstrumentor().instrument()
        logger.info("Celery OTel instrumentation attached")
    except Exception as exc:
        logger.warning(f"Celery instrumentation failed: {exc}")


def current_trace_context() -> dict:
    """
    Return trace_id and span_id for the active span, suitable for log injection.

    Returns empty dict if no active span or OTel is disabled.
    """
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if not ctx.is_valid:
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }
