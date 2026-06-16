"""OpenTelemetry instrumentation for the Sentry agent.

Phase 2 scope: tracer setup plus a ``with_span`` wrapper applied at graph
construction time. By default no exporter is configured, so spans are created
but discarded (near-zero cost). ``init_tracing(enable_console=True)`` adds a
stdout exporter that pretty-prints each span — useful for dev debugging.
"""

from collections.abc import Callable
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

_TRACER_NAME = "sentry"
_initialized = False


def init_tracing(*, enable_console: bool = False) -> None:
    """Configure the global OpenTelemetry tracer provider.

    Idempotent: subsequent calls are no-ops. With ``enable_console=True``,
    spans are exported to stdout for local debugging; without it spans are
    still created but not exported.
    """
    global _initialized
    if _initialized:
        return

    provider = TracerProvider()
    if enable_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


def with_span(
    name: str,
    fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Wrap a graph-node function in a span named ``name``.

    Captures the result-dict keys as the ``result.keys`` attribute so traces
    record what fields each node wrote into state. Exceptions are recorded on
    the span and re-raised; the span's status is set to ERROR so failed steps
    are obvious in any trace viewer.
    """

    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        tracer = trace.get_tracer(_TRACER_NAME)
        with tracer.start_as_current_span(name) as span:
            try:
                result = fn(*args, **kwargs)
                span.set_attribute(
                    "result.keys", ",".join(sorted(result.keys()))
                )
                return result
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(
                    trace.Status(trace.StatusCode.ERROR, str(exc))
                )
                raise

    return wrapped