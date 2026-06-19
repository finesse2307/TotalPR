"""OpenTelemetry instrumentation for the Sentry agent.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

_TRACER_NAME = "sentry"
_DEFAULT_SERVICE_NAME = "sentry"
_initialized = False


def init_tracing(
    *,
    enable_console: bool = False,
    service_name: str = _DEFAULT_SERVICE_NAME,
) -> None:
    """Configure the global OpenTelemetry tracer provider.

    Idempotent: subsequent calls are no-ops. With ``enable_console=True``,
    spans are exported to stdout for local debugging. ``service_name`` is
    attached to every span via the OTel Resource so traces are correctly
    attributed in any multi-service backend.
    """
    global _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if enable_console:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True


@contextmanager
def run_span(name: str = "review_pr") -> Iterator[None]:
    """Wrap one agent invocation in an outer span.

    Used by callers (smoke runner, eval harness, webhook handler) to surround
    ``graph.invoke(...)`` so that node-level spans created via ``with_span``
    nest under this one and the run appears as a single trace tree.
    """
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name):
        yield


def with_span(
    name: str,
    fn: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    """Wrap a graph-node function in a span named ``name``.

    Captures the result-dict keys as the ``result.keys`` attribute so traces
    record what fields each node wrote into state. Exceptions are recorded
    on the span and re-raised; the span's status is set to ERROR.
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