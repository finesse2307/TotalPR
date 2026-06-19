"""Tests for the telemetry module.
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from sentry.telemetry import init_tracing, with_span


def test_with_span_returns_function_result() -> None:
    """The wrapper returns whatever the wrapped function returned."""

    def f(x: int) -> dict[str, int]:
        return {"doubled": x * 2}

    wrapped = with_span("test", f)

    assert wrapped(5) == {"doubled": 10}


def test_with_span_propagates_exceptions() -> None:
    """Exceptions raised inside the wrapped function are re-raised."""

    def f() -> dict[str, int]:
        raise RuntimeError("inside the span")

    wrapped = with_span("test", f)

    with pytest.raises(RuntimeError, match="inside the span"):
        wrapped()


def test_init_tracing_is_idempotent() -> None:
    """Calling init_tracing multiple times does not raise or duplicate setup."""
    init_tracing()
    init_tracing(enable_console=True)
    init_tracing()
    # No exception = pass.


def test_with_span_records_result_keys_attribute() -> None:
    """``result.keys`` span attribute lists the sorted keys of the result dict."""
    init_tracing()
    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    def f() -> dict[str, int]:
        return {"beta": 2, "alpha": 1}

    with_span("my_op", f)()

    finished = exporter.get_finished_spans()
    my_op = next(s for s in finished if s.name == "my_op")
    assert my_op.attributes is not None
    assert my_op.attributes.get("result.keys") == "alpha,beta"

def test_run_span_provides_active_recording_span() -> None:
    """Inside a run_span block, the current span is a real recording span."""
    from opentelemetry import trace

    from sentry.telemetry import init_tracing, run_span

    init_tracing()
    with run_span("test-run"):
        ctx = trace.get_current_span().get_span_context()
        assert ctx.span_id != 0  # the default no-op span has span_id == 0
        assert ctx.trace_id != 0