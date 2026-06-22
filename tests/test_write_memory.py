"""Tests for the write_memory node.

Mocks the MemoryStore so tests don't depend on Postgres. Real-database
behavior of store() is covered in tests/test_memory.py.
"""

from unittest.mock import MagicMock

from sentry.nodes.write_memory import make_write_memory_node
from sentry.state import (
    AgentState,
    Category,
    Finding,
    PRMetadata,
    Severity,
)


def _make_state(findings: list[Finding] | None = None) -> AgentState:
    return AgentState(
        pr=PRMetadata(
            repo="acme/api",
            pr_number=42,
            head_sha="a",
            base_sha="b",
            author="alice",
            title="Add user lookup",
        ),
        raw_diff="diff content",
        findings=findings or [],
    )


def test_no_memory_is_no_op() -> None:
    """make_write_memory_node(None) returns a no-op even with findings present."""
    node = make_write_memory_node(None)
    result = node(
        _make_state(
            findings=[
                Finding(
                    category=Category.BUG,
                    severity=Severity.LOW,
                    file="x.py",
                    message="msg",
                ),
            ]
        )
    )
    assert result == {}


def test_all_findings_written_in_one_store_findings_call() -> None:
    """write_memory routes every finding through a single store_findings call."""
    memory = MagicMock()
    memory.store_findings.return_value = [101, 102]
    node = make_write_memory_node(memory)
    findings = [
        Finding(
            category=Category.SECURITY, severity=Severity.HIGH,
            file="a.py", message="first",
        ),
        Finding(
            category=Category.BUG, severity=Severity.MEDIUM,
            file="b.py", message="second",
        ),
    ]
    node(_make_state(findings=findings))

    memory.store_findings.assert_called_once()
    kwargs = memory.store_findings.call_args.kwargs
    assert kwargs["repo"] == "acme/api"
    assert kwargs["diff_text"] == "diff content"
    assert kwargs["findings"] == findings
    assert kwargs["was_accepted"] is None


def test_empty_findings_skips_store_findings() -> None:
    """A clean PR (no findings) means no DB writes at all."""
    memory = MagicMock()
    node = make_write_memory_node(memory)
    result = node(_make_state(findings=[]))

    assert result == {}
    memory.store_findings.assert_not_called()