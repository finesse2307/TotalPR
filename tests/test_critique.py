"""Tests for the critique node.

Covers: happy path (findings populated from a valid submit_findings call),
prompt-shape verification (diff + tool outputs + tool errors all surface),
error paths (no submit_findings, validation failure, missing diff), and the
empty-findings outcome for clean PRs.
"""

import os

import psycopg
import pytest

from sentry.embedding import DeterministicMockEmbeddingClient
from sentry.llm import LLMResponse, LLMToolCall, MockLLMClient
from sentry.memory import MemoryStore
from sentry.nodes.critique import make_critique_node
from sentry.state import (
    AgentState,
    Category,
    DiffFile,
    DiffHunk,
    ParsedDiff,
    PRMetadata,
    Severity,
    ToolName,
    ToolResult,
)


def _make_state(
    *,
    with_diff: bool = True,
    tool_results: list[ToolResult] | None = None,
) -> AgentState:
    diff = (
        ParsedDiff(
            files=[
                DiffFile(
                    path="src/users.py",
                    language="python",
                    hunks=[
                        DiffHunk(
                            header="@@ -1,2 +1,3 @@",
                            content=" class UserRepo:\n+    def get(self, uid): ...",
                        )
                    ],
                )
            ]
        )
        if with_diff
        else None
    )
    return AgentState(
        pr=PRMetadata(
            repo="acme/x",
            pr_number=1,
            head_sha="a",
            base_sha="b",
            author="alice",
            title="Add user lookup",
        ),
        raw_diff="(unused)",
        diff=diff,
        tool_results=tool_results or [],
    )


def _submit_findings_response(
    findings: list[dict[str, object]] | None = None,
) -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            LLMToolCall(
                id="call_1",
                name="submit_findings",
                arguments={"findings": findings if findings is not None else []},
            )
        ],
    )


def test_happy_path_populates_findings() -> None:
    """A valid submit_findings call produces structured Finding records."""
    mock = MockLLMClient(
        [
            _submit_findings_response(
                findings=[
                    {
                        "category": "security",
                        "severity": "high",
                        "file": "src/users.py",
                        "line": 2,
                        "message": "SQL injection via f-string; use parameterized queries.",
                    },
                    {
                        "category": "maintainability",
                        "severity": "low",
                        "file": "src/users.py",
                        "message": "Method get() is missing a docstring.",
                    },
                ]
            )
        ]
    )
    findings = make_critique_node(mock)(_make_state())["findings"]

    assert len(findings) == 2
    assert findings[0].category is Category.SECURITY
    assert findings[0].severity is Severity.HIGH
    assert findings[0].file == "src/users.py"
    assert findings[0].line == 2
    assert "SQL injection" in findings[0].message
    assert findings[1].line is None  # omitted in input, defaults to None


def test_prompt_contains_diff_and_tool_evidence_including_errors() -> None:
    """User message includes PR title, file path, tool output, and tool errors."""
    tool_results = [
        ToolResult(
            tool=ToolName.SEMGREP,
            arguments={"config": "p/security"},
            output="users.py:2: SQL injection (rule: python.lang.security.sql-injection)",
            duration_ms=42,
        ),
        ToolResult(
            tool=ToolName.RIPGREP,
            arguments={"pattern": "get"},
            output="",
            error="RuntimeError: docker daemon offline",
            duration_ms=3,
        ),
    ]
    mock = MockLLMClient([_submit_findings_response()])
    make_critique_node(mock)(_make_state(tool_results=tool_results))

    assert len(mock.calls) == 1
    messages, _system, tools, _ = mock.calls[0]
    prompt = messages[0].content

    assert "Add user lookup" in prompt
    assert "src/users.py" in prompt
    assert "SQL injection" in prompt
    assert "ERROR: RuntimeError: docker daemon offline" in prompt
    assert tools is not None
    assert [t.name for t in tools] == ["submit_findings"]


def test_no_submit_findings_raises() -> None:
    """If the LLM returns text only, the node raises."""
    mock = MockLLMClient([LLMResponse(text="I have no findings to share.")])
    node = make_critique_node(mock)

    with pytest.raises(ValueError, match="did not call submit_findings"):
        node(_make_state())


def test_invalid_findings_args_raises() -> None:
    """Bad category enum triggers validation failure."""
    bad = LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            LLMToolCall(
                id="call_1",
                name="submit_findings",
                arguments={
                    "findings": [
                        {
                            "category": "cosmic-rays",
                            "severity": "high",
                            "file": "src/users.py",
                            "message": "...",
                        }
                    ]
                },
            )
        ],
    )
    mock = MockLLMClient([bad])
    node = make_critique_node(mock)

    with pytest.raises(ValueError, match="failed validation"):
        node(_make_state())


def test_missing_diff_raises_without_calling_llm() -> None:
    """Critique refuses to run, and does not call the LLM, when diff is None."""
    mock = MockLLMClient([])
    node = make_critique_node(mock)

    with pytest.raises(ValueError, match="parse_diff"):
        node(_make_state(with_diff=False))

    assert mock.calls == []


def test_empty_findings_is_valid() -> None:
    """A clean PR produces an empty findings list, not an error."""
    mock = MockLLMClient([_submit_findings_response(findings=[])])
    findings = make_critique_node(mock)(_make_state())["findings"]

    assert findings == []

_DSN = os.environ.get(
    "POSTGRES_TEST_DSN",
    "postgresql://sentry:sentry_dev_password@localhost:5432/sentry",
)


@pytest.fixture
def memory_store() -> MemoryStore:
    """A MemoryStore against a freshly truncated memories table.

    Skips if Postgres isn't reachable so this file still passes in DB-less CI.
    """
    try:
        with psycopg.connect(_DSN, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("TRUNCATE memories RESTART IDENTITY")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable: {exc}")
    return MemoryStore(
        dsn=_DSN,
        embedder=DeterministicMockEmbeddingClient(dimension=1024),
    )


def test_memory_section_absent_when_no_matches(
    memory_store: MemoryStore,
) -> None:
    """Empty memory store → no memory section appended to prompt."""
    mock = MockLLMClient([_submit_findings_response()])
    make_critique_node(mock, memory=memory_store)(_make_state())

    prompt = mock.calls[0][0][0].content
    assert "Past Review Outcomes" not in prompt


def test_memory_section_present_when_matches_exist(
    memory_store: MemoryStore,
) -> None:
    """Stored labeled memories appear as a section in the prompt with outcomes."""
    memory_store.store(
        repo="acme/x",
        diff_text="(unused)",
        finding_text="Use parameterized queries to prevent SQL injection",
        category=Category.SECURITY,
        severity=Severity.HIGH,
        was_accepted=True,
    )
    memory_store.store(
        repo="acme/x",
        diff_text="(unused)",
        finding_text="Prefer list comprehension over explicit loops",
        category=Category.STYLE,
        severity=Severity.LOW,
        was_accepted=False,
    )

    mock = MockLLMClient([_submit_findings_response()])
    make_critique_node(mock, memory=memory_store)(_make_state())

    prompt = mock.calls[0][0][0].content
    assert "Past Review Outcomes" in prompt
    assert "[ACCEPTED]" in prompt
    assert "[REJECTED]" in prompt
    assert "parameterized queries" in prompt
    assert "list comprehension" in prompt