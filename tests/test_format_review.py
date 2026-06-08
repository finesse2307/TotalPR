"""Tests for the format_review node.

Assertions are by substring and relative ordering — that captures the rendering
contract without coupling to incidental whitespace or wording.
"""

from sentry.nodes.format_review import format_review
from sentry.state import (
    AgentState,
    Category,
    Finding,
    PRMetadata,
    Severity,
)


def _make_state(findings: list[Finding]) -> AgentState:
    return AgentState(
        pr=PRMetadata(
            repo="acme/x",
            pr_number=1,
            head_sha="a",
            base_sha="b",
            author="alice",
            title="t",
        ),
        raw_diff="(unused)",
        findings=findings,
    )


def _finding(
    category: Category = Category.BUG,
    severity: Severity = Severity.MEDIUM,
    file: str = "src/x.py",
    line: int | None = None,
    message: str = "msg",
) -> Finding:
    return Finding(
        category=category,
        severity=severity,
        file=file,
        line=line,
        message=message,
    )


def test_no_findings_returns_clean_message() -> None:
    body = format_review(_make_state([]))["review_body"]

    assert body.startswith("## Sentry Code Review")
    assert "No findings" in body
    assert "###" not in body  # no severity sections rendered


def test_single_finding_renders_category_file_line_and_message() -> None:
    body = format_review(
        _make_state(
            [
                _finding(
                    category=Category.SECURITY,
                    severity=Severity.HIGH,
                    file="src/users.py",
                    line=12,
                    message="SQL injection",
                )
            ]
        )
    )["review_body"]

    assert "### High severity" in body
    assert "**[security]**" in body
    assert "`src/users.py:12`" in body
    assert "SQL injection" in body


def test_finding_without_line_omits_line_suffix() -> None:
    body = format_review(
        _make_state(
            [_finding(file="src/users.py", line=None, message="Missing docstring.")]
        )
    )["review_body"]

    assert "`src/users.py`" in body
    assert "src/users.py:" not in body  # no colon-N suffix


def test_severity_sections_in_high_medium_low_order() -> None:
    findings = [
        _finding(severity=Severity.LOW, message="low one"),
        _finding(severity=Severity.HIGH, message="high one"),
        _finding(severity=Severity.MEDIUM, message="medium one"),
    ]
    body = format_review(_make_state(findings))["review_body"]

    pos_high = body.index("### High severity")
    pos_medium = body.index("### Medium severity")
    pos_low = body.index("### Low severity")
    assert pos_high < pos_medium < pos_low


def test_findings_sorted_by_file_then_line_within_severity() -> None:
    findings = [
        _finding(severity=Severity.HIGH, file="src/b.py", line=10, message="b10"),
        _finding(severity=Severity.HIGH, file="src/a.py", line=20, message="a20"),
        _finding(severity=Severity.HIGH, file="src/a.py", line=5, message="a5"),
    ]
    body = format_review(_make_state(findings))["review_body"]

    pos_a5 = body.index("a5")
    pos_a20 = body.index("a20")
    pos_b10 = body.index("b10")
    assert pos_a5 < pos_a20 < pos_b10


def test_summary_line_pluralization_and_counts() -> None:
    one = format_review(_make_state([_finding(severity=Severity.HIGH)]))["review_body"]
    many = format_review(
        _make_state(
            [
                _finding(severity=Severity.HIGH),
                _finding(severity=Severity.HIGH),
                _finding(severity=Severity.LOW),
            ]
        )
    )["review_body"]

    assert "Found 1 finding (1 high)" in one  # singular
    assert "Found 3 findings (2 high, 1 low)" in many  # plural; no medium listed