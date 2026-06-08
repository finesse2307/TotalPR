"""format_review node: render structured findings as a markdown review body.

Deterministic, no LLM call. Produces the markdown ``post_comment`` will post to
GitHub. Findings are grouped by severity (high → medium → low) and sorted by
file/line within each group for stable, diff-friendly output across runs.
"""

from sentry.state import AgentState, Finding, Severity

_SEVERITY_ORDER: list[Severity] = [Severity.HIGH, Severity.MEDIUM, Severity.LOW]
_SEVERITY_HEADING = {
    Severity.HIGH: "High severity",
    Severity.MEDIUM: "Medium severity",
    Severity.LOW: "Low severity",
}


def _finding_sort_key(f: Finding) -> tuple[str, int]:
    """Sort key within a severity group: file, then line (None sorts last)."""
    return (f.file, f.line if f.line is not None else 1_000_000)


def _format_finding(f: Finding) -> str:
    location = f"`{f.file}:{f.line}`" if f.line is not None else f"`{f.file}`"
    return f"- **[{f.category.value}]** {location} — {f.message}"


def _summary_line(findings: list[Finding]) -> str:
    counts = {s: 0 for s in _SEVERITY_ORDER}
    for f in findings:
        counts[f.severity] += 1
    parts = [f"{counts[s]} {s.value}" for s in _SEVERITY_ORDER if counts[s]]
    total = len(findings)
    plural = "s" if total != 1 else ""
    return f"Found {total} finding{plural} ({', '.join(parts)})."


def format_review(state: AgentState) -> dict[str, str]:
    """Render the structured findings into a markdown review body."""
    if not state.findings:
        return {
            "review_body": (
                "## Sentry Code Review\n\nNo findings. The diff looks clean."
            )
        }

    sections: list[str] = [
        "## Sentry Code Review",
        "",
        _summary_line(state.findings),
    ]

    for severity in _SEVERITY_ORDER:
        group = sorted(
            (f for f in state.findings if f.severity is severity),
            key=_finding_sort_key,
        )
        if not group:
            continue
        sections.append("")
        sections.append(f"### {_SEVERITY_HEADING[severity]}")
        sections.append("")
        for f in group:
            sections.append(_format_finding(f))

    return {"review_body": "\n".join(sections)}