"""critique node: turns the diff + tool evidence into structured findings.

Reads ``state.diff`` and ``state.tool_results``, formats both into a prompt,
and calls the LLM with a single ``submit_findings`` tool. The tool's arguments
are validated into a list of ``Finding`` records. ``make_critique_node`` injects
the LLMClient at graph-construction time, same pattern as the planner.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError

from sentry.llm import LLMClient, Message, ToolDef
from sentry.memory import Memory, MemoryStore
from sentry.state import AgentState, Finding

_SYSTEM_PROMPT = """\
You are the critique component of an automated code review system. You are given:
- The PR diff (files and hunks that changed)
- Results from review tools that were run on the diff (Ruff, Semgrep, ripgrep, docs)
- Optionally, a "Past Review Outcomes" section showing how similar past \
feedback was received by the team

Produce structured findings a human reviewer would care about. Defer to tool \
output for things tools catch (Ruff for style, Semgrep for security patterns); \
add findings tools can't catch (logic errors, design issues, missing tests, \
intent mismatches).

If a tool reported an error, do not invent findings from it; treat that tool's \
evidence as missing.

If a "Past Review Outcomes" section is present, use it to calibrate: avoid \
raising findings similar to ones the team has REJECTED, and prioritize \
patterns the team has ACCEPTED.

Call submit_findings with the list of findings. An empty list is a valid answer \
when the diff has no actionable issues.
"""

_SUBMIT_FINDINGS_TOOL = ToolDef(
    name="submit_findings",
    description=(
        "Submit the list of code review findings. "
        "This is the only tool you may call."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "security",
                                "bug",
                                "performance",
                                "style",
                                "maintainability",
                                "testing",
                                "docs",
                            ],
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                        },
                        "file": {"type": "string"},
                        "line": {"type": "integer"},
                        "message": {"type": "string"},
                    },
                    "required": ["category", "severity", "file", "message"],
                },
            },
        },
        "required": ["findings"],
    },
)


class _SubmitFindingsArgs(BaseModel):
    """Schema for parsing the LLM's submit_findings tool-call arguments."""

    findings: list[Finding] = Field(default_factory=list)


def _format_prompt(state: AgentState) -> str:
    """Render diff and tool results into a prompt-friendly string."""
    lines: list[str] = [
        f'PR #{state.pr.pr_number} in {state.pr.repo}: "{state.pr.title}"',
        "",
        "## Diff",
    ]
    if state.diff is None or not state.diff.files:
        lines.append("(no changes)")
    else:
        for f in state.diff.files:
            lines.append("")
            lines.append(f"File: {f.path} (language: {f.language or 'unknown'})")
            for i, hunk in enumerate(f.hunks, start=1):
                lines.append(f"Hunk {i}: {hunk.header}")
                lines.append(hunk.content)

    lines.append("")
    lines.append("## Tool results")
    if not state.tool_results:
        lines.append("(no tools were run)")
    else:
        for r in state.tool_results:
            lines.append("")
            lines.append(f"Tool: {r.tool.value}")
            lines.append(f"Arguments: {r.arguments}")
            if r.error:
                lines.append(f"ERROR: {r.error}")
            else:
                lines.append("Output:")
                lines.append(r.output)

    return "\n".join(lines)

def _format_memory_section(memories: list[Memory]) -> str:
    """Render retrieved memories as a markdown prompt section.

    Outcomes are surfaced explicitly so the LLM can weight them: ACCEPTED
    findings are signal to repeat; REJECTED findings are signal to suppress.
    """
    lines = [
        "## Past Review Outcomes",
        "",
        (
            "Similar past feedback from this repository, with team outcomes. "
            "Avoid raising findings similar to REJECTED ones; prioritize "
            "patterns similar to ACCEPTED ones."
        ),
        "",
    ]
    for i, m in enumerate(memories, start=1):
        outcome = "ACCEPTED" if m.was_accepted else "REJECTED"
        lines.append(
            f"{i}. [{outcome}] [{m.category.value}/{m.severity.value}] "
            f"{m.finding_text}"
        )
    return "\n".join(lines)

def make_critique_node(
    llm: LLMClient,
    *,
    memory: MemoryStore | None = None,
) -> Callable[[AgentState], dict[str, list[Finding]]]:
    """Build a critique-node bound to a specific LLMClient and optional memory."""

    def critique_node(state: AgentState) -> dict[str, list[Finding]]:
        if state.diff is None:
            raise ValueError(
                "critique_node requires state.diff to be populated; "
                "did parse_diff run?"
            )

        prompt = _format_prompt(state)

        if memory is not None:
            similar = memory.retrieve_similar(
                diff_text=state.raw_diff,
                repo=state.pr.repo,
                k=3,
                only_labeled=True,
            )
            if similar:
                prompt = _format_memory_section(similar) + "\n\n" + prompt

        response = llm.complete(
            messages=[Message(role="user", content=prompt)],
            system=_SYSTEM_PROMPT,
            tools=[_SUBMIT_FINDINGS_TOOL],
        )

        submit_calls = [
            tc for tc in response.tool_calls if tc.name == "submit_findings"
        ]
        if not submit_calls:
            raise ValueError(
                "critique LLM did not call submit_findings; got "
                f"stop_reason={response.stop_reason!r}, "
                f"tool_calls={[tc.name for tc in response.tool_calls]}"
            )

        try:
            args = _SubmitFindingsArgs.model_validate(submit_calls[0].arguments)
        except ValidationError as exc:
            raise ValueError(
                f"submit_findings arguments failed validation: {exc}"
            ) from exc

        return {"findings": args.findings}

    return critique_node