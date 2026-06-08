"""plan node: asks the LLM to choose which review tools to run for this PR.

The planner formats the parsed diff into a prompt and calls the LLM with a
single ``submit_plan`` tool. The model's tool-call arguments are validated into
a ``Plan`` that downstream nodes execute. The ``make_plan_node`` factory injects
the LLMClient so the same node works with the real client or the mock.
"""

from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError

from sentry.llm import LLMClient, Message, ToolDef
from sentry.state import AgentState, Plan, ToolCall

_SYSTEM_PROMPT = """\
You are the planning component of an automated code review system. Given a PR \
diff, decide which review tools to run.

Available review tools (reference them in submit_plan; do not call them directly):
- ruff: Python linter. Fast and deterministic. Use for Python files when the \
diff is likely to contain mechanical style or simple-bug issues.
- semgrep: Security and pattern static analyzer. Use when the diff touches \
code paths with potential security implications (auth, queries, \
deserialization, secrets).
- ripgrep: Codebase search. Use to check whether a changed symbol is used \
elsewhere, confirm a constant existed, or find similar patterns.
- docs_lookup: External library/API documentation. Use to verify claims about \
unfamiliar library behavior or API contracts.

Call submit_plan with your reasoning and an ordered list of tool invocations. \
Prefer fewer, well-justified calls over many.
"""

_SUBMIT_PLAN_TOOL = ToolDef(
    name="submit_plan",
    description=(
        "Submit the review plan: reasoning plus an ordered list of tool calls. "
        "This is the only tool you may call."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Short explanation of why these tools, in this order.",
            },
            "calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "enum": ["ruff", "semgrep", "ripgrep", "docs_lookup"],
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Tool-specific arguments as a string map.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Why this specific call.",
                        },
                    },
                    "required": ["tool", "rationale"],
                },
            },
        },
        "required": ["reasoning", "calls"],
    },
)


class _SubmitPlanArgs(BaseModel):
    """Schema for parsing the LLM's submit_plan tool-call arguments."""

    reasoning: str
    calls: list[ToolCall] = Field(default_factory=list)


def _format_diff_for_prompt(state: AgentState) -> str:
    """Render the parsed diff into a prompt-friendly string."""
    if state.diff is None or not state.diff.files:
        return "(no changes)"

    lines: list[str] = [
        f'PR #{state.pr.pr_number} in {state.pr.repo}: "{state.pr.title}"',
        "",
        "Changed files:",
    ]
    for f in state.diff.files:
        lines.append("")
        lines.append(f"File: {f.path} (language: {f.language or 'unknown'})")
        for i, hunk in enumerate(f.hunks, start=1):
            lines.append(f"Hunk {i}: {hunk.header}")
            lines.append(hunk.content)
    return "\n".join(lines)


def make_plan_node(
    llm: LLMClient,
) -> Callable[[AgentState], dict[str, Plan]]:
    """Build a plan-node bound to a specific LLMClient.

    The returned function is the actual LangGraph node:
    ``(state) -> {"plan": Plan}``. Injecting the client at graph-construction
    time lets us swap the mock for the real client without touching the node.
    """

    def plan_node(state: AgentState) -> dict[str, Plan]:
        if state.diff is None:
            raise ValueError(
                "plan_node requires state.diff to be populated; "
                "did parse_diff run?"
            )

        response = llm.complete(
            messages=[
                Message(role="user", content=_format_diff_for_prompt(state)),
            ],
            system=_SYSTEM_PROMPT,
            tools=[_SUBMIT_PLAN_TOOL],
        )

        submit_calls = [
            tc for tc in response.tool_calls if tc.name == "submit_plan"
        ]
        if not submit_calls:
            raise ValueError(
                "planner LLM did not call submit_plan; got "
                f"stop_reason={response.stop_reason!r}, "
                f"tool_calls={[tc.name for tc in response.tool_calls]}"
            )

        try:
            args = _SubmitPlanArgs.model_validate(submit_calls[0].arguments)
        except ValidationError as exc:
            raise ValueError(
                f"submit_plan arguments failed validation: {exc}"
            ) from exc

        return {"plan": Plan(reasoning=args.reasoning, calls=args.calls)}

    return plan_node