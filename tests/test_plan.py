"""Tests for the plan node.

Covers: happy path (plan populated from a valid submit_plan call), the LLM-call
shape (system prompt, message, tools), and error paths (no submit_plan, bad
args, missing diff). Empty plan is treated as a valid outcome.
"""

import pytest

from sentry.llm import LLMResponse, LLMToolCall, MockLLMClient
from sentry.nodes.plan import make_plan_node
from sentry.state import (
    AgentState,
    DiffFile,
    DiffHunk,
    ParsedDiff,
    PRMetadata,
    ToolName,
)


def _make_state(*, with_diff: bool = True) -> AgentState:
    diff = (
        ParsedDiff(
            files=[
                DiffFile(
                    path="src/users.py",
                    language="python",
                    hunks=[
                        DiffHunk(
                            header="@@ -1,2 +1,3 @@",
                            content=" class UserRepo:\n+    def get(...): ...",
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
        raw_diff="(unused; diff is set directly)",
        diff=diff,
    )


def _submit_plan_response(
    reasoning: str = "default reasoning",
    calls: list[dict[str, object]] | None = None,
) -> LLMResponse:
    return LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            LLMToolCall(
                id="call_1",
                name="submit_plan",
                arguments={
                    "reasoning": reasoning,
                    "calls": calls if calls is not None else [],
                },
            )
        ],
    )


def test_happy_path_populates_plan() -> None:
    """A valid submit_plan call produces a fully-populated Plan."""
    mock = MockLLMClient(
        [
            _submit_plan_response(
                reasoning="SQL injection pattern; semgrep first.",
                calls=[
                    {
                        "tool": "semgrep",
                        "arguments": {"config": "p/security-audit"},
                        "rationale": "Check for SQL injection.",
                    },
                    {
                        "tool": "ripgrep",
                        "arguments": {"pattern": "get_user"},
                        "rationale": "Find other callers.",
                    },
                ],
            )
        ]
    )
    node = make_plan_node(mock)
    plan = node(_make_state())["plan"]

    assert plan.reasoning.startswith("SQL injection")
    assert len(plan.calls) == 2
    assert plan.calls[0].tool is ToolName.SEMGREP
    assert plan.calls[0].arguments == {"config": "p/security-audit"}
    assert plan.calls[1].tool is ToolName.RIPGREP
    assert plan.calls[1].rationale == "Find other callers."


def test_planner_sends_correct_system_and_tools() -> None:
    """The planner sends the system prompt and exposes only submit_plan."""
    mock = MockLLMClient([_submit_plan_response()])
    make_plan_node(mock)(_make_state())

    assert len(mock.calls) == 1
    messages, system, tools, _ = mock.calls[0]
    assert system is not None
    assert "submit_plan" in system
    assert tools is not None
    assert [t.name for t in tools] == ["submit_plan"]
    assert "Add user lookup" in messages[0].content
    assert "src/users.py" in messages[0].content


def test_no_submit_plan_call_raises() -> None:
    """If the LLM returns text only (no submit_plan), the node raises."""
    mock = MockLLMClient([LLMResponse(text="I won't plan today.")])
    node = make_plan_node(mock)

    with pytest.raises(ValueError, match="did not call submit_plan"):
        node(_make_state())


def test_invalid_submit_plan_args_raises() -> None:
    """Bad enum value in the plan triggers a validation failure."""
    bad_response = LLMResponse(
        stop_reason="tool_use",
        tool_calls=[
            LLMToolCall(
                id="call_1",
                name="submit_plan",
                arguments={
                    "reasoning": "ok",
                    "calls": [
                        {
                            "tool": "not-a-real-tool",
                            "arguments": {},
                            "rationale": "n/a",
                        }
                    ],
                },
            )
        ],
    )
    mock = MockLLMClient([bad_response])
    node = make_plan_node(mock)

    with pytest.raises(ValueError, match="failed validation"):
        node(_make_state())


def test_missing_diff_raises_without_calling_llm() -> None:
    """The planner refuses to run, and does not call the LLM, when diff is None."""
    mock = MockLLMClient([])  # zero scripted responses; any call would explode
    node = make_plan_node(mock)

    with pytest.raises(ValueError, match="parse_diff"):
        node(_make_state(with_diff=False))

    assert mock.calls == []


def test_empty_calls_is_valid() -> None:
    """A plan with zero tool calls is valid (planner judged no tools needed)."""
    mock = MockLLMClient(
        [_submit_plan_response(reasoning="No security or perf concerns; skip tools.")]
    )
    node = make_plan_node(mock)
    plan = node(_make_state())["plan"]

    assert plan.calls == []
    assert "skip tools" in plan.reasoning