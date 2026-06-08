"""State schema for the Sentry code-review agent.

The state object is the single source of truth that flows through the LangGraph
state machine. Each node reads relevant fields, does its work, and returns an
update; LangGraph merges updates into the running state.
"""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field


class Category(StrEnum):
    """Vocabulary for finding categories. Keep this small and stable."""

    SECURITY = "security"
    BUG = "bug"
    PERFORMANCE = "performance"
    STYLE = "style"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    DOCS = "docs"


class Severity(StrEnum):
    """Vocabulary for finding severity. Keep this small and stable."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ToolName(StrEnum):
    """The four sandboxed tools the agent can call."""

    RUFF = "ruff"
    SEMGREP = "semgrep"
    RIPGREP = "ripgrep"
    DOCS_LOOKUP = "docs_lookup"


class PostStatus(StrEnum):
    """Outcome of posting the review back to GitHub."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class PRMetadata(BaseModel):
    """Identifying info about the PR under review. Set once at the start of a run."""

    repo: str
    pr_number: int
    head_sha: str
    base_sha: str
    author: str
    title: str


class DiffHunk(BaseModel):
    """A single hunk within a changed file."""

    header: str
    content: str


class DiffFile(BaseModel):
    """A single changed file in the PR."""

    path: str
    language: str | None = None
    hunks: list[DiffHunk] = Field(default_factory=list)


class ParsedDiff(BaseModel):
    """The PR diff after parsing into structured form."""

    files: list[DiffFile] = Field(default_factory=list)


class ToolCall(BaseModel):
    """One planned tool invocation: which tool, with what args, and why."""

    tool: ToolName
    arguments: dict[str, str] = Field(default_factory=dict)
    rationale: str


class Plan(BaseModel):
    """The agent's plan: ordered tool calls plus the planner's reasoning."""

    reasoning: str
    calls: list[ToolCall] = Field(default_factory=list)


class ToolResult(BaseModel):
    """The outcome of one executed tool call."""

    tool: ToolName
    arguments: dict[str, str]
    output: str
    error: str | None = None
    duration_ms: int


class Finding(BaseModel):
    """A single review finding produced by the critique node."""

    category: Category
    severity: Severity
    file: str
    line: int | None = None
    message: str


class RunMeta(BaseModel):
    """Observability and cost metadata for one agent run."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    ended_at: datetime | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """Top-level state flowing through the LangGraph state machine.

    Populated incrementally by nodes:
        parse_diff → plan → run_tool → critique → format_review → post_comment.

    Only ``pr`` must be supplied at construction; everything else has a sensible
    default so partial states are valid mid-run.
    """

    pr: PRMetadata
    raw_diff: str
    diff: ParsedDiff | None = None
    plan: Plan | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    review_body: str | None = None
    post_status: PostStatus = PostStatus.PENDING
    post_url: str | None = None
    meta: RunMeta = Field(default_factory=RunMeta)