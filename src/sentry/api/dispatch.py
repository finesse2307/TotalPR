"""Background webhook dispatch: fetch PR diff, run agent graph, post review.

Called from a FastAPI BackgroundTask after the webhook handler has returned
200 to GitHub. The handler must not block on agent work — GitHub's webhook
timeout is 10 seconds and a real review takes longer.

``should_dispatch`` is split out as a pure function so the decision logic
(repo filter, action filter, kill switch) is testable without mocking every
downstream dependency.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from sentry.anthropic_client import AnthropicLLMClient
from sentry.api.main import Settings
from sentry.budget import BudgetedLLMClient
from sentry.cache import SQLiteCacheLLMClient
from sentry.embedding import VoyageEmbeddingClient
from sentry.github.auth import GitHubAppAuth
from sentry.github.client import GitHubClient
from sentry.github.poster import GitHubPoster
from sentry.graph import build_graph
from sentry.memory import MemoryStore
from sentry.nodes.run_tool import ToolRegistry
from sentry.state import AgentState, PRMetadata, ToolName
from sentry.telemetry import run_span
from sentry.tools.docs_lookup_tool import make_docs_lookup_tool
from sentry.tools.ripgrep_tool import make_ripgrep_tool
from sentry.tools.ruff_tool import make_ruff_tool
from sentry.tools.semgrep_tool import make_semgrep_tool
from sentry.workspace import materialize_workspace

logger = logging.getLogger(__name__)

_ACTIONABLE_ACTIONS = {"opened", "synchronize", "reopened"}


def should_dispatch(
    payload: dict[str, Any], settings: Settings
) -> tuple[bool, str]:
    """Return ``(decision, reason)`` for whether to review this PR webhook.

    Decision is False if any of:
    - reviews are globally disabled via the kill switch
    - the repo doesn't match the configured test repo
    - the action isn't one that signals new code (opened/synchronize/reopened)
    """
    if not settings.reviews_enabled:
        return False, "reviews disabled by kill switch"

    repo = payload.get("repository", {}).get("full_name", "")
    if repo != settings.github_test_repo:
        return False, f"repo not in scope: {repo}"

    action = payload.get("action", "")
    if action not in _ACTIONABLE_ACTIONS:
        return False, f"action not actionable: {action}"

    return True, "ok"


def dispatch_pull_request(
    payload: dict[str, Any], settings: Settings
) -> None:
    """Process a pull_request webhook payload end-to-end.

    Decides via ``should_dispatch``, then runs the agent. All exceptions are
    caught and logged — this runs in a BackgroundTask whose failures don't
    propagate to the webhook's HTTP response.
    """
    decision, reason = should_dispatch(payload, settings)
    logger.info(
        "dispatch decision: %s (%s) repo=%s action=%s",
        decision, reason,
        payload.get("repository", {}).get("full_name", "?"),
        payload.get("action", "?"),
    )
    if not decision:
        return

    repo = payload["repository"]["full_name"]
    pr_number = payload["pull_request"]["number"]
    logger.info("dispatch start: %s#%d", repo, pr_number)

    try:
        _run_review(payload, settings)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "dispatch failed for %s#%d: %s", repo, pr_number, exc
        )


def _postgres_dsn() -> str:
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'sentry')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'sentry_dev_password')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'sentry')}"
    )


def _run_review(payload: dict[str, Any], settings: Settings) -> None:
    """Build the full agent stack and run one PR review end-to-end."""
    pr_data = payload["pull_request"]
    repo = payload["repository"]["full_name"]
    pr_number = pr_data["number"]

    auth = GitHubAppAuth(
        app_id=settings.github_app_id,
        installation_id=settings.github_installation_id,
        private_key_path=settings.github_private_key_path,
    )
    gh = GitHubClient(auth=auth)
    poster = GitHubPoster(client=gh)

    raw_diff = gh.get_pr_diff(repo=repo, pr_number=pr_number)

    real_llm = AnthropicLLMClient()
    cached_llm = SQLiteCacheLLMClient(
        real_llm,
        db_path=Path(".cache/llm.sqlite"),
        namespace=real_llm.model,
    )
    budgeted_llm = BudgetedLLMClient(cached_llm, cap_usd=0.50)

    memory = MemoryStore(
        dsn=_postgres_dsn(), embedder=VoyageEmbeddingClient()
    )

    initial = AgentState(
        pr=PRMetadata(
            repo=repo,
            pr_number=pr_number,
            head_sha=pr_data["head"]["sha"],
            base_sha=pr_data["base"]["sha"],
            author=pr_data["user"]["login"],
            title=pr_data["title"],
        ),
        raw_diff=raw_diff,
    )

    with tempfile.TemporaryDirectory(prefix="sentry-pr-") as ws_str:
        workspace = Path(ws_str)
        materialize_workspace(raw_diff, workspace)
        tools: ToolRegistry = {
            ToolName.RUFF: make_ruff_tool(),
            ToolName.SEMGREP: make_semgrep_tool(workspace_path=workspace),
            ToolName.RIPGREP: make_ripgrep_tool(workspace_path=workspace),
            ToolName.DOCS_LOOKUP: make_docs_lookup_tool(),
        }
        graph = build_graph(
            llm=budgeted_llm, tools=tools, poster=poster, memory=memory,
        )
        with run_span(f"review_pr:{repo}#{pr_number}"):
            final = graph.invoke(initial)

    findings_count = len(final.get("findings") or [])
    logger.info(
        "review complete: %s#%d findings=%d post_status=%s spend=$%.4f",
        repo, pr_number, findings_count,
        final.get("post_status"), budgeted_llm.total_spend_usd,
    )