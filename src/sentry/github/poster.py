"""GitHubPoster: ``CommentPoster`` Protocol implementation backed by GitHub API.

Drop-in replacement for ``NoopPoster``. Wraps a ``GitHubClient`` and translates
its ``GitHubAPIError`` into a ``CommentResult`` with ``status=FAILED`` so the
graph's contract ("posters never raise") is preserved.
"""

import logging

from sentry.github.client import GitHubAPIError, GitHubClient
from sentry.posting import CommentResult
from sentry.state import PostStatus, PRMetadata

logger = logging.getLogger(__name__)


class GitHubPoster:
    """Post review comments to GitHub via an installation-authenticated client."""

    def __init__(self, *, client: GitHubClient) -> None:
        self._client = client

    def post(self, pr: PRMetadata, body: str) -> CommentResult:
        try:
            url = self._client.post_pr_comment(
                repo=pr.repo,
                pr_number=pr.pr_number,
                body=body,
            )
        except GitHubAPIError as exc:
            logger.warning(
                "github comment post failed for %s#%d: %s",
                pr.repo, pr.pr_number, exc,
            )
            return CommentResult(status=PostStatus.FAILED, error=str(exc))
        return CommentResult(status=PostStatus.SUCCESS, url=url)