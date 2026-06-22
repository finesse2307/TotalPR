"""Tests for GitHubPoster."""

from unittest.mock import MagicMock

from sentry.github.client import GitHubAPIError
from sentry.github.poster import GitHubPoster
from sentry.state import PostStatus, PRMetadata


def _pr() -> PRMetadata:
    return PRMetadata(
        repo="finesse2307/totalpr-test",
        pr_number=1,
        head_sha="abc",
        base_sha="def",
        author="alice",
        title="smoke",
    )


def test_post_success_returns_success_with_url() -> None:
    """A successful post produces SUCCESS status and the returned URL."""
    client = MagicMock()
    client.post_pr_comment.return_value = "https://github.com/x/y/pull/1#issuecomment-1"

    poster = GitHubPoster(client=client)
    result = poster.post(_pr(), "review body")

    assert result.status == PostStatus.SUCCESS
    assert result.url == "https://github.com/x/y/pull/1#issuecomment-1"
    assert result.error is None
    client.post_pr_comment.assert_called_once_with(
        repo="finesse2307/totalpr-test", pr_number=1, body="review body"
    )


def test_post_failure_returns_failed_without_raising() -> None:
    """A GitHubAPIError is swallowed and returned as FAILED — posters never raise."""
    client = MagicMock()
    client.post_pr_comment.side_effect = GitHubAPIError("403 forbidden")

    poster = GitHubPoster(client=client)
    result = poster.post(_pr(), "review body")

    assert result.status == PostStatus.FAILED
    assert result.url is None
    assert "403" in (result.error or "")