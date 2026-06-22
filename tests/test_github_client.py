"""Tests for GitHubClient."""

from unittest.mock import MagicMock

import httpx
import pytest

from sentry.github.client import GitHubAPIError, GitHubClient


def _ok_response(text: str) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.text = text
    return r


def _make_auth(token: str = "ghs_test_token") -> MagicMock:
    auth = MagicMock()
    auth.installation_token.return_value = token
    return auth


def test_get_pr_diff_sends_correct_request() -> None:
    """The request hits the right URL with the diff Accept header and bearer token."""
    captured: dict[str, object] = {}

    def fake_get(url: str, **kwargs: object) -> MagicMock:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _ok_response("diff body")

    http = MagicMock()
    http.get.side_effect = fake_get

    client = GitHubClient(auth=_make_auth(), http_client=http)
    client.get_pr_diff(repo="finesse2307/totalpr-test", pr_number=1)

    assert captured["url"] == (
        "https://api.github.com/repos/finesse2307/totalpr-test/pulls/1"
    )
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer ghs_test_token"
    assert headers["Accept"] == "application/vnd.github.diff"


def test_get_pr_diff_returns_response_text() -> None:
    """The PR diff body is returned verbatim."""
    http = MagicMock()
    http.get.return_value = _ok_response(
        "diff --git a/x.py b/x.py\n+print('hi')\n"
    )
    client = GitHubClient(auth=_make_auth(), http_client=http)

    result = client.get_pr_diff(repo="a/b", pr_number=42)
    assert result.startswith("diff --git a/x.py")
    assert "print('hi')" in result


def test_get_pr_diff_non_200_raises() -> None:
    """A non-200 response becomes a GitHubAPIError."""
    bad = MagicMock()
    bad.status_code = 404
    bad.text = "Not Found"
    http = MagicMock()
    http.get.return_value = bad

    client = GitHubClient(auth=_make_auth(), http_client=http)
    with pytest.raises(GitHubAPIError, match="404"):
        client.get_pr_diff(repo="a/b", pr_number=99)


def test_get_pr_diff_http_error_raises() -> None:
    """A network failure becomes a GitHubAPIError, not a bare httpx error."""
    http = MagicMock()
    http.get.side_effect = httpx.ConnectError("connection refused")

    client = GitHubClient(auth=_make_auth(), http_client=http)
    with pytest.raises(GitHubAPIError, match="failed to reach"):
        client.get_pr_diff(repo="a/b", pr_number=1)

def _created_response(html_url: str) -> MagicMock:
    r = MagicMock()
    r.status_code = 201
    r.json.return_value = {"html_url": html_url}
    return r


def test_post_pr_comment_sends_correct_request() -> None:
    """The post hits the issues/comments endpoint with the body in JSON."""
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> MagicMock:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        captured["json"] = kwargs.get("json", {})
        return _created_response("https://github.com/x/y/issues/1#issuecomment-99")

    http = MagicMock()
    http.post.side_effect = fake_post

    client = GitHubClient(auth=_make_auth(), http_client=http)
    url = client.post_pr_comment(repo="x/y", pr_number=1, body="hello")

    assert captured["url"] == (
        "https://api.github.com/repos/x/y/issues/1/comments"
    )
    headers = captured["headers"]
    assert isinstance(headers, dict)
    assert headers["Authorization"] == "Bearer ghs_test_token"
    assert captured["json"] == {"body": "hello"}
    assert url == "https://github.com/x/y/issues/1#issuecomment-99"


def test_post_pr_comment_non_201_raises() -> None:
    """A non-201 response from GitHub becomes a GitHubAPIError."""
    bad = MagicMock()
    bad.status_code = 403
    bad.text = "Resource not accessible by integration"
    http = MagicMock()
    http.post.return_value = bad

    client = GitHubClient(auth=_make_auth(), http_client=http)
    with pytest.raises(GitHubAPIError, match="403"):
        client.post_pr_comment(repo="x/y", pr_number=1, body="hello")