"""Minimal GitHub REST client for installation-authenticated calls.

Phase 4 only needs two operations:
- ``get_pr_diff`` — fetch the unified diff for a PR
- ``post_pr_comment`` — post an issue comment on a PR (lands in step 57)

The client takes a ``GitHubAppAuth`` and pulls an installation token on every
request. Caching lives in the auth object; the client is stateless.
"""

import httpx

from sentry.github.auth import GitHubAppAuth


class GitHubAPIError(Exception):
    """Raised when a GitHub REST API call fails."""


class GitHubClient:
    """REST client for installation-authenticated GitHub operations."""

    GITHUB_API_BASE = "https://api.github.com"

    def __init__(
        self,
        *,
        auth: GitHubAppAuth,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._auth = auth
        self._http = http_client or httpx.Client(timeout=30.0)

    def get_pr_diff(self, *, repo: str, pr_number: int) -> str:
        """Fetch the unified diff of a PR.

        ``repo`` is the ``owner/name`` slug. Content negotiation via
        ``Accept: application/vnd.github.diff`` returns the diff directly
        as text instead of JSON.
        """
        url = f"{self.GITHUB_API_BASE}/repos/{repo}/pulls/{pr_number}"
        headers = {
            "Authorization": f"Bearer {self._auth.installation_token()}",
            "Accept": "application/vnd.github.diff",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = self._http.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise GitHubAPIError(f"failed to reach GitHub: {exc}") from exc

        if response.status_code != 200:
            raise GitHubAPIError(
                f"PR diff request failed: "
                f"{response.status_code} {response.text[:200]}"
            )
        return response.text
    
    def post_pr_comment(
        self, *, repo: str, pr_number: int, body: str
    ) -> str:
        """Post a top-level comment on a PR. Returns the comment's HTML URL.

        Uses the issue-comments endpoint since GitHub treats PRs as issues
        for top-level discussion. Review (inline) comments would use the
        pulls/comments endpoint with path+position+commit_id.
        """
        url = (
            f"{self.GITHUB_API_BASE}/repos/{repo}/issues/{pr_number}/comments"
        )
        headers = {
            "Authorization": f"Bearer {self._auth.installation_token()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = self._http.post(
                url, headers=headers, json={"body": body}
            )
        except httpx.HTTPError as exc:
            raise GitHubAPIError(f"failed to reach GitHub: {exc}") from exc

        if response.status_code != 201:
            raise GitHubAPIError(
                f"comment post failed: "
                f"{response.status_code} {response.text[:200]}"
            )
        return str(response.json()["html_url"])