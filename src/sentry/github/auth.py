"""GitHub App installation authentication.

Implements the two-step GitHub App auth flow:

1. Sign a short-lived JWT with the app's private RSA key (``iat`` ~now,
   ``exp`` +9 min, ``iss`` = app id, signed with RS256).
2. POST it to ``/app/installations/<id>/access_tokens`` to receive a roughly
   one-hour installation access token, used as the bearer for subsequent
   GitHub API calls.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import jwt


class GitHubAuthError(Exception):
    """Raised when JWT signing or installation-token exchange fails."""


@dataclass
class InstallationToken:
    """An installation access token with its absolute expiry time."""

    token: str
    expires_at: float  # Unix timestamp


class GitHubAppAuth:
    """Authenticator for a single GitHub App installation."""

    JWT_LIFETIME_SECONDS = 9 * 60       # GitHub caps app JWTs at 10 min
    REFRESH_BUFFER_SECONDS = 5 * 60     # Refresh tokens 5 min before expiry
    GITHUB_API_BASE = "https://api.github.com"

    def __init__(
        self,
        *,
        app_id: str,
        installation_id: int,
        private_key_path: Path,
        http_client: httpx.Client | None = None,
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._app_id = app_id
        self._installation_id = installation_id
        self._private_key = private_key_path.read_text()
        self._http = http_client or httpx.Client(timeout=10.0)
        self._now = now_fn
        self._cached: InstallationToken | None = None

    def installation_token(self) -> str:
        """Return a valid installation token, refreshing if near expiry."""
        now = self._now()
        if (
            self._cached is not None
            and self._cached.expires_at - now > self.REFRESH_BUFFER_SECONDS
        ):
            return self._cached.token
        self._cached = self._fetch_new_token()
        return self._cached.token

    def _make_jwt(self) -> str:
        """Sign a short-lived JWT identifying this app to GitHub."""
        now = int(self._now())
        payload: dict[str, Any] = {
            "iat": now - 60,                      # clock-skew tolerance
            "exp": now + self.JWT_LIFETIME_SECONDS,
            "iss": self._app_id,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    def _fetch_new_token(self) -> InstallationToken:
        """Exchange the app JWT for an installation access token."""
        url = (
            f"{self.GITHUB_API_BASE}/app/installations/"
            f"{self._installation_id}/access_tokens"
        )
        headers = {
            "Authorization": f"Bearer {self._make_jwt()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            response = self._http.post(url, headers=headers)
        except httpx.HTTPError as exc:
            raise GitHubAuthError(f"failed to reach GitHub: {exc}") from exc

        if response.status_code != 201:
            raise GitHubAuthError(
                f"installation token request failed: "
                f"{response.status_code} {response.text[:200]}"
            )

        data = response.json()
        expires_at = datetime.fromisoformat(data["expires_at"]).timestamp()
        return InstallationToken(token=data["token"], expires_at=expires_at)