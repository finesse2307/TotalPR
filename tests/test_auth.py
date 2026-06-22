"""Tests for GitHubAppAuth."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from sentry.github.auth import GitHubAppAuth, GitHubAuthError


@pytest.fixture(scope="module")
def rsa_key_pair() -> tuple[bytes, bytes]:
    """Generate a single RSA key pair for the whole test module."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return private_pem, public_pem


@pytest.fixture
def private_key_path(
    tmp_path: Path, rsa_key_pair: tuple[bytes, bytes]
) -> Path:
    path = tmp_path / "test.pem"
    path.write_bytes(rsa_key_pair[0])
    return path


def _ok_response(token: str, expires_at_iso: str) -> MagicMock:
    r = MagicMock()
    r.status_code = 201
    r.json.return_value = {"token": token, "expires_at": expires_at_iso}
    return r


def test_jwt_has_iat_exp_iss_with_app_id(
    private_key_path: Path, rsa_key_pair: tuple[bytes, bytes]
) -> None:
    """The JWT claims set matches GitHub's required shape."""
    fixed_now = 1_700_000_000.0
    auth = GitHubAppAuth(
        app_id="4112785",
        installation_id=141818869,
        private_key_path=private_key_path,
        http_client=MagicMock(),
        now_fn=lambda: fixed_now,
    )
    token = auth._make_jwt()
    decoded = jwt.decode(
        token,
        rsa_key_pair[1],
        algorithms=["RS256"],
        options={"verify_exp": False},
    )
    assert decoded["iss"] == "4112785"
    assert decoded["iat"] == int(fixed_now) - 60
    assert decoded["exp"] == int(fixed_now) + 9 * 60


def test_fetches_token_on_first_call(private_key_path: Path) -> None:
    """The first installation_token() call hits GitHub."""
    http = MagicMock()
    http.post.return_value = _ok_response(
        "ghs_first", "2030-01-01T00:00:00+00:00"
    )
    auth = GitHubAppAuth(
        app_id="123",
        installation_id=456,
        private_key_path=private_key_path,
        http_client=http,
    )

    token = auth.installation_token()
    assert token == "ghs_first"
    assert http.post.call_count == 1


def test_returns_cached_token_when_fresh(private_key_path: Path) -> None:
    """Subsequent calls within the buffer window return the cached token."""
    http = MagicMock()
    http.post.return_value = _ok_response(
        "ghs_first", "2030-01-01T00:00:00+00:00"
    )
    auth = GitHubAppAuth(
        app_id="123",
        installation_id=456,
        private_key_path=private_key_path,
        http_client=http,
        now_fn=lambda: 1_700_000_000.0,
    )

    auth.installation_token()
    auth.installation_token()
    auth.installation_token()
    assert http.post.call_count == 1


def test_refetches_token_when_near_expiry(private_key_path: Path) -> None:
    """Once the cached token is within REFRESH_BUFFER of expiry, refetch."""
    fake_clock = [1_700_000_000.0]

    def now() -> float:
        return fake_clock[0]

    exp_1 = datetime.fromtimestamp(fake_clock[0] + 360, tz=UTC).isoformat()
    exp_2 = datetime.fromtimestamp(fake_clock[0] + 3600, tz=UTC).isoformat()

    http = MagicMock()
    http.post.side_effect = [
        _ok_response("ghs_first", exp_1),
        _ok_response("ghs_second", exp_2),
    ]
    auth = GitHubAppAuth(
        app_id="123",
        installation_id=456,
        private_key_path=private_key_path,
        http_client=http,
        now_fn=now,
    )

    assert auth.installation_token() == "ghs_first"
    # Advance to 2 minutes before expiry (within 5-minute REFRESH_BUFFER)
    fake_clock[0] += 240
    assert auth.installation_token() == "ghs_second"
    assert http.post.call_count == 2


def test_non_201_response_raises(private_key_path: Path) -> None:
    """A non-201 from GitHub becomes a GitHubAuthError."""
    bad = MagicMock()
    bad.status_code = 401
    bad.text = "Bad credentials"
    http = MagicMock()
    http.post.return_value = bad

    auth = GitHubAppAuth(
        app_id="123",
        installation_id=456,
        private_key_path=private_key_path,
        http_client=http,
    )
    with pytest.raises(GitHubAuthError, match="401"):
        auth.installation_token()


def test_http_error_raises(private_key_path: Path) -> None:
    """Network failures become GitHubAuthError, not bare httpx errors."""
    http = MagicMock()
    http.post.side_effect = httpx.ConnectError("connection refused")
    auth = GitHubAppAuth(
        app_id="123",
        installation_id=456,
        private_key_path=private_key_path,
        http_client=http,
    )
    with pytest.raises(GitHubAuthError, match="failed to reach"):
        auth.installation_token()