"""Tests for the FastAPI webhook handler."""
import hashlib
import hmac
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from sentry.api.main import Settings, create_app, verify_signature

SECRET = "test-secret-not-real"


def _sign(body: bytes, secret: str = SECRET) -> str:
    return (
        "sha256="
        + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(
        github_app_id="123",
        github_private_key_path=Path("/dev/null"),
        github_webhook_secret=SECRET,
        github_installation_id=456,
        github_test_repo="x/y",
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    return TestClient(
        create_app(settings=settings, dispatch_fn=lambda _payload, _s: None)
    )


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_verify_signature_accepts_valid() -> None:
    body = b'{"hello": "world"}'
    assert verify_signature(body, _sign(body), SECRET) is True


def test_verify_signature_rejects_invalid_hex() -> None:
    body = b'{"hello": "world"}'
    assert verify_signature(body, "sha256=deadbeef", SECRET) is False


def test_verify_signature_rejects_missing_header() -> None:
    body = b'{"hello": "world"}'
    assert verify_signature(body, "", SECRET) is False


def test_verify_signature_rejects_wrong_prefix() -> None:
    body = b'{"hello": "world"}'
    assert verify_signature(body, "md5=abcdef", SECRET) is False


def test_webhook_valid_signature_returns_200(client: TestClient) -> None:
    body = json.dumps(
        {"action": "opened", "repository": {"full_name": "x/y"}}
    ).encode()
    r = client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(body),
            "X-GitHub-Event": "pull_request",
            "X-GitHub-Delivery": "abc-123",
            "Content-Type": "application/json",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["event"] == "pull_request"
    assert data["action"] == "opened"


def test_webhook_invalid_signature_returns_401(client: TestClient) -> None:
    body = b'{"action": "opened"}'
    r = client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": "sha256=invalid",
            "X-GitHub-Event": "pull_request",
        },
    )
    assert r.status_code == 401


def test_webhook_missing_signature_returns_401(client: TestClient) -> None:
    body = b'{"action": "opened"}'
    r = client.post(
        "/webhook",
        content=body,
        headers={"X-GitHub-Event": "pull_request"},
    )
    assert r.status_code == 401


def test_webhook_invalid_json_returns_400(client: TestClient) -> None:
    body = b"not json {"
    r = client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(body),
            "X-GitHub-Event": "ping",
        },
    )
    assert r.status_code == 400

def test_webhook_pull_request_triggers_dispatch(
    settings: Settings,
) -> None:
    """A pull_request webhook with valid signature schedules the dispatch."""
    calls: list[dict[str, object]] = []

    def fake_dispatch(payload: dict[str, object], _settings: Settings) -> None:
        calls.append(payload)

    client = TestClient(
        create_app(settings=settings, dispatch_fn=fake_dispatch)
    )

    body = json.dumps(
        {
            "action": "opened",
            "repository": {"full_name": "x/y"},
            "pull_request": {"number": 42},
        }
    ).encode()
    r = client.post(
        "/webhook",
        content=body,
        headers={
            "X-Hub-Signature-256": _sign(body),
            "X-GitHub-Event": "pull_request",
        },
    )
    assert r.status_code == 200
    # TestClient runs background tasks synchronously before returning
    assert len(calls) == 1
    assert calls[0]["action"] == "opened"

def test_duplicate_delivery_skipped(settings: Settings) -> None:
    """A delivery_id seen before is skipped without invoking dispatch."""
    from sentry.dedupe import Deduper

    calls: list[dict[str, object]] = []

    def capture_dispatch(payload: dict[str, object], _s: Settings) -> None:
        calls.append(payload)

    class OnceDeduper:
        def __init__(self) -> None:
            self.seen: set[str] = set()

        def mark_seen(self, delivery_id: str) -> bool:
            if delivery_id in self.seen:
                return False
            self.seen.add(delivery_id)
            return True

    deduper: Deduper = OnceDeduper()
    client = TestClient(
        create_app(
            settings=settings,
            dispatch_fn=capture_dispatch,
            deduper=deduper,
        )
    )

    body = json.dumps(
        {
            "action": "opened",
            "repository": {"full_name": "x/y"},
            "pull_request": {"number": 1},
        }
    ).encode()
    headers = {
        "X-Hub-Signature-256": _sign(body),
        "X-GitHub-Event": "pull_request",
        "X-GitHub-Delivery": "delivery-XYZ",
    }

    r1 = client.post("/webhook", content=body, headers=headers)
    r2 = client.post("/webhook", content=body, headers=headers)

    assert r1.status_code == 200
    assert r1.json().get("duplicate") is not True
    assert r2.status_code == 200
    assert r2.json().get("duplicate") is True
    assert len(calls) == 1  # only the first dispatch fired