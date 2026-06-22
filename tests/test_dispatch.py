"""Tests for should_dispatch decision logic."""

from pathlib import Path

import pytest

from sentry.api.dispatch import should_dispatch
from sentry.api.main import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        github_app_id="123",
        github_private_key_path=Path("/dev/null"),
        github_webhook_secret="x",
        github_installation_id=1,
        github_test_repo="finesse2307/totalpr-test",
        reviews_enabled=True,
    )


def _payload(*, repo: str, action: str) -> dict[str, object]:
    return {
        "action": action,
        "repository": {"full_name": repo},
        "pull_request": {"number": 1},
    }


def test_dispatches_opened_on_test_repo(settings: Settings) -> None:
    ok, _ = should_dispatch(
        _payload(repo="finesse2307/totalpr-test", action="opened"), settings
    )
    assert ok is True


def test_dispatches_synchronize(settings: Settings) -> None:
    ok, _ = should_dispatch(
        _payload(
            repo="finesse2307/totalpr-test", action="synchronize"
        ),
        settings,
    )
    assert ok is True


def test_skips_non_test_repo(settings: Settings) -> None:
    ok, reason = should_dispatch(
        _payload(repo="someone/else", action="opened"), settings
    )
    assert ok is False
    assert "not in scope" in reason


def test_skips_non_actionable_action(settings: Settings) -> None:
    ok, reason = should_dispatch(
        _payload(repo="finesse2307/totalpr-test", action="closed"),
        settings,
    )
    assert ok is False
    assert "not actionable" in reason


def test_kill_switch_disables_all(settings: Settings) -> None:
    settings.reviews_enabled = False
    ok, reason = should_dispatch(
        _payload(repo="finesse2307/totalpr-test", action="opened"),
        settings,
    )
    assert ok is False
    assert "kill switch" in reason