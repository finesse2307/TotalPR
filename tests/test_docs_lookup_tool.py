"""Tests for the docs_lookup tool.

Mocks ``urllib.request.urlopen`` rather than hitting real PyPI. Covers the
short-circuit (no package arg), the happy path with parsed metadata, the
three error paths (404, generic HTTPError, URLError, TimeoutError), and the
parameterized URL template.
"""

import json
import urllib.error
from typing import Any

import pytest

from sentry.tools.docs_lookup_tool import make_docs_lookup_tool


class _FakeResponse:
    """Minimal context-manager stand-in for urlopen's HTTPResponse."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _pypi_payload(
    *,
    version: str = "1.2.3",
    summary: str = "A test package",
    project_url: str = "https://example.com",
    author: str = "Test Author",
    url_count: int = 2,
) -> bytes:
    body = {
        "info": {
            "version": version,
            "summary": summary,
            "project_url": project_url,
            "author": author,
        },
        "urls": [{"filename": f"f{i}.whl"} for i in range(url_count)],
    }
    return json.dumps(body).encode("utf-8")


def test_missing_package_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty or whitespace-only package arg returns early without an HTTP call."""
    called = False

    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _FakeResponse(b"{}")

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    tool = make_docs_lookup_tool()
    assert tool({}) == "(no package provided)"
    assert tool({"package": "   "}) == "(no package provided)"
    assert called is False


def test_happy_path_returns_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    """A successful fetch is rendered as a multi-line summary."""

    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        return _FakeResponse(
            _pypi_payload(
                version="6.0.1",
                summary="YAML parser and emitter",
                project_url="https://pyyaml.org",
                author="Kirill Simonov",
                url_count=4,
            )
        )

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    out = make_docs_lookup_tool()({"package": "pyyaml"})

    assert "package: pyyaml" in out
    assert "latest version: 6.0.1" in out
    assert "summary: YAML parser and emitter" in out
    assert "project url: https://pyyaml.org" in out
    assert "author: Kirill Simonov" in out
    assert "release files: 4" in out


def test_404_returns_not_found_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 from PyPI maps to a clear 'not found' string."""

    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.HTTPError(
            url="https://pypi.org/pypi/nopepkg/json",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    out = make_docs_lookup_tool()({"package": "nopepkg"})
    assert out == "(package 'nopepkg' not found on PyPI)"


def test_other_http_error_returns_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-404 HTTP errors surface the status code and reason."""

    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.HTTPError(
            url="https://pypi.org/pypi/x/json",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    out = make_docs_lookup_tool()({"package": "x"})
    assert "HTTP error from PyPI: 503" in out
    assert "Service Unavailable" in out


def test_timeout_returns_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TimeoutError becomes a user-readable timeout message."""

    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("simulated")

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    out = make_docs_lookup_tool(timeout_seconds=5)({"package": "pyyaml"})
    assert out == "(timeout after 5s reaching PyPI)"


def test_network_error_returns_network_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A URLError (DNS / connection refused) becomes a network-error string."""

    def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
        raise urllib.error.URLError("nodename nor servname provided")

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    out = make_docs_lookup_tool()({"package": "pyyaml"})
    assert "network error reaching PyPI" in out
    assert "nodename nor servname provided" in out


def test_custom_url_template_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """A custom pypi_url_template is substituted with the package name."""
    captured: dict[str, Any] = {}

    def fake_urlopen(req: Any, **kwargs: Any) -> Any:
        captured["url"] = req.full_url
        return _FakeResponse(_pypi_payload())

    monkeypatch.setattr(
        "sentry.tools.docs_lookup_tool.urllib.request.urlopen", fake_urlopen
    )

    make_docs_lookup_tool(
        pypi_url_template="https://mirror.example.com/{package}.json"
    )({"package": "requests"})

    assert captured["url"] == "https://mirror.example.com/requests.json"