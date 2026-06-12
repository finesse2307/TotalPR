"""Tests for the Ruff tool.

Mocks subprocess.run rather than shelling out to Docker. Covers: empty input
shortcut, docker command construction, findings output, no-findings output,
timeout, missing-docker, and broken-invocation error paths.
"""

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentry.tools.ruff_tool import RuffToolError, make_ruff_tool


def _fake_completed(
    *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def test_empty_code_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whitespace-only code returns early without calling subprocess."""
    called = False

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _fake_completed()

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    result = make_ruff_tool()({"code": "   "})

    assert result == "(no code provided)"
    assert called is False


def test_command_is_sandboxed_docker_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructed command includes sandbox flags and the target image."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _fake_completed(stdout="")

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    make_ruff_tool(image="my-ruff:test")({"code": "print('hi')\n"})

    cmd = captured["cmd"]
    assert cmd[0:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--network=none" in cmd
    assert "--memory=256m" in cmd
    assert "--cpus=0.5" in cmd
    assert "my-ruff:test" in cmd
    assert "check" in cmd
    assert "--output-format=concise" in cmd
    assert "input.py" in cmd
    assert captured["kwargs"]["timeout"] > 0


def test_findings_returned_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Ruff reports issues (rc != 0 with stdout), stdout is returned."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(
            returncode=1,
            stdout="input.py:1:1: F401 [*] `os` imported but unused\n",
        )

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    result = make_ruff_tool()({"code": "import os\n"})

    assert "F401" in result
    assert "imported but unused" in result


def test_no_findings_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A clean Ruff run (rc=0, no stdout) returns the no-findings sentinel."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(returncode=0, stdout="")

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    assert make_ruff_tool()({"code": "x = 1\n"}) == "(no findings)"


def test_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A subprocess timeout becomes RuffToolError."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=30)

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    with pytest.raises(RuffToolError, match="timed out"):
        make_ruff_tool(timeout_seconds=30)({"code": "x = 1\n"})


def test_docker_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the docker binary is absent, a clear RuffToolError is raised."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    with pytest.raises(RuffToolError, match="docker command not found"):
        make_ruff_tool()({"code": "x = 1\n"})


def test_invocation_failure_with_no_output_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """rc != 0 with no stdout (e.g. missing image) raises RuffToolError."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(
            returncode=125, stdout="", stderr="Unable to find image"
        )

    monkeypatch.setattr("sentry.tools.ruff_tool.subprocess.run", fake_run)

    with pytest.raises(
        RuffToolError, match=r"docker run failed.*Unable to find image"
    ):
        make_ruff_tool()({"code": "x = 1\n"})