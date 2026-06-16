"""Tests for the Semgrep tool.
"""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentry.tools.semgrep_tool import (
    DEFAULT_CONFIG,
    SemgrepToolError,
    make_semgrep_tool,
)


def _fake_completed(
    *, returncode: int = 0, stdout: str = "", stderr: str = ""
) -> MagicMock:
    result = MagicMock()
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result


def _workspace_with_file(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "x.py").write_text("print('hi')\n")
    return ws


def test_empty_workspace_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty workspace returns early without calling subprocess."""
    called = False

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _fake_completed()

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    empty = tmp_path / "ws"
    empty.mkdir()

    tool = make_semgrep_tool(workspace_path=empty)
    assert tool({}) == "(workspace is empty)"
    assert called is False


def test_command_is_sandboxed_docker_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The constructed command mounts the workspace and uses the right image."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return _fake_completed(stdout="")

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    make_semgrep_tool(workspace_path=ws, image="semgrep:test")({})

    cmd = captured["cmd"]
    assert cmd[0:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--memory=512m" in cmd
    assert f"{ws.resolve()}:/work:ro" in cmd
    assert "semgrep:test" in cmd
    assert "scan" in cmd
    assert f"--config={DEFAULT_CONFIG}" in cmd
    assert captured["kwargs"]["timeout"] > 0


def test_findings_returned_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When Semgrep reports issues (rc=1 with stdout), stdout is returned."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(
            returncode=1,
            stdout=(
                "x.py\n"
                "   sql-injection (python.lang.security.sql-injection)\n"
                '   12: query = f"SELECT * FROM users WHERE id={uid}"\n'
            ),
        )

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    out = make_semgrep_tool(workspace_path=ws)({})

    assert "sql-injection" in out
    assert "SELECT" in out


def test_no_findings_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A clean Semgrep run (rc=0, empty stdout) returns the placeholder."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(returncode=0, stdout="")

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    assert make_semgrep_tool(workspace_path=ws)({}) == "(no findings)"


def test_args_config_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """args['config'] takes precedence over the construction-time default."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        return _fake_completed(stdout="")

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    make_semgrep_tool(workspace_path=ws)({"config": "p/python"})

    assert "--config=p/python" in captured["cmd"]
    assert f"--config={DEFAULT_CONFIG}" not in captured["cmd"]


def test_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subprocess timeout becomes SemgrepToolError."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=60)

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    with pytest.raises(SemgrepToolError, match="timed out"):
        make_semgrep_tool(workspace_path=ws)({})


def test_docker_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the docker binary is absent, SemgrepToolError is raised."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    with pytest.raises(SemgrepToolError, match="docker command not found"):
        make_semgrep_tool(workspace_path=ws)({})


def test_invocation_failure_with_no_output_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unexpected rc (not 0 or 1) with empty stdout raises SemgrepToolError."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(
            returncode=125, stdout="", stderr="Unable to find image"
        )

    monkeypatch.setattr("sentry.tools.semgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    with pytest.raises(SemgrepToolError, match="docker run failed"):
        make_semgrep_tool(workspace_path=ws)({})