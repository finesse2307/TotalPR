"""Tests for the ripgrep tool.
"""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from sentry.tools.ripgrep_tool import RipgrepToolError, make_ripgrep_tool


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


def test_missing_pattern_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing or whitespace pattern returns early."""
    called = False

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _fake_completed()

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    tool = make_ripgrep_tool(workspace_path=ws)

    assert tool({}) == "(no pattern provided)"
    assert tool({"pattern": "   "}) == "(no pattern provided)"
    assert called is False


def test_empty_workspace_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An empty workspace returns early even with a valid pattern."""
    called = False

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        nonlocal called
        called = True
        return _fake_completed()

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    empty = tmp_path / "ws"
    empty.mkdir()

    result = make_ripgrep_tool(workspace_path=empty)({"pattern": "foo"})
    assert result == "(workspace is empty)"
    assert called is False


def test_command_includes_sandbox_flags_and_pattern(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The constructed command mounts the workspace, sandboxes, includes the pattern."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        return _fake_completed(stdout="")

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    make_ripgrep_tool(workspace_path=ws, image="ripgrep:test")(
        {"pattern": "get_user"}
    )

    cmd = captured["cmd"]
    assert cmd[0:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--network=none" in cmd
    assert "--memory=128m" in cmd
    assert f"{ws.resolve()}:/work:ro" in cmd
    assert "ripgrep:test" in cmd
    assert "get_user" in cmd


def test_glob_arg_adds_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When args['glob'] is provided, --glob is added to the command."""
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> Any:
        captured["cmd"] = cmd
        return _fake_completed(stdout="")

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    make_ripgrep_tool(workspace_path=ws)({"pattern": "TODO", "glob": "*.py"})

    cmd = captured["cmd"]
    glob_idx = cmd.index("--glob")
    assert cmd[glob_idx + 1] == "*.py"


def test_matches_returned_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Matches pass through to the caller as ripgrep formatted them."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(
            returncode=0,
            stdout="./src/x.py:12:    def get_user(self, uid):\n",
        )

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    out = make_ripgrep_tool(workspace_path=ws)({"pattern": "get_user"})

    assert "src/x.py:12" in out
    assert "def get_user" in out


def test_no_matches_returns_placeholder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """rc=1 (ripgrep's 'no matches') maps to a placeholder, not an error."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(returncode=1, stdout="")

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    result = make_ripgrep_tool(workspace_path=ws)({"pattern": "x"})
    assert result == "(no matches)"


def test_real_ripgrep_error_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """rc=2 (genuine ripgrep error, e.g. bad regex) raises RipgrepToolError."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        return _fake_completed(
            returncode=2,
            stdout="",
            stderr="regex parse error: unclosed group",
        )

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    with pytest.raises(RipgrepToolError, match="ripgrep failed"):
        make_ripgrep_tool(workspace_path=ws)({"pattern": "(["})


def test_timeout_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A subprocess timeout becomes RipgrepToolError."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=15)

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    with pytest.raises(RipgrepToolError, match="timed out"):
        make_ripgrep_tool(workspace_path=ws)({"pattern": "x"})


def test_docker_missing_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When the docker binary is absent, RipgrepToolError is raised."""

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("docker not found")

    monkeypatch.setattr("sentry.tools.ripgrep_tool.subprocess.run", fake_run)

    ws = _workspace_with_file(tmp_path)
    with pytest.raises(RipgrepToolError, match="docker command not found"):
        make_ripgrep_tool(workspace_path=ws)({"pattern": "x"})