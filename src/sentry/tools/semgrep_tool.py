"""Semgrep tool: scan workspace files for security and pattern findings.

Runs Semgrep inside the official Docker image against a workspace directory
bound at construction time via ``make_semgrep_tool``. Findings are returned
verbatim as text for the critique node to interpret.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

DEFAULT_IMAGE = "semgrep/semgrep:latest"
DEFAULT_CONFIG = "p/security-audit"
DEFAULT_TIMEOUT_SECONDS = 60

ToolFn = Callable[[dict[str, str]], str]


class SemgrepToolError(RuntimeError):
    """Raised when the Semgrep sandbox fails to execute."""


def make_semgrep_tool(
    *,
    workspace_path: Path,
    image: str = DEFAULT_IMAGE,
    config: str = DEFAULT_CONFIG,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ToolFn:
    """Build a ToolFn bound to a workspace path.

    ``args["config"]`` overrides the default ruleset for a single call. The
    default ``p/security-audit`` is the registry pack most relevant to code
    review (SQL injection, secrets, unsafe deserialization, ...).
    """

    def semgrep_tool(args: dict[str, str]) -> str:
        if not workspace_path.exists() or not any(workspace_path.iterdir()):
            return "(workspace is empty)"

        chosen_config = args.get("config", config)

        cmd = [
            "docker", "run",
            "--rm",
            "--memory=512m",
            "--cpus=0.5",
            "--volume", f"{workspace_path.resolve()}:/work:ro",
            "--workdir", "/work",
            image,
            "scan",
            f"--config={chosen_config}",
            "--quiet",
            "--no-rewrite-rule-ids",
            ".",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SemgrepToolError(
                f"Semgrep timed out after {timeout_seconds}s"
            ) from exc
        except FileNotFoundError as exc:
            raise SemgrepToolError(
                "docker command not found; is Docker installed and in PATH?"
            ) from exc

        # Semgrep exits 0 (no findings) or 1 (findings present). Other codes
        # combined with empty stdout indicate a real invocation failure.
        if result.returncode not in (0, 1) and not result.stdout.strip():
            raise SemgrepToolError(
                f"docker run failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        return result.stdout.strip() or "(no findings)"

    return semgrep_tool