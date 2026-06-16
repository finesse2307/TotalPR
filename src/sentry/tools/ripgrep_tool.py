"""ripgrep tool: search the workspace for a planner-supplied pattern.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

DEFAULT_IMAGE = "sentry-ripgrep:latest"
DEFAULT_TIMEOUT_SECONDS = 15
DEFAULT_MAX_COUNT = 50  # cap matches per file to keep output bounded

ToolFn = Callable[[dict[str, str]], str]


class RipgrepToolError(RuntimeError):
    """Raised when the ripgrep sandbox fails to execute."""


def make_ripgrep_tool(
    *,
    workspace_path: Path,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_count: int = DEFAULT_MAX_COUNT,
) -> ToolFn:
    """Build a ToolFn bound to a workspace path.

    The returned function expects ``args["pattern"]`` (the regex to search for)
    and optionally ``args["glob"]`` (a glob to restrict file matches, e.g.
    ``"*.py"``). Missing pattern returns a clear error string rather than
    silently scanning everything.
    """

    def ripgrep_tool(args: dict[str, str]) -> str:
        pattern = args.get("pattern", "").strip()
        if not pattern:
            return "(no pattern provided)"

        if not workspace_path.exists() or not any(workspace_path.iterdir()):
            return "(workspace is empty)"

        cmd = [
            "docker", "run",
            "--rm",
            "--network=none",
            "--memory=128m",
            "--cpus=0.5",
            "--volume", f"{workspace_path.resolve()}:/work:ro",
            "--workdir", "/work",
            image,
            "--line-number",
            "--with-filename",
            "--max-count", str(max_count),
        ]
        if glob := args.get("glob", "").strip():
            cmd += ["--glob", glob]
        cmd += [pattern, "."]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RipgrepToolError(
                f"ripgrep timed out after {timeout_seconds}s"
            ) from exc
        except FileNotFoundError as exc:
            raise RipgrepToolError(
                "docker command not found; is Docker installed and in PATH?"
            ) from exc

        # ripgrep exits 0 (matches), 1 (no matches), 2 (real error).
        if result.returncode == 1:
            return "(no matches)"
        if result.returncode != 0:
            raise RipgrepToolError(
                f"ripgrep failed (rc={result.returncode}): "
                f"{result.stderr.strip()}"
            )

        return result.stdout.strip() or "(no matches)"

    return ripgrep_tool