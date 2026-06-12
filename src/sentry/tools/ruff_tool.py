"""Ruff tool: lint Python source by shelling out to a sandboxed Docker container.

Accepts a single argument ``args["code"]`` containing Python source. Writes it
to a temp file mounted read-only into a one-shot container with no network,
limited CPU and memory. Returns Ruff's stdout (findings) or "(no findings)" on
a clean run.

The image ``sentry-ruff:latest`` must exist; build it with::

    docker build -f docker/ruff.Dockerfile -t sentry-ruff:latest .
"""

import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

DEFAULT_IMAGE = "sentry-ruff:latest"
DEFAULT_TIMEOUT_SECONDS = 30

ToolFn = Callable[[dict[str, str]], str]


class RuffToolError(RuntimeError):
    """Raised when the Ruff sandbox fails to execute properly."""


def make_ruff_tool(
    *,
    image: str = DEFAULT_IMAGE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> ToolFn:
    """Build a ToolFn that runs Ruff on supplied Python source.

    Image and timeout are configurable for testing and for environments with
    custom builds.
    """

    def ruff_tool(args: dict[str, str]) -> str:
        code = args.get("code", "")
        if not code.strip():
            return "(no code provided)"

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir).resolve()
            (tmp_dir_path / "input.py").write_text(code)

            cmd = [
                "docker", "run",
                "--rm",
                "--network=none",
                "--memory=256m",
                "--cpus=0.5",
                "--volume", f"{tmp_dir_path}:/work:ro",
                "--workdir", "/work",
                image,
                "check",
                "--output-format=concise",
                "input.py",
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
                raise RuffToolError(
                    f"Ruff timed out after {timeout_seconds}s"
                ) from exc
            except FileNotFoundError as exc:
                raise RuffToolError(
                    "docker command not found; is Docker installed and in PATH?"
                ) from exc

            # Ruff exits non-zero whenever it finds issues — that's expected,
            # not a failure. Only treat invocation errors (no stdout, non-zero
            # rc) as broken.
            if result.returncode != 0 and not result.stdout.strip():
                raise RuffToolError(
                    f"docker run failed (rc={result.returncode}): "
                    f"{result.stderr.strip()}"
                )

            return result.stdout.strip() or "(no findings)"

    return ruff_tool