"""docs_lookup tool: query PyPI for package metadata.

Given ``args["package"]`` (a Python package name), fetches PyPI's JSON API and
returns a summary: latest version, one-line summary, project URL, author, and
release-file count. Useful for verifying that an imported package exists, what
it claims to do, and whether it has recent releases.

No Docker sandbox: we make the HTTP call ourselves, not run user code. Errors
(404, network failure, timeout) are caught and returned as text so the
critique node sees them as evidence rather than a crash.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable

DEFAULT_TIMEOUT_SECONDS = 10
PYPI_URL = "https://pypi.org/pypi/{package}/json"

ToolFn = Callable[[dict[str, str]], str]


def make_docs_lookup_tool(
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    pypi_url_template: str = PYPI_URL,
) -> ToolFn:
    """Build a ToolFn that looks up Python packages on PyPI.

    ``pypi_url_template`` is parameterized for tests (point at a local fixture
    server, a mock, or another registry mirror).
    """

    def docs_lookup_tool(args: dict[str, str]) -> str:
        package = args.get("package", "").strip()
        if not package:
            return "(no package provided)"

        url = pypi_url_template.format(package=package)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "sentry-code-review-agent/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return f"(package '{package}' not found on PyPI)"
            return f"(HTTP error from PyPI: {exc.code} {exc.reason})"
        except TimeoutError:
            return f"(timeout after {timeout_seconds}s reaching PyPI)"
        except urllib.error.URLError as exc:
            return f"(network error reaching PyPI: {exc.reason})"

        info = data.get("info", {}) or {}
        urls = data.get("urls") or []

        latest = info.get("version") or "?"
        summary = info.get("summary") or "(no summary)"
        project_url = (
            info.get("project_url")
            or info.get("home_page")
            or "(no URL)"
        )
        author = info.get("author") or info.get("author_email") or "(unknown)"

        return (
            f"package: {package}\n"
            f"latest version: {latest}\n"
            f"summary: {summary}\n"
            f"project url: {project_url}\n"
            f"author: {author}\n"
            f"release files: {len(urls)}"
        )

    return docs_lookup_tool