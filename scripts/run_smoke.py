"""Manual smoke test: run the agent against one eval case using real Anthropic.
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from sentry.anthropic_client import AnthropicLLMClient
from sentry.budget import BudgetedLLMClient
from sentry.cache import SQLiteCacheLLMClient
from sentry.graph import build_graph
from sentry.nodes.run_tool import ToolRegistry
from sentry.posting import NoopPoster
from sentry.state import AgentState, PRMetadata, ToolName
from sentry.telemetry import init_tracing, run_span
from sentry.tools.docs_lookup_tool import make_docs_lookup_tool
from sentry.tools.ripgrep_tool import make_ripgrep_tool
from sentry.tools.ruff_tool import make_ruff_tool
from sentry.tools.semgrep_tool import make_semgrep_tool
from sentry.workspace import materialize_workspace


def load_env_file(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line. Strips surrounding quotes."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def synth_unified_diff(filename: str, diff: str) -> str:
    """Wrap a hunks-only eval-set diff with a standard unified-diff file header."""
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"{diff}\n"
    )


def pick_case(
    cases: list[dict[str, Any]], selector: str | None
) -> dict[str, Any]:
    if selector is None:
        return cases[0]
    if selector.isdigit():
        return cases[int(selector) - 1]
    for c in cases:
        if c["id"] == selector:
            return c
    raise SystemExit(
        f"No case matches '{selector}'. Known ids: "
        f"{[c['id'] for c in cases]}"
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", nargs="?", help="Case id (e.g. case-001) or 1-indexed position."
    )
    parser.add_argument(
        "--cap", type=float, default=0.50,
        help="Spend cap in USD (default 0.50).",
    )
    parser.add_argument(
        "--cache-db", type=Path, default=Path(".cache/llm.sqlite"),
        help="SQLite cache file (default .cache/llm.sqlite).",
    )
    parser.add_argument(
        "--eval-set", type=Path, default=Path("evals/eval_set.json"),
        help="Path to eval_set.json.",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Enable console export of OpenTelemetry spans.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))
    init_tracing(enable_console=args.trace)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY not set in env or .env",
            file=sys.stderr,
        )
        return 1

    real_llm = AnthropicLLMClient()
    cached_llm = SQLiteCacheLLMClient(
        real_llm, db_path=args.cache_db, namespace=real_llm.model
    )
    budgeted_llm = BudgetedLLMClient(cached_llm, cap_usd=args.cap)

    eval_data = json.loads(args.eval_set.read_text())
    case = pick_case(eval_data["cases"], args.case)
    raw_diff = synth_unified_diff(case["filename"], case["diff"])

    print(f"Running case: {case['id']} ({case['name']})")
    print(f"Model:    {real_llm.model}")
    print(f"Cap:      ${args.cap:.2f}")
    print(f"Cache:    {args.cache_db}")
    print()

    initial = AgentState(
        pr=PRMetadata(
            repo="local/smoke",
            pr_number=0,
            head_sha="smoke",
            base_sha="smoke",
            author="smoke",
            title=case["name"],
        ),
        raw_diff=raw_diff,
    )

    with tempfile.TemporaryDirectory(prefix="sentry-ws-") as ws_str:
        workspace = Path(ws_str)
        materialize_workspace(raw_diff, workspace)

        tools: ToolRegistry = {
            ToolName.RUFF: make_ruff_tool(),
            ToolName.SEMGREP: make_semgrep_tool(workspace_path=workspace),
            ToolName.RIPGREP: make_ripgrep_tool(workspace_path=workspace),
            ToolName.DOCS_LOOKUP: make_docs_lookup_tool(),
        }

        graph = build_graph(llm=budgeted_llm, tools=tools, poster=NoopPoster())
        with run_span("review_pr"):
            final = graph.invoke(initial)

    print("=" * 72)
    print("REVIEW BODY")
    print("=" * 72)
    print(final.get("review_body") or "(none)")
    print()

    print("=" * 72)
    print("FINDINGS")
    print("=" * 72)
    findings = final.get("findings") or []
    for f in findings:
        loc = f"{f.file}:{f.line}" if f.line else f.file
        print(f"  [{f.severity.value}/{f.category.value}] {loc} — {f.message}")
    if not findings:
        print("(none)")
    print()

    print("=" * 72)
    print("STATS")
    print("=" * 72)
    print(f"Cache:    {cached_llm.hits} hits / {cached_llm.misses} misses")
    print(
        f"Tokens:   {budgeted_llm.total_input_tokens} input / "
        f"{budgeted_llm.total_output_tokens} output"
    )
    print(
        f"Spend:    ${budgeted_llm.total_spend_usd:.4f} (cap ${args.cap:.2f})"
    )
    print(f"Post:     {final.get('post_status')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())