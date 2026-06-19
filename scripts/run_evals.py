"""Eval harness: run every eval_set.json case through the agent and score it.

Per case, builds a fresh LLM stack (so spend and cache counters are per-case),
materializes a workspace, invokes the graph, and matches each predicted finding
against the case's ``expected_findings``:
"""

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentry.anthropic_client import AnthropicLLMClient
from sentry.budget import BudgetedLLMClient, BudgetExceededError
from sentry.cache import SQLiteCacheLLMClient
from sentry.graph import build_graph
from sentry.nodes.run_tool import ToolRegistry
from sentry.posting import NoopPoster
from sentry.state import AgentState, Finding, PRMetadata, ToolName
from sentry.tools.docs_lookup_tool import make_docs_lookup_tool
from sentry.tools.ripgrep_tool import make_ripgrep_tool
from sentry.tools.ruff_tool import make_ruff_tool
from sentry.tools.semgrep_tool import make_semgrep_tool
from sentry.workspace import materialize_workspace


def load_env_file(path: Path) -> None:
    """Minimal .env loader: KEY=VALUE per line, strips surrounding quotes."""
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
    return (
        f"diff --git a/{filename} b/{filename}\n"
        f"--- a/{filename}\n"
        f"+++ b/{filename}\n"
        f"{diff}\n"
    )


def _matches(pred: Finding, exp: dict[str, Any]) -> bool:
    """Whether a predicted finding satisfies an expected_findings entry.

    Match criteria: category equality plus all ``must_mention`` substrings
    present in the message (case-insensitive). Severity is NOT a match gate
    because severity calibration is inherently subjective; it's tracked as a
    separate metric on matched pairs.
    """
    if pred.category.value != exp["category"]:
        return False
    msg_lower = pred.message.lower()
    return all(
        keyword.lower() in msg_lower
        for keyword in exp.get("must_mention", [])
    )


def _score(
    findings: list[Finding], expected: list[dict[str, Any]]
) -> tuple[int, int, int, list[dict[str, Any]], int]:
    """Greedy match. Returns (tp, fp, fn, unmatched_expected, severity_agreements)."""
    unmatched_idx = list(range(len(expected)))
    matched_predictions = 0
    severity_agreements = 0

    for pred in findings:
        for slot, exp_idx in enumerate(unmatched_idx):
            exp = expected[exp_idx]
            if _matches(pred, exp):
                matched_predictions += 1
                if pred.severity.value == exp["severity"]:
                    severity_agreements += 1
                del unmatched_idx[slot]
                break

    tp = matched_predictions
    fp = len(findings) - tp
    fn = len(unmatched_idx)
    return tp, fp, fn, [expected[i] for i in unmatched_idx], severity_agreements


def _finding_to_dict(f: Finding) -> dict[str, Any]:
    return {
        "file": f.file,
        "line": f.line,
        "category": f.category.value,
        "severity": f.severity.value,
        "message": f.message,
    }


def _percentile(xs: list[int], pct: int) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = min(int(len(s) * pct / 100), len(s) - 1)
    return s[k]


def _run_case(
    case: dict[str, Any],
    *,
    cache_db: Path,
    per_case_cap: float,
) -> dict[str, Any]:
    real_llm = AnthropicLLMClient()
    cached_llm = SQLiteCacheLLMClient(
        real_llm, db_path=cache_db, namespace=real_llm.model
    )
    budgeted_llm = BudgetedLLMClient(cached_llm, cap_usd=per_case_cap)

    raw_diff = synth_unified_diff(case["filename"], case["diff"])
    initial = AgentState(
        pr=PRMetadata(
            repo="local/eval",
            pr_number=0,
            head_sha="eval",
            base_sha="eval",
            author="eval",
            title=case["name"],
        ),
        raw_diff=raw_diff,
    )

    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="sentry-eval-") as ws_str:
            workspace = Path(ws_str)
            materialize_workspace(raw_diff, workspace)
            tools: ToolRegistry = {
                ToolName.RUFF: make_ruff_tool(),
                ToolName.SEMGREP: make_semgrep_tool(workspace_path=workspace),
                ToolName.RIPGREP: make_ripgrep_tool(workspace_path=workspace),
                ToolName.DOCS_LOOKUP: make_docs_lookup_tool(),
            }
            graph = build_graph(
                llm=budgeted_llm, tools=tools, poster=NoopPoster()
            )
            final = graph.invoke(initial)
    except BudgetExceededError as exc:
        return {
            "id": case["id"],
            "name": case["name"],
            "status": "budget_exceeded",
            "error": str(exc),
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "spend_usd": budgeted_llm.total_spend_usd,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(case.get("expected_findings", [])),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "id": case["id"],
            "name": case["name"],
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "duration_ms": int((time.perf_counter() - start) * 1000),
            "spend_usd": budgeted_llm.total_spend_usd,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": len(case.get("expected_findings", [])),
        }

    duration_ms = int((time.perf_counter() - start) * 1000)
    findings = final.get("findings") or []
    expected = case.get("expected_findings", [])
    tp, fp, fn, unmatched, sev_agree = _score(findings, expected)

    return {
        "id": case["id"],
        "name": case["name"],
        "status": "ok",
        "duration_ms": duration_ms,
        "spend_usd": budgeted_llm.total_spend_usd,
        "input_tokens": budgeted_llm.total_input_tokens,
        "output_tokens": budgeted_llm.total_output_tokens,
        "cache_hits": cached_llm.hits,
        "cache_misses": cached_llm.misses,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "predicted_findings": [_finding_to_dict(f) for f in findings],
        "unmatched_expected": unmatched,
        "severity_agreements": sev_agree,
    }


def _aggregate(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in case_results if r["status"] == "ok"]
    tp = sum(r["true_positives"] for r in ok)
    fp = sum(r["false_positives"] for r in ok)
    fn = sum(r["false_negatives"] for r in ok)
    sev_agree = sum(r.get("severity_agreements", 0) for r in ok)
    sev_total = sum(r["true_positives"] for r in ok)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    durations = [r["duration_ms"] for r in ok]
    return {
        "cases_total": len(case_results),
        "cases_ok": len(ok),
        "cases_failed": len(case_results) - len(ok),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_spend_usd": sum(r["spend_usd"] for r in case_results),
        "total_input_tokens": sum(r.get("input_tokens", 0) for r in ok),
        "total_output_tokens": sum(r.get("output_tokens", 0) for r in ok),
        "p50_duration_ms": _percentile(durations, 50),
        "p95_duration_ms": _percentile(durations, 95),
        "severity_agreement_rate": sev_agree / sev_total if sev_total else 0.0,
    }


def _print_summary(
    summary: dict[str, Any], case_results: list[dict[str, Any]]
) -> None:
    print()
    print("=" * 72)
    print("PER-CASE")
    print("=" * 72)
    for r in case_results:
        if r["status"] != "ok":
            print(f"  {r['id']:<10} FAILED ({r['status']}): {r.get('error', '')}")
            continue
        print(
            f"  {r['id']:<10} "
            f"tp={r['true_positives']} fp={r['false_positives']} "
            f"fn={r['false_negatives']}  "
            f"${r['spend_usd']:.4f}  {r['duration_ms']:>5}ms"
        )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Cases:     {summary['cases_ok']}/{summary['cases_total']} succeeded")
    print(f"Precision: {summary['precision']:.3f}")
    print(f"Recall:    {summary['recall']:.3f}")
    print(f"F1:        {summary['f1']:.3f}")
    print(
        f"TP/FP/FN:  {summary['true_positives']}/"
        f"{summary['false_positives']}/{summary['false_negatives']}"
    )
    print(
        f"SevAgree:  {summary['severity_agreement_rate']:.3f} "
        f"(of matched findings)"
    )
    print(f"Spend:     ${summary['total_spend_usd']:.4f}")
    print(
        f"Latency:   p50={summary['p50_duration_ms']}ms "
        f"p95={summary['p95_duration_ms']}ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-set", type=Path, default=Path("evals/eval_set.json"),
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("evals/results"),
    )
    parser.add_argument(
        "--cache-db", type=Path, default=Path(".cache/llm.sqlite"),
    )
    parser.add_argument(
        "--per-case-cap", type=float, default=0.20,
        help="Per-case spend cap in USD (default 0.20).",
    )
    parser.add_argument(
        "--total-cap", type=float, default=1.00,
        help="Total spend cap across the run (default 1.00).",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run cases whose id contains this substring.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in env or .env", file=sys.stderr)
        return 1

    eval_data = json.loads(args.eval_set.read_text())
    cases: list[dict[str, Any]] = eval_data["cases"]
    if args.filter:
        cases = [c for c in cases if args.filter in c["id"]]
        if not cases:
            print(f"No cases match filter '{args.filter}'", file=sys.stderr)
            return 1

    case_results: list[dict[str, Any]] = []
    total_spend = 0.0

    for case in cases:
        if total_spend >= args.total_cap:
            print(
                f"\nTotal cap ${args.total_cap:.2f} reached after "
                f"${total_spend:.4f}; skipping remaining cases."
            )
            break

        print(f"Running {case['id']:<10} ({case['name']})...", end=" ", flush=True)
        result = _run_case(
            case, cache_db=args.cache_db, per_case_cap=args.per_case_cap
        )
        case_results.append(result)
        total_spend += result["spend_usd"]

        if result["status"] == "ok":
            print(
                f"tp={result['true_positives']} "
                f"fp={result['false_positives']} "
                f"fn={result['false_negatives']}  "
                f"${result['spend_usd']:.4f}  {result['duration_ms']}ms"
            )
        else:
            print(f"FAILED ({result['status']}): {result.get('error', '')}")

    summary = _aggregate(case_results)
    _print_summary(summary, case_results)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.results_dir / f"run-{ts}.json"
    report_path.write_text(
        json.dumps(
            {"timestamp": ts, "summary": summary, "cases": case_results},
            indent=2,
        )
    )
    print(f"\nReport: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())