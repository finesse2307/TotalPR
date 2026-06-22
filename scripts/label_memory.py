"""Interactive CLI for labeling memories as accepted or rejected.

Walks through memories with ``was_accepted IS NULL`` one at a time. For each
one, prompt for a single character:

    a — accept (was_accepted = TRUE)
    r — reject (was_accepted = FALSE)
    s — skip (leave unlabeled, move on)
    q — quit (stop now, no changes to remaining)

Usage:
    python scripts/label_memory.py
    python scripts/label_memory.py --repo acme/api
    python scripts/label_memory.py --limit 5
"""

import argparse
import os
import sys
from pathlib import Path

from sentry.embedding import DeterministicMockEmbeddingClient
from sentry.memory import MemoryStore


def load_env_file(path: Path) -> None:
    """Minimal .env loader matching the rest of scripts/."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)


def build_dsn_from_env() -> str:
    user = os.environ.get("POSTGRES_USER", "sentry")
    password = os.environ.get("POSTGRES_PASSWORD", "sentry_dev_password")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "sentry")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=str, default=None,
        help="Only review memories from this repo.",
    )
    parser.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of memories to review (default 20).",
    )
    parser.add_argument(
        "--dsn", type=str, default=None,
        help="Postgres DSN. Defaults to building from POSTGRES_* env vars.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))
    dsn = args.dsn or build_dsn_from_env()

    # The embedder isn't used for label() or list_unlabeled() — only store().
    # Pass a mock so the constructor is satisfied without depending on VOYAGE_API_KEY.
    store = MemoryStore(
        dsn=dsn,
        embedder=DeterministicMockEmbeddingClient(dimension=1024),
    )

    unlabeled = store.list_unlabeled(repo=args.repo, limit=args.limit)
    if not unlabeled:
        scope = f" in {args.repo}" if args.repo else ""
        print(f"No unlabeled memories{scope}.")
        return 0

    print(f"Found {len(unlabeled)} unlabeled memory/memories.")
    print("Keys: [a]ccept  [r]eject  [s]kip  [q]uit")
    print()

    labeled = 0
    for i, m in enumerate(unlabeled, start=1):
        print(f"--- {i}/{len(unlabeled)} | id={m.id} | repo={m.repo} ---")
        print(f"  [{m.severity.value}/{m.category.value}] {m.finding_text}")
        print()

        while True:
            choice = input("Label (a/r/s/q): ").strip().lower()
            if choice in ("a", "r", "s", "q"):
                break
            print("Invalid. Use a, r, s, or q.")

        if choice == "a":
            store.label(m.id, was_accepted=True)
            labeled += 1
            print("  -> ACCEPTED\n")
        elif choice == "r":
            store.label(m.id, was_accepted=False)
            labeled += 1
            print("  -> REJECTED\n")
        elif choice == "s":
            print("  -> skipped\n")
        else:  # q
            print("  -> quitting")
            break

    print(f"Labeled {labeled} memory/memories.")
    return 0


if __name__ == "__main__":
    sys.exit(main())