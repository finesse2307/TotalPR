"""Apply migrations to the Sentry Postgres database.

Reads each ``.sql`` file in ``migrations/`` in lexicographic order and
executes its contents against the database configured by ``POSTGRES_*`` env
vars or by ``--dsn``. Idempotent: every migration uses ``CREATE ... IF NOT
EXISTS``, so re-running is safe.

Usage:
    docker compose up -d postgres   # bring up the database
    python scripts/init_db.py       # apply all migrations
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg


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
    """Build a libpq DSN string from POSTGRES_* env vars (matches .env.example)."""
    user = os.environ.get("POSTGRES_USER", "sentry")
    password = os.environ.get("POSTGRES_PASSWORD", "sentry_dev_password")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "sentry")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--migrations-dir", type=Path, default=Path("migrations"),
    )
    parser.add_argument(
        "--dsn", type=str, default=None,
        help="Postgres DSN. Defaults to building from POSTGRES_* env vars.",
    )
    args = parser.parse_args()

    load_env_file(Path(".env"))
    dsn = args.dsn or build_dsn_from_env()

    sql_files = sorted(args.migrations_dir.glob("*.sql"))
    if not sql_files:
        print(f"No .sql files in {args.migrations_dir}", file=sys.stderr)
        return 1

    # Redact password from the printed DSN — only show host/port/db.
    safe_target = dsn.split("@")[-1] if "@" in dsn else dsn
    print(f"Connecting to {safe_target}...")

    with psycopg.connect(dsn, autocommit=False) as conn:
        for path in sql_files:
            sql = path.read_text()
            print(f"  Applying {path.name}...", end=" ", flush=True)
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            print("ok")

    print(f"\nApplied {len(sql_files)} migration(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())