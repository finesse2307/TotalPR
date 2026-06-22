"""Integration tests for MemoryStore.

Talk to a real Postgres at the DSN given by ``POSTGRES_TEST_DSN`` or the
default localhost:5432 from docker-compose. Each test truncates ``memories``
before running so tests don't leak data into one another. If Postgres is not
reachable, the whole module is skipped.
"""

import os

import psycopg
import pytest

from sentry.embedding import DeterministicMockEmbeddingClient
from sentry.memory import MemoryStore
from sentry.state import Category, Severity

DEFAULT_DSN = "postgresql://sentry:sentry_dev_password@localhost:5432/sentry"


@pytest.fixture(scope="session")
def dsn() -> str:
    """Return a usable Postgres DSN, or skip the module if unreachable."""
    test_dsn = os.environ.get("POSTGRES_TEST_DSN", DEFAULT_DSN)
    try:
        with psycopg.connect(test_dsn, connect_timeout=2):
            pass
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable: {exc}")
    return test_dsn


@pytest.fixture
def store(dsn: str) -> MemoryStore:
    """Return a MemoryStore against a freshly truncated memories table."""
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE memories RESTART IDENTITY")
    return MemoryStore(
        dsn=dsn,
        embedder=DeterministicMockEmbeddingClient(dimension=1024),
    )


def test_store_returns_id_and_persists_row(
    store: MemoryStore, dsn: str
) -> None:
    """store() returns the new row's id and the row is queryable."""
    mid = store.store(
        repo="acme/api",
        diff_text="some diff",
        finding_text="found something",
        category=Category.SECURITY,
        severity=Severity.HIGH,
    )
    assert isinstance(mid, int) and mid > 0

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT repo, finding_text, was_accepted FROM memories WHERE id = %s",
            (mid,),
        )
        row = cur.fetchone()
    assert row == ("acme/api", "found something", None)


def test_store_persists_all_fields(store: MemoryStore) -> None:
    """All fields round-trip through store + retrieve."""
    mid = store.store(
        repo="acme/api",
        diff_text="diff body",
        finding_text="msg",
        category=Category.BUG,
        severity=Severity.MEDIUM,
        was_accepted=True,
    )
    [m] = store.retrieve_similar(diff_text="diff body", k=1)
    assert m.id == mid
    assert m.repo == "acme/api"
    assert m.diff_text == "diff body"
    assert m.finding_text == "msg"
    assert m.category == Category.BUG
    assert m.severity == Severity.MEDIUM
    assert m.was_accepted is True


def test_retrieve_returns_identity_match_first(store: MemoryStore) -> None:
    """A stored diff is the closest match when queried with the same text."""
    store.store(
        repo="r", diff_text="alpha", finding_text="a",
        category=Category.BUG, severity=Severity.LOW,
    )
    store.store(
        repo="r", diff_text="beta", finding_text="b",
        category=Category.BUG, severity=Severity.LOW,
    )
    results = store.retrieve_similar(diff_text="alpha", k=2)
    assert len(results) == 2
    assert results[0].diff_text == "alpha"


def test_retrieve_respects_k(store: MemoryStore) -> None:
    for i in range(5):
        store.store(
            repo="r", diff_text=f"d{i}", finding_text=f"f{i}",
            category=Category.BUG, severity=Severity.LOW,
        )
    assert len(store.retrieve_similar(diff_text="d0", k=2)) == 2
    assert len(store.retrieve_similar(diff_text="d0", k=10)) == 5


def test_retrieve_filters_by_repo(store: MemoryStore) -> None:
    store.store(
        repo="acme", diff_text="x", finding_text="a",
        category=Category.BUG, severity=Severity.LOW,
    )
    store.store(
        repo="other", diff_text="x", finding_text="b",
        category=Category.BUG, severity=Severity.LOW,
    )
    results = store.retrieve_similar(diff_text="x", repo="acme", k=5)
    assert len(results) == 1
    assert results[0].repo == "acme"


def test_retrieve_only_labeled_excludes_null(store: MemoryStore) -> None:
    store.store(
        repo="r", diff_text="x", finding_text="labeled",
        category=Category.BUG, severity=Severity.LOW,
        was_accepted=True,
    )
    store.store(
        repo="r", diff_text="x", finding_text="unlabeled",
        category=Category.BUG, severity=Severity.LOW,
    )
    results = store.retrieve_similar(diff_text="x", k=5, only_labeled=True)
    assert len(results) == 1
    assert results[0].finding_text == "labeled"


def test_retrieve_similarity_is_in_valid_range(store: MemoryStore) -> None:
    """Returned similarity is a cosine value in [-1, 1]."""
    store.store(
        repo="r", diff_text="x", finding_text="a",
        category=Category.BUG, severity=Severity.LOW,
    )
    [m] = store.retrieve_similar(diff_text="y", k=1)
    assert m.similarity is not None
    assert -1.0 <= m.similarity <= 1.0


def test_label_updates_was_accepted(store: MemoryStore) -> None:
    """label() flips was_accepted between True, False, and back."""
    mid = store.store(
        repo="r", diff_text="x", finding_text="msg",
        category=Category.BUG, severity=Severity.LOW,
    )
    store.label(mid, was_accepted=True)
    [m1] = store.retrieve_similar(diff_text="x", k=1)
    assert m1.was_accepted is True

    store.label(mid, was_accepted=False)
    [m2] = store.retrieve_similar(diff_text="x", k=1)
    assert m2.was_accepted is False


def test_label_unknown_id_raises(store: MemoryStore) -> None:
    """Labeling a nonexistent id raises KeyError."""
    with pytest.raises(KeyError, match="999"):
        store.label(999, was_accepted=True)

def test_list_unlabeled_returns_oldest_first(store: MemoryStore) -> None:
    """list_unlabeled returns NULL-only rows, oldest first, respecting repo and limit."""
    a = store.store(
        repo="acme", diff_text="a", finding_text="first",
        category=Category.BUG, severity=Severity.LOW,
    )
    store.store(
        repo="acme", diff_text="b", finding_text="labeled",
        category=Category.BUG, severity=Severity.LOW, was_accepted=True,
    )
    c = store.store(
        repo="acme", diff_text="c", finding_text="second",
        category=Category.BUG, severity=Severity.LOW,
    )
    store.store(
        repo="other", diff_text="d", finding_text="wrong repo",
        category=Category.BUG, severity=Severity.LOW,
    )

    results = store.list_unlabeled(repo="acme", limit=10)

    ids = [m.id for m in results]
    assert ids == [a, c]  # b excluded (labeled); 'other' excluded (repo filter)
    assert results[0].finding_text == "first"
    assert results[1].finding_text == "second"

def test_store_findings_embeds_once_and_returns_ids(
    store: MemoryStore, dsn: str
) -> None:
    """store_findings embeds the diff exactly once and inserts one row per finding."""
    from sentry.state import Finding

    findings = [
        Finding(
            category=Category.SECURITY, severity=Severity.HIGH,
            file="a.py", message="first issue",
        ),
        Finding(
            category=Category.BUG, severity=Severity.MEDIUM,
            file="b.py", message="second issue",
        ),
    ]

    # The fixture's embedder is a DeterministicMockEmbeddingClient — it records
    # every call so we can assert it was called exactly once for the batch.
    ids = store.store_findings(
        repo="acme",
        diff_text="shared diff",
        findings=findings,
    )

    assert len(ids) == 2
    # Each id is a fresh row
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT finding_text FROM memories WHERE id = ANY(%s) ORDER BY id",
            (ids,),
        )
        texts = [r[0] for r in cur.fetchall()]
    assert texts == ["first issue", "second issue"]

    # Confirm only one embed call was made for the whole batch.
    # isinstance narrows the Protocol-typed _embedder to the mock so mypy
    # accepts the .calls attribute access.
    embedder = store._embedder
    assert isinstance(embedder, DeterministicMockEmbeddingClient)
    assert len(embedder.calls) == 1
    assert embedder.calls[0] == ["shared diff"]