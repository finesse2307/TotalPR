"""Sentry's memory store: diff-conditioned review memories in Postgres+pgvector.

Stores ``(diff, finding, accept/reject)`` tuples with embeddings, retrieves
nearest-neighbor matches by cosine distance, and supports manual labeling.

Embedding is delegated to an ``EmbeddingClient`` so the store doesn't know
about Voyage or any specific embedding service — it only requires vectors of
the configured dimension.
"""

from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector  # type: ignore[import-untyped]
from pydantic import BaseModel

from sentry.embedding import EmbeddingClient
from sentry.state import Category, Finding, Severity


class Memory(BaseModel):
    """One stored review memory.

    ``similarity`` is populated only by ``retrieve_similar`` (it's the cosine
    similarity to the query, in ``[-1, 1]``) and is ``None`` after a fresh
    ``store`` call.
    """

    id: int
    repo: str
    diff_text: str
    finding_text: str
    category: Category
    severity: Severity
    was_accepted: bool | None
    created_at: datetime
    similarity: float | None = None


class MemoryStore:
    """Read/write store for review memories with vector similarity search."""

    def __init__(self, *, dsn: str, embedder: EmbeddingClient) -> None:
        self._dsn = dsn
        self._embedder = embedder

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self._dsn)
        register_vector(conn)
        return conn

    def store(
        self,
        *,
        repo: str,
        diff_text: str,
        finding_text: str,
        category: Category | str,
        severity: Severity | str,
        was_accepted: bool | None = None,
    ) -> int:
        """Embed ``diff_text``, insert a row, return its id."""
        [embedding] = self._embedder.embed([diff_text])
        cat = category.value if isinstance(category, Category) else category
        sev = severity.value if isinstance(severity, Severity) else severity

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO memories
                    (repo, diff_text, finding_text, category, severity,
                     was_accepted, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (repo, diff_text, finding_text, cat, sev,
                 was_accepted, embedding),
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("INSERT...RETURNING returned no row")
            return int(row[0])
    
    def store_findings(
        self,
        *,
        repo: str,
        diff_text: str,
        findings: list[Finding],
        was_accepted: bool | None = None,
    ) -> list[int]:
        """Store multiple Findings sharing one diff. Embeds the diff once.

        All inserted rows get the same ``was_accepted`` value (default ``None``)
        since they share a review batch. To label individual findings
        differently later, use ``label(id, was_accepted=...)`` per id.
        """
        if not findings:
            return []

        [embedding] = self._embedder.embed([diff_text])
        ids: list[int] = []
        with self._connect() as conn, conn.cursor() as cur:
            for f in findings:
                cur.execute(
                    """
                    INSERT INTO memories
                        (repo, diff_text, finding_text, category, severity,
                         was_accepted, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (repo, diff_text, f.message, f.category.value,
                     f.severity.value, was_accepted, embedding),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("INSERT...RETURNING returned no row")
                ids.append(int(row[0]))
        return ids

    def retrieve_similar(
        self,
        *,
        diff_text: str,
        repo: str | None = None,
        k: int = 3,
        only_labeled: bool = False,
    ) -> list[Memory]:
        """Return ``k`` memories most similar to ``diff_text`` by cosine similarity.

        Args:
            diff_text: Query text. Embedded once and matched against stored rows.
            repo: If set, only memories from this repo are returned.
            k: Number of memories to return.
            only_labeled: If True, exclude memories with ``was_accepted IS NULL``.
        """
        if k <= 0:
            return []

        [embedding] = self._embedder.embed([diff_text])

        where_clauses: list[str] = []
        params: list[Any] = [embedding]
        if repo is not None:
            where_clauses.append("repo = %s")
            params.append(repo)
        if only_labeled:
            where_clauses.append("was_accepted IS NOT NULL")

        where_sql = (
            " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        )
        sql = f"""
            SELECT id, repo, diff_text, finding_text, category, severity,
                   was_accepted, created_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            {where_sql}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params.append(embedding)
        params.append(k)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            Memory(
                id=row[0],
                repo=row[1],
                diff_text=row[2],
                finding_text=row[3],
                category=row[4],
                severity=row[5],
                was_accepted=row[6],
                created_at=row[7],
                similarity=float(row[8]),
            )
            for row in rows
        ]

    def label(self, memory_id: int, *, was_accepted: bool) -> None:
        """Mark a memory as accepted or rejected.

        Raises:
            KeyError: if no memory exists with the given id.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE memories SET was_accepted = %s WHERE id = %s",
                (was_accepted, memory_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"No memory with id {memory_id}")
    
    def list_unlabeled(
        self,
        *,
        repo: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """Return unlabeled memories (oldest first), optionally filtered by repo.

        Used by the labeling CLI to walk through ``was_accepted IS NULL`` rows.
        ``similarity`` is left as ``None`` on returned Memory records since
        this query doesn't compare against any embedding.
        """
        where_clauses = ["was_accepted IS NULL"]
        params: list[Any] = []
        if repo is not None:
            where_clauses.append("repo = %s")
            params.append(repo)
        where_sql = " WHERE " + " AND ".join(where_clauses)
        sql = f"""
            SELECT id, repo, diff_text, finding_text, category, severity,
                   was_accepted, created_at
            FROM memories
            {where_sql}
            ORDER BY created_at ASC
            LIMIT %s
        """
        params.append(limit)

        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            Memory(
                id=row[0],
                repo=row[1],
                diff_text=row[2],
                finding_text=row[3],
                category=row[4],
                severity=row[5],
                was_accepted=row[6],
                created_at=row[7],
            )
            for row in rows
        ]