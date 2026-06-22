"""Embedding client protocol plus a Voyage AI implementation.

Mirrors the ``llm.py`` pattern: a small Protocol, a real implementation
(``VoyageEmbeddingClient``), and a deterministic mock for tests
(``DeterministicMockEmbeddingClient``). Call sites depend only on the
Protocol so production and test setups swap cleanly.
"""

import hashlib
import os
from typing import Any, Protocol

import voyageai


class EmbeddingClient(Protocol):
    """Convert text to fixed-dimension float vectors."""

    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text. Empty input → empty list."""
        ...


class VoyageEmbeddingClient:
    """EmbeddingClient backed by Voyage AI's hosted embedding API.

    Defaults to ``voyage-code-3`` at 1024 dimensions. The ``client`` parameter
    accepts a pre-constructed ``voyageai.Client`` for test injection; in
    production it's built from ``VOYAGE_API_KEY``.
    """

    DEFAULT_MODEL = "voyage-code-3"
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        dimension: int = DEFAULT_DIMENSION,
    ) -> None:
        if client is None:
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError(
                    "VOYAGE_API_KEY not set. Sign up at voyageai.com and "
                    "add the key to your .env file."
                )
            client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        self._client = client
        self.model = model
        self.dimension = dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(
            texts=texts,
            model=self.model,
            input_type="document",
            output_dimension=self.dimension,
        )
        return list(result.embeddings)


class DeterministicMockEmbeddingClient:
    """Deterministic mock embedder for tests.

    Hashes each input to SHA-256 and produces a fixed-dimension vector by
    scaling bytes to ``[-1, 1]``. Same text → same vector across calls and
    instances. Vectors are *not* semantically meaningful — two semantically
    similar texts won't have higher similarity than unrelated ones. Use real
    Voyage for similarity-quality tests; use this for plumbing.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode()).digest()
        return [(h[i % len(h)] - 128) / 128.0 for i in range(self.dimension)]