"""Tests for the embedding clients."""

from unittest.mock import MagicMock

import pytest

from sentry.embedding import (
    DeterministicMockEmbeddingClient,
    VoyageEmbeddingClient,
)


def test_mock_is_deterministic_across_instances() -> None:
    """Same text yields the same vector across calls and instances."""
    [vec_a] = DeterministicMockEmbeddingClient().embed(["hello"])
    [vec_b] = DeterministicMockEmbeddingClient().embed(["hello"])
    assert vec_a == vec_b


def test_mock_distinct_inputs_produce_distinct_outputs() -> None:
    """Different texts produce different vectors."""
    [a, b] = DeterministicMockEmbeddingClient().embed(["alpha", "beta"])
    assert a != b


def test_mock_respects_dimension() -> None:
    """The dimension constructor argument controls vector length."""
    [vec] = DeterministicMockEmbeddingClient(dimension=64).embed(["hello"])
    assert len(vec) == 64


def test_mock_records_calls() -> None:
    """Each embed() call is recorded with its inputs for assertion."""
    mock = DeterministicMockEmbeddingClient()
    mock.embed(["one", "two"])
    mock.embed(["three"])
    assert mock.calls == [["one", "two"], ["three"]]


def test_mock_empty_input_returns_empty_list() -> None:
    assert DeterministicMockEmbeddingClient().embed([]) == []


def test_voyage_passes_correct_args_to_sdk() -> None:
    """VoyageEmbeddingClient forwards texts, model, input_type, and dimension."""
    fake_result = MagicMock()
    fake_result.embeddings = [[0.1] * 1024, [0.2] * 1024]
    fake_client = MagicMock()
    fake_client.embed.return_value = fake_result

    voyage = VoyageEmbeddingClient(client=fake_client)
    result = voyage.embed(["hello", "world"])

    assert result == [[0.1] * 1024, [0.2] * 1024]
    fake_client.embed.assert_called_once_with(
        texts=["hello", "world"],
        model="voyage-code-3",
        input_type="document",
        output_dimension=1024,
    )


def test_voyage_empty_input_skips_api_call() -> None:
    """Empty input list short-circuits without touching the SDK."""
    fake_client = MagicMock()
    voyage = VoyageEmbeddingClient(client=fake_client)
    assert voyage.embed([]) == []
    fake_client.embed.assert_not_called()


def test_voyage_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing without an injected client or VOYAGE_API_KEY raises."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageEmbeddingClient()


def test_voyage_custom_model_and_dimension() -> None:
    """Model and dimension overrides flow through to the SDK call."""
    fake_result = MagicMock()
    fake_result.embeddings = [[0.1] * 512]
    fake_client = MagicMock()
    fake_client.embed.return_value = fake_result

    voyage = VoyageEmbeddingClient(
        client=fake_client, model="voyage-code-2", dimension=512
    )
    voyage.embed(["x"])

    fake_client.embed.assert_called_once_with(
        texts=["x"],
        model="voyage-code-2",
        input_type="document",
        output_dimension=512,
    )