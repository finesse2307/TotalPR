"""Tests for Deduper implementations."""

from unittest.mock import MagicMock

from sentry.dedupe import KEY_PREFIX, NullDeduper, RedisDeduper


def test_null_always_accepts() -> None:
    """NullDeduper treats every delivery as new — best-effort dedupe."""
    d = NullDeduper()
    assert d.mark_seen("abc") is True
    assert d.mark_seen("abc") is True  # same id, still True


def test_redis_first_call_returns_true_and_sets_key() -> None:
    """First mark_seen returns True; uses SET NX EX with the configured TTL."""
    client = MagicMock()
    client.set.return_value = True

    d = RedisDeduper(client=client, ttl_seconds=3600)
    assert d.mark_seen("delivery-1") is True

    client.set.assert_called_once_with(
        KEY_PREFIX + "delivery-1", "1", nx=True, ex=3600
    )


def test_redis_duplicate_returns_false() -> None:
    """Redis returns None when the key already exists; we surface that as False."""
    client = MagicMock()
    client.set.return_value = None

    d = RedisDeduper(client=client)
    assert d.mark_seen("delivery-1") is False


def test_redis_custom_prefix() -> None:
    """Custom key_prefix is honored."""
    client = MagicMock()
    client.set.return_value = True

    d = RedisDeduper(client=client, key_prefix="custom:")
    d.mark_seen("xyz")

    args = client.set.call_args.args
    assert args[0] == "custom:xyz"