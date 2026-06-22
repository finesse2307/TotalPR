"""Webhook delivery deduplication using Redis SET NX EX.

GitHub retries failed deliveries (and connection-level resends can replay
successful ones). Each delivery carries an immutable ``X-GitHub-Delivery``
UUID — we use it as the dedupe key with a TTL covering GitHub's retry window.

``mark_seen`` is the only operation: it atomically reserves the key. If the
key already existed, the delivery is a duplicate and the caller should skip.

When no Redis client is configured (local dev without Upstash), the
``NullDeduper`` accepts every delivery — best-effort dedupe is acceptable
in dev. Production hands the dispatcher a real ``RedisDeduper``.
"""

from typing import Protocol

import redis

DEFAULT_TTL_SECONDS = 24 * 3600  # GitHub retries within ~8h; 24h is generous
KEY_PREFIX = "webhook:delivery:"


class Deduper(Protocol):
    """Atomically reserve a delivery id. Returns True if first time seen."""

    def mark_seen(self, delivery_id: str) -> bool:
        ...


class NullDeduper:
    """No-op deduper for local dev — every delivery is treated as new."""

    def mark_seen(self, delivery_id: str) -> bool:  # noqa: ARG002
        return True


class RedisDeduper:
    """SET NX EX-backed deduper. Returns True iff the key was newly set."""

    def __init__(
        self,
        *,
        client: redis.Redis,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        key_prefix: str = KEY_PREFIX,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._prefix = key_prefix

    def mark_seen(self, delivery_id: str) -> bool:
        """Return True if this is the first time we've seen ``delivery_id``.

        Uses ``SET key value NX EX ttl`` — atomic check-and-set. The redis
        client returns ``True`` on success (key was new), ``None`` if the
        key already existed.
        """
        key = self._prefix + delivery_id
        result = self._client.set(key, "1", nx=True, ex=self._ttl)
        return bool(result)