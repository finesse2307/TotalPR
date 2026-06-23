"""FastAPI service entry point for the TotalPR code-review agent.

Hosts:
- ``GET /health`` — liveness probe used by Fly.io and other health checkers
- ``POST /webhook`` — receives GitHub webhook deliveries, verifies the HMAC
  signature, and acknowledges receipt. Dispatching to the agent graph lands
  in a later step.
"""

import hashlib
import hmac
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic_settings import BaseSettings, SettingsConfigDict

from sentry.dedupe import Deduper, NullDeduper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

def _load_env_file(path: Path) -> None:
    """Populate os.environ from a .env file. Matches the loader used by scripts/."""
    import os

    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), value)

def _materialize_private_key() -> None:
    """Write GITHUB_PRIVATE_KEY contents to GITHUB_PRIVATE_KEY_PATH if both are set.

    Fly secrets can't ship file contents directly — we pass the PEM as an env
    var and write it to the path the rest of the code expects.
    """
    import os

    contents = os.environ.get("GITHUB_PRIVATE_KEY")
    target = os.environ.get("GITHUB_PRIVATE_KEY_PATH")
    if not contents or not target:
        return
    target_path = Path(target)
    if target_path.exists():
        return
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(contents)
    target_path.chmod(0o600)


_materialize_private_key()


# Load .env on import so non-Settings env vars (ANTHROPIC_API_KEY, VOYAGE_API_KEY,
# POSTGRES_*) are available to the rest of the agent stack at dispatch time.
_load_env_file(Path(".env"))


logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Runtime configuration loaded from environment + .env file."""

    github_app_id: str
    github_private_key_path: Path
    github_webhook_secret: str
    github_installation_id: int
    github_test_repo: str
    reviews_enabled: bool = True
    redis_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


def verify_signature(
    body: bytes, signature_header: str, secret: str
) -> bool:
    """Constant-time HMAC-SHA256 verification of a GitHub webhook signature.

    GitHub sends the signature as ``sha256=<hex>`` in ``X-Hub-Signature-256``.
    Returns ``False`` for absent or malformed headers and never raises.
    """
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def create_app(
    settings: Settings | None = None,
    dispatch_fn: "Callable[[dict[str, Any], Settings], None] | None" = None,
    deduper: Deduper | None = None,
) -> FastAPI:
    """FastAPI factory. ``dispatch_fn`` and ``deduper`` are injectable for tests."""
    from sentry.api.dispatch import dispatch_pull_request

    cfg = settings or Settings()  # type: ignore[call-arg]
    dispatch = dispatch_fn or dispatch_pull_request
    if deduper is not None:
        dedupe: Deduper = deduper
    elif cfg.redis_url:
        import redis as redis_lib

        from sentry.dedupe import RedisDeduper
        dedupe = RedisDeduper(
            client=redis_lib.Redis.from_url(cfg.redis_url, decode_responses=True)
        )
        logger.info("dedupe: redis enabled")
    else:
        dedupe = NullDeduper()
        logger.info("dedupe: disabled (no REDIS_URL)")
    app = FastAPI(title="TotalPR", version="0.4.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhook")
    async def webhook(
        request: Request, background: BackgroundTasks
    ) -> dict[str, Any]:
        body = await request.body()
        signature = request.headers.get("X-Hub-Signature-256", "")

        if not verify_signature(body, signature, cfg.github_webhook_secret):
            raise HTTPException(status_code=401, detail="invalid signature")

        event_type = request.headers.get("X-GitHub-Event", "unknown")
        delivery_id = request.headers.get("X-GitHub-Delivery", "unknown")

        try:
            payload = json.loads(body) if body else {}
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, detail="invalid json"
            ) from None

        action = payload.get("action", "")
        repo = payload.get("repository", {}).get("full_name", "")

        logger.info(
            "webhook received event=%s action=%s repo=%s delivery=%s",
            event_type, action, repo, delivery_id,
        )

        if event_type == "pull_request":
            if not dedupe.mark_seen(delivery_id):
                logger.info(
                    "duplicate delivery skipped: delivery=%s", delivery_id
                )
                return {"status": "ok", "event": event_type, "duplicate": True}
            logger.info("scheduling dispatch for delivery=%s", delivery_id)
            background.add_task(dispatch, payload, cfg)
        else:
            logger.info(
                "no dispatch: event_type=%s not in {pull_request}", event_type
            )

        return {"status": "ok", "event": event_type, "action": action}

    return app


def get_app() -> FastAPI:
    """Lazy app factory. Uvicorn calls this; tests don't.

    Constructing the FastAPI app eagerly at import would call
    ``Settings()``, which fails if no ``.env`` is present and no GitHub
    secrets are in the environment — breaking ``pytest`` and fresh-clone
    collection. Deferring lets tests import ``create_app`` without
    instantiating it.
    """
    return create_app()


# Uvicorn target. Use the factory mode: ``uvicorn sentry.api.main:get_app --factory``
app = None