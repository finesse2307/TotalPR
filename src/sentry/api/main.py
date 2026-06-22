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
) -> FastAPI:
    """FastAPI factory. ``dispatch_fn`` is injectable for tests."""
    from sentry.api.dispatch import dispatch_pull_request

    cfg = settings or Settings()  # type: ignore[call-arg]
    dispatch = dispatch_fn or dispatch_pull_request
    app = FastAPI(title="TotalPR", version="0.4.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/webhook")
    async def webhook(
        request: Request, background: BackgroundTasks
    ) -> dict[str, str]:
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
            logger.info("scheduling dispatch for delivery=%s", delivery_id)
            background.add_task(dispatch, payload, cfg)
        else:
            logger.info(
                "no dispatch: event_type=%s not in {pull_request}", event_type
            )

        return {"status": "ok", "event": event_type, "action": action}

    return app


app = create_app()