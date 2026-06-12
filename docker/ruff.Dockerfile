# Sentry's sandboxed Ruff runner.
# Build with: docker build -f docker/ruff.Dockerfile -t sentry-ruff:latest .

FROM python:3.12-slim

RUN pip install --no-cache-dir ruff

ENTRYPOINT ["ruff"]
