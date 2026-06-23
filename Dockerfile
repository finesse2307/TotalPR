FROM python:3.12-slim

WORKDIR /app

# Install minimal system deps. We do NOT install docker-cli or the tool images
# (ruff, semgrep, ripgrep) in this container — tool execution will be disabled
# in the deployed environment. The full tool stack runs in local dev only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY migrations/ ./migrations/

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "sentry.api.main:app", "--host", "0.0.0.0", "--port", "8000"]