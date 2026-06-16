# Sentry's sandboxed ripgrep runner.
# Build with: docker build -f docker/ripgrep.Dockerfile -t sentry-ripgrep:latest .

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ripgrep \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["rg"]