-- Initial schema for the Sentry memory store.
-- Stores diff-conditioned review memories with embeddings for similarity search.
--
-- ``was_accepted`` is nullable on purpose: NULL = unlabeled (agent produced
-- but no human verdict yet), TRUE = accepted, FALSE = rejected.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memories (
    id              SERIAL PRIMARY KEY,
    repo            TEXT NOT NULL,
    diff_text       TEXT NOT NULL,
    finding_text    TEXT NOT NULL,
    category        TEXT NOT NULL,
    severity        TEXT NOT NULL,
    was_accepted    BOOLEAN,
    embedding       VECTOR(1024) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS memories_repo_idx
    ON memories (repo);

CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);