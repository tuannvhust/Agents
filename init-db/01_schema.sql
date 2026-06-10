-- ============================================================
-- Agent System — canonical database schema (fresh installs)
-- Runs once when agent-postgres starts with an empty data volume.
-- For existing volumes, 02_upgrade.sql applies idempotent patches.
-- ============================================================

-- ── Helpers ─────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ── Tables ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_runs (
    id               UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name       TEXT        NOT NULL,
    run_id           TEXT        NOT NULL UNIQUE,
    task             TEXT        NOT NULL,
    final_answer     TEXT,
    success          BOOLEAN     NOT NULL DEFAULT FALSE,
    reflection_count INTEGER     NOT NULL DEFAULT 0,
    minio_artifacts  JSONB       NOT NULL DEFAULT '[]',
    run_trace        JSONB       NOT NULL DEFAULT '{}',
    run_status       TEXT        NOT NULL DEFAULT 'completed',
    error_message    TEXT,
    job_id           TEXT,
    input_file       TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGSERIAL   PRIMARY KEY,
    agent_name  TEXT        NOT NULL,
    key         TEXT        NOT NULL,
    value       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (agent_name, key)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id          BIGSERIAL   PRIMARY KEY,
    run_id      TEXT        NOT NULL REFERENCES agent_runs (run_id) ON DELETE CASCADE,
    tool_name   TEXT        NOT NULL,
    input_args  JSONB,
    output      TEXT,
    success     BOOLEAN     NOT NULL DEFAULT TRUE,
    error       TEXT,
    called_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS file_artifacts (
    id           BIGSERIAL    PRIMARY KEY,
    run_id       TEXT         NOT NULL REFERENCES agent_runs (run_id) ON DELETE CASCADE,
    agent_name   TEXT         NOT NULL,
    file_path    TEXT         NOT NULL,
    file_size    INTEGER,
    content_type TEXT         NOT NULL DEFAULT 'text/plain',
    written_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_configs (
    name        TEXT        PRIMARY KEY,
    config      JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes ─────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_name ON agent_runs (agent_name, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_run_id     ON agent_runs (run_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status     ON agent_runs (run_status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_job_id     ON agent_runs (job_id) WHERE job_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_agent_memory_lookup    ON agent_memory (agent_name, key);
CREATE INDEX IF NOT EXISTS idx_tool_calls_run_id      ON tool_calls (run_id);
CREATE INDEX IF NOT EXISTS idx_file_artifacts_run_id  ON file_artifacts (run_id);
CREATE INDEX IF NOT EXISTS idx_file_artifacts_agent   ON file_artifacts (agent_name, written_at DESC);
CREATE INDEX IF NOT EXISTS idx_file_artifacts_file_path ON file_artifacts (file_path);

-- ── Triggers ────────────────────────────────────────────────────────────────

CREATE OR REPLACE TRIGGER agent_runs_updated_at
    BEFORE UPDATE ON agent_runs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER agent_memory_updated_at
    BEFORE UPDATE ON agent_memory
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE OR REPLACE TRIGGER agent_configs_updated_at
    BEFORE UPDATE ON agent_configs
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
