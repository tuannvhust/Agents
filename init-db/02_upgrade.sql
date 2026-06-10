-- ============================================================
-- Agent System — idempotent upgrades for existing databases
-- Safe to re-run: every statement uses IF NOT EXISTS / OR REPLACE.
-- Also executed by app/worker bootstrap on startup.
-- ============================================================

CREATE TABLE IF NOT EXISTS agent_configs (
    name        TEXT        PRIMARY KEY,
    config      JSONB       NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS run_trace JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS run_status TEXT NOT NULL DEFAULT 'completed';
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS job_id TEXT;
ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS input_file TEXT;

CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs (run_status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_runs_job_id ON agent_runs (job_id) WHERE job_id IS NOT NULL;
