-- Add persisted run trace for existing databases (idempotent).
ALTER TABLE agent_runs
    ADD COLUMN IF NOT EXISTS run_trace JSONB NOT NULL DEFAULT '{}'::jsonb;
