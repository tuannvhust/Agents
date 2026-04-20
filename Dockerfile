# ── Stage 1a: base Python packages (API) ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Use setuptools (matches pyproject.toml build-backend)
RUN pip install --no-cache-dir "setuptools>=45" wheel

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 1b: chat extras (Chainlit + base) ────────────────────────────────────
FROM builder AS builder-chat

RUN pip install --no-cache-dir --prefix=/install ".[chat]"

# ── Stage 2: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Agent System"
LABEL description="Multi-agent system with LangGraph, MCP, MinIO, ElasticSearch"

# ── Install Node.js 20 LTS ────────────────────────────────────────────────────
# Required so the MCP client can spawn MCP servers via `npx` as child processes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/* \
    # Verify
    && node --version && npx --version

# ── Non-root user (with home directory so npm cache works) ────────────────────
RUN groupadd -r appuser \
    && useradd -r -m -d /home/appuser -g appuser appuser

# Pre-warm the npx cache AS appuser so the cache is owned correctly at runtime.
# This avoids EACCES errors when the MCP client spawns `npx` during startup.
USER appuser
RUN npx --yes @modelcontextprotocol/server-postgres --help 2>/dev/null || true
USER root

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy source, skills, and guardrails (SafetyPlugin local rules)
COPY src/ src/
COPY skills/ skills/
COPY guardrails/ guardrails/

RUN chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

EXPOSE 8000

# Single worker — agent cache is in-process; use a shared store (Redis/DB) before scaling out
CMD ["uvicorn", "agent_system.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]


# ── Stage 3: Chainlit chat UI ──────────────────────────────────────────────────
FROM python:3.11-slim AS chat

LABEL maintainer="Agent System"
LABEL description="Chainlit chat UI for the Agent System"

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get purge -y --auto-remove curl gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && node --version && npx --version

RUN groupadd -r appuser \
    && useradd -r -m -d /home/appuser -g appuser appuser

USER appuser
RUN npx --yes @modelcontextprotocol/server-postgres --help 2>/dev/null || true
USER root

WORKDIR /app

# Use the chat-extras builder layer
COPY --from=builder-chat /install /usr/local

COPY src/ src/
COPY skills/ skills/
COPY guardrails/ guardrails/

RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

# Chainlit reads chainlit.md and .chainlit/ from the working directory
CMD ["chainlit", "run", "src/agent_system/chat/app.py", \
     "--host", "0.0.0.0", "--port", "8501"]
