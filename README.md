# Agent System

Production-ready **multi-agent orchestration** framework: LangGraph, FastAPI, RabbitMQ workers, PostgreSQL, MinIO, Redis, Langfuse.

This guide walks you through **starting the project from scratch** on a clean machine.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker + Docker Compose | v2 recommended |
| OpenRouter API key | [https://openrouter.ai](https://openrouter.ai) |
| ~8 GB RAM | Full stack includes Langfuse, ElasticSearch, MinIO, Postgres, RabbitMQ |

---

## Step 1 — Clone and configure `.env`

```bash
git clone <your-repo-url> Agents
cd Agents
cp .env.example .env
```

Edit `.env` — minimum required:

```env
OPENROUTER_API_KEY=sk-or-your-key-here

# Vision LLM for OCR node (cccd_agent)
OCR_MODEL=qwen/qwen3-vl-8b-instruct
OCR_MODEL_SOURCE=openrouter

# Job queue (default on)
QUEUE_ENABLED=true
```

Leave `API_KEY=` blank for local dev (no auth). Set it in production.

---

## Step 2 — Start the stack

```bash
docker compose up -d --build
```

Wait until core services are healthy:

```bash
docker compose ps
```

Expected: `agent-system`, `agent-postgres`, `agent-minio`, `agent-rabbitmq`, `agent-cache-redis`, `worker` → **healthy** or **Up**.

Scale workers for more concurrent runs:

```bash
docker compose up -d --scale worker=3
```

> **Do not** run `agent-worker` on the host while Docker `worker` is running — both consume the same RabbitMQ queue and cause stale config / duplicate jobs.

---

## Step 3 — Verify health

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

Optional UIs:

| Service | URL | Credentials |
|---------|-----|-------------|
| API docs (Swagger) | [http://localhost:8080/docs](http://localhost:8080/docs) | API key if set |
| Chainlit chat | [http://localhost:8501](http://localhost:8501) | — |
| RabbitMQ Management | [http://localhost:15672](http://localhost:15672) | `agent` / `agent` |
| MinIO Console | [http://localhost:9101](http://localhost:9101) | `minioadmin` / `minioadmin` |
| Langfuse | [http://localhost:3001](http://localhost:3001) | see `.env` |
| pgAdmin (if on `common-net`) | [http://localhost:5050](http://localhost:5050) | `admin@example.com` / `admin123456` |

---

## Step 4 — Shell helpers

Paste once per terminal session:

```bash
BASE="http://localhost:8080"
AUTH=()
KEY="$(awk -F= '/^API_KEY=/{print $2}' .env | tr -d '[:space:]')"
[ -n "$KEY" ] && AUTH=(-H "X-API-Key: $KEY")

poll_run() {
  local agent="$1" run_id="$2" trace="${3:-false}"
  local url="$BASE/agents/$agent/runs/$run_id"
  [ "$trace" = "true" ] && url="$url?include_trace=true"
  while true; do
    resp="$(curl -s "$url" ${AUTH[@]:+"${AUTH[@]}"})"
    run_status="$(printf '%s' "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin).get('run_status',''))")"
    printf '%s\n' "$resp" | python3 -m json.tool
    case "$run_status" in
      queued|running) echo "… still $run_status, waiting 2s" >&2; sleep 2 ;;
      *) return 0 ;;
    esac
  done
}
```

> All agent runs use **`POST /agents/{name}/run`** with **`multipart/form-data`** (not JSON). The API returns **HTTP 202** immediately; poll until `run_status` is `completed`, `failed`, or `awaiting_approval`.

---

## Step 5 — Register agents

Sub-agents **must** be registered before the coordinator.

### 5a. Sub-agents

```bash
# Researcher
curl -s -X POST "$BASE/agents" ${AUTH[@]:+"${AUTH[@]}"} \
  -H "Content-Type: application/json" \
  -d '{"name":"researcher","skill_name":"roles/researcher","role":"subagent","plugins":["safety"]}' \
  | python3 -m json.tool

# Analyst
curl -s -X POST "$BASE/agents" ${AUTH[@]:+"${AUTH[@]}"} \
  -H "Content-Type: application/json" \
  -d '{"name":"analyst","skill_name":"roles/analyst","role":"subagent","plugins":["safety"]}' \
  | python3 -m json.tool

# CCCD processor (OCR-enabled)
curl -s -X POST "$BASE/agents" ${AUTH[@]:+"${AUTH[@]}"} \
  -H "Content-Type: application/json" \
  -d '{
    "name": "cccd_agent",
    "skill_name": "agents/cccd_processor",
    "role": "subagent",
    "enable_ocr": true,
    "ocr_skill_name": "ocr/cccd",
    "tools": ["calculate"]
  }' | python3 -m json.tool
```

Do **not** set `ocr_model` on registration — inherit `OCR_MODEL` from `.env`.

### 5b. Coordinator

```bash
curl -s -X POST "$BASE/agents" ${AUTH[@]:+"${AUTH[@]}"} \
  -H "Content-Type: application/json" \
  -d '{
    "name": "coordinator",
    "skill_name": "agents/coordinator",
    "role": "coordinator",
    "sub_agents": ["researcher", "analyst", "cccd_agent"],
    "plugins": ["safety"]
  }' | python3 -m json.tool
```

### 5c. Verify

```bash
curl -s "$BASE/agents" ${AUTH[@]:+"${AUTH[@]}"} | python3 -m json.tool
```

You should see 4 agents. Coordinator `tools` includes `invoke_researcher`, `invoke_analyst`, `invoke_cccd_agent`.

---

## Step 6 — Run a simple agent

```bash
# Enqueue
curl -s -X POST "$BASE/agents/researcher/run" ${AUTH[@]:+"${AUTH[@]}"} \
  -F "task=What is the latest stable version of Python?" \
  | python3 -m json.tool

# Copy run_id from response, then poll
RUN_ID="<run_id>"
poll_run researcher "$RUN_ID"
```

Expected: `"run_status": "completed"` with a `final_answer`.

Monitor the queue: RabbitMQ UI → **Queues** → `agent.jobs` (Ready / Unacked / Consumers).

---

## Step 7 — Run CCCD extraction (OCR)

### Upload a local PDF or image (recommended)

```bash
RUN_ID="$(curl -s -X POST "$BASE/agents/cccd_agent/run" ${AUTH[@]:+"${AUTH[@]}"} \
  -F "task=Extract all fields. The person is married and works as a doctor." \
  -F "image=@ocr_input/test1.pdf;type=application/pdf" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")"

echo "RUN_ID=$RUN_ID"
poll_run cccd_agent "$RUN_ID"
```

Use a real CCCD scan/PDF — test files with no readable card data return `null` fields.

### Remote image URL

```bash
RUN_ID="$(curl -s -X POST "$BASE/agents/cccd_agent/run" ${AUTH[@]:+"${AUTH[@]}"} \
  -F "task=Extract all fields." \
  -F "image_url=https://example.com/cccd.jpg" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")"
poll_run cccd_agent "$RUN_ID"
```

### Via coordinator (production pattern)

```bash
RUN_ID="$(curl -s -X POST "$BASE/agents/coordinator/run" ${AUTH[@]:+"${AUTH[@]}"} \
  -F "task=Extract all CCCD fields. Married, works as a doctor." \
  -F "image_url=https://example.com/cccd.jpg" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['run_id'])")"
poll_run coordinator "$RUN_ID"
```

Worker logs should show:

```
[ROUTE] START → ocr (image_url present)
[OCR NODE] extracted N chars from image
```

```bash
docker compose logs -f worker | grep -E "ROUTE|OCR NODE|AGENT RUN"
```

---

## Step 8 — Open Chainlit (optional)

Go to [http://localhost:8501](http://localhost:8501), pick an agent in ⚙️ settings, and chat.

Type `/reload` after registering new agents via the API without restarting containers.

---

## Architecture (short)

```
Client  →  POST /agents/{name}/run  (multipart)  →  202 + run_id
                ↓
         RabbitMQ queue agent.jobs
                ↓
         worker(s)  →  LangGraph  →  Postgres + MinIO
                ↓
Client  ←  GET /agents/{name}/runs/{run_id}  (poll)
```

| Role | Graph | Reflection |
|------|-------|------------|
| `coordinator` | agent → tools → **reflect** → END | ✅ |
| `subagent` | agent → tools → END | ❌ |
| `cccd_agent` (+ OCR) | **ocr** → agent → tools → END | ❌ |

OCR node runs only when `image_url` is present (from uploaded file or `image_url` form field). Coordinators forward `image_url` to OCR-enabled sub-agents via `invoke_*`.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check (public) |
| `POST` | `/agents` | Register agent |
| `GET` | `/agents` | List agents |
| `DELETE` | `/agents/{name}` | Delete agent |
| `POST` | `/agents/{name}/run` | Enqueue run (`multipart`: `task`, optional `image` or `image_url`) → **202** |
| `GET` | `/agents/{name}/runs/{run_id}` | Poll result (`?include_trace=true`) |
| `POST` | `/agents/{name}/runs/{run_id}/resume` | Resume after human approval → **202** |
| `GET` | `/review/ui` | Reviewer UI |
| `GET` | `/files` | List file artifacts |

Full docs: [http://localhost:8080/docs](http://localhost:8080/docs)

### `POST /agents/{name}/run` form fields

| Field | Required | Description |
|-------|----------|-------------|
| `task` | ✅ | Instruction for the agent |
| `image` | ❌ | Local PDF/JPEG/PNG (OCR agents) |
| `image_url` | ❌ | Remote image URL (ignored if `image` file is sent) |
| `session_id` | ❌ | Custom run ID |
| `include_trace` | ❌ | `true` / `false` |

---

## Configuration reference

```env
# Models (optional overrides; unset = OPENROUTER_DEFAULT_MODEL)
ORCHESTRATOR_MODEL=...
SUBAGENT_MODEL=...
OCR_MODEL=qwen/qwen3-vl-8b-instruct

# Queue
QUEUE_ENABLED=true
QUEUE_MAX_JOBS=10          # concurrent jobs per worker
RABBITMQ_URL=amqp://agent:agent@localhost:5672/

# Cache (optional)
CACHE_ENABLED=false
CACHE_REDIS_URL=redis://localhost:6380/0
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `curl: option : blank argument` | Use `${AUTH[@]:+"${AUTH[@]}"}` not `"${AUTH[@]}"` in zsh |
| `is not a valid model ID` | Delete agent, re-register **without** `ocr_model`; restart worker |
| All OCR fields `null` | Use a real CCCD image; check `[OCR NODE] extracted N chars` in worker logs |
| `Run not found for agent 'coordinator'` | Poll with the same agent you enqueued (`cccd_agent` vs `coordinator`) |
| Jobs stuck in queue | `docker compose ps worker`; RabbitMQ → Consumers should be ≥ 1 |
| Stale agent config | `docker compose restart worker`; kill stray host `agent-worker` |
| `ocr_agent_1` reload warning in chat | Old agent in DB — delete via `DELETE /agents/{name}` |

```bash
# Check for stray host workers
ps aux | grep agent-worker | grep -v grep
kill <PID>   # if found
```

---

## Useful commands

```bash
docker compose logs -f app          # API logs
docker compose logs -f worker       # Worker logs
docker compose up --build -d app worker   # Rebuild after code changes

# Postgres
psql postgresql://agent:agent@localhost:5433/agentdb
psql postgresql://agent:agent@localhost:5433/agentdb \
  -c "SELECT run_id, agent_name, run_status, length(final_answer) FROM agent_runs ORDER BY created_at DESC LIMIT 5;"

# Tests
pip install -e ".[dev]"
pytest -v
```

---

## Project layout

```
prompts/
  roles/          # researcher, analyst, writer, …
  agents/         # coordinator, cccd_processor
  ocr/            # VLM extraction prompts (cccd.md)
init-db/
  01_schema.sql   # Fresh Postgres schema
  02_upgrade.sql  # Idempotent migrations on startup
ocr_input/        # Sample PDFs for local OCR tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for deeper design notes.
