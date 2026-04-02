# Agent System

Production-ready multi-agent framework built with LangGraph, LangChain, FastAPI, MinIO, PostgreSQL, ElasticSearch, and Langfuse.

---

## What This Project Includes

- Multi-agent orchestration with reflection loop (DONE / RETRY / FAIL)
- Skills loaded from local files or Langfuse (`local`, `langfuse`, `hybrid` with TTL cache)
- Built-in tools + MCP tools
- Async PostgreSQL pool (`asyncpg`) for config/runs/memory/artifacts
- Optional API key security (`X-API-Key`) for all routes except `/health`
- Request middleware (`X-Request-ID`, `X-Response-Time`)
- MinIO artifact storage + `file_artifacts` tracking
- Langfuse v3 tracing + prompt management

---

## Services and Default URLs

| Service | URL / Port | Notes |
|---|---|---|
| API (`agent-system`) | [http://localhost:8080](http://localhost:8080) | Swagger at `/docs` |
| MinIO API | `localhost:9100` | Object storage API |
| MinIO Console | [http://localhost:9101](http://localhost:9101) | Login: `minioadmin / minioadmin` |
| Agent PostgreSQL | `localhost:5433` | DB: `agentdb` |
| ElasticSearch | [http://localhost:9200](http://localhost:9200) | Logs index: `agent-system-logs` |
| Kibana | [http://localhost:5601](http://localhost:5601) | Log visualization |
| Langfuse v3 | [http://localhost:3001](http://localhost:3001) | Traces + prompts |

---

## Step-by-Step: Run the Project

### 1) Prepare environment

```bash
cp .env.example .env
```

Set at least:

```env
OPENROUTER_API_KEY=sk-or-your-key-here
```

Optional:

```env
API_KEY=your-secret-key   # leave empty to disable auth
```

### 2) Build and start containers

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
```

### 3) Verify health

```bash
curl -s http://localhost:8080/health | python3 -m json.tool
```

`/health` is always open (no API key required).

### 4) Open API docs

- [http://localhost:8080/docs](http://localhost:8080/docs)

---

## Step-by-Step: Test with Specific cURL Commands

### 0) Prepare reusable variables

```bash
BASE_URL="http://localhost:8080"
API_KEY_VALUE="$(awk -F= '/^API_KEY=/{print $2}' .env | tr -d '[:space:]')"
AUTH_HEADER=()
if [ -n "$API_KEY_VALUE" ]; then AUTH_HEADER=(-H "X-API-Key: $API_KEY_VALUE"); fi
```

### 1) Health check (public route)

```bash
curl -s "$BASE_URL/health" | python3 -m json.tool
```

### 2) Create 3 agents

```bash
curl -s -X POST "$BASE_URL/agents" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "coder",
    "skill_name": "coder",
    "model": "qwen/qwen3-30b-a3b-thinking-2507",
    "model_source": "openrouter",
    "max_reflections": 3,
    "tools": [
      "web_search",
      "calculate",
      "fetch_url",
      "write_file",
      "create_word_file",
      "read_file",
      "list_files",
      "get_datetime",
      "summarise_text",
      "memory_save",
      "memory_get",
      "query"
    ]
  }' | python3 -m json.tool

curl -s -X POST "$BASE_URL/agents" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "researcher",
    "skill_name": "researcher",
    "model": "qwen/qwen3-30b-a3b-thinking-2507",
    "model_source": "openrouter",
    "max_reflections": 3,
    "tools": [
      "web_search",
      "calculate",
      "fetch_url",
      "write_file",
      "create_word_file",
      "read_file",
      "list_files",
      "get_datetime",
      "summarise_text",
      "memory_save",
      "memory_get",
      "query"
    ]
  }' | python3 -m json.tool

curl -s -X POST "$BASE_URL/agents" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "analyst",
    "skill_name": "analyst",
    "model": "qwen/qwen3-30b-a3b-thinking-2507",
    "model_source": "openrouter",
    "max_reflections": 3,
    "tools": [
      "web_search",
      "calculate",
      "fetch_url",
      "write_file",
      "create_word_file",
      "read_file",
      "list_files",
      "get_datetime",
      "summarise_text",
      "memory_save",
      "memory_get",
      "query"
    ]
  }' | python3 -m json.tool
```

### 3) List agents

```bash
curl -s "$BASE_URL/agents" "${AUTH_HEADER[@]}" | python3 -m json.tool
```

### 4) Run test tasks

```bash
curl -s -X POST "$BASE_URL/agents/researcher/run" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Compare LangGraph vs CrewAI in a concise table with pros/cons.",
    "session_id": "session-researcher-001"
  }' | python3 -m json.tool

curl -s -X POST "$BASE_URL/agents/analyst/run" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Summarize top 5 AI agent frameworks and rank by enterprise readiness.",
    "session_id": "session-analyst-001"
  }' | python3 -m json.tool

curl -s -X POST "$BASE_URL/agents/coder/run" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Create a markdown report and then save a Word file named reports/pro_plus_demo.docx using create_word_file with format=markdown, include_toc=true, and a short header/footer.",
    "session_id": "session-coder-001"
  }' | python3 -m json.tool
```

### 5) Verify stored file artifacts

```bash
curl -s "$BASE_URL/files?limit=20" "${AUTH_HEADER[@]}" | python3 -m json.tool
curl -s "$BASE_URL/files/agents/coder?limit=20" "${AUTH_HEADER[@]}" | python3 -m json.tool
```

### 6) Download a saved artifact (presigned URL)

```bash
curl -sG "$BASE_URL/files/download" "${AUTH_HEADER[@]}" \
  --data-urlencode "file_path=reports/pro_plus_demo.docx" \
  --data-urlencode "expires=3600" | python3 -m json.tool
```

### 7) Debug endpoints (auth-protected)

```bash
curl -s "$BASE_URL/debug/skills" "${AUTH_HEADER[@]}" | python3 -m json.tool
curl -s "$BASE_URL/debug/tracing" "${AUTH_HEADER[@]}" | python3 -m json.tool
```

### 8) Validate middleware headers

```bash
curl -si "$BASE_URL/agents" "${AUTH_HEADER[@]}" | sed -n '1,20p'
```

You should see:
- `x-request-id`
- `x-response-time`

### 9) Watch logs live

```bash
docker logs -f agent-system
```

Look for lines like:
- `DB pool ready (asyncpg)`
- `REQUEST GET /agents ...`
- run/tool/reflection step logs

---

## API Auth Behavior

- `/health`: always public
- `/agents`, `/files`, `/debug`: require `X-API-Key` only when `API_KEY` is set in `.env`
- If `API_KEY` is blank, auth is disabled for local dev

---

## Built-in Tools (Current)

| Tool | Description |
|---|---|
| `web_search` | Web search via `ddgs` provider |
| `calculate` | Safe math evaluator |
| `fetch_url` | HTTP GET and return text |
| `write_file` | Save text to MinIO |
| `create_word_file` | Create `.docx` (plain/markdown/pro+) and save to MinIO |
| `read_file` | Read file content from MinIO |
| `list_files` | List object keys in MinIO |
| `get_datetime` | Current UTC timestamp |
| `summarise_text` | Excerpt long text |
| `memory_save` | Save key/value memory in PostgreSQL |
| `memory_get` | Load key/value memory from PostgreSQL |

---

## ADK Output Regression Workflow

This repo includes Google ADK proxy agents and eval placeholders so you can lock
ideal outputs as regression baselines.

### 1) Install ADK dependencies

```bash
pip install -e ".[adk,dev]"
```

### 2) Start backend API first

```bash
docker compose up -d app
curl -s http://localhost:8080/health | python3 -m json.tool
```

### 3) Start ADK Web

```bash
./scripts/adk_web.sh
```

Then open [http://localhost:8000](http://localhost:8000), and select one of:
- `coder_proxy`
- `researcher_proxy`
- `analyst_proxy`

### 4) Capture benchmark from an ideal session

1. Chat with the proxy agent in ADK Web.
2. When you get an ideal response, open the **Eval** tab.
3. Select the matching eval file (`coder_proxy` / `researcher_proxy` / `analyst_proxy`).
4. Click **Add current session**.

This writes/updates the eval case in:
- `eval/adk/coder_proxy.test.json`
- `eval/adk/researcher_proxy.test.json`
- `eval/adk/analyst_proxy.test.json`

The saved `final_response` becomes your ground-truth benchmark.

### 5) Run regression from CLI (`adk eval`)

```bash
# run all proxy eval sets
./scripts/adk_eval.sh all

# run one proxy only
./scripts/adk_eval.sh researcher
```

Equivalent raw command pattern:

```bash
adk eval <agent_module_path> <eval_file_path> --print_detailed_results
```

### 6) Run regression from pytest

```bash
pytest -v tests/test_adk_regression.py
```

Notes:
- the pytest wrapper auto-skips when ADK is not installed, no eval cases exist yet,
  backend API is unavailable, or `GOOGLE_API_KEY` is missing.
- once eval cases exist, pytest becomes your automated output-regression gate.

### 7) Useful ADK env vars

```env
GOOGLE_API_KEY=...                      # required by ADK model runtime
ADK_MODEL=gemini-2.0-flash              # optional model override
ADK_TARGET_API_BASE_URL=http://localhost:8080
ADK_TARGET_API_KEY=                      # optional; needed only if your API_KEY auth is enabled
ADK_WEB_PORT=8000
```

---

## Useful Commands

```bash
# Rebuild app only
docker compose build --no-cache app && docker compose up -d app

# Check service status
docker compose ps

# API logs
docker logs -f agent-system

# Enter postgres
psql postgresql://agent:agent@localhost:5433/agentdb

# Run tests
pytest -v

# Stop stack (keep volumes)
docker compose down

# Full reset (remove volumes)
docker compose down -v
```
