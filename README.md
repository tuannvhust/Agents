# Agent System

Production-ready multi-agent framework built with LangGraph, LangChain, FastAPI, MinIO, PostgreSQL, ElasticSearch, and Langfuse.

---

## What This Project Includes

- Multi-agent orchestration with reflection loop (DONE / RETRY / FAIL), including a **process review** of plan, first tool choice, and arguments (see trace + reflection)
- **Run traces**: structured export of assistant text, planned tools with exact arguments, executions, and reflection steps (`include_trace`, MinIO `trace.json`, PostgreSQL `run_trace`)
- **Human-in-the-loop**: optional pause before executing configured **high-stakes** tools; Reviewer UI + resume API (`/review/*`, `POST /agents/{name}/runs/{run_id}/resume`)
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
| API (`agent-system`) | [http://localhost:8080](http://localhost:8080) | Swagger at `/docs`; human review UI at `/review/ui` |
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
    ],
    "plugins": ["safety"]
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
    ],
    "tools_requiring_approval":["create_word_file"],
    "plugins": ["safety"]
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
    ],
    "plugins": ["safety"]
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
    "session_id": "session-researcher-001",
    "include_trace": true
  }' | python3 -m json.tool

curl -s -X POST "$BASE_URL/agents/analyst/run" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Summarize top 5 AI agent frameworks and rank by enterprise readiness.",
    "session_id": "session-analyst-001",
    "include_trace": true
  }' | python3 -m json.tool

curl -s -X POST "$BASE_URL/agents/coder/run" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Create a markdown report and then save a Word file named reports/pro_plus_demo.docx using create_word_file with format=markdown, include_toc=true, and a short header/footer.",
    "include_trace": true,
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
- `run_trace run_id=...` (INFO on `agent_system.core.trace`) and full trace JSON at DEBUG

---

## Run traces (structured logging & export)

Each completed run can expose a **trace document** (`schema_version`, `steps`, `tool_invocations`):

| Where | What |
|-------|------|
| **API** | `POST /agents/{name}/run` with `"include_trace": true` → response field `trace` |
| **MinIO** | `runs/{agent_name}/{run_id}/_exports/trace.json` and `result.json` when `MINIO_SCOPE_PATHS_TO_RUN=true`; else `exports/{agent}/{run_id}/` (listed in `stored_artifacts`) |
| **PostgreSQL** | Column `agent_runs.run_trace` (JSONB) |

**Steps** include: `agent` (assistant text, `tool_calls_planned` with full `arguments`), `tools` (executions with exact args and outputs), `reflect` (decision / reason / suggestions).

**Database upgrade:** if your database was created before `run_trace` existed, apply once:

```bash
docker exec -i agent-postgres psql -U agent -d agentdb < init-db/02_run_trace_column.sql
```

(Adjust user/host/port if you are not using the default compose mapping.)

---

## Reflection process review

The reflection evaluator receives the **process trace** and must answer (in its structured reply) whether the initial plan was logical, whether the **first executed tool** was appropriate, and whether **tool arguments** were correct. That analysis is merged into the stored reflection reason and appears in trace `reflect` steps.

---

## Human-in-the-loop: high-stakes tool approval

Some tool calls can be configured to **pause** until a human approves or rejects the **entire planned batch** for that turn.

### Configure an agent

Add **`tools_requiring_approval`** (list of tool names) when creating the agent. If the model plans **any** tool in that list, execution stops **before** the tools node.

### API flow

1. **`POST /agents/{name}/run`** — response may have `"run_status": "awaiting_approval"` and `"approval_request": { ... }` (planned tools, args, message digest, trace tail).
2. **Reviewer UI** — open [http://localhost:8080/review/ui](http://localhost:8080/review/ui) (same auth as API). Set base URL and `X-API-Key` if needed, refresh pending, inspect payload, **Approve** or **Reject** (reject requires a reason).
3. **Resume** (choose one):
   - `POST /review/{run_id}/decide` with body `{"action":"approve"}` or `{"action":"reject","reason":"..."}`
   - `POST /agents/{name}/runs/{run_id}/resume` with the same JSON body

**Approve** → tools run as planned. **Reject** → the model sees rejection tool messages and continues from the agent node.

### Operational notes

- Approvals use an in-memory LangGraph **checkpoint** (`MemorySaver`) keyed by `run_id`. Pending work is **lost if the API process restarts**. The default Docker image runs **one uvicorn worker**, which matches this design.
- For multiple high-stakes steps in one run, you may need to approve **more than once**.

---

## Step-by-step: test recently added features

Follow these in order the first time you validate traces, reflection, and human approval.

### A) Automated tests (no Docker required)

From the repo root, with dev dependencies:

```bash
pip install -e ".[dev]"
pytest -q tests/test_trace.py tests/test_reflection.py tests/test_human_approval_routing.py
```

Optional full suite (may require env/MCP-related skips):

```bash
pytest -v
```

### B) Database trace column (existing deployments only)

If `agent_runs` has no `run_trace` column, run the migration in the table above, then restart the app container if needed.

### C) End-to-end trace via API

1. Start the stack (`docker compose up -d --build`) and ensure `OPENROUTER_API_KEY` is set.
2. Use the **“Prepare reusable variables”** snippet from this README (`BASE_URL`, `AUTH_HEADER`).
3. Create or use an agent, then run with trace:

```bash
curl -s -X POST "$BASE_URL/agents/researcher/run" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Say hello in one short sentence.",
    "session_id": "trace-test-001",
    "include_trace": true
  }' | python3 -m json.tool
```

4. **Check:** `run_status` is `completed`, `trace.steps` is a non-empty array, `trace.tool_invocations` matches any tools that ran.
5. **Check:** `stored_artifacts` contains `.../_exports/trace.json` (scoped) or `.../exports/.../trace.json` (unscoped) when MinIO is up.
6. **Check DB:** `SELECT run_id, run_trace->'schema_version' FROM agent_runs WHERE run_id = 'trace-test-001';`

### D) Reflection + process review in the trace

Use a task that triggers **web_search** and possibly **RETRY** (e.g. compare frameworks). Run with `"include_trace": true`. In `trace.steps`, find objects with `"type": "reflect"` and read `reason` — it should reflect plan / first tool / arguments when the model follows the reflection prompt format.

### E) Human approval gate

1. Register an agent that lists both normal tools and **`tools_requiring_approval`** (example: gate file writes):

```bash
curl -s -X POST "$BASE_URL/agents" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "researcher-gated",
    "skill_name": "researcher",
    "model_source": "openrouter",
    "max_reflections": 3,
    "tools": ["web_search", "create_word_file", "calculate"],
    "tools_requiring_approval": ["create_word_file"]
  }' | python3 -m json.tool
```

2. Run a task that will eventually call **`create_word_file`** (e.g. ask for a comparison saved as `.docx` in MinIO).

3. **Expect:** HTTP 200 with `"run_status": "awaiting_approval"`, non-null `approval_request`, `success: false`.

4. **List pending:**

```bash
curl -s "$BASE_URL/review/pending" "${AUTH_HEADER[@]}" | python3 -m json.tool
```

5. **Decide** (replace `RUN_ID` from the response):

```bash
RUN_ID="your-run-id-here"
curl -s -X POST "$BASE_URL/review/$RUN_ID/decide" "${AUTH_HEADER[@]}" \
  -H "Content-Type: application/json" \
  -d '{"action": "approve"}' | python3 -m json.tool
```

6. **Expect:** Either `run_status: completed` (run finished) or `awaiting_approval` again if another gated tool batch is planned.

7. **Reject path:** use `"action": "reject", "reason": "Use a different path"` and confirm the agent continues with that feedback (new assistant turn without executing tools).

---

## API Auth Behavior

- `/health`: always public
- `/agents`, `/files`, `/debug`, `/review`: require `X-API-Key` only when `API_KEY` is set in `.env`
- If `API_KEY` is blank, auth is disabled for local dev

---

## Built-in Tools (Current)

| Tool | Description |
|---|---|
| `web_search` | Web search via `ddgs` provider |
| `calculate` | Safe math evaluator |
| `fetch_url` | HTTP GET and return text |
| `write_file` | Save text to MinIO (run-scoped path when `MINIO_SCOPE_PATHS_TO_RUN=true`) |
| `create_word_file` | Create `.docx` (plain/markdown/pro+) and save to MinIO (same) |
| `read_file` | Read file content from MinIO (same) |
| `list_files` | List object keys under the current run workspace (same) |
| `get_datetime` | Current UTC timestamp |
| `summarise_text` | Excerpt long text |
| `memory_save` | Save key/value memory in PostgreSQL |
| `memory_get` | Load key/value memory from PostgreSQL |

### MinIO paths and sessions (`run_id`)

When **`MINIO_SCOPE_PATHS_TO_RUN`** is `true` (default in `.env.example`), tool-written objects are stored under:

`runs/{agent_name}/{run_id}/<path-you-pass>`

The **`run_id`** is the same identifier as the optional **`session_id`** field on `POST /agents/{name}/run` (auto-generated if omitted). Different sessions therefore no longer overwrite the same logical path (for example `reports/summary.docx`).

The agent should keep using **run-relative** paths in tool calls (e.g. `reports/out.docx`). **`read_file`** / **`list_files`** apply the same prefix for the active run. To reference an object **outside** the current run (or a legacy flat key), pass a full key that already starts with `runs/`.

**Framework exports** for each session (`result.json`, `trace.json`) follow the same rules: with scoping on they are under **`runs/{agent_name}/{run_id}/_exports/`** (avoiding clashes with a tool-written `result.json` in the workspace root). With scoping off they use **`exports/{agent_name}/{run_id}/`**.

Set **`MINIO_SCOPE_PATHS_TO_RUN=false`** only if you need the old flat layout for tool paths. **`/files/download`** and **`file_artifacts`** use the **full** object key stored after writes.

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
