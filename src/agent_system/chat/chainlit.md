# Agent System — Chat UI

Talk directly to your registered agents. Each message you send becomes a **task** for the selected agent.

## Getting started

1. Select an agent from the **⚙️ Settings** panel (top-right).
2. Type your task and press **Enter**.
3. Watch the agent think, call tools, and reflect in real time.

## Tips

| Command | Effect |
|---------|--------|
| `/reload` | Re-sync agents from the database |
| Any text | Runs as a task on the selected agent |

## Status indicators

| Icon | Meaning |
|------|---------|
| 🤔 Thinking… | LLM is planning the next action |
| 🔧 Calling tools… | Executing one or more tool calls |
| 🔄 Evaluating output… | Reflection node deciding DONE / RETRY / FAIL |
| ⏸ Waiting for approval… | High-stakes tool paused for your review |
| ✅ / ❌ | Tool result (success / failure) |

Artifacts written to MinIO during a run are listed at the bottom of the response.
