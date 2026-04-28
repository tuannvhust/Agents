"""Chainlit chat UI — talk to registered agents with streaming status updates.

Run:
    chainlit run src/agent_system/chat/app.py [-w]

The app connects to the same Postgres DB as the FastAPI backend and loads agents
from it.  CHAT_API_BASE_URL / CHAT_API_KEY in .env are only needed if you want
the optional "Reload agents" button to call the REST API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select

from agent_system.chat.runner import (
    ensure_initialized,
    get_agent,
    list_agents,
    reload_agents,
)

logger = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_info(name: str) -> str:
    agent = get_agent(name)
    if agent is None:
        return ""
    tools = [t.name for t in agent._tools]
    n_tools = len(tools)
    needs_approval = list(agent.config.tools_requiring_approval or [])
    lines = [
        f"**Skill**: `{agent.config.skill_name}`",
        f"**Model**: `{agent.config.model or 'default'}`  ({agent.config.model_source})",
        f"**Tools ({n_tools})**: {', '.join(f'`{t}`' for t in tools) or '—'}",
    ]
    if needs_approval:
        lines.append(f"**Approval gates**: {', '.join(f'`{t}`' for t in needs_approval)}")
    if agent.config.plugins:
        lines.append(f"**Plugins**: {', '.join(f'`{p}`' for p in agent.config.plugins)}")
    return "\n".join(lines)


async def _close_step(step: cl.Step | None) -> None:
    if step is not None:
        try:
            await step.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


async def _sync_agent_dropdown(preferred: str | None = None) -> bool:
    """Redraw the ⚙️ Agent dropdown from the in-memory cache. Returns False if none."""
    agents = list_agents()
    if not agents:
        cl.user_session.set("agent_name", None)
        return False
    pref = preferred if preferred is not None else cl.user_session.get("agent_name")
    initial = pref if pref in agents else agents[0]
    settings = await cl.ChatSettings(
        [
            Select(
                id="agent",
                label="Agent",
                values=agents,
                initial_value=initial,
                description="Choose which agent to talk to.",
            ),
        ]
    ).send()
    selected = settings.get("agent", initial)
    cl.user_session.set("agent_name", selected)
    return True


# ── Lifecycle ─────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise infrastructure and show agent selector."""
    await ensure_initialized()
    # Pick up API-side create/delete without requiring a Chainlit process restart.
    await reload_agents()

    if not await _sync_agent_dropdown():
        await cl.Message(
            content=(
                "⚠️ **No agents found.**\n\n"
                "Register an agent via the REST API first, then **start a new chat** "
                "or type `/reload`."
            ),
        ).send()
        return

    selected = cl.user_session.get("agent_name")
    await cl.Message(
        content=(
            f"👋 Connected to **{selected}**\n\n"
            f"{_agent_info(selected)}\n\n"
            "---\nType your task and press **Enter** to run."
        ),
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    agent_name = settings.get("agent")
    cl.user_session.set("agent_name", agent_name)
    agent = get_agent(agent_name or "")
    if agent:
        await cl.Message(
            content=f"Switched to **{agent_name}**\n\n{_agent_info(agent_name)}",
        ).send()


# ── Action: reload agents ──────────────────────────────────────────────────────

@cl.action_callback("reload_agents")
async def _cb_reload(action: cl.Action) -> None:
    await reload_agents()
    names = list_agents()
    if not await _sync_agent_dropdown():
        await cl.Message(
            content="🔄 Reloaded — **no agents** in Postgres. Register via the API.",
        ).send()
        return
    sel = cl.user_session.get("agent_name")
    await cl.Message(
        content=(
            f"🔄 Agents reloaded. Available: {', '.join(f'`{a}`' for a in names) or 'none'}\n\n"
            f"Selected: **`{sel}`**"
            + (f"\n\n{_agent_info(sel)}" if sel else "")
        ),
    ).send()


# ── Main message handler ───────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message) -> None:
    agent_name: str | None = cl.user_session.get("agent_name")

    # Special commands
    if message.content.strip().lower() in ("/reload", "/refresh"):
        await reload_agents()
        names = list_agents()
        if not await _sync_agent_dropdown():
            await cl.Message(
                content="🔄 Reloaded — **no agents** in Postgres. Register via the API.",
            ).send()
            return
        sel = cl.user_session.get("agent_name")
        await cl.Message(
            content=(
                f"🔄 Agents reloaded. Available: {', '.join(f'`{a}`' for a in names) or 'none'}\n\n"
                f"Selected: **`{sel}`**"
                + (f"\n\n{_agent_info(sel)}" if sel else "")
            ),
        ).send()
        return

    if not agent_name:
        await cl.Message(
            content="⚠️ No agent selected. Use the ⚙️ settings panel to pick one.",
        ).send()
        return

    agent = get_agent(agent_name)
    if agent is None:
        await cl.Message(
            content=(
                f"⚠️ Agent **{agent_name}** not found. "
                "Type `/reload` to refresh the agent list."
            ),
        ).send()
        return

    await _run_agent(agent, message.content.strip())


async def _run_agent(agent: Any, task: str) -> None:
    """Stream an agent run, rendering status steps and the final answer."""
    answer_msg = cl.Message(content="")
    active_step: cl.Step | None = None
    tool_results_buffer: list[dict] = []
    streamed_tokens = False

    async def open_step(name: str) -> None:
        nonlocal active_step
        # Flush any buffered tool results into the closing step
        if tool_results_buffer and active_step is not None:
            lines = []
            for r in tool_results_buffer:
                preview = (r["text"] or "")[:300].replace("\n", " ")
                lines.append(f"{r['icon']} **{r['tool_name']}**: {preview}")
            active_step.output = "\n".join(lines)
            tool_results_buffer.clear()
        await _close_step(active_step)
        s = cl.Step(name=name, type="run")
        await s.__aenter__()
        active_step = s

    async def close_current_step() -> None:
        nonlocal active_step
        if tool_results_buffer and active_step is not None:
            lines = []
            for r in tool_results_buffer:
                preview = (r["text"] or "")[:300].replace("\n", " ")
                lines.append(f"{r['icon']} **{r['tool_name']}**: {preview}")
            active_step.output = "\n".join(lines)
            tool_results_buffer.clear()
        await _close_step(active_step)
        active_step = None

    try:
        async for event in agent.stream_run(task=task):
            kind = event.get("kind")

            if kind == "status":
                await open_step(event["text"])

            elif kind == "token":
                # First token: close any open step and start streaming the answer
                if active_step is not None:
                    await close_current_step()
                streamed_tokens = True
                await answer_msg.stream_token(event["text"])

            elif kind == "tool_results":
                # Buffer tool results — they're flushed when the next step opens
                tool_results_buffer.extend(event.get("results", []))

            elif kind == "done":
                await close_current_step()

                final_answer: str = event.get("final_answer", "")
                artifacts: list[str] = event.get("artifacts") or []
                n_reflects = event.get("reflection_count", 0)
                success: bool = event.get("success", True)
                run_id: str = event.get("run_id", "")

                if not streamed_tokens:
                    answer_msg.content = final_answer

                footer_parts = [f"run `{run_id[:8]}…`"]
                if n_reflects:
                    footer_parts.append(f"{n_reflects} reflection(s)")
                if not success:
                    footer_parts.append("⚠️ run marked as failed")
                footer = "  \n*" + " · ".join(footer_parts) + "*"

                if artifacts:
                    art_list = "\n".join(f"- `{a}`" for a in artifacts)
                    footer += f"\n\n📁 **Stored artifacts:**\n{art_list}"

                answer_msg.content += footer
                await answer_msg.send()
                return

            elif kind == "approval_needed":
                await close_current_step()
                run_id = event.get("run_id", "")
                payload = event.get("payload") or {}
                planned = payload.get("planned_tools") or []

                tool_lines = "\n".join(
                    f"- **{t['name']}**  `{str(t.get('arguments', {}))[:120]}`"
                    for t in planned
                )

                res = await cl.AskActionMessage(
                    content=(
                        f"⏸️ **Approval required** — run `{run_id[:8]}…`\n\n"
                        f"The agent wants to call:\n{tool_lines or '(no details)'}\n\n"
                        "What should it do?"
                    ),
                    actions=[
                        cl.Action(name="approve", value="approve", label="✅ Approve & run"),
                        cl.Action(name="reject", value="reject", label="❌ Reject"),
                    ],
                    timeout=300,
                ).send()

                if res is None or res.get("value") != "approve":
                    # Ask for rejection reason
                    reason_res = await cl.AskUserMessage(
                        content="📝 Enter a rejection reason (or press Enter to skip):",
                        timeout=120,
                    ).send()
                    reason = (
                        (reason_res.get("output") or "Rejected by operator.").strip()
                        if reason_res
                        else "Rejected by operator."
                    )
                    decision: dict = {"action": "reject", "reason": reason}
                    await cl.Message(content=f"❌ Rejected: *{reason}*").send()
                else:
                    decision = {"action": "approve"}
                    await cl.Message(content="✅ Approved — resuming…").send()

                # Resume streaming
                resume_answer = cl.Message(content="")
                streamed_tokens = False
                active_step = None

                async for ev2 in agent.resume_stream_run(run_id=run_id, decision=decision):
                    k2 = ev2.get("kind")
                    if k2 == "status":
                        await open_step(ev2["text"])
                    elif k2 == "token":
                        if active_step is not None:
                            await close_current_step()
                        streamed_tokens = True
                        await resume_answer.stream_token(ev2["text"])
                    elif k2 == "tool_results":
                        tool_results_buffer.extend(ev2.get("results", []))
                    elif k2 == "done":
                        await close_current_step()
                        final = ev2.get("final_answer", "")
                        if not streamed_tokens:
                            resume_answer.content = final
                        await resume_answer.send()
                        return
                    elif k2 == "error":
                        await close_current_step()
                        await cl.Message(content=f"❌ **Error**: {ev2.get('text', '')}").send()
                        return
                return

            elif kind == "error":
                await close_current_step()
                await cl.Message(
                    content=f"❌ **Error**: {event.get('text', 'Unknown error')}",
                ).send()
                return

    except asyncio.CancelledError:
        await close_current_step()
        await cl.Message(content="⚠️ Run cancelled.").send()
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error in _run_agent: %s", exc)
        await close_current_step()
        await cl.Message(content=f"❌ **Unexpected error**: {exc}").send()
