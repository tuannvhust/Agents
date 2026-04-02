from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

pytestmark = pytest.mark.asyncio


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _eval_cases_available(eval_file: Path) -> bool:
    if not eval_file.exists():
        return False
    try:
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        return bool(data.get("eval_cases"))
    except Exception:  # noqa: BLE001
        return False


def _backend_available(base_url: str) -> bool:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.adk
async def test_adk_output_regression_eval_sets() -> None:
    """Run ADK eval cases for coder/researcher/analyst proxy agents.

    This test only runs when:
    - google-adk is installed
    - at least one eval file contains real eval_cases
    - backend API is reachable
    - GOOGLE_API_KEY is configured (required by ADK LLM agent runtime)
    """
    try:
        from google.adk.evaluation.agent_evaluator import AgentEvaluator
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"google-adk not installed or unavailable: {exc}")

    if not os.getenv("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY is not set; skipping ADK regression test.")

    base_url = os.getenv("ADK_TARGET_API_BASE_URL", "http://localhost:8080")
    if not _backend_available(base_url):
        pytest.skip(f"Backend API not reachable at {base_url}; skipping ADK regression.")

    root = _repo_root()
    targets = [
        (
            root / "src/agent_system/adk/coder_proxy",
            root / "eval/adk/coder_proxy.test.json",
        ),
        (
            root / "src/agent_system/adk/researcher_proxy",
            root / "eval/adk/researcher_proxy.test.json",
        ),
        (
            root / "src/agent_system/adk/analyst_proxy",
            root / "eval/adk/analyst_proxy.test.json",
        ),
    ]

    runnable = [(agent, eval_file) for agent, eval_file in targets if _eval_cases_available(eval_file)]
    if not runnable:
        pytest.skip("No ADK eval cases found yet. Capture baseline via ADK Web first.")

    for agent_module_path, eval_file in runnable:
        await AgentEvaluator.evaluate(
            agent_module=str(agent_module_path),
            eval_dataset_file_path_or_dir=str(eval_file),
        )

