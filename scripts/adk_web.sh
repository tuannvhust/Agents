#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADK_ROOT="${ROOT_DIR}/src/agent_system/adk"
PORT="${ADK_WEB_PORT:-8000}"

if [ ! -d "${ADK_ROOT}" ]; then
  echo "ADK root directory not found: ${ADK_ROOT}" >&2
  exit 1
fi

echo "[adk-web] starting from: ${ADK_ROOT}"
echo "[adk-web] url: http://localhost:${PORT}"
echo "[adk-web] note: set GOOGLE_API_KEY and ADK_TARGET_API_BASE_URL if needed"

cd "${ADK_ROOT}"
adk web --port "${PORT}"

