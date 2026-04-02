#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AGENT_KEY="${1:-all}"        # all | coder | researcher | analyst
CONFIG_FILE="${2:-}"         # optional config file path

run_eval() {
  local module_path="$1"
  local eval_file="$2"

  if [ ! -f "${eval_file}" ]; then
    echo "[adk-eval] skip missing file: ${eval_file}"
    return 0
  fi

  echo "[adk-eval] module=${module_path}"
  echo "[adk-eval] eval=${eval_file}"
  if [ -n "${CONFIG_FILE}" ]; then
    adk eval "${module_path}" "${eval_file}" --config_file_path "${CONFIG_FILE}" --print_detailed_results
  else
    adk eval "${module_path}" "${eval_file}" --print_detailed_results
  fi
}

case "${AGENT_KEY}" in
  coder)
    run_eval "${ROOT_DIR}/src/agent_system/adk/coder_proxy" "${ROOT_DIR}/eval/adk/coder_proxy.test.json"
    ;;
  researcher)
    run_eval "${ROOT_DIR}/src/agent_system/adk/researcher_proxy" "${ROOT_DIR}/eval/adk/researcher_proxy.test.json"
    ;;
  analyst)
    run_eval "${ROOT_DIR}/src/agent_system/adk/analyst_proxy" "${ROOT_DIR}/eval/adk/analyst_proxy.test.json"
    ;;
  all)
    run_eval "${ROOT_DIR}/src/agent_system/adk/coder_proxy" "${ROOT_DIR}/eval/adk/coder_proxy.test.json"
    run_eval "${ROOT_DIR}/src/agent_system/adk/researcher_proxy" "${ROOT_DIR}/eval/adk/researcher_proxy.test.json"
    run_eval "${ROOT_DIR}/src/agent_system/adk/analyst_proxy" "${ROOT_DIR}/eval/adk/analyst_proxy.test.json"
    ;;
  *)
    echo "Usage: scripts/adk_eval.sh [all|coder|researcher|analyst] [optional_config_file]" >&2
    exit 2
    ;;
esac

