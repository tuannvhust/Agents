# ADK Eval Assets

This folder stores ADK regression datasets (`*.test.json`) for proxy agents:

- `coder_proxy.test.json`
- `researcher_proxy.test.json`
- `analyst_proxy.test.json`

## How to capture benchmark responses

1. Start ADK web from `src/agent_system/adk`.
2. Chat with one proxy agent (`coder_proxy`, `researcher_proxy`, or `analyst_proxy`).
3. When the response is ideal, open **Eval** tab.
4. Select the matching eval set file and click **Add current session**.
5. Save/commit the updated `.test.json` file as ground truth.

## Naming convention

- One eval file per proxy agent.
- `eval_set_id` equals the filename stem.
- Keep `session_input.app_name` aligned with proxy folder name.

