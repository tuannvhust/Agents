"""Google ADK adapter package for Agent System.

This package contains lightweight proxy agents used by ADK web/eval flows.
Each proxy forwards user tasks to the existing FastAPI endpoint
``/agents/{name}/run`` and returns the final answer text as benchmarkable output.
"""

