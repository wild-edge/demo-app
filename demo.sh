#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WILDEDGE_DSN:-}" ]]; then
  echo 'Warning: WILDEDGE_DSN is not set. Running without WildEdge cloud reporting.' >&2
  echo '  To enable monitoring, set WILDEDGE_DSN="https://<secret>@ingest.wildedge.dev/<key>"' >&2
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo 'Warning: OPENROUTER_API_KEY is not set. Ambiguous articles will return a fallback message instead of a remote summary.' >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

uv sync

# wildedge run execs uvicorn with wildedge/autoload/ prepended to PYTHONPATH.
# sitecustomize.py bootstraps the runtime before the app loads, instrumenting
# ONNX Runtime, llama-cpp-python, and the OpenAI client automatically.
#
# --reload spawns a fresh worker process via exec each time code changes.
# The module-level guard in sitecustomize.py ensures that worker bootstraps
# correctly (unlike the old os.environ guard, which propagated across exec
# and blocked the worker's init).
uv run wildedge run \
  --print-startup-report \
  --integrations onnx,gguf,openai \
  -- uvicorn app.main:app --reload --port 8002

# Test with:
#   curl -s -X POST http://localhost:8002/articles \
#     -H "Content-Type: application/json" \
#     -d '{"text": "Apple unveiled a groundbreaking new chip today."}' | jq .
