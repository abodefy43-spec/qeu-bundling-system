#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="python3"
if [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
fi

PORT=5000
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=5001
  echo "Port 5000 is busy. Falling back to port 5001."
fi

echo "Opening http://127.0.0.1:${PORT}"
open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
"$PYTHON_BIN" -m qeu_bundling.cli serve --host 127.0.0.1 --port "$PORT"
