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

echo "============================================================"
echo "Localhost Full Pipeline + Dashboard (Port 5000)"
echo "============================================================"
echo

PORT=5000
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=5001
  echo "Port 5000 is busy. Falling back to port 5001."
  echo
fi

"$PYTHON_BIN" -m qeu_bundling.cli run full
status=$?
if [ "$status" -ne 0 ]; then
  echo "Full pipeline failed. Check logs above."
  read -r -p "Press Enter to close..."
  exit "$status"
fi

echo
echo "Opening http://127.0.0.1:${PORT}"
open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
"$PYTHON_BIN" -m qeu_bundling.cli serve --host 127.0.0.1 --port "$PORT"
read -r -p "Press Enter to close..."
