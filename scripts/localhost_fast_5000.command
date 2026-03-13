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
echo "Localhost Fast Refresh + Dashboard (Port 5000)"
echo "============================================================"
echo

PORT=5000
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=5001
  echo "Port 5000 is busy. Falling back to port 5001."
  echo
fi

"$PYTHON_BIN" -m qeu_bundling.cli run quick
if [ "$?" -ne 0 ]; then
  echo "Quick refresh failed quality gates. Using latest available outputs to launch dashboard anyway."
fi

echo
echo "Starting dashboard server in this window..."
export QEU_LOCAL_FAST_MODE=1
export QEU_DASHBOARD_DEFAULT_PERSON_COUNT=5
echo "Browser will open automatically once the local dashboard is ready..."

(
  deadline=$((SECONDS + 120))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1; then
      open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
      exit 0
    fi
    sleep 0.5
  done
  exit 1
) &

"$PYTHON_BIN" -m qeu_bundling.cli serve --host 127.0.0.1 --port "$PORT"
