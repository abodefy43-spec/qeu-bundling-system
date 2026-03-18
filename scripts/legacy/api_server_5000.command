#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN=""
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
  PYTHON_BIN="python"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python not found. Install Python or create .venv first."
  read -r -p "Press Enter to close..."
  exit 1
fi

LAN_IP="$("$PYTHON_BIN" - <<'PY'
import socket

ip = ""
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.connect(("8.8.8.8", 80))
    ip = sock.getsockname()[0]
except OSError:
    pass
finally:
    sock.close()
print(ip)
PY
)"

PORT=5002
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port 5002 is busy. Stop the existing process and retry."
  read -r -p "Press Enter to close..."
  exit 1
fi

"$PYTHON_BIN" -m qeu_bundling.api.server --host 0.0.0.0 --port "$PORT"
read -r -p "Press Enter to close..."
