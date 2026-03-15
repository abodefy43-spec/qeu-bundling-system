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
else
  echo "Python not found. Install Python or create .venv first."
  read -r -p "Press Enter to close..."
  exit 1
fi

PORT=5000
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=5001
  if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Ports 5000 and 5001 are busy. Stop other servers and try again."
    read -r -p "Press Enter to close..."
    exit 1
  fi
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
if [ -z "$LAN_IP" ]; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
fi

echo "============================================================"
echo "Network Share Mode"
echo "============================================================"
echo
echo "This starts the dashboard on this Mac and shares it on LAN."
export QEU_DASHBOARD_DEFAULT_PERSON_COUNT="${QEU_DASHBOARD_DEFAULT_PERSON_COUNT:-1}"
echo "Dashboard startup mode: QEU_DASHBOARD_DEFAULT_PERSON_COUNT=${QEU_DASHBOARD_DEFAULT_PERSON_COUNT}"
echo "Local URL: http://127.0.0.1:${PORT}"
if [ -n "$LAN_IP" ]; then
  echo "LAN URL:   http://${LAN_IP}:${PORT}"
else
  echo "LAN URL:   Could not auto-detect. Run ifconfig to get your IPv4."
fi
echo

(
  sleep 2
  open "http://127.0.0.1:${PORT}" >/dev/null 2>&1 || true
) &

"$PYTHON_BIN" -m qeu_bundling.cli serve --host 0.0.0.0 --port "$PORT"
read -r -p "Press Enter to close..."
