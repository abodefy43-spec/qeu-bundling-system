#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "QEU Bundling System - Easy Run"
echo "============================================================"
echo
echo "[1] Quick refresh + open dashboard (Phases 6-9)"
echo "[2] Full rebuild + open dashboard (Phases 0-9)"
echo "[3] Open dashboard only"
echo "[4] Exit"
echo
read -r -p "Choose an option (1-4): " choice

case "$choice" in
  1)
    echo
    echo "Running quick refresh (new results) + presentation..."
    bash "$SCRIPT_DIR/localhost_fast_5000.command"
    status=$?
    if [ "$status" -ne 0 ]; then
      echo
      echo "Something failed. Check logs above and try again."
    fi
    ;;
  2)
    echo
    echo "Running full pipeline..."
    bash "$SCRIPT_DIR/localhost_full_5000.command"
    status=$?
    if [ "$status" -ne 0 ]; then
      echo
      echo "Something failed. Check logs above and try again."
    fi
    ;;
  3)
    echo
    echo "Opening dashboard URL only..."
    bash "$SCRIPT_DIR/open_localhost_5000.command"
    ;;
  4)
    ;;
  *)
    echo "Invalid option."
    ;;
esac

echo
read -r -p "Press Enter to close..."
