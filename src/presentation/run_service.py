"""Background pipeline run management for the presentation app."""

from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunState:
    running: bool = False
    last_started_utc: str = ""
    last_finished_utc: str = ""
    last_exit_code: int | None = None
    last_error: str = ""
    log_tail: deque[str] = field(default_factory=lambda: deque(maxlen=400))


_state = RunState()
_lock = threading.Lock()


def get_status() -> dict[str, object]:
    with _lock:
        return {
            "running": _state.running,
            "last_started_utc": _state.last_started_utc,
            "last_finished_utc": _state.last_finished_utc,
            "last_exit_code": _state.last_exit_code,
            "last_error": _state.last_error,
            "log_tail": list(_state.log_tail),
        }


def start_pipeline_run(project_root: Path) -> tuple[bool, str]:
    """Start a background run if not already running."""
    with _lock:
        if _state.running:
            return False, "A pipeline refresh is already running."
        _state.running = True
        _state.last_started_utc = _utc_now_iso()
        _state.last_finished_utc = ""
        _state.last_exit_code = None
        _state.last_error = ""
        _state.log_tail.clear()
        _state.log_tail.append(f"[{_state.last_started_utc}] Refresh started.")

    t = threading.Thread(
        target=_run_pipeline,
        args=(project_root,),
        daemon=True,
        name="bundle-pipeline-runner",
    )
    t.start()
    return True, "Pipeline refresh started."


def _run_pipeline(project_root: Path) -> None:
    cmd = [sys.executable, str(project_root / "src" / "run_pipeline.py")]
    proc: subprocess.Popen[str] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            with _lock:
                _state.log_tail.append(line)
        code = proc.wait()
        with _lock:
            _state.running = False
            _state.last_finished_utc = _utc_now_iso()
            _state.last_exit_code = int(code)
            if code == 0:
                _state.log_tail.append(f"[{_state.last_finished_utc}] Refresh completed successfully.")
            else:
                _state.last_error = f"Pipeline exited with code {code}"
                _state.log_tail.append(f"[{_state.last_finished_utc}] {_state.last_error}")
    except Exception as exc:
        with _lock:
            _state.running = False
            _state.last_finished_utc = _utc_now_iso()
            _state.last_exit_code = -1
            _state.last_error = f"Pipeline run failed: {exc}"
            _state.log_tail.append(f"[{_state.last_finished_utc}] {_state.last_error}")
        if proc and proc.poll() is None:
            proc.kill()

