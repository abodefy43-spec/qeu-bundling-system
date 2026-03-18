"""Run the Phase 0 API without needing PYTHONPATH tweaks."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from api.app import run_api

    run_api()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
