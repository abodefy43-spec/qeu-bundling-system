"""Run the bundle-universe materialization pipeline without PYTHONPATH tweaks."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from pipelines.bundle_universe import build_bundle_universe

    result = build_bundle_universe(project_root=root)
    print(result.artifact_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
