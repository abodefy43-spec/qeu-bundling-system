"""Unified local entrypoint for the cleaned repository."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    src_dir = Path(__file__).resolve().parent / "src"
    src_path = str(src_dir)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def main(argv: list[str] | None = None) -> int:
    _bootstrap_src_path()
    from pipelines.cli import main as pipeline_main

    return int(pipeline_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
