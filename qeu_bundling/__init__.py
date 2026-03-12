"""Compatibility shim so `python -m qeu_bundling...` works without installation."""

from __future__ import annotations

from pathlib import Path

# Add src/qeu_bundling to this package path so submodules resolve.
_src_pkg = Path(__file__).resolve().parent.parent / "src" / "qeu_bundling"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[name-defined]
