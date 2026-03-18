"""Compatibility shim for the archived pre-reset QEU package."""

from __future__ import annotations

from pathlib import Path

_legacy_pkg = Path(__file__).resolve().parent.parent / "src" / "legacy" / "qeu_bundling"
if _legacy_pkg.exists():
    __path__.append(str(_legacy_pkg))  # type: ignore[name-defined]
