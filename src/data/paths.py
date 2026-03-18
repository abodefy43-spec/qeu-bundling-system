"""Canonical project paths for the cleaned repository."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LEGACY_RUNTIME_DIRS = ("output", "feedback", "rollback_safety_v2")


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src_dir: Path
    tests_dir: Path
    scripts_dir: Path
    legacy_dir: Path
    data_dir: Path
    raw_dir: Path
    reference_dir: Path
    processed_dir: Path
    features_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    runs_dir: Path


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_project_paths(project_root: str | Path | None = None) -> ProjectPaths:
    env_root = str(os.environ.get("QEU_PROJECT_ROOT", "") or "").strip()
    root = Path(project_root or env_root or _default_project_root()).resolve()
    data_dir = root / "data"
    processed_dir = data_dir / "processed"
    return ProjectPaths(
        root=root,
        src_dir=root / "src",
        tests_dir=root / "tests",
        scripts_dir=root / "scripts",
        legacy_dir=root / "legacy",
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        reference_dir=data_dir / "reference",
        processed_dir=processed_dir,
        features_dir=processed_dir / "features",
        artifacts_dir=processed_dir / "artifacts",
        reports_dir=processed_dir / "reports",
        runs_dir=processed_dir / "runs",
    )


def ensure_project_layout(paths: ProjectPaths | None = None) -> ProjectPaths:
    resolved = paths or get_project_paths()
    for directory in (
        resolved.src_dir,
        resolved.tests_dir,
        resolved.scripts_dir,
        resolved.legacy_dir,
        resolved.data_dir,
        resolved.raw_dir,
        resolved.reference_dir,
        resolved.processed_dir,
        resolved.features_dir,
        resolved.artifacts_dir,
        resolved.reports_dir,
        resolved.runs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return resolved


def detect_legacy_runtime_paths(project_root: str | Path | None = None) -> list[str]:
    paths = get_project_paths(project_root=project_root)
    found: list[str] = []
    for name in LEGACY_RUNTIME_DIRS:
        candidate = paths.root / name
        if candidate.exists():
            found.append(name)
    return found
