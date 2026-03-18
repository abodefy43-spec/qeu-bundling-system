"""Centralized project path configuration."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_raw_dir: Path
    data_processed_dir: Path
    data_reference_dir: Path
    output_dir: Path
    output_runs_dir: Path
    output_latest_dir: Path
    output_logs_dir: Path
    output_review_dir: Path
    output_seeds_dir: Path
    data_processed_candidates_dir: Path
    data_processed_diagnostics_dir: Path


def _default_project_root() -> Path:
    # paths.py -> config -> qeu_bundling -> src -> project root
    return Path(__file__).resolve().parents[3]


def _resolve_dir(
    env_name: str,
    default: Path,
    root: Path,
) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def get_paths(project_root: Path | None = None) -> ProjectPaths:
    root = (project_root or _default_project_root()).resolve()
    data_processed_dir = _resolve_dir("QEU_DATA_PROCESSED_DIR", root / "data" / "processed", root)
    output_dir = _resolve_dir("QEU_OUTPUT_DIR", root / "output", root)
    return ProjectPaths(
        project_root=root,
        data_raw_dir=_resolve_dir("QEU_DATA_RAW_DIR", root / "data" / "raw", root),
        data_processed_dir=data_processed_dir,
        data_reference_dir=_resolve_dir("QEU_DATA_REFERENCE_DIR", root / "data" / "reference", root),
        output_dir=output_dir,
        output_runs_dir=output_dir / "runs",
        output_latest_dir=output_dir / "latest",
        output_logs_dir=output_dir / "logs",
        output_review_dir=output_dir / "review",
        output_seeds_dir=output_dir / "seeds",
        data_processed_candidates_dir=data_processed_dir / "candidates",
        data_processed_diagnostics_dir=data_processed_dir / "diagnostics",
    )


def ensure_layout(paths: ProjectPaths) -> None:
    paths.data_raw_dir.mkdir(parents=True, exist_ok=True)
    paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    paths.data_reference_dir.mkdir(parents=True, exist_ok=True)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    paths.output_runs_dir.mkdir(parents=True, exist_ok=True)
    paths.output_latest_dir.mkdir(parents=True, exist_ok=True)
    paths.output_logs_dir.mkdir(parents=True, exist_ok=True)
    paths.output_review_dir.mkdir(parents=True, exist_ok=True)
    paths.output_seeds_dir.mkdir(parents=True, exist_ok=True)
    paths.data_processed_candidates_dir.mkdir(parents=True, exist_ok=True)
    paths.data_processed_diagnostics_dir.mkdir(parents=True, exist_ok=True)


def run_output_dir(run_id: str, project_root: Path | None = None) -> Path:
    paths = get_paths(project_root=project_root)
    out_dir = paths.output_runs_dir / str(run_id).strip()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def latest_manifest_path(project_root: Path | None = None) -> Path:
    paths = get_paths(project_root=project_root)
    paths.output_latest_dir.mkdir(parents=True, exist_ok=True)
    return paths.output_latest_dir / "latest_manifest.json"


def log_path_for(mode: str, project_root: Path | None = None) -> Path:
    paths = get_paths(project_root=project_root)
    paths.output_logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_mode = str(mode or "run").strip().lower().replace(" ", "_")
    return paths.output_logs_dir / f"{safe_mode}_{stamp}.log"


def move_root_temp_logs(project_root: Path | None = None) -> list[Path]:
    paths = get_paths(project_root=project_root)
    moved: list[Path] = []
    for path in sorted(paths.project_root.glob(".tmp_*")):
        if not path.is_file():
            continue
        dest = paths.output_logs_dir / path.name.lstrip(".")
        suffix = 1
        while dest.exists():
            dest = paths.output_logs_dir / f"{path.stem.lstrip('.')}_{suffix}{path.suffix}"
            suffix += 1
        try:
            shutil.move(str(path), str(dest))
            moved.append(dest)
        except OSError:
            continue
    return moved


def migrate_legacy_data(paths: ProjectPaths) -> list[str]:
    """Migrate legacy data paths to the canonical layout."""
    ensure_layout(paths)
    moved: list[str] = []
    root = paths.project_root
    legacy_data_first = root / "data first"
    legacy_data = root / "data"

    if legacy_data_first.exists():
        for item in legacy_data_first.iterdir():
            dest = paths.data_raw_dir / item.name
            if item.resolve() == dest.resolve():
                continue
            if dest.exists():
                continue
            shutil.move(str(item), str(dest))
            moved.append(f"{item} -> {dest}")

    reference_names = {
        "recipe_data.json",
        "product_families.json",
        "theme_tokens.json",
        "category_importance.csv",
    }
    if legacy_data.exists():
        for item in legacy_data.iterdir():
            if not item.is_file():
                continue
            if item.name in reference_names:
                dest = paths.data_reference_dir / item.name
            else:
                dest = paths.data_processed_dir / item.name
            if item.resolve() == dest.resolve():
                continue
            if dest.exists():
                continue
            shutil.move(str(item), str(dest))
            moved.append(f"{item} -> {dest}")

    if legacy_data_first.exists():
        try:
            legacy_data_first.rmdir()
        except OSError:
            pass
    return moved
