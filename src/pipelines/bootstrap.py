"""Phase 0 bootstrap pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from data.paths import detect_legacy_runtime_paths, ensure_project_layout, get_project_paths
from features.registry import list_feature_specs
from shared.registry import list_engine_descriptors
from utils.runtime import timestamp_utc, write_json


@dataclass(frozen=True)
class Phase0BootstrapResult:
    manifest_path: Path
    status_path: Path
    report: dict[str, object]


def _build_report(project_root: str | Path | None = None) -> dict[str, object]:
    paths = ensure_project_layout(get_project_paths(project_root=project_root))
    engines = [descriptor.as_dict() for descriptor in list_engine_descriptors()]
    features = [spec.as_dict() for spec in list_feature_specs()]
    return {
        "phase": "phase0_cleanup",
        "generated_at": timestamp_utc(),
        "status": "ready_for_engine_implementation",
        "paths": {
            "root": str(paths.root),
            "raw": str(paths.raw_dir),
            "reference": str(paths.reference_dir),
            "processed": str(paths.processed_dir),
            "features": str(paths.features_dir),
            "artifacts": str(paths.artifacts_dir),
            "reports": str(paths.reports_dir),
            "runs": str(paths.runs_dir),
        },
        "engines": engines,
        "features": features,
        "legacy_runtime_paths": detect_legacy_runtime_paths(project_root=project_root),
    }


def bootstrap_phase0(project_root: str | Path | None = None) -> Phase0BootstrapResult:
    paths = ensure_project_layout(get_project_paths(project_root=project_root))
    report = _build_report(project_root=paths.root)
    stamp = timestamp_utc().replace(":", "").replace("-", "").replace("+00:00", "Z")
    manifest_path = paths.runs_dir / f"phase0_manifest_{stamp}.json"
    status_path = paths.reports_dir / "phase0_status.json"
    write_json(manifest_path, report)
    write_json(status_path, report)
    return Phase0BootstrapResult(
        manifest_path=manifest_path,
        status_path=status_path,
        report=report,
    )


def load_latest_status(project_root: str | Path | None = None) -> dict[str, object]:
    paths = ensure_project_layout(get_project_paths(project_root=project_root))
    status_path = paths.reports_dir / "phase0_status.json"
    if not status_path.exists():
        return bootstrap_phase0(project_root=paths.root).report
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["status_path"] = str(status_path)
    return payload
