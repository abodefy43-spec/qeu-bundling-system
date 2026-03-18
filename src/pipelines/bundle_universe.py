"""Offline pipeline for materializing the reusable bundle universe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from engines.bundles.universe import (
    BundleUniverseMaterializationResult,
    materialize_bundle_universe,
)


@dataclass(frozen=True)
class BundleUniversePipelineResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, object]


def build_bundle_universe(
    *,
    project_root: str | Path | None = None,
    target_size: int = 100_000,
    per_root_limit: int = 18,
    root_limit: int | None = None,
) -> BundleUniversePipelineResult:
    result: BundleUniverseMaterializationResult = materialize_bundle_universe(
        project_root=project_root,
        target_size=target_size,
        per_root_limit=per_root_limit,
        root_limit=root_limit,
    )
    return BundleUniversePipelineResult(
        artifact_path=result.artifact_path,
        report_path=result.report_path,
        report=result.report,
    )
