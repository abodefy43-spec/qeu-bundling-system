"""Materialize final API recommendations only from existing artifacts."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from qeu_bundling.config.paths import ensure_layout, get_paths
from qeu_bundling.core.final_recommendations import (
    bundle_ids_artifact_path,
    fallback_bundle_bank_artifact_path,
    materialize_final_recommendations_by_user,
    resolve_final_recommendations_max_users,
    resolve_final_recommendations_max_users_from_env,
    resolve_final_recommendations_random_seed,
    resolve_final_recommendations_random_seed_from_env,
    resolve_final_recommendations_user_selection_mode_from_env,
)

LOGGER = logging.getLogger("qeu_bundling.runners.run_materialize_final")


def _env_str(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or default).strip()


def _download_file_from_s3(bucket: str, key: str, target: Path, artifact_name: str) -> None:
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for S3 artifact bootstrap but is unavailable") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    LOGGER.info(
        "[batch] downloading artifact=%s s3://%s/%s -> %s",
        artifact_name,
        bucket,
        key,
        target,
    )
    try:
        boto3.client("s3").download_file(bucket, key, str(partial))
        partial.replace(target)
    except Exception as exc:
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        raise RuntimeError(f"Failed to download artifact {artifact_name} from s3://{bucket}/{key}") from exc

    LOGGER.info(
        "[batch] downloaded artifact=%s bytes=%d duration_sec=%.3f",
        artifact_name,
        int(target.stat().st_size) if target.exists() else 0,
        time.perf_counter() - started,
    )


def _bootstrap_required_artifacts(base_dir: Path) -> None:
    bucket = _env_str("QEU_ARTIFACTS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("QEU_ARTIFACTS_S3_BUCKET is required for materialize-final batch mode")

    paths = get_paths(project_root=base_dir)
    artifacts = [
        (
            "filtered_orders",
            paths.data_processed_dir / "filtered_orders.pkl",
            _env_str("QEU_S3_FILTERED_ORDERS_KEY", "processed/filtered_orders.pkl"),
        ),
        (
            "person_candidates_scored",
            paths.output_dir / "person_candidates_scored.csv",
            _env_str("QEU_S3_SCORED_CANDIDATES_KEY", "output/person_candidates_scored.csv"),
        ),
        (
            "person_candidate_pairs",
            paths.data_processed_candidates_dir / "person_candidate_pairs.csv",
            _env_str("QEU_S3_CANDIDATE_PAIRS_KEY", "processed/candidates/person_candidate_pairs.csv"),
        ),
    ]

    for artifact_name, target, key in artifacts:
        if target.exists():
            continue
        if not key:
            raise RuntimeError(f"S3 key for {artifact_name} is empty")
        _download_file_from_s3(bucket=bucket, key=key, target=target, artifact_name=artifact_name)


def main(
    max_users: int | None = None,
    random_sample: bool = False,
    random_seed: int | None = None,
) -> int:
    paths = get_paths()
    ensure_layout(paths)
    base_dir = paths.project_root.resolve()

    _bootstrap_required_artifacts(base_dir)

    resolved_max_users = (
        resolve_final_recommendations_max_users(max_users)
        if max_users is not None
        else resolve_final_recommendations_max_users_from_env()
    )
    resolved_selection_mode = "random" if random_sample else resolve_final_recommendations_user_selection_mode_from_env()
    resolved_random_seed = (
        resolve_final_recommendations_random_seed(random_seed)
        if random_seed is not None
        else resolve_final_recommendations_random_seed_from_env()
    )

    result = materialize_final_recommendations_by_user(
        base_dir=base_dir,
        max_users=resolved_max_users,
        user_selection_mode=resolved_selection_mode,
        random_seed=resolved_random_seed,
    )
    print(
        json.dumps(
            {
                "path": str(result.path),
                "run_id": str(result.run_id),
                "user_count": int(result.user_count),
                "profile_count": int(result.profile_count),
                "max_users": None if resolved_max_users is None else int(resolved_max_users),
                "selection_mode": str(resolved_selection_mode),
                "random_seed": None if resolved_random_seed is None else int(resolved_random_seed),
                "fallback_bundle_bank_path": str(fallback_bundle_bank_artifact_path(base_dir)),
                "bundle_ids_path": str(bundle_ids_artifact_path(base_dir)),
            },
            ensure_ascii=False,
        )
    )
    return 0
