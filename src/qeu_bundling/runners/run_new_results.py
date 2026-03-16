"""Canonical fast runner for phases 6-9 using structured phase names."""

from __future__ import annotations

import json
import secrets
import time
import shutil
from pathlib import Path

from qeu_bundling.config.paths import ensure_layout, get_paths, move_root_temp_logs, run_output_dir
from qeu_bundling.core.data_review_pack import generate_review_pack
from qeu_bundling.core.evaluate_bundle_quality import evaluate_quality
from qeu_bundling.core.final_recommendations import materialize_final_recommendations_by_user
from qeu_bundling.core.run_manifest import (
    append_seed_history,
    new_run_id,
    utc_now_iso,
    write_run_manifest,
)
from qeu_bundling.pipeline import phase_06_bundle_selection as p06
from qeu_bundling.pipeline import phase_07_train_models as p07
from qeu_bundling.pipeline import phase_08_predict as p08
from qeu_bundling.pipeline import phase_09_optimize as p09

REQUIRED_QUICK_MODEL_ARTIFACTS = (
    "free_item_model.pkl",
    "discount_model.pkl",
    "preprocessor.pkl",
)


def resolve_quick_seed(seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    return int(secrets.randbelow(2**31 - 1) + 1)


def _write_run_metadata(mode: str, seed: int, started_at: str, finished_at: str, eval_slice: bool = False) -> None:
    paths = get_paths()
    ensure_layout(paths)
    payload = {
        "mode": mode,
        "seed": int(seed),
        "started_at": started_at,
        "finished_at": finished_at,
        "eval_slice": bool(eval_slice),
    }
    with (paths.output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _copy_artifacts_to_run_dir(run_dir: Path, artifact_paths: dict[str, Path]) -> dict[str, Path]:
    copied: dict[str, Path] = {}
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for key, src in artifact_paths.items():
        if not src.exists():
            copied[key] = src
            continue
        dest = artifacts_dir / src.name
        try:
            shutil.copy2(src, dest)
            copied[key] = dest
        except OSError:
            copied[key] = src
    return copied


def _log_event(run_id: str, phase: str, event: str, **kwargs: object) -> None:
    payload = {
        "ts_utc": utc_now_iso(),
        "run_id": run_id,
        "phase": phase,
        "event": event,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False))


def _phase_record(phase: str, started: float, status: str, **kwargs: object) -> dict[str, object]:
    elapsed = max(0.0, time.time() - started)
    return {"phase": phase, "status": status, "duration_sec": round(float(elapsed), 3), **kwargs}


def _quick_model_artifact_status() -> tuple[bool, list[str]]:
    output_dir = get_paths().output_dir
    missing = [name for name in REQUIRED_QUICK_MODEL_ARTIFACTS if not (output_dir / name).exists()]
    return (not missing, missing)


def main(seed: int | None = None, eval_slice: bool = False, retrain_models: bool = False):
    moved_logs = move_root_temp_logs()
    if moved_logs:
        print(f"Moved {len(moved_logs)} root temp log(s) into output/logs.")
    run_seed = resolve_quick_seed(seed)
    if eval_slice and seed is None:
        run_seed = 20260201
    started_at = utc_now_iso()
    t0 = time.time()
    run_id = new_run_id("quick")
    phase_records: list[dict[str, object]] = []

    print("=" * 60)
    print("NEW RESULTS - Phases 6, 7, 8, 9 only (uses cached data)")
    print(f"Seed: {run_seed}")
    if eval_slice:
        print("Eval slice mode: deterministic quick run (uses cached processed artifacts)")
    print("=" * 60)

    print("\nPHASE 6: Bundle Selection")
    print("-" * 40)
    phase_t = time.time()
    _log_event(run_id, "phase_06", "start")
    p06.run(run_seed=run_seed)
    phase_records.append(_phase_record("phase_06", phase_t, "completed"))
    _log_event(run_id, "phase_06", "completed")

    phase_t = time.time()
    models_ready, missing_models = _quick_model_artifact_status()
    if retrain_models:
        print("\nPHASE 7: ML Model Training")
        print("-" * 40)
        print("Forced model retraining enabled")
        _log_event(run_id, "phase_07", "start", forced=True)
        p07.run()
        phase_records.append(_phase_record("phase_07", phase_t, "completed", forced=True))
        _log_event(run_id, "phase_07", "completed", forced=True)
    elif models_ready:
        print("\nPHASE 7: ML Model Training")
        print("-" * 40)
        print("Reusing existing trained models; skipping phase 07")
        phase_records.append(
            _phase_record(
                "phase_07",
                phase_t,
                "skipped",
                reused_existing_models=True,
                required_artifacts=list(REQUIRED_QUICK_MODEL_ARTIFACTS),
            )
        )
        _log_event(
            run_id,
            "phase_07",
            "skipped",
            reused_existing_models=True,
            required_artifacts=list(REQUIRED_QUICK_MODEL_ARTIFACTS),
        )
    else:
        print("\nPHASE 7: ML Model Training")
        print("-" * 40)
        print(
            "Model artifacts missing; running phase 07 "
            f"({', '.join(missing_models)})"
        )
        _log_event(run_id, "phase_07", "start", missing_artifacts=missing_models)
        p07.run()
        phase_records.append(
            _phase_record(
                "phase_07",
                phase_t,
                "completed",
                missing_artifacts=missing_models,
            )
        )
        _log_event(run_id, "phase_07", "completed", missing_artifacts=missing_models)

    print("\nPHASE 8: Bundle Prediction & Output")
    print("-" * 40)
    phase_t = time.time()
    _log_event(run_id, "phase_08", "start")
    p08.run(run_seed=run_seed)
    phase_records.append(_phase_record("phase_08", phase_t, "completed"))
    _log_event(run_id, "phase_08", "completed")

    print("\nPHASE 8B: Materialize API Final Recommendations")
    print("-" * 40)
    phase_t = time.time()
    _log_event(run_id, "phase_08b", "start")
    final_reco = materialize_final_recommendations_by_user()
    phase_records.append(
        _phase_record(
            "phase_08b",
            phase_t,
            "completed",
            profile_count=int(final_reco.profile_count),
            user_count=int(final_reco.user_count),
        )
    )
    _log_event(
        run_id,
        "phase_08b",
        "completed",
        profile_count=int(final_reco.profile_count),
        user_count=int(final_reco.user_count),
    )
    print(f"  Output: {final_reco.path} (users={final_reco.user_count:,})")

    print("\nPHASE 9: Performance Optimization")
    print("-" * 40)
    phase_t = time.time()
    _log_event(run_id, "phase_09", "start")
    p09.run()
    phase_records.append(_phase_record("phase_09", phase_t, "completed"))
    _log_event(run_id, "phase_09", "completed")

    phase_t = time.time()
    _log_event(run_id, "quality_eval", "start")
    quality_metrics = evaluate_quality(save=True)
    quality_passed = bool(quality_metrics.get("critical_gates_passed", False))
    phase_records.append(
        _phase_record(
            "quality_eval",
            phase_t,
            "completed" if quality_passed else "failed",
            critical_gates_passed=quality_passed,
        )
    )
    _log_event(run_id, "quality_eval", "completed", critical_gates_passed=quality_passed)

    elapsed = time.time() - t0
    finished_at = utc_now_iso()
    _write_run_metadata("quick", run_seed, started_at, finished_at, eval_slice=eval_slice)

    paths = get_paths()
    run_dir = run_output_dir(run_id)
    run_metadata_src = paths.output_dir / "run_metadata.json"
    run_metadata_dest = run_dir / "run_metadata.json"
    if run_metadata_src.exists():
        try:
            shutil.copy2(run_metadata_src, run_metadata_dest)
        except OSError:
            pass

    artifact_paths = {
        "person_candidate_pairs": paths.data_processed_candidates_dir / "person_candidate_pairs.csv",
        "person_candidates_scored": paths.output_dir / "person_candidates_scored.csv",
        "person_reco_quality": paths.output_dir / "person_reco_quality.json",
        "final_recommendations_by_user": paths.output_dir / "final_recommendations_by_user.json",
        "bundle_quality_metrics": paths.output_dir / "bundle_quality_metrics.json",
        "pair_scoring_breakdown": paths.data_processed_diagnostics_dir / "pair_scoring_breakdown.csv",
        "suspicious_pairs_audit": paths.data_processed_diagnostics_dir / "suspicious_pairs_audit.csv",
    }
    copied_artifacts = _copy_artifacts_to_run_dir(run_dir, artifact_paths)
    manifest_path = write_run_manifest(
        mode="quick",
        run_id=run_id,
        seed=run_seed,
        started_at=started_at,
        finished_at=finished_at,
        phases=phase_records,
        artifact_paths=copied_artifacts,
    )
    append_seed_history(
        mode="quick",
        run_id=run_id,
        seed=run_seed,
        started_at=started_at,
        finished_at=finished_at,
    )
    review_paths = generate_review_pack()
    review_artifacts = {
        f"review_{k}": Path(v)
        for k, v in review_paths.items()
    }
    manifest_path = write_run_manifest(
        mode="quick",
        run_id=run_id,
        seed=run_seed,
        started_at=started_at,
        finished_at=finished_at,
        phases=phase_records,
        artifact_paths={**copied_artifacts, **review_artifacts},
    )
    print("\n" + "=" * 60)
    print(f"DONE  ({elapsed:.1f}s)")
    print(f"  Output: {paths.output_dir / 'person_candidates_scored.csv'}")
    print(f"          {paths.data_processed_candidates_dir / 'person_candidate_pairs.csv'}")
    print(f"          {paths.output_dir / 'run_metadata.json'}")
    print(f"          {manifest_path}")
    print("=" * 60)
    if not quality_passed:
        raise RuntimeError("Critical quality gates failed. See output/bundle_quality_metrics.json")


if __name__ == "__main__":
    main()
