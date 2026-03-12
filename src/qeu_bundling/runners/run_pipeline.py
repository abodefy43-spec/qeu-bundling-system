"""Canonical full pipeline runner using structured phase names."""

from __future__ import annotations

import json
import time
import shutil
from pathlib import Path

from qeu_bundling.config.paths import ensure_layout, get_paths, move_root_temp_logs, run_output_dir
from qeu_bundling.core.data_review_pack import generate_review_pack
from qeu_bundling.core.evaluate_bundle_quality import evaluate_quality
from qeu_bundling.core.run_manifest import (
    append_seed_history,
    new_run_id,
    utc_now_iso,
    write_run_manifest,
)
from qeu_bundling.pipeline import phase_00_extract_campaigns as p00
from qeu_bundling.pipeline import phase_01_load_data as p01
from qeu_bundling.pipeline import phase_02_embeddings as p02
from qeu_bundling.pipeline import phase_03_copurchase as p03
from qeu_bundling.pipeline import phase_04_categories as p04
from qeu_bundling.pipeline import phase_05_recipe_scoring as p05
from qeu_bundling.pipeline import phase_06_bundle_selection as p06
from qeu_bundling.pipeline import phase_07_train_models as p07
from qeu_bundling.pipeline import phase_08_predict as p08
from qeu_bundling.pipeline import phase_09_optimize as p09


def resolve_full_seed(seed: int | None) -> int:
    return 42 if seed is None else int(seed)


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
    return {
        "phase": phase,
        "status": status,
        "duration_sec": round(float(elapsed), 3),
        **kwargs,
    }


def main(seed: int | None = None, eval_slice: bool = False):
    moved_logs = move_root_temp_logs()
    if moved_logs:
        print(f"Moved {len(moved_logs)} root temp log(s) into output/logs.")
    run_seed = resolve_full_seed(seed)
    if eval_slice and seed is None:
        run_seed = 20260201
    started_at = utc_now_iso()
    t0 = time.time()
    run_id = new_run_id("full")
    eval_start = "2026-02-01"
    eval_end = "2026-02-15"
    phase_records: list[dict[str, object]] = []

    print("=" * 60)
    print("PHASE 0: Extract Campaigns & Product Pictures")
    print("=" * 60)
    print(f"Seed: {run_seed}")
    if eval_slice:
        print(f"Eval slice enabled: {eval_start} -> {eval_end}")
    phase_t = time.time()
    _log_event(run_id, "phase_00", "start")
    p00.run()
    phase_records.append(_phase_record("phase_00", phase_t, "completed"))
    _log_event(run_id, "phase_00", "completed")
    print()

    print("=" * 60)
    print("PHASE 1: Data Loading & Preprocessing")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_01", "start")
    data = p01.run(
        force_rebuild=True,
        start_date=eval_start if eval_slice else p01.DEFAULT_START_DATE,
        end_date=eval_end if eval_slice else p01.DEFAULT_END_DATE,
    )
    phase_records.append(
        _phase_record(
            "phase_01",
            phase_t,
            "completed",
            orders_rows=int(len(data.orders)),
            eval_slice=bool(eval_slice),
        )
    )
    _log_event(run_id, "phase_01", "completed", orders_rows=int(len(data.orders)))
    print(f"  Orders: {len(data.orders):,} | Ingredients: {len(data.ingredient_index):,}\n")

    print("=" * 60)
    print("PHASE 2: Product Embeddings (Sentence Transformers, optimized)")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_02", "start")
    p02.run()
    phase_records.append(_phase_record("phase_02", phase_t, "completed"))
    _log_event(run_id, "phase_02", "completed")
    print()

    print("=" * 60)
    print("PHASE 3: Co-purchase Analysis")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_03", "start")
    p03.run()
    phase_records.append(_phase_record("phase_03", phase_t, "completed"))
    _log_event(run_id, "phase_03", "completed")
    print()

    print("=" * 60)
    print("PHASE 4: Category Assignment")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_04", "start")
    p04.run()
    phase_records.append(_phase_record("phase_04", phase_t, "completed"))
    _log_event(run_id, "phase_04", "completed")
    print()

    print("=" * 60)
    print("PHASE 5: Recipe Scoring")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_05", "start")
    p05.run()
    phase_records.append(_phase_record("phase_05", phase_t, "completed"))
    _log_event(run_id, "phase_05", "completed")
    print()

    print("=" * 60)
    print("PHASE 6: Bundle Selection")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_06", "start")
    p06.run(run_seed=run_seed)
    phase_records.append(_phase_record("phase_06", phase_t, "completed"))
    _log_event(run_id, "phase_06", "completed")
    print()

    print("=" * 60)
    print("PHASE 7: ML Model Training")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_07", "start")
    p07.run()
    phase_records.append(_phase_record("phase_07", phase_t, "completed"))
    _log_event(run_id, "phase_07", "completed")
    print()

    print("=" * 60)
    print("PHASE 8: Bundle Prediction & Final Output")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_08", "start")
    result = p08.run(run_seed=run_seed)
    phase_records.append(_phase_record("phase_08", phase_t, "completed", final_rows=int(len(result))))
    _log_event(run_id, "phase_08", "completed", final_rows=int(len(result)))
    print()

    print("=" * 60)
    print("PHASE 9: Performance Optimization")
    print("=" * 60)
    phase_t = time.time()
    _log_event(run_id, "phase_09", "start")
    p09.run()
    phase_records.append(_phase_record("phase_09", phase_t, "completed"))
    _log_event(run_id, "phase_09", "completed")
    print()

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
    _write_run_metadata("full", run_seed, started_at, finished_at, eval_slice=eval_slice)
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
        "model_metrics": paths.output_dir / "model_metrics.json",
        "bundle_quality_metrics": paths.output_dir / "bundle_quality_metrics.json",
        "pair_scoring_breakdown": paths.data_processed_diagnostics_dir / "pair_scoring_breakdown.csv",
        "suspicious_pairs_audit": paths.data_processed_diagnostics_dir / "suspicious_pairs_audit.csv",
    }
    copied_artifacts = _copy_artifacts_to_run_dir(run_dir, artifact_paths)
    manifest_path = write_run_manifest(
        mode="full",
        run_id=run_id,
        seed=run_seed,
        started_at=started_at,
        finished_at=finished_at,
        phases=phase_records,
        artifact_paths=copied_artifacts,
    )
    append_seed_history(
        mode="full",
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
        mode="full",
        run_id=run_id,
        seed=run_seed,
        started_at=started_at,
        finished_at=finished_at,
        phases=phase_records,
        artifact_paths={**copied_artifacts, **review_artifacts},
    )

    print("=" * 60)
    print(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  Scored candidate rows: {len(result)}")
    print(f"  Output: {paths.output_dir / 'person_candidates_scored.csv'}")
    print(f"  Metadata: {paths.output_dir / 'run_metadata.json'}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Models: {paths.output_dir / 'free_item_model.pkl'}")
    print(f"          {paths.output_dir / 'discount_model.pkl'}")
    print(f"          {paths.output_dir / 'preprocessor.pkl'}")
    print("=" * 60)
    if not quality_passed:
        raise RuntimeError("Critical quality gates failed. See output/bundle_quality_metrics.json")


if __name__ == "__main__":
    main()
