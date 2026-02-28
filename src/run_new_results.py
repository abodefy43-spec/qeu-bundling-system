"""Run only phases 6-9 to get new bundle results (skips data loading, embeddings, etc.)."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

for path in (str(PROJECT_ROOT), str(SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _import_phase(filename: str):
    module_name = filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()

    print("=" * 60)
    print("NEW RESULTS — Phases 6, 7, 8, 9 only (uses cached data)")
    print("=" * 60)

    # Phase 6: Bundle selection
    print("\nPHASE 6: Bundle Selection")
    print("-" * 40)
    m = _import_phase("06_bundle_selection.py")
    m.select_bundles()

    # Phase 7: ML model training
    print("\nPHASE 7: ML Model Training")
    print("-" * 40)
    m = _import_phase("07_train_models.py")
    m.train_models()

    # Phase 8: Prediction & output
    print("\nPHASE 8: Bundle Prediction & Output")
    print("-" * 40)
    m = _import_phase("08_predict.py")
    result = m.predict_bundles()

    # Phase 9: Artifact summary
    print("\nPHASE 9: Performance Optimization")
    print("-" * 40)
    m = _import_phase("09_optimize.py")
    m.summarize_artifact_sizes()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"DONE  ({elapsed:.1f}s)")
    print(f"  Output: output/top_10_bundles.csv")
    print(f"          output/final_bundles.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
