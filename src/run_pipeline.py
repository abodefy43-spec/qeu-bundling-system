"""Run the full QEU bundling pipeline (Phases 1-9)."""

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
    """Import a module whose filename starts with a digit (e.g. 01_load_data)."""
    module_name = filename.removesuffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    t0 = time.time()

    # Phase 1
    print("=" * 60)
    print("PHASE 1: Data Loading & Preprocessing")
    print("=" * 60)
    m = _import_phase("01_load_data.py")
    data = m.load_all(force_rebuild=True)
    print(f"  Orders: {len(data.orders):,} | Ingredients: {len(data.ingredient_index):,}\n")

    # Phase 2
    print("=" * 60)
    print("PHASE 2: Product Embeddings (Sentence Transformers, optimized)")
    print("=" * 60)
    m = _import_phase("02_embeddings.py")
    m.generate_embeddings()
    print()

    # Phase 3
    print("=" * 60)
    print("PHASE 3: Co-purchase Analysis")
    print("=" * 60)
    m = _import_phase("03_copurchase.py")
    m.compute_copurchase_scores()
    print()

    # Phase 4
    print("=" * 60)
    print("PHASE 4: Category Assignment")
    print("=" * 60)
    m = _import_phase("04_categories.py")
    m.assign_categories()
    print()

    # Phase 5
    print("=" * 60)
    print("PHASE 5: Recipe Scoring")
    print("=" * 60)
    m = _import_phase("05_recipe_scoring.py")
    m.compute_recipe_scores()
    print()

    # Phase 6
    print("=" * 60)
    print("PHASE 6: Bundle Selection (30/35/30)")
    print("=" * 60)
    m = _import_phase("06_bundle_selection.py")
    m.select_bundles()
    print()

    # Phase 7
    print("=" * 60)
    print("PHASE 7: ML Model Training")
    print("=" * 60)
    m = _import_phase("07_train_models.py")
    m.train_models()
    print()

    # Phase 8
    print("=" * 60)
    print("PHASE 8: Bundle Prediction & Final Output")
    print("=" * 60)
    m = _import_phase("08_predict.py")
    result = m.predict_bundles()
    print()

    # Phase 9
    print("=" * 60)
    print("PHASE 9: Performance Optimization")
    print("=" * 60)
    m = _import_phase("09_optimize.py")
    m.summarize_artifact_sizes()
    print()

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"PIPELINE COMPLETE  ({elapsed:.1f}s)")
    print(f"  Final bundles: {len(result)} rows")
    print(f"  Output: output/final_bundles.csv")
    print(f"  Models: output/free_item_model.pkl")
    print(f"          output/discount_model.pkl")
    print(f"          output/preprocessor.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
