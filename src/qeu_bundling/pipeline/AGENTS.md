# SUBDIRECTORY KNOWLEDGE BASE

## OVERVIEW
- `pipeline/` owns the phased artifact workflow from `phase_00` through `phase_09`, with full and quick runners consuming different slices of the sequence.

## WHERE TO LOOK
- Full phase order - `runners/run_pipeline.py`
- Quick phase order - `runners/run_new_results.py`
- Data ingestion and preprocessed artifact contract - `phase_01_load_data.py`
- Ranking and candidate artifact hotspot - `phase_06_bundle_selection.py`
- Model training hotspot - `phase_07_train_models.py`
- Prediction/output hotspot - `phase_08_predict.py`
- Phase-level checks - `tests/test_phase05_recipe_scoring.py`, `tests/test_phase07_model_candidates.py`, `tests/test_phase08_predict.py`

## CONVENTIONS
- Full runs execute `phase_00` to `phase_09`; quick runs reuse cached artifacts and run phases 6-9 only.
- Pipeline modules ingest `data/raw`, `data/reference`, and `data/processed`, then write candidate, diagnostics, model, and scored outputs consumed by later phases and runners.
- Keep phase artifact names stable across pipeline steps; runner and core layers own manifest writing after phase execution.
- `phase_01_load_data.py`, `phase_06_bundle_selection.py`, `phase_07_train_models.py`, and `phase_08_predict.py` are the main hotspots before changing data contracts, ranking, training, or scoring behavior.
- Path resolution still flows through `get_paths()` even when modules create candidate or diagnostics directories.

## ANTI-PATTERNS
- Do not mix Flask, dashboard, `templates/`, or `static/` guidance into this file.
- Do not describe root command catalogs when the local concern is artifact sequencing and phase ownership.
