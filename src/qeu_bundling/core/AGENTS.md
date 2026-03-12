# SUBDIRECTORY KNOWLEDGE BASE

## OVERVIEW
- `core/` is the shared helper layer for manifests, review artifacts, quality evaluation, feedback persistence, and translation/utility code reused by other package areas.

## WHERE TO LOOK
- Run metadata and latest-pointer logic - `run_manifest.py`
- Review artifact generation - `data_review_pack.py`
- Offline quality gates - `evaluate_bundle_quality.py`
- Pair-level feedback persistence - `feedback_memory.py`
- Deterministic people review export - `people_predictions_review_export.py`
- Arabic-to-English helper with cache/fallback behavior - `product_name_translation.py`
- Local test anchors - `tests/test_run_manifest_latest.py`, `tests/test_data_review_pack.py`, `tests/test_feedback_memory.py`

## CONVENTIONS
- Shared helpers should serve other package areas; keep ownership here on reusable IO, manifests, diagnostics, and evaluation utilities.
- Use `get_paths()` or path helpers built from it for filesystem writes, especially under `output/`, review, seeds, and processed feedback files.
- `run_manifest.py` owns latest-pointer and seed-history behavior; downstream readers should consume those helpers instead of rebuilding path logic.
- `data_review_pack.py` and `evaluate_bundle_quality.py` assume artifact-driven workflows and should stay aligned with generated file names.
- `feedback_memory.py` persists normalized pair-level feedback that presentation and scoring code can reuse.

## ANTI-PATTERNS
- Do not move phase sequencing or Flask route/session behavior into this file.
- Do not duplicate package-parent path guidance verbatim; keep only the local helper consequences of `get_paths()`.
