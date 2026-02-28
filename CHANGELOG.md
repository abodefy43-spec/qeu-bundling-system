# Changelog

## 2026-02-28 — Bug Fixes & Improvements

All 10 issues from HANDOFF.md have been addressed.

### Critical Fixes

**1. Non-Reproducible Results (Issue #1)**
- `src/06_bundle_selection.py`: Removed random noise injection (`rng.uniform(0, 15.0)`) that was seeded with `time.time()`, causing different output on every run. Pool shuffle now uses `random_state=42`.
- `src/08_predict.py`: Same fix — removed `rng.uniform(0, 12.0)` noise and time-seeded RNG in `_select_top_compatible()`. Pool shuffle now uses `random_state=42`.
- Removed unused `import time` from both files.

**2. Insecure Flask Secret Key (Issue #2)**
- `app.py`: Replaced hardcoded `"qeu-presentation-secret"` fallback with `secrets.token_hex(32)`. The app now generates a secure random key when `FLASK_SECRET_KEY` env var is not set.

**3. Duplicate Model Files (Issue #3)**
- `src/07_train_models.py`: Removed the duplicate saves of `bundle_free_item_model.pkl` and `bundle_discount_model.pkl`. Only `free_item_model.pkl`, `discount_model.pkl`, and `preprocessor.pkl` are saved now.
- `src/08_predict.py`: Removed the fallback logic that tried `bundle_*` filenames when standard names were missing.
- Deleted the existing duplicate `.pkl` files from `output/`.

### Output Quality

**4. Arabic Names in Output (Issue #6)**
- `src/08_predict.py`: Added a translation step that runs `translate_arabic_to_english()` on both `product_a_name` and `product_b_name` before writing any output CSVs. Uses the existing `arabic_translations_cache.json` for performance.

**5. Same Product Type Bundles (Issues #5, #7)**
- `data/product_families.json`: Added 12 new product families to block same-type pairings:
  - `tomato_paste` — blocks tomato paste + tomato paste
  - `cooking_oil` — blocks sunflower oil + corn oil
  - `tea` — blocks black tea + green tea
  - `juice` — blocks orange juice + apple juice
  - `bread` — blocks white bread + brown bread
  - `yogurt` — blocks plain yogurt + greek yogurt
  - `chicken` — blocks chicken breast + chicken thigh
  - `sugar` — blocks white sugar + brown sugar
  - `flour` — blocks all-purpose flour + whole wheat flour
  - `tuna_canned_fish` — blocks tuna + canned tuna
  - `cardamom_spices` — blocks cardamom variants
  - All families include Arabic keyword variants.

**6. Theme Dominance (Issue #8)**
- `src/06_bundle_selection.py`: Reduced `MAX_BUNDLES_PER_THEME` from 12 to 3, ensuring no single theme (e.g. cardamom, spices) dominates the top bundles.

### Housekeeping

**7. Missing .gitignore (Issue #4)**
- Created `.gitignore` with exclusions for `__pycache__/`, `*.pkl`, `*.npy`, `*.npz`, `output/`, `.idea/`, `.vscode/`, `.DS_Store`, etc.

**8. Output File Documentation (Issue #9)**
- Created `output/README.md` documenting the purpose of each output file.

**9. Data Folder Documentation (Issue #10)**
- Created `data first/README.md` explaining that it contains original raw data files.

### Files Modified
- `src/06_bundle_selection.py`
- `src/07_train_models.py`
- `src/08_predict.py`
- `app.py`
- `data/product_families.json`

### Files Created
- `.gitignore`
- `output/README.md`
- `data first/README.md`

### Files Deleted
- `output/bundle_free_item_model.pkl`
- `output/bundle_discount_model.pkl`
