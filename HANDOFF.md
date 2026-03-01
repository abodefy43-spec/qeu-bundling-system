# Bundle Quality Recovery

## What will be fixed

- Remove huge history paragraphs from people cards and show only concise product-name lists.
- Push recommendations toward products customers usually buy (frequency-aware relevance).
- Stop weak pairing artifacts like flour+corn-flakes from single-token overlap logic.
- Rank high-similarity pairs lower (favor embedding diversity).
- Upgrade embedding quality with a stronger multilingual model and full rebuild.

## Files to update

- [d:/Machine learning/Qeu/Bundling system/templates/dashboard.html](d:/Machine learning/Qeu/Bundling system/templates/dashboard.html)
- [d:/Machine learning/Qeu/Bundling system/src/presentation/bundle_view.py](d:/Machine learning/Qeu/Bundling system/src/presentation/bundle_view.py)
- [d:/Machine learning/Qeu/Bundling system/src/06_bundle_selection.py](d:/Machine learning/Qeu/Bundling system/src/06_bundle_selection.py)
- [d:/Machine learning/Qeu/Bundling system/src/02_embeddings.py](d:/Machine learning/Qeu/Bundling system/src/02_embeddings.py)

## Implementation plan

- **Recommendation text cleanup**
  - In `bundle_view.py`, limit the “based on” list to clean product names only (no long paragraph dump), deduplicate names, and cap display count (e.g., top 8-12 names).
  - In `dashboard.html`, render compact lines only:
    - chosen bundle product names
    - free product name
    - short “based on products” name list
  - Remove verbose count-heavy text blocks that hurt readability.
- **Make frequent buys more important**
  - In `bundle_view.py`, compute customer-profile product frequency from available purchase history and include a `history_popularity_score` for each candidate bundle.
  - Update person recommendation hybrid score to weight:
    - history match
    - purchase frequency relevance
    - bundle quality (`final_score`)
    - copurchase strength (`purchase_score`)
  - Keep diversity guard so the 10 people cards don’t collapse to near-identical bundles.
- **Block weak lexical artifacts (corn-only issue)**
  - In `06_bundle_selection.py`, add a lightweight name-based penalty/guard for pairs that only overlap on generic root tokens (e.g., single common token like “corn”) without stronger category evidence.
  - Require stronger evidence via specific shared tags and/or copurchase support before such pairs survive top ranking.
- **Invert embedding preference for ranking**
  - In `06_bundle_selection.py`, keep the similarity filter but stop rewarding high similarity in scoring.
  - Replace positive similarity contribution with a diversity-friendly score (e.g., `embedding_diversity_score = 100 - embedding_score`) and use that in `final_score` weighting.
  - Tighten similarity threshold if needed after inspection (e.g., reduce max allowed similarity).
- **Upgrade embeddings (full rebuild path)**
  - In `02_embeddings.py`, switch from current MiniLM multilingual model to a stronger multilingual sentence-transformer model for product-name semantics.
  - Keep normalized embeddings and sparse top-product similarity export unchanged so downstream phases remain compatible.
  - Rebuild embeddings via full pipeline phases (Phase 2+) so ranking uses improved vectors.

## Validation

- Run syntax/lint checks on changed files.
- Run full pipeline rebuild (includes embedding regeneration), then `new_results` flow.
- Verify `output/top_10_bundles.csv` for:
  - fewer nonsensical token-only pairs
  - stronger relevance to commonly bought items
  - continued free-item correctness (cheaper item free)
- Verify dashboard “People Recommendations” shows concise product-name text only (no giant history paragraph).

## Success criteria

- Recommendation cards are concise and readable.
- Top bundles are more behavior-relevant and less token-artifact driven.
- Embedding signal favors complementary diversity instead of near-duplicate similarity.
# QEU Product Bundling System — Handoff for Improvement

**Purpose:** This project creates "Buy X, Get Y Free/Cheap" bundle offers for a Saudi grocery retailer. Please fix all the issues below and improve the results.

**Zip file:** `QEU_Bundling_System.zip` (~230 MB) — share via Google Drive, OneDrive, WeTransfer, etc.

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline:**
   ```bash
   python src/run_pipeline.py
   ```
   Or `run_pipeline.bat` on Windows.

3. **Data location:** Order data is in `data first/order_items.csv`. The pipeline looks there automatically.

4. **Flask dashboard (optional):**
   ```bash
   python app.py
   ```
   Or `run_presentation.bat` — web UI to view bundles.

---

## CRITICAL ISSUES TO FIX

### Issue 1: Non-Reproducible Results (Random Noise)

**File:** `src/06_bundle_selection.py` (lines ~359-361)

**Problem:** Random noise (0-15) is added to scores on every run, so output changes every time. Makes it impossible to validate improvements.

**Fix:** Remove the random noise, or use a fixed seed (e.g. 42) for reproducibility. Same for the pool shuffle — use fixed `random_state=42`.

---

### Issue 2: Insecure Flask Secret Key

**File:** `app.py` (line ~17)

**Problem:** Hardcoded fallback `"qeu-presentation-secret"` is insecure.

**Fix:** Use `secrets.token_hex(32)` when `FLASK_SECRET_KEY` env var is not set, or require the env var in production.

---

### Issue 3: Duplicate Model Files

**File:** `src/07_train_models.py` (lines ~297-310)

**Problem:** Models saved twice (`free_item_model.pkl` + `bundle_free_item_model.pkl`, etc.) — wastes ~10MB.

**Fix:** Save only one set (`free_item_model.pkl`, `discount_model.pkl`). Phase 8 already has fallback to old names — update it to use only the standard names and remove the fallback.

---

### Issue 4: Missing .gitignore

**Problem:** No `.gitignore` for Python, data, output, IDE files.

**Fix:** Create `.gitignore` in project root with `__pycache__/`, `*.pkl`, `*.npy`, `*.npz`, `venv/`, `output/`, `.idea/`, etc.

---

### Issue 5: Same Product Type Bundles (e.g. Tomato Paste + Tomato Paste)

**File:** `src/06_bundle_selection.py` or `data/product_families.json`

**Problem:** Bundle #6 pairs two tomato paste brands (Saudi Tomato Paste + جورجينا معجون طماطم). Same product type, different brands — should be blocked.

**Fix:** Add `tomato_paste` family to `product_families.json`, or add brand/product-type differentiation logic to block same-product-type pairs.

---

### Issue 6: Arabic Names in Output

**Problem:** Many product names appear in Arabic (e.g. ساديا صدور دجاج متبل بالزبادي, جورجينا معجون طماطم). Translation exists (`product_name_translation.py`, `deep-translator`) but isn't applied in final output.

**Fix:** Ensure all product names are translated to English before writing `top_10_bundles.csv` and `final_bundles.csv`.

---

### Issue 7: Same-Product / Substitute Bundles

**Problem:** Despite product-family blocking, some bundles still pair similar items (chips + chips, rice + rice). `product_families.json` and `MAX_EMBEDDING_SIMILARITY` may need tuning.

**Fix:** Stronger same-product detection; add more product families; consider stricter embedding similarity.

---

### Issue 8: Theme Dominance

**Problem:** Previously cardamom dominated the top 10. Theme capping (`MAX_BUNDLES_PER_THEME`) helps but may need tuning.

**Fix:** Ensure diverse themes in top 10 (spices, dairy, snacks, beverages, etc.).

---

### Issue 9: Output File Confusion

**Problem:** Multiple output files (`final_bundles.csv`, `results_100_final.csv`, `results_final.csv`, `top_10_bundles.csv`) — purpose unclear.

**Fix:** Add `output/README.md` documenting each file's purpose.

---

### Issue 10: Data Folder Confusion

**Problem:** Raw data in `data first/` but processed in `data/` — unclear which is which.

**Fix:** Add `data first/README.md` explaining raw vs processed data.

---

## WHAT TO KEEP (Do NOT Change)

- **Complementary pairings** — Items that go together (bread + chicken, rice + sauce) are good. Odd/creative combos are desired.
- **ML-predicted discounts** — Keep 0–30% cap. Variation in discount % is fine.
- **Pipeline variation** — After fixing Issue 1, you may optionally add back *controlled* variation (e.g. fixed-seed shuffle) if you want different runs to surface different good bundles — but reproducibility should be the default.

---

## What Good Results Look Like

- **All product names in English**
- **No same-product variants** (no chips + chips, rice + rice, tomato paste + tomato paste)
- **Complementary pairings** (items that go together)
- **Diverse themes and categories**
- **Ramadan/Saudi recipe items** prioritized where relevant
- **Reproducible** — same output when run twice

---

## Key Files

| File | Purpose |
|------|---------|
| `src/01_load_data.py` | Load orders, recipe data; product name parsing |
| `src/04_categories.py` | Category assignment, product families |
| `src/06_bundle_selection.py` | Bundle scoring, theme capping, filters |
| `src/08_predict.py` | ML prediction, top-10 selection, output |
| `data/product_families.json` | Product family taxonomy |
| `data/theme_tokens.json` | Theme keywords |
| `prompts/cursor_prompt_5.md` | Detailed fix instructions for Issues 1–5 |

---

## Priority Order

1. Issue 1 (reproducibility) — most critical  
2. Issue 2 (secret key) — security  
3. Issue 3 (duplicate models) — cleanup  
4. Issue 6 (Arabic names) — output quality  
5. Issue 5 (tomato paste) — bundle quality  
6. Issue 7 (same-product bundles) — bundle quality  
7. Issues 4, 8, 9, 10 — improvements  

---

## Verification

After fixes:

- [ ] Running pipeline twice produces same output
- [ ] No duplicate `bundle_*.pkl` files in output/
- [ ] All product names in `top_10_bundles.csv` are in English
- [ ] No tomato paste + tomato paste (or similar) in top 10
- [ ] `.gitignore` and READMEs in place

Thank you!
