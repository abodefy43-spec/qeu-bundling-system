---
name: Bundle Quality Recovery
overview: Simplify recommendation text, prioritize frequently bought products, eliminate weak lexical matches (like corn-only), invert embedding similarity preference, and upgrade embeddings with a full Phase 2 rebuild.
todos:
  - id: clean-reco-text
    content: Simplify people recommendation text to concise product-name lists only
    status: pending
  - id: freq-weighted-reco
    content: Add frequency-aware relevance to person recommendation scoring
    status: pending
  - id: lexical-artifact-guard
    content: Add guard against weak token-only pairing artifacts in phase 6
    status: pending
  - id: invert-embedding-score
    content: Use diversity-oriented embedding contribution in bundle ranking
    status: pending
  - id: upgrade-embedding-model
    content: Upgrade embedding model in phase 2 and keep downstream compatibility
    status: pending
  - id: rebuild-and-validate
    content: Run full rebuild + new results and verify quality/output requirements
    status: pending
isProject: false
---

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

