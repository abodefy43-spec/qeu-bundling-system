# Recommender Diversity and Fallback Improvement Plan

## Diagnosis

1. **Fallback Dominance (83%)**: 
   - `_score_personal_candidate` calculates `_personal_score` (which includes the `top_bundle_bonus`), but the code immediately discards it and only uses `_premium_complement_score`. Top-bundle candidates thus lose their primary advantage and tie with or lose to fallbacks.
   - `top_bundle_rows_by_anchor` only contains ~1156 rows total globally. When a profile's history anchors aren't in this tiny pool, the system strictly falls back to `context.neighbors` (copurchase_fallback) and `FALLBACK_LANE_TEMPLATES`.

2. **Diversity Collapse (`unique_family_count_top10` = 1)**:
   - Hard diversity blockers (`blocked_themes`, `blocked_pair_fingerprints`, `blocked_groups`) are currently *only* enforced for `LANE_SNACK`. `LANE_MEAL` and `LANE_OCCASION` are allowed to freely duplicate families.
   - The soft penalty `family_penalty = min(0.24, 0.12 * family_reuse)` in `_premium_complement_score` is too weak to overcome the high base scores of fallback templates, leading to repeated `chips+soda` or `coffee+evap_milk` pairs across different anchors.

3. **Artifact Consistency**:
   - `run quick --eval-slice` produces `person_candidates_scored.csv`, which is then loaded by `evaluate_bundle_quality.py` and `people_predictions_review_export.py`. However, `review_export` creates its own subset of profiles and runs the prediction logic again to generate its `output/review/*.json`. The final gate metrics rely heavily on both `person_candidates_scored.csv` and `person_reco_quality.json`. We need to ensure that the scoring changes reflect in the exported JSON accurately.

## Execution Strategy

### 1. Fix Personalization Scoring (Restore Top-Bundle Priority)
- In `_evaluate_candidate` (inside `_pick_candidate_for_anchor`), merge `_personal_score` and `_premium_complement_score` so that `top_bundle_bonus`, `history_bonus`, and `brand_signal` are properly added to the final candidate score.
- Ensure the `_candidate_rank_key` explicitly sorts by `source` tier (0 for top_bundle, 1 for copurchase_fallback, 2 for template) so fallbacks cannot arbitrarily beat valid top-bundles on marginal score differences.

### 2. Enforce Global Diversity Constraints
- Apply `blocked_pair_fingerprints` and `blocked_themes` across *all* food lanes, not just `LANE_SNACK`.
- Increase the soft `family_penalty` for repeated complement families across the entire profile to forcefully demote redundant pairs (e.g. increase from max `0.24` to max `0.45`).

### 3. Verify Artifacts
- Run `python -m py_compile ...`
- Run the full test suite (`test_person_predictions.py`, `test_bundle_semantics.py`, `test_evaluate_bundle_quality.py`).
- Regenerate the review payload using `people_predictions_review_export.py`.
- Run `evaluate_bundle_quality.py` and `run quick --eval-slice` to confirm `fallback_share` drops significantly and `unique_family_count_top10` increases, while keeping semantic gates passing.

## Expected Outcomes
- `fallback_share` drops below 0.60.
- `unique_family_count_top10` increases to >= 4.
- `template_dup_rate` drops significantly.
- Strict semantic checks remain fully enforced.