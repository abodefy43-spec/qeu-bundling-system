# Serving Refactor Report: 2 Personalized + Curated Fallback-to-3

## Scope
This report covers the serving-path refactor in:
- `src/qeu_bundling/presentation/person_predictions.py`
- `scripts/validate_serving_output.py`
- `tests/test_person_predictions.py`

Goal: keep hard business constraints, simplify online selection, and serve exactly 3 bundles by using curated fallback when personalized slots are insufficient.

## Bottlenecks Found
1. Tier orchestration overhead from legacy multi-tier emergency logic, repeated rescue paths, and random resampling.
2. Personalized stage could occupy slots with candidates later rejected at finalize time (score-floor mismatch), preventing fallback fill.
3. Fallback path often failed in real data due strict quality assumptions on missing copurchase fields.
4. Nonfood fallback could dominate in some runs when not explicitly food-first.

## Architecture Changes
### New serving behavior
- Personalized stage targets at most 2 bundles, with optional 3rd only if explicitly strong.
- Fallback stage fills remaining slots to 3 from curated fallback candidates through the same final hygiene checks.
- Emergency tier3 online generation and random resample replacement were removed from the live path.

### Final selection / constraints preserved
- Hard no-item-reuse within person (`selected_item_ids` for both items in each bundle).
- Compatibility reject rules remain enforced.
- Final-stage tuna penalty and repetition penalty remain enforced.
- Fixed-margin pricing path unchanged.

### Fallback implementation details
- Built a curated fallback pool once per request (`_build_curated_fallback_pool`).
- Added robust scoring priors for missing copurchase metadata in curated rows.
- Food-first fallback fill; nonfood fallback only used when `include_nonfood` gate is on and after food attempts.
- Tier selection now applies score-floor checks before slot acceptance so invalid tier1 selections cannot block fallback fill.

## Validation Script Updates
`validate_serving_output.py` now reports:
- personalized vs fallback slot usage
- fallback user share and slot share
- top repeated pairs overall, personalized-only, fallback-only
- latency and stage timing summaries (avg/p50/p90/p95)

## Test Updates
- Updated tests tied to removed resample/emergency behavior.
- Added/kept coverage for:
  - exact-3 serving when feasible
  - no-item-reuse across personalized + fallback
  - compatibility and pricing constraints in filler behavior
  - fallback fill behavior and output mode semantics

Result: `137 passed` in `tests/test_person_predictions.py`.

## Before / After Metrics
### Latency (same 3-profile benchmark, instrumentation disabled)
- Before (pre-refactor snapshot):
  - `output/review/latency_after_s3_v3_noprofile/summary.json`
  - serving runtime: **69,122.626 ms**
- After (current refactor):
  - `output/review/latency_after_2p_s3_noprofile/summary.json`
  - serving runtime: **22,660.327 ms**

Approx improvement on this benchmark: **~67.2% faster**.

### Serving quality/coverage (50-profile run)
- `output/review/serving_validation_2p_fallback_v5/summary.json`
- Coverage:
  - 3 bundles: **50/50**
  - 2/1/0 bundles: **0/0/0**
- Hard rule compliance:
  - item reuse violations: **0**
  - compatibility violations: **0**
  - pricing violations: **0**
- Tier usage:
  - personalized slots: **19**
  - fallback slots: **131**
  - fallback user share: **0.94**

## Behavior-Preserving vs Approximation Changes
### Behavior-preserving
- No item reuse within person.
- Compatibility blocking and tuna/repetition penalties retained in final ranking.
- Pricing formula unchanged.

### Approximation / policy changes
- Personalized serving intentionally capped to 2 (optional strong 3rd).
- Removed random profile resampling path.
- Fallback copurchase evidence now uses deterministic priors when source fields are missing.
- Score-floor check moved into tier acceptance to avoid dead-slot selection.

## Remaining Risks
1. Fallback overuse is high in current 50-profile validation (expected quality risk if fallback library is too narrow).
2. Cross-user fallback repetition is still concentrated (top fallback pairs repeat heavily).
3. Candidate build stage remains dominant latency cost (~1.7s/profile average in profiled 50-user run).

## Next Refactors Recommended
1. Expand curated fallback library breadth (especially meal/snack/occasion alternatives) to reduce repetition.
2. Add stronger fallback de-duplication across users (motif and pair family penalties) tuned for fallback-only slots.
3. Cache anchor-level candidate retrieval across similar profiles to reduce candidate build time.
4. Add a strict `min_personalized_slots` KPI dashboard to monitor fallback over-reliance per run.
