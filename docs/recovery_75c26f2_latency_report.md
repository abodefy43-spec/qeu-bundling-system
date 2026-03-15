# Recovery + Latency Report (Baseline `75c26f2`)

## 1) Recovery Baseline
- Baseline commit: `75c26f2`
- Recovery branch: `recovery_from_75c26f2`
- Branch creation command used: `git switch -c recovery_from_75c26f2 75c26f2`
- Serving path validated:
  - `src/qeu_bundling/presentation/app.py` -> `build_recommendations_for_profiles()`
  - `src/qeu_bundling/api/server.py` -> `build_recommendations_for_profiles()`
  - Core logic: `src/qeu_bundling/presentation/person_predictions.py`

## 2) Baseline Sample Audit (Diagnosis Only)
- Deterministic sample method:
  - `build_default_profiles(load_order_pool(...), count=25, rng=random.Random(20260315))`
- Real serving run used:
  - `build_recommendations_for_profiles(..., run_id="baseline_audit_75c26f2", rng_salt="baseline_audit")`
- Raw artifact:
  - `output/serving_audit/baseline_75c26f2_sample25_raw.json`
- Summary artifact:
  - `output/serving_audit/baseline_75c26f2_sample25_summary.json`

### Baseline observations
- Coverage:
  - `25/25` users returned `3` bundles.
- Origin mix:
  - `top_bundle: 40`, `copurchase_fallback: 34`, `fallback_cleaning: 1` (75 total bundles).
- Repetition:
  - Top repeated pair appeared `5` times.
  - Multiple motifs repeated 3-5 times across only 25 users.
- Within-user item reuse:
  - `5` users had repeated product IDs across their 3 bundles.
- Weak/suspicious bundles seen:
  - Example: tuna + sweetened condensed milk.
  - Example motif repeated: instant noodles + toast bread.
- Latency:
  - Total: `169.53s` for 25 users.
  - Average: `6.78s/user`.

### Likely baseline causes
- Heavy fallback-template path dominates runtime.
- Repeated pair semantic evaluation across templates.
- Repeated product text normalization in hot loops.
- Large monolithic per-profile orchestration with repeated checks.

### What looked acceptable and preserved
- Deterministic outputs for fixed run seed/salt.
- Strong bundles still present for many users.
- Exposure-aware scoring and origin labeling still usable.

## 3) Latency Refactor (Behavior-Preserving Focus)

## Scope
- Files changed:
  - `src/qeu_bundling/presentation/person_predictions.py`
  - `scripts/validate_serving_output.py` (new)
  - `tests/test_person_predictions.py`

## Main changes
1. Added env-gated profiling (`QEU_SERVING_PROFILE`) in serving path:
   - Per-profile timing.
   - Stage timing: preparation, candidate generation, scoring/ranking, selection/fill, output assembly, total.
   - Exposed via `get_last_serving_profile_metrics()`.
2. Cached normalization and product text:
   - `_normalise_text_cached(...)`
   - `_product_text_from_components(...)`
3. Optimized fallback template evaluation:
   - Precomputed product text per PID inside `_fallback_candidates_for_lane`.
   - Added per-pair pass cache for expensive gates/semantic checks.
4. Reduced repeated nonfood anchor scanning:
   - Precomputed nonfood anchor list once per request.
5. Added reusable validation utility:
   - `scripts/validate_serving_output.py` for deterministic sampling + raw/summary artifacts + optional before/after comparison.
6. Added regression test for profiling-mode metrics presence.

## Change classification
- Behavior-preserving:
  - Profiling instrumentation (env-gated, default off).
  - Text normalization cache.
  - Product text cache by normalized components.
  - Nonfood anchor precomputation.
- Likely behavior-preserving:
  - Per-pair fallback gate cache in `_fallback_candidates_for_lane` (template-independent gating reused per oriented pair).
- Behavior-affecting:
  - None intentionally introduced.

## 4) Before/After Measurements (Same deterministic sample)
- Sample: 25 profiles, seed `20260315`, run id `baseline_audit_75c26f2`, rng salt `baseline_audit`.

### Runtime
- Before: `169.53s` total (`6.78s/user`)
- After: `68.49s` total (`2.74s/user`)
- Improvement: ~`59.6%` faster total runtime

### Output stability check
- Comparison key: `source_order_ids` signature per profile.
- Compared artifacts:
  - Before: `output/serving_audit/baseline_75c26f2_sample25_raw.json`
  - After: `output/serving_audit/after_opt_75c26f2_sample25_seed20260315_raw.json`
- Result:
  - `changed_profiles: 0`
  - `missing_profiles: 0`

### Stage timing (after, profiling-enabled run)
- Stage totals:
  - `candidate_generation`: `59.77s`
  - `selection_fill`: `0.41s`
  - `profile_preparation`: `0.14s`
  - `scoring_ranking`: `0.04s`
  - `output_assembly`: `0.05s`
- Per-profile latency distribution:
  - `p50: 1.68s`
  - `p90: 2.74s`
  - `p95: 3.05s`

## 5) Remaining Bottlenecks
- Candidate generation remains dominant stage.
- Fallback generation is still the primary hotspot under profiling.
- Additional speedups are possible, but most likely require deeper structural precomputation/indexing work.

## 6) Not Yet Implemented (Recommended Next)
- Quality fixes for baseline issues (item reuse violations, suspicious motifs, stronger compatibility) were intentionally not changed in this latency pass.
- If approved next:
  - tighten no-item-reuse enforcement at final selection,
  - strengthen compatibility blocks for noodles+bread and tuna+dairy-dessert variants,
  - improve exposure controls for high-frequency motifs.

