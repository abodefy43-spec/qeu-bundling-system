# Candidate-First Bundle Rules Plan

## Goal

Bring the people-serving recommender into strict compliance with the product rules while keeping deterministic 2-item bundles, preserving the existing payload shape, and minimizing churn outside the serving layer.

## Files

- `src/qeu_bundling/presentation/person_predictions.py`
- `src/qeu_bundling/presentation/app.py`
- `src/qeu_bundling/presentation/bundle_semantics.py`
- `tests/test_person_predictions.py`
- `tests/test_bundle_semantics.py`

## Change Sequence

1. Add focused characterization tests in `tests/test_person_predictions.py` for exact-3 output, unique anchors, unique pairs, composition bans, cleaning cap, deterministic fallback retrieval, exposure penalty, recency preference, template-over-copurchase priority, and strict 2-item bundles.
2. Add explicit deterministic curated fallback libraries in `src/qeu_bundling/presentation/person_predictions.py` with stable metadata for meal, snack, occasion, and household use.
3. Add shared helper logic in `src/qeu_bundling/presentation/person_predictions.py` for:
   - domain/composition classification
   - recency and recent-intent scoring from profile history order/counts
   - fallback retrieval scoring
   - batch exposure penalty
   - final candidate constraint checks
4. Refactor serving in `src/qeu_bundling/presentation/person_predictions.py` from lane-first finalization to a shared candidate pool built from `top_bundle`, curated fallback templates/library, and copurchase candidates, then select exactly 3 bundles under constraints:
   - exactly 3 bundles total
   - max 1 cleaning/household bundle
   - cleaning only after food options are exhausted or clearly weaker
   - unique anchors per user
   - unique pairs per user
   - no banned food/household/appliance composition
   - mild deterministic diversity and exposure penalty
   - source priority `top_bundle > fallback_template > copurchase_fallback`
5. Keep all candidate sources flowing through the same semantic and composition filters so fallback cannot bypass safety.
6. Remove or neutralize dashboard-only cleaning insertion in `src/qeu_bundling/presentation/app.py` so the Flask display layer no longer changes recommendation count or rule compliance after serving.
7. Verify deterministic behavior, focused tests, syntax, and relevant suite coverage.

## Guardrails

- Do not add neural models or stochastic ranking.
- Do not change bundle cardinality away from 2 items.
- Do not weaken semantic or composition validity to fill slots.
- Do not allow food + cleaning, food + appliance, or other cross-domain leaks in any source path.
- Do not show more than one cleaning/household bundle for a person.
- Do not allow duplicate anchors or duplicate pairs for a person.
- Keep tie-breaks stable and reviewable.
- Prefer serving-layer changes over pipeline rewrites unless a serving-only fix is impossible.

## Verification

- `python -m unittest tests.test_person_predictions`
- `python -m unittest tests.test_bundle_semantics`
- `python -m unittest tests.test_run_seed_behavior`
- `python -m py_compile src/qeu_bundling/presentation/person_predictions.py src/qeu_bundling/presentation/app.py src/qeu_bundling/presentation/bundle_semantics.py tests/test_person_predictions.py tests/test_bundle_semantics.py`
- If touched behavior reaches broader stable paths, run `python -m unittest discover -s tests -p "test_*.py"`

## QA Scenarios

### Exact output and uniqueness

- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_each_profile_gets_exactly_three_two_item_bundles`
- Setup: build a profile with mixed food history and a bundle table with at least 3 valid candidates plus distractors.
- Expected result: `len(recs[0]["bundles"]) == 3`, every bundle contains only `product_a` and `product_b`, anchors are unique, and oriented pair keys are unique.

### Composition safety

- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_cross_domain_pairs_are_rejected_in_main_and_fallback_paths`
- Setup: include food-cleaning, food-appliance, and valid cleaning-cleaning candidates in both top-bundle and fallback pools.
- Expected result: food-cleaning and food-appliance never appear in output or fallback candidate sets; one valid cleaning-cleaning pair may appear only when food slots cannot reach 3.
- Command: `python -m unittest tests.test_bundle_semantics.BundleSemanticsTests`
- Expected result: dessert-dessert and known nonsense pairs remain semantically blocked.

### Candidate-first selection and source priority

- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_source_priority_prefers_top_bundle_over_template_and_copurchase`
- Setup: create a lane where `top_bundle`, `fallback_template`, and `copurchase_fallback` are all valid competitors for the same profile.
- Expected result: the chosen bundle has `recommendation_origin == "top_bundle"` and matches the top-bundle pair.
- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_candidate_first_prefers_template_over_copurchase_when_top_is_missing`
- Setup: create one lane where both fallback template and copurchase are valid and comparable, with no valid top-bundle winner.
- Expected result: selected `recommendation_origin` starts with `fallback_template:` and the chosen pair matches the template candidate.
- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_candidate_first_can_choose_later_better_candidate`
- Expected result: a later candidate with higher deterministic score wins over an earlier anchor-local copurchase path.

### Fallback quality and determinism

- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_curated_fallback_library_fills_to_exact_three_deterministically`
- Setup: patch main-path candidate picking to fail or provide fewer than 3 valid food bundles.
- Expected result: output still has exactly 3 bundles, fallback origins are deterministic across repeated runs, and no invalid fallback pair is admitted.
- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_cleaning_bundle_is_capped_at_one_and_only_used_when_needed`
- Expected result: zero cleaning bundles when 3 valid food bundles exist, otherwise exactly one cleaning bundle at most.

### Personalization and batch control

- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_recent_intent_beats_generic_bundle_when_quality_is_close`
- Setup: use a profile whose most recent items imply tea/biscuit or coffee/evaporated-milk intent while a generic staple pair has slightly stronger base popularity.
- Expected result: the recent-intent pair is selected ahead of the generic staple pair.
- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_recent_brand_affinity_breaks_close_ties`
- Expected result: among otherwise close candidates, the recent preferred-brand complement wins deterministically.
- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_exposure_penalty_reduces_batch_pair_dominance_deterministically`
- Setup: build a batch of profiles with the same strong generic pair competing against good alternatives.
- Expected result: the most repeated pair appears for fewer later profiles than it would without the penalty, while output stays deterministic and valid.

### Dashboard cleanup

- Command: `python -m unittest tests.test_people_only_ui`
- Command: `python -m unittest tests.test_person_predictions.PersonPredictionsTests.test_dashboard_layer_does_not_inject_extra_cleaning_bundle`
- Expected result: the Flask/display path does not append a new cleaning card or alter the serving-generated bundle count after recommendations are built.

## Atomic Commit Strategy

- `test: lock bundle serving rule regressions`
- `fix: switch serving to candidate-first exact-three selection`
- `fix: add deterministic curated fallback retrieval and batch-aware penalties`
