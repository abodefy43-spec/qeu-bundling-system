# Recommender Structural Refactor Plan

## Goal

Refactor the deterministic recommendation pipeline so semantic classification, lane admission, ranking, and fallback behavior are easier to reason about without weakening hard-invalid protections, freshness handling, or reproducibility.

## Files

- `src/qeu_bundling/presentation/bundle_semantics.py`
- `src/qeu_bundling/presentation/person_predictions.py`
- `src/qeu_bundling/core/run_manifest.py`
- `tests/test_bundle_semantics.py`
- `tests/test_person_predictions.py`
- `tests/test_evaluate_bundle_quality.py`

## Bounded Change Strategy

1. Keep public functions and current payload contracts stable.
2. In `bundle_semantics.py`, introduce shared semantic facts and token-pattern helpers so relation, strength, hard-invalid, and visible-lane checks consume the same inputs instead of duplicating order-sensitive substring rules.
3. In `person_predictions.py`, introduce a shared internal pair-analysis layer for product text, roles, groups, semantic snapshot, and deterministic tie-break metadata, then reuse it across anchor admission, pair filtering, complement gating, ranking, and fallback selection.
4. Keep `run_manifest.py` behaviorally equivalent, limiting changes to readability or small helper extraction only if it reduces ambiguity around freshest-artifact selection.
5. Add tests that verify real behavior rather than only boolean admission: ranking order, semantic-vs-score interaction, deterministic tie-breaks, fallback orientation, finalize-choice payload integrity, and latest-artifact freshness selection.

## Targeted Refactors

### `bundle_semantics.py`

- Centralize normalized text token checks and recurring pattern families.
- Separate three decision layers more clearly:
  - semantic fact inference
  - lane-independent pair classification and hard-invalid logic
  - lane visibility and fit logic
- Remove contradictory cases where a pair is weak-valid in one layer but effectively blocked later by duplicated pattern checks.
- Preserve current strong keepers and current hard-invalid protections.

### `person_predictions.py`

- Add an internal immutable pair-analysis object for:
  - anchor/complement names, categories, families, prices
  - group labels and semantic groups
  - semantic snapshot from `bundle_semantics`
  - snack pattern and pair fingerprint metadata
  - deterministic ordering keys
- Route these consumers through the shared analysis object instead of recomputing product facts independently:
  - `_anchor_allowed_for_lane`
  - `_passes_pair_filters`
  - `_passes_complement_gate`
  - `_score_personal_candidate`
  - `_rank_anchors_by_lane`
  - `_pick_candidate_for_anchor`
  - `_fallback_candidates_for_lane`
- Make tie-break behavior explicit and centralized so semantic strength cannot be accidentally outranked by weaker but noisier signals.
- Preserve missing-lane behavior rather than forcing weak filler bundles.

### `run_manifest.py`

- Keep the current preference for a newer fallback artifact over a stale manifest pointer.
- If changed at all, extract comparison logic into a small helper and retain exact selection semantics.

## Guardrails

- Do not loosen global gates.
- Do not add generic junk fallbacks to recover coverage.
- Do not suppress type errors or leave broken code paths.
- Prefer removing or consolidating brittle rules over adding more one-off constants.
- Preserve culturally and semantically strong pairs such as:
  - `tea + evaporated milk`
  - `coffee + evaporated milk`
  - `tea + biscuit`
  - `dates + cream`
  - `dates + milk`
  - `dessert + fresh cream`
  - `eggs + bread`
  - `chips + cola`
- Continue rejecting nonsense or low-expression combinations such as:
  - `oats + tuna`
  - `rice + tomato paste`
  - carb + produce-only meal pairs
  - milk + allium
  - cooking-heavy snack pairs
  - non-food contamination in food lanes

## Verification

- `python -m py_compile src/qeu_bundling/presentation/bundle_semantics.py src/qeu_bundling/presentation/person_predictions.py src/qeu_bundling/core/run_manifest.py`
- `python -m unittest discover -s tests -p "test_bundle_semantics.py"`
- `python -m unittest discover -s tests -p "test_person_predictions.py"`
- `python -m unittest discover -s tests -p "test_evaluate_bundle_quality.py"`
- `python -m qeu_bundling.core.evaluate_bundle_quality`
- `python -m qeu_bundling.cli run quick --eval-slice`

## QA Scenarios

### `bundle_semantics.py`

1. Keeper visibility remains intact.
   - Tool: `tests/test_bundle_semantics.py`
   - Cases:
     - `tea + evaporated milk` => occasion visible
     - `coffee + evaporated milk` => occasion visible
     - `tea + biscuit` => occasion visible
     - `dates + fresh cream` => occasion visible
     - `eggs + toast bread` => meal visible
   - Expected result: relation/strength stay lane-appropriate and `visible_lane_expression_ok(...)` remains true.
2. Hard-invalid and low-expression meal noise still reject.
   - Tool: `tests/test_bundle_semantics.py`
   - Cases:
     - `full cream milk + fresh onions` => hard invalid
     - `oats + tuna` => not visible meal
     - `rice + tomato paste` => not visible meal
     - carb + produce-only pair => not visible meal
   - Expected result: rejection reason remains deterministic and no weaker classification path re-admits the pair later.
3. Occasion serving context stays explicit.
   - Tool: `tests/test_bundle_semantics.py`
   - Cases:
     - `dessert pudding + milk powder` => not visible occasion
     - `dessert pudding + triangle cheese` => not visible occasion
     - `dates + cooking cream` => not visible occasion
   - Expected result: dessert/occasion rescue requires serving dairy, not generic dairy-like text.

### `person_predictions.py`

1. Deterministic ranking and tie-breaks remain stable.
   - Tool: `tests/test_person_predictions.py`
   - Setup: same profile with different `rng_salt` values and repeated calls.
   - Expected result: identical anchor/complement ordering and identical chosen pairs.
2. Semantic gate outranks noisy score.
   - Tool: `tests/test_person_predictions.py`
   - Cases:
     - high-score weak or semantically bad pair loses to lower-score semantically valid pair
     - packaging/non-food candidate cannot enter food lane even with strong copurchase and recipe scores
   - Expected result: semantic rejection happens before score can force admission.
3. Fallback preserves orientation and does not force junk lane fill.
   - Tool: `tests/test_person_predictions.py`
   - Cases:
     - fallback chooses semantically aligned complement for a valid anchor
     - weak-only fallback candidate set leaves lane missing instead of filling with noise
   - Expected result: fallback output matches lane semantics and missing-lane behavior remains allowed.
4. Finalized payload stays intact.
   - Tool: `tests/test_person_predictions.py`
   - Cases: selected bundle record still includes anchor/complement ids, lane, free-product orientation, pricing fields, semantic fields, and deterministic metadata.
   - Expected result: refactor changes internals, not output shape.

### `run_manifest.py`

1. Freshness resolution prefers a newer fallback artifact over a stale manifest pointer.
   - Tool: `tests/test_evaluate_bundle_quality.py`
   - Setup: stale manifest points to older `person_reco_quality.json`, root output contains newer one.
   - Expected result: evaluation reads the newer fallback artifact and preserves the gate-passing metric.
2. Matching paths remain stable.
   - Tool: `tests/test_evaluate_bundle_quality.py`
   - Setup: manifest artifact path and fallback path refer to the same file.
   - Expected result: resolver returns the same file without regressing path handling.

## Expected Outcome

- Smaller decision surface per function.
- Shared pair reasoning across semantic filtering, ranking, and fallback logic.
- Deterministic ranking with clearer tie-breaks.
- No regression in hard-invalid protections, quality gates, or freshest-artifact selection.
