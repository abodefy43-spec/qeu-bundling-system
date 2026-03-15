# Within-User Item Reuse Bug Fix (`recovery_75c26f2_latency_pass`)

## Scope
- Branch: `recovery_75c26f2_latency_pass`
- Baseline behavior source: `75c26f2` serving path
- Goal: fix within-user item reuse only, with minimal logic change and no recommender redesign.

## Reproduction (Deterministic Audit Sample)
- Command:
  - `python3 scripts/validate_serving_output.py --sample-size 25 --seed 20260315 --label reuse_bug_before_fix --run-id baseline_audit_75c26f2 --rng-salt baseline_audit`
- Before-fix raw artifact:
  - `output/serving_audit/reuse_bug_before_fix_sample25_seed20260315_raw.json`
- Before-fix summary:
  - `output/serving_audit/reuse_bug_before_fix_sample25_seed20260315_summary.json`

### Reproduced violating profiles (5)
- `person_6045e90c5c60`:
  - type: complement reused later as anchor (`complement_reused_from_anchor`)
- `person_49ddf348a686`:
  - type: complement reused later as anchor (`complement_reused_from_anchor`)
- `person_761e84a996be`:
  - type: complement reused as complement (`complement_reused_as_complement`)
- `person_70c3c053f449`:
  - type: complement reused as complement (`complement_reused_as_complement`)
- `person_607af77269db`:
  - type: anchor reused after being complement (`anchor_reused_from_complement`)

## Root Cause
- Location: `_try_select()` inside `build_recommendations_for_profiles()` in `src/qeu_bundling/presentation/person_predictions.py`.
- Existing guard only blocked:
  - duplicate anchor reuse (`selected_anchors`)
  - duplicate exact pair reuse (`selected_pairs`)
- Missing guard:
  - complement reuse across selected bundles
  - any cross-role reuse (complement -> anchor, anchor -> complement)

## Bug Location Classification
- Candidate generation: not root cause (can generate overlapping items by design).
- Final selection: **root cause** (acceptance guard incomplete).
- Fallback/copurchase selection: contributes candidates but not the direct bug.
- Output assembly: not root cause.

## Exact Fix (Smallest Correct Change)
- File: `src/qeu_bundling/presentation/person_predictions.py`
- Change:
  - Added `selected_item_ids: set[int]` in final selection state.
  - In `_try_select()`, reject candidate if:
    - `anchor_id in selected_item_ids`
    - `complement_id in selected_item_ids`
    - existing anchor/pair checks still apply
  - On accept, add both `anchor_id` and `complement_id` to `selected_item_ids`.

This keeps selection flow/scoring intact and only enforces the missing hard constraint.

## Focused Test Coverage Added
- File: `tests/test_person_predictions.py`
- Added:
  - `test_no_anchor_reuse_within_person_selection`
  - `test_no_complement_reuse_within_person_selection`
  - `test_no_cross_role_item_reuse_within_person_selection`

These tests stub candidate generation and exercise final selection behavior directly.

## After-Fix Validation
- Command:
  - `python3 scripts/validate_serving_output.py --sample-size 25 --seed 20260315 --label reuse_bug_after_fix --run-id baseline_audit_75c26f2 --rng-salt baseline_audit --compare-path output/serving_audit/reuse_bug_before_fix_sample25_seed20260315_raw.json`
- After-fix raw artifact:
  - `output/serving_audit/reuse_bug_after_fix_sample25_seed20260315_raw.json`
- After-fix summary:
  - `output/serving_audit/reuse_bug_after_fix_sample25_seed20260315_summary.json`

### Results
- Within-user reuse violations:
  - before: `5`
  - after: `0`
- Profiles changed:
  - `5`
  - Matches the 5 violating source-order signatures.
- Latency impact on same sample:
  - before: `61.51s` total (`2.46s/profile`)
  - after: `67.54s` total (`2.70s/profile`)
  - delta: `+6.03s` total (~`+9.8%`)
- Suspicious bundle check:
  - heuristic suspicious count stayed `1` before and `1` after
  - no net increase in suspicious-pair count.

## Notes
- This is a correctness fix in final selection only.
- No scoring philosophy or fallback architecture redesign was introduced.
- Existing latency optimizations remain in place.

