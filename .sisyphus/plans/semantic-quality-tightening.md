# Semantic Quality Tightening Plan

## Goal

Tighten visible bundle semantics with minimal disruption by improving semantic expression, weak-pair admission, and fallback template membership while preserving the existing lanes, API/UI contract, hard-invalid safety, and missing-lane behavior.

## Files

- `src/qeu_bundling/presentation/bundle_semantics.py`
- `src/qeu_bundling/presentation/person_predictions.py`
- `tests/test_bundle_semantics.py`
- `tests/test_person_predictions.py`

## Change Sequence

1. Add characterization tests for must-keep strong visible pairs and current weak leak cases.
2. Tighten semantic role and visible-lane expression logic in `bundle_semantics.py`:
   - make meal visible expression less permissive for pantry/prep pairs
   - make occasion dessert rescue require serving context rather than generic dairy
   - make snack visible expression less permissive for prep dairy and broad snack-ish products
3. Tighten new-semantic weak-pair admission in `person_predictions.py` so `weak_valid` food pairs require stronger evidence before entering ranking.
4. Tighten fallback template membership and fallback weak-pair acceptance in `person_predictions.py` so templates do not force-fill visible food lanes.
5. Verify that missing food lanes remain allowed when no strong visible pair survives.

## Guardrails

- Do not change lane names, payload fields, or UI/API behavior.
- Do not weaken or bypass `hard_invalid` rules.
- Do not allow feedback to admit pairs that fail semantic safety checks.
- Prefer missing visible lanes over weak visible fillers.
- Preserve culturally valid strong pairs such as `tea + evaporated milk`, `coffee + evaporated milk`, `tea + biscuit`, `dates + milk`, `dates + cream`, `dessert + serving cream`, `eggs + bread`, and `chocolate + milk`.

## Verification

- `python -m pytest tests/test_bundle_semantics.py -q`
- `python -m pytest tests/test_person_predictions.py -q`
- Add focused assertions for weak visible leakage, fallback leakage, missing-lane behavior, and feedback not overriding semantic safety.

## QA Scenarios

### Keepers that must remain visible

- `tea + evaporated milk` => occasion allowed, visible expression true
- `coffee + evaporated milk` => occasion allowed, visible expression true
- `tea + biscuit` => occasion allowed, visible expression true
- `dates + cream` => occasion allowed, visible expression true
- `dates + milk` => occasion allowed, visible expression true
- `dessert + fresh cream` => occasion allowed, visible expression true
- `eggs + bread` => meal allowed, visible expression true
- `chocolate + milk` => snack allowed, visible expression true

### Main-path leaks that must be rejected or demoted

- `oats + eggs` => not visible meal
- `oats + tuna` => not visible meal
- `rice + carrots` => not visible meal
- `bread + oats` => not visible meal
- `chicken + milk` => not visible meal
- `dessert + powdered milk` => not visible occasion
- `dessert + cheese` => not visible occasion
- `dessert + biscuit` without host context => not visible occasion

### Fallback leaks that must not refill lanes

- `chips + cooking cheese` => not accepted as snack fallback
- `milk + biscuit` with non-serving milk => not accepted as occasion fallback
- `condensed + biscuit` => not accepted as occasion fallback
- weak fallback-only snack/occasion candidate set => lane remains missing

### Safety invariants

- feedback cannot admit a pair that fails hard-invalid or visible expression checks
- nonfood/personal-care items cannot enter food lanes during anchor or pair admission
- missing visible lanes are allowed when no strong visible pair survives

## Atomic Commit Strategy

- `test: characterize semantic leaks and preserve strong visible pairs`
- `fix: tighten visible meal and occasion semantic expression`
- `fix: narrow snack and occasion fallback template membership`
- `fix: require stronger evidence for weak semantic fallback pairs`
- `test: lock missing-lane and semantic-safety regressions`
