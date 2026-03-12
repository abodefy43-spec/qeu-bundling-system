# AI_CONTEXT.md

## Project identity
### What this system is
QEU Product Bundling is an offline-first grocery recommendation platform that:
- runs phased data processing and ML scoring,
- produces ranked two-item bundle candidates,
- serves people-based recommendations in a Flask dashboard,
- collects user feedback for ranking adjustments.

### What this system is not
- Not a real-time microservice backed by an online feature store.
- Not a DB-centric application (no relational schema migrations found).
- Not an infrastructure-as-code repository (no Docker/K8s/Terraform files found).

### Main product/use case
Generate semantically plausible grocery bundles for person cards, grouped by lane (`meal`, `snack`, `occasion`, optional `nonfood`), with item-2 forced free in presentation payloads.

### Current maturity level
Late prototype / production-like internal tool:
- substantial rules and test coverage,
- deterministic run manifests and review exports,
- active semantic hardening iteration,
- still tuning recommendation quality/coverage trade-offs.

## Current working model
### End-to-end behavior
1. CLI (`python -m qeu_bundling.cli`) runs full or quick pipeline.
2. Pipeline writes artifacts under `data/processed/` and `output/`.
3. `phase_06_bundle_selection.py` builds person candidate pairs.
4. `phase_07_train_models.py` trains discount/free-item models.
5. `phase_08_predict.py` scores candidates and writes `output/person_candidates_scored.csv`.
6. `phase_09_optimize.py` validates quality and emits diagnostics.
7. Flask dashboard (`presentation/app.py`) loads latest artifacts and builds per-person bundles via `person_predictions.py`.
8. Dashboard feedback writes to `data/processed/person_feedback.csv`; curated review feedback is read from `output/review/bundle_feedback.csv`.

### Critical user flows
- Run quick refresh and open dashboard.
- Add random/manual people, shuffle cards, regenerate a single person.
- Submit feedback (`like/dislike/wrong_pair/too_expensive`).
- Export deterministic 100-person review (`people_predictions_review_export.py`).

### Business logic that appears central
- Semantic lane control: `presentation/bundle_semantics.py`.
- Candidate picking/ranking/gating: `presentation/person_predictions.py`.
- Pipeline candidate generation and score shaping: `pipeline/phase_06_bundle_selection.py` and `pipeline/phase_08_predict.py`.
- Quality gates for run pass/fail: `core/evaluate_bundle_quality.py`.

## Important assumptions
### Architectural assumptions
- File artifacts are the contract between phases (CSV/PKL/NPY/NPZ/JSON).
- Latest run resolution depends on `output/latest/latest_manifest.json`.
- Dashboard state is in-memory per process (`_PERSON_STATES`), not persistent.

### Product assumptions
- Visible food lanes remain `meal`, `snack`, `occasion`; `nonfood` optional.
- Free product display convention is `free_product = product_b` in final payload shaping.
- Missing lane is acceptable when semantic quality is insufficient.

### Developer assumptions
- Determinism is intentional when seed/run_id/rng_salt are fixed.
- Recommendation quality must prioritize semantic safety over forced coverage.
- Existing safeguards are expected to remain: global pair dedupe, anchor cap, per-person product dedupe, nonfood isolation.

### Inferred constraints (evidence-based)
- Backward compatibility with dashboard form payload fields is important (feedback endpoint expects canonical IDs/names).
- Semantic feature flags are actively used as kill-switches.
- Quick/full runners can intentionally fail if quality gates fail.

## Editing rules for AI agents
### Patterns to preserve
- Route contracts in `presentation/app.py`.
- Artifact path resolution via `config/paths.py` and `core/run_manifest.py`.
- Deterministic tie-breaking and seed behavior.
- Forced free-item-B shaping in person recommendation payloads.

### Safety-critical modules
- `src/qeu_bundling/presentation/person_predictions.py`
- `src/qeu_bundling/presentation/bundle_semantics.py`
- `src/qeu_bundling/pipeline/phase_06_bundle_selection.py`
- `src/qeu_bundling/core/evaluate_bundle_quality.py`

### Anti-patterns to avoid
- Adding random behavior without stable seeding/tie-breaks.
- Bypassing hard-invalid semantic rules via feedback.
- Mixing nonfood into food lanes.
- Changing dashboard payload keys used by forms/templates without synchronized updates/tests.

### Naming/style conventions found in repo
- Constants are upper snake case.
- Lane constants (`LANE_MEAL`, `LANE_SNACK`, `LANE_OCCASION`, `LANE_NONFOOD`).
- Helper-heavy functional style in pipeline/presentation modules.
- Tests use `unittest` (not pytest fixtures).

### What must be checked before changing code
- Existing tests around touched logic (`tests/test_person_predictions.py`, `tests/test_bundle_semantics.py`, phase tests).
- Manifest/artifact consumers (`bundle_view.py`, review pack, evaluate quality).
- Route/form payload compatibility.

### What to update when a feature changes
- `README.md` for user-facing command/behavior changes.
- `docs/ARCHITECTURE.md` if flow/contracts change.
- `docs/ROADMAP.md` if priorities/phase status changes.
- Relevant test files and review export expectations.

## Project vocabulary
- **Anchor**: primary product (`product_a` logical source during selection).
- **Complement**: paired product (`product_b` logical source during selection).
- **Lane**: recommendation intent bucket (`meal`, `snack`, `occasion`, `nonfood`).
- **Top-bundle**: candidate sourced from pre-scored pair rows tied to anchor.
- **Copurchase fallback**: candidate sourced from neighbor copurchase graph.
- **Template fallback**: deterministic lane template fallback when normal path fails.
- **Hard invalid**: absolute semantic reject.
- **Visible expression floor**: pair may be semantically possible but too weak for visible lane.
- **Person reco quality**: `output/person_reco_quality.json` summary diagnostics.
- **Review export**: deterministic sample files under `output/review/people_predictions_review_100*`.

## Source-of-truth map
- **Architecture flow**: `docs/ARCHITECTURE.md`
- **Environment/path config**: `src/qeu_bundling/config/paths.py`
- **CLI/API surface**: `src/qeu_bundling/cli/__main__.py`, `src/qeu_bundling/presentation/app.py`
- **Business logic**: `src/qeu_bundling/presentation/person_predictions.py`, `src/qeu_bundling/presentation/bundle_semantics.py`
- **Feedback logic**: `src/qeu_bundling/core/feedback_memory.py`, `src/qeu_bundling/review/feedback_loader.py`
- **Run lineage/manifests**: `src/qeu_bundling/core/run_manifest.py`
- **UI patterns**: `templates/layout.html`, `templates/dashboard.html`, `static/styles.css`, `static/dashboard_people.js`
- **Tests**: `tests/` (especially `test_person_predictions.py`, `test_bundle_semantics.py`)
- **Deployment/runtime scripts**: `scripts/*.bat`

## Safe workflow for future AI agents
1. Read `README.md`.
2. Read `docs/AI_CONTEXT.md`.
3. Read `docs/ARCHITECTURE.md`.
4. Inspect the exact modules you will change.
5. Make the smallest coherent change that preserves contracts.
6. Update docs if behavior/contracts changed.
7. Run targeted tests first, then broader suite as needed.
8. If output artifacts matter, run the relevant CLI flow and confirm generated files.

## 30-Second Context for AI Agents
This repository is a file-artifact-driven grocery bundling system with a phased offline pipeline and a Flask dashboard. The highest-risk logic lives in `person_predictions.py` (lane selection, gating, ranking, fallback) and `bundle_semantics.py` (hard invalid + lane compatibility). Public contract stability matters at route/form/template boundaries. Determinism and semantic safety are explicit goals. If you change recommendation behavior, update tests and verify outputs through run manifests and review artifacts.
