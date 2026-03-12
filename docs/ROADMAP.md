# ROADMAP.md

## Purpose
This roadmap tracks practical, low-risk evolution of QEU Product Bundling with emphasis on semantic recommendation quality, determinism, and contract stability.

## Current baseline (as of latest repo audit)
- Pipeline phases 00-09 are implemented and runnable through CLI.
- People-based dashboard flow is active and lane-based.
- Semantic engine v2 is integrated with feature flags.
- Review export and quality metrics artifacts are available.
- Test coverage is broad, with heavy emphasis on person prediction and semantics.

## Guardrails for all roadmap work
- Preserve Flask route and template payload compatibility.
- Preserve deterministic behavior for fixed seeds/run_id/rng_salt.
- Preserve recommendation safety controls (pair dedupe, anchor cap, nonfood isolation, free-item convention).
- Prefer missing lane over low-quality recommendation leakage.

## Near-term priorities (0-2 weeks)
1. Semantic quality hardening
- Continue eliminating weak visible lane pairs in `meal` and `occasion`.
- Keep hard-invalid strict and feedback unable to bypass hard safety.

2. Fallback quality and coverage balance
- Reduce fallback overuse without relaxing semantic gates.
- Improve top-bundle candidate survivability where safe.

3. Encoding and UI polish cleanup
- Remove mojibake artifacts in templates/static/review summaries.
- Keep bilingual RTL behavior stable.

4. Script reliability
- Fix or replace stale launcher references (for example `easy_run.bat` -> missing `run_pipeline.bat`).

## Short-term priorities (2-6 weeks)
1. Recommendation observability
- Expand diagnostics for rejection reasons and lane selection decisions.
- Add easier side-by-side comparison between runs in review artifacts.

2. Feedback loop maturity
- Improve curated feedback ingestion workflows and documentation for reviewers.
- Add validation utilities for feedback CSV quality and class balance.

3. Modularization safety pass
- Gradually extract dense logic from `person_predictions.py` into smaller modules without contract changes.

## Mid-term priorities (6-12 weeks)
1. Performance tuning
- Reduce quick-run latency and dashboard generation latency while keeping deterministic outcomes.

2. Quality regression tooling
- Add deterministic benchmark fixtures for recommendation quality trends across commits.

3. Serving hardening
- Improve session-state robustness and refresh behavior under multiple users.

## Long-term options (12+ weeks)
1. Controlled online integration
- If needed, expose recommendation service endpoints with strict contract versioning.

2. Model lifecycle governance
- Add explicit model registry/version metadata beyond current file outputs.

3. Infrastructure formalization
- Introduce deployable packaging (container/runtime manifests) if production hosting becomes a requirement.

## Definition of done (for behavior-changing PRs)
- Tests updated and passing for touched semantics and recommendation paths.
- README and docs updated when contracts/flows change.
- Relevant run/review artifact regenerated and checked.
- No regression in deterministic behavior for fixed-seed workflows.

## Open uncertainties
- Production deployment target and SLOs are not codified in-repo.
- Review process ownership and cadence are implied by artifacts but not formally documented.
- Tolerance thresholds for fallback share and lane-level missing rates are not centrally defined.
