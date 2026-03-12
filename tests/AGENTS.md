# SUBDIRECTORY KNOWLEDGE BASE

## OVERVIEW
- `tests/` uses `unittest` with patch-heavy unit tests, temporary filesystem fixtures, and deterministic seed checks across presentation, pipeline, and core code.

## WHERE TO LOOK
- Canonical test command - `.github/workflows/ci.yml`
- Largest hotspot and mocking style reference - `test_person_predictions.py`
- Translation/cache tests - `test_dashboard_i18n.py`
- Seed determinism checks - `test_run_seed_behavior.py`
- Review-pack temp artifact tests - `test_data_review_pack.py`
- UI route response checks - `test_people_only_ui.py`

## CONVENTIONS
- Run tests with `python -m unittest discover -s tests -p "test_*.py"`.
- Prefer `unittest` and `unittest.mock.patch` for isolating file paths, translators, and personalization context.
- Use `tempfile.TemporaryDirectory` when a test needs disposable processed/output trees or cache files.
- Preserve deterministic behavior checks around seeds and randomized sampling; several tests assert stable outputs for fixed seeds.
- `test_person_predictions.py` is the main hotspot for patch-based recommendation logic and regression coverage.

## ANTI-PATTERNS
- Do not introduce non-unittest concepts or fixture vocabulary into this directory guidance.
- Do not couple tests to generated repo outputs when a temporary directory can isolate the scenario.
