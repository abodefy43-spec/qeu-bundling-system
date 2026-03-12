# SUBDIRECTORY KNOWLEDGE BASE

## OVERVIEW
- `presentation/` owns the Flask dashboard, session-scoped person recommendations, and dashboard-only translation/rendering behavior.

## WHERE TO LOOK
- Flask app, routes, session state, refresh hooks - `app.py`
- Background refresh runner and log tail state - `run_service.py`
- Latest-run artifact loading and bundle shaping boundary - `bundle_view.py`
- Recommendation heuristics and profile generation hotspot - `person_predictions.py`
- Arabic dashboard translation/cache behavior - `dashboard_i18n.py`
- Root template coupling - `templates/dashboard.html`
- Root static companion script - `static/dashboard_people.js`
- UI behavior checks - `tests/test_people_only_ui.py`
- Translation/cache checks - `tests/test_dashboard_i18n.py`

## CONVENTIONS
- The Flask app points at repo-root `templates/` and `static/`; UI edits usually span Python plus those root asset folders.
- Treat dashboard recommendation state as session-scoped and run-aware; `app.py` keys state by session and latest manifest run id.
- `run_service.py` shells out to `python -m qeu_bundling.cli run full`; refresh behavior is a presentation hook over pipeline artifacts, not a second orchestration stack.
- `person_predictions.py` is the main hotspot for profile generation, ranking, feedback multipliers, and lane heuristics.
- `dashboard_i18n.py` is dashboard-only translation logic with persistent cache files and fallback behavior when translation is unavailable.
- Keep edits presentation-local; dashboard routes consume pipeline outputs rather than redefining phase logic.

## ANTI-PATTERNS
- Do not document or edit generated output artifacts as if they live in this package.
- Do not move pipeline phase rules or generic test command lists into this file.
