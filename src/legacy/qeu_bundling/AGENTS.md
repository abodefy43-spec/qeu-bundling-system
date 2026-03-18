# SUBDIRECTORY KNOWLEDGE BASE

## OVERVIEW
- `src/qeu_bundling/` is the canonical package root; use this file for package-wide boundaries and handoffs between child directories.

## WHERE TO LOOK
- CLI entry surface - `cli/__main__.py`
- Runtime shim that preserves `python -m qeu_bundling...` - `qeu_bundling/__init__.py`
- Shared path contract - `config/paths.py`
- Full orchestration - `runners/run_pipeline.py`
- Quick orchestration - `runners/run_new_results.py`
- Review feedback normalization - `review/feedback_loader.py`
- Presentation-local rules - `presentation/AGENTS.md`
- Pipeline-local rules - `pipeline/AGENTS.md`
- Core-local rules - `core/AGENTS.md`

## CONVENTIONS
- Import within the canonical package under `qeu_bundling`, not from legacy top-level scripts.
- Keep the repo-root `qeu_bundling/` package as a runtime bridge only; implementation changes belong under `src/qeu_bundling/`.
- Route all filesystem access through `get_paths()` and helpers derived from it.
- `cli/`, `config/`, `review/`, and `runners/` stay covered here; they are small coordination layers, not separate child domains.
- `presentation/`, `pipeline/`, and `core/` own their local hotspot guidance in child files; keep this parent focused on cross-package rules.
- `runners/` orchestrates phases and review/quality side effects; phase details belong in `pipeline/AGENTS.md`.

## ANTI-PATTERNS
- Do not duplicate the root command catalog here.
- Do not treat the repo-root shim as the canonical implementation package.
- Do not place tests guidance here; `tests/AGENTS.md` owns test-only conventions.
- Do not restate Flask, phase sequencing, or core helper details that are already child-local.
