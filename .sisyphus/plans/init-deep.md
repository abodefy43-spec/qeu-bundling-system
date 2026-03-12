# Init Deep for QEU Bundling System

## TL;DR
> **Summary**: Generate a root `AGENTS.md` plus a minimal child hierarchy for the active Python code and test areas, using repo-truth sources only and avoiding redundant files in generated or archival trees.
> **Deliverables**:
> - Root `AGENTS.md`
> - `src/qeu_bundling/AGENTS.md`
> - `src/qeu_bundling/presentation/AGENTS.md`
> - `src/qeu_bundling/pipeline/AGENTS.md`
> - `src/qeu_bundling/core/AGENTS.md`
> - `tests/AGENTS.md`
> **Effort**: Medium
> **Parallel**: YES - 3 waves
> **Critical Path**: 1 -> 2 -> 3 -> (4,5,6,7) -> (8,9,10)

## Context
### Original Request
- Run `/init-deep` in default update mode.
- Generate hierarchical `AGENTS.md` files using discovery, scoring, generation, and review.
- Maximize search effort before deciding locations.

### Interview Summary
- The repo is a Python project centered on `src/qeu_bundling` with active domains in `presentation`, `pipeline`, `core`, and `tests`.
- There are no existing `AGENTS.md` or `CLAUDE.md` files, so update mode becomes create-only for this repo.
- `data/`, `output/`, and `rollback_safety_v2/` are mostly input/generated/archive areas already described by local `README.md` files or by root docs.
- Validation should focus on file placement, repo-truth commands, local-only child guidance, and parent/child non-redundancy.

### Metis Review (gaps addressed)
- Added explicit truth-source precedence and fail-closed behavior if a target `AGENTS.md` unexpectedly appears before execution.
- Added a concrete create-vs-skip rubric so placement is algorithmic rather than subjective.
- Added explicit exclusion rules for generated/artifact/archive trees and a separate audit task for non-target directories.

## Work Objectives
### Core Objective
- Create a repo-specific `AGENTS.md` hierarchy that helps coding agents work in the active code and test areas without repeating parent guidance or documenting generated trees as if they were source areas.

### Deliverables
- `AGENTS.md`
- `src/qeu_bundling/AGENTS.md`
- `src/qeu_bundling/presentation/AGENTS.md`
- `src/qeu_bundling/pipeline/AGENTS.md`
- `src/qeu_bundling/core/AGENTS.md`
- `tests/AGENTS.md`
- `.sisyphus/evidence/task-*-*.txt` validation artifacts

### Definition of Done (verifiable conditions with commands)
- `python -c "from pathlib import Path; req=['AGENTS.md','src/qeu_bundling/AGENTS.md','src/qeu_bundling/presentation/AGENTS.md','src/qeu_bundling/pipeline/AGENTS.md','src/qeu_bundling/core/AGENTS.md','tests/AGENTS.md']; missing=[p for p in req if not Path(p).exists()]; assert not missing, missing"`
- `python -c "from pathlib import Path; forbidden=['docs/AGENTS.md','data/AGENTS.md','data/raw/AGENTS.md','data/reference/AGENTS.md','data/processed/AGENTS.md','output/AGENTS.md','rollback_safety_v2/AGENTS.md']; bad=[p for p in forbidden if Path(p).exists()]; assert not bad, bad"`
- `python -c "from pathlib import Path; files=['AGENTS.md','src/qeu_bundling/AGENTS.md','src/qeu_bundling/presentation/AGENTS.md','src/qeu_bundling/pipeline/AGENTS.md','src/qeu_bundling/core/AGENTS.md','tests/AGENTS.md'];
for p in files:
    text=Path(p).read_text(encoding='utf-8')
    assert '## OVERVIEW' in text and '## WHERE TO LOOK' in text, p
    assert len(text.splitlines()) <= (150 if p=='AGENTS.md' else 80), p"`
- `python -c "from pathlib import Path; text=Path('AGENTS.md').read_text(encoding='utf-8'); assert 'python -m qeu_bundling.cli run full' in text and 'python -m unittest discover -s tests -p \"test_*.py\"' in text"`

### Must Have
- Root file covers repo overview, structure, commands, truth sources, and explicit skip/exclusion notes.
- Child files cover only local deltas for their directory.
- Root header is exactly `# PROJECT KNOWLEDGE BASE`; every child header is exactly `# SUBDIRECTORY KNOWLEDGE BASE`.
- Command references come from repo truth sources only: `README.md`, `docs/architecture.md`, `docs/operations.md`, subdir `README.md`, `.github/workflows/ci.yml`, and active source layout.
- Placement uses the rubric below, not intuition.
- Any unexpected preexisting target `AGENTS.md` is read first and only overwritten when it already matches the generated header; otherwise that path fails closed.

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- Must NOT create `AGENTS.md` in generated or archival trees: `data/raw`, `data/processed`, `data/reference`, `output`, `rollback_safety_v2`, or their descendants.
- Must NOT restate parent sections verbatim inside child files.
- Must NOT invent workflows, commands, package names, or tools missing from the repo.
- Must NOT rewrite `README.md`, docs, source code, tests, or generated data.
- Must NOT include generic “follow best practices” filler.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: `none` for code-level tests; use content-validation commands plus repo truth checks. Existing framework: `unittest` is documented in CI but does not need to run because no code paths change.
- QA policy: every task includes happy-path and failure-path scenarios using exact files and commands.
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.txt`

## Execution Strategy
### Truth-Source Precedence
- 1. Executable repo truth in active source layout and command entrypoints
- 2. `README.md`
- 3. `docs/architecture.md`
- 4. `docs/operations.md`
- 5. Subdirectory `README.md`
- 6. `.github/workflows/ci.yml`
- 7. External AGENTS examples for formatting only, never for repo facts

### Placement Rubric
- Score inputs:
  - `+3` file count > 20
  - `+2` immediate subdir count > 5
  - `+2` code ratio > 70%
  - `+2` boundary marker exists (`__init__.py` for package code, dedicated test root for `tests`)
  - `+2` at least one local hotspot file > 500 lines
  - `+3` distinct local workflow exists that would otherwise force root duplication (Flask/session/UI, phase sequencing/artifacts, shared utility IO/manifests, or unittest/mock conventions)
- Create rule: generate a child file when score >= 8.
- Borderline rule: if score is 6-7, generate only when at least two local-only conventions are provable from repo files.
- Skip rule: do not generate in generated/input/archive trees even if raw score qualifies, unless a preexisting managed `AGENTS.md` already exists there.
- Depth rule: `max-depth=3` counts from repo root; deeper descendants stay covered by the nearest generated parent in this repo.

### Parallel Execution Waves
Wave 1: placement matrix, root file, package-parent file
Wave 2: presentation, pipeline, core, and tests child files
Wave 3: exclusion audit, command/path validation, parent-child dedup review

### Dependency Matrix (full, all tasks)
| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | none | 2,3,4,5,6,7,8 |
| 2 | 1 | 3,9,10 |
| 3 | 1,2 | 4,5,6,9,10 |
| 4 | 1,3 | 10 |
| 5 | 1,3 | 10 |
| 6 | 1,3 | 10 |
| 7 | 1,2 | 10 |
| 8 | 1 | 10 |
| 9 | 2,3,4,5,6,7 | 10 |
| 10 | 4,5,6,7,8,9 | Final Verification Wave |

### Agent Dispatch Summary (wave -> task count -> categories)
- Wave 1 -> 3 tasks -> `unspecified-low`, `writing`
- Wave 2 -> 4 tasks -> `writing`
- Wave 3 -> 3 tasks -> `unspecified-low`, `writing`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. Lock Placement Matrix and Exclusion Rules
- [x] 2. Write Root `AGENTS.md`
- [x] 3. Write Package-Parent `src/qeu_bundling/AGENTS.md`
- [x] 4. Write `src/qeu_bundling/presentation/AGENTS.md`
- [x] 5. Write `src/qeu_bundling/pipeline/AGENTS.md`
- [x] 6. Write `src/qeu_bundling/core/AGENTS.md`
- [x] 7. Write `tests/AGENTS.md`
- [x] 8. Audit Excluded Directories Stay Parent-Covered
- [x] 9. Validate Commands and Repo References Inside All Generated Files
- [x] 10. Deduplicate Parent/Child Content and Finalize Hierarchy

  **What to do**: Run a final pass that trims overlap between parent and child files, confirms each child adds local-only value, and records the finalized file list plus line counts.
  **Must NOT do**: Do not leave children that merely restate parent sections. Do not exceed the line-budget caps from the Definition of Done.

  **Recommended Agent Profile**:
  - Category: `writing` - Reason: concise rewrite/review of generated markdown only
  - Skills: `[]` - No specialized skill needed
  - Omitted: [`playwright`, `git-master`] - Not relevant

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification Wave | Blocked By: 4,5,6,7,8,9

  **References** (executor has NO interview context - be exhaustive):
  - External: `https://agents.md/` - Child files should be nearest-context overrides, not parent copies
  - Pattern: `AGENTS.md` - Root file after generation
  - Pattern: `src/qeu_bundling/AGENTS.md` - Package parent after generation
  - Pattern: `src/qeu_bundling/presentation/AGENTS.md` - Local delta candidate
  - Pattern: `src/qeu_bundling/pipeline/AGENTS.md` - Local delta candidate
  - Pattern: `src/qeu_bundling/core/AGENTS.md` - Local delta candidate
  - Pattern: `tests/AGENTS.md` - Local delta candidate

  **Acceptance Criteria** (agent-executable only):
  - [ ] `python -c "from pathlib import Path; files=['AGENTS.md','src/qeu_bundling/AGENTS.md','src/qeu_bundling/presentation/AGENTS.md','src/qeu_bundling/pipeline/AGENTS.md','src/qeu_bundling/core/AGENTS.md','tests/AGENTS.md']; lines={p: len(Path(p).read_text(encoding='utf-8').splitlines()) for p in files}; assert lines['AGENTS.md'] <= 150; assert all(lines[p] <= 80 for p in files if p != 'AGENTS.md'); Path('.sisyphus/evidence/task-10-dedup.txt').write_text('\n'.join(f'{k}\t{v}' for k,v in lines.items()), encoding='utf-8')"`
  - [ ] `python -c "from pathlib import Path; parent=Path('src/qeu_bundling/AGENTS.md').read_text(encoding='utf-8'); child=Path('src/qeu_bundling/presentation/AGENTS.md').read_text(encoding='utf-8'); assert child != parent; assert 'Flask' in child and 'Flask' not in parent"`

  **QA Scenarios** (MANDATORY - task incomplete without these):
  ```text
  Scenario: Every child adds local-only value
    Tool: Bash
    Steps: Run the line-budget command and compare parent vs child presence of local anchor terms such as `Flask`, `phase_06`, `run_manifest.py`, and `patch`.
    Expected: Each child contains local anchor terms missing from the parent and stays within line limits.
    Evidence: .sisyphus/evidence/task-10-dedup.txt

  Scenario: A child file is effectively a parent copy
    Tool: Bash
    Steps: Compare parent and child files for identical bodies or missing local anchor terms.
    Expected: No child is identical to its parent; any such child must be trimmed or deleted before final review.
    Evidence: .sisyphus/evidence/task-10-dedup-error.txt
  ```

  **Commit**: NO | Message: `docs(agents): deduplicate hierarchy` | Files: `AGENTS.md`, `src/qeu_bundling/AGENTS.md`, `src/qeu_bundling/presentation/AGENTS.md`, `src/qeu_bundling/pipeline/AGENTS.md`, `src/qeu_bundling/core/AGENTS.md`, `tests/AGENTS.md`

## Final Verification Wave (4 parallel agents, ALL must APPROVE)
- [x] F1. Plan Compliance Audit - oracle
- [x] F2. Code Quality Review - unspecified-high
- [x] F3. Real Manual QA - unspecified-high
- [x] F4. Scope Fidelity Check - deep

## Commit Strategy
- No commit is part of this plan unless the user explicitly asks for one after review.
- If a commit is later requested, use one commit for the finished AGENTS hierarchy rather than per-file commits.

## Success Criteria
- Exactly six `AGENTS.md` files exist at the approved paths and nowhere else.
- Root guidance matches repo truth and child files add local guidance only.
- Generated/artifact/archive trees stay uncovered by child AGENTS files.
- Validation evidence proves structure, command accuracy, and low parent/child redundancy without human inspection.
