# QEU Bundling System

Phase 0 turns the repository into a clean scaffold for three separate recommendation engines:

- `compatible_products`
- `frequently_bought_together`
- `personalized_bundles`

The engines are intentionally stubbed. This reset is for cleanup, stable execution, and architecture alignment only.

## Active Structure

```text
src/
  api/
  data/
  engines/
    compatible/
    fbt/
    bundles/
  features/
  pipelines/
  shared/
  utils/
tests/
data/
scripts/
legacy/
```

`src/legacy/qeu_bundling` and `tests/legacy` keep the old implementation isolated from the active runtime.

## What Each Engine Will Do

- `compatible_products`: anchor-led compatibility and substitution recommendations
- `frequently_bought_together`: transaction-driven co-purchase recommendations
- `personalized_bundles`: customer-level bundle recommendations

## Canonical Data Layout

Active code reads and writes only through `data/`:

- `data/raw/`
- `data/reference/`
- `data/processed/features/`
- `data/processed/artifacts/`
- `data/processed/reports/`
- `data/processed/runs/`

Legacy runtime folders like `output/` are no longer part of the active execution path.

## Run

```bash
python3 -m pip install -e .[dev]
python3 main.py bootstrap
python3 scripts/run_pipeline.py
python3 main.py api --host 127.0.0.1 --port 8000
pytest
```
