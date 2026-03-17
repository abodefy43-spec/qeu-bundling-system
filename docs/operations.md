# Operations

## Install

```bash
pip install -r requirements.txt
```

## Migrate Legacy Data Layout (one-time)

```bash
python -m qeu_bundling.cli migrate-data
```

This moves legacy files from:

- `data first/` -> `data/raw/`
- `data/*.json|csv` (reference configs) -> `data/reference/`
- remaining `data/*` generated files -> `data/processed/`

## Common Commands

### API (24/7 service process)

```bash
python -m qeu_bundling.cli serve --host 0.0.0.0 --port 5000
```

### Daily batch compute (full uncapped run)

```bash
unset QEU_FINAL_RECOMMENDATIONS_MAX_USERS
python -m qeu_bundling.cli run full
```

### Daily batch compute (materialize final artifact only, uncapped)

```bash
unset QEU_FINAL_RECOMMENDATIONS_MAX_USERS
python -m qeu_bundling.cli run materialize-final
```

Batch runs are one-off processes and exit after artifact upload. The API keeps serving the currently loaded artifact until you roll/restart API tasks.

### Fast test mode (10 users, same materialization path)

```bash
python -m qeu_bundling.cli run materialize-final --max-users 10
```

### Fast test mode (10 users random sample, reproducible)

```bash
python -m qeu_bundling.cli run materialize-final --max-users 10 --random-sample --random-seed 42
```

### Quick refresh (phases 6-9 + final materialization)

```bash
python -m qeu_bundling.cli run quick
```

### Dashboard (local)

```bash
python -m qeu_bundling.cli serve --host 127.0.0.1 --port 5000
```

### Usage Summary

```bash
python -m qeu_bundling.cli usage --csv d:\usage-events-YYYY-MM-DD.csv --save output/usage_summary_latest.txt
```

### Interactive Review

```bash
python -m qeu_bundling.cli review
```

## Troubleshooting

- If pipeline cannot locate `order_items.csv`, place it in `data/raw/`.
- If phases fail on missing reference files, verify `data/reference/` contains:
  - `recipe_data.json`
  - `product_families.json`
  - `theme_tokens.json`
  - `category_importance.csv`
- If dashboard shows no data, run `python -m qeu_bundling.cli run full` first.
