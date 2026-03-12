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

### Full Pipeline

```bash
python -m qeu_bundling.cli run full
```

### Quick Refresh

```bash
python -m qeu_bundling.cli run quick
```

### Dashboard

```bash
python -m qeu_bundling.cli serve
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
