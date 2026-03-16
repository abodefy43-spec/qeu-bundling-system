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

### Full Pipeline (100-user precompute test mode)

```bash
QEU_FINAL_RECOMMENDATIONS_MAX_USERS=100 python -m qeu_bundling.cli run full
```

### Materialize Final Recommendations Only (No Phase Rerun)

```bash
QEU_FINAL_RECOMMENDATIONS_MAX_USERS=100 \
QEU_FINAL_RECOMMENDATIONS_USER_SELECTION=random \
QEU_FALLBACK_BUNDLE_BANK_ENABLED=1 \
QEU_FALLBACK_BUNDLE_BANK_TARGET_SIZE=1000 \
QEU_FALLBACK_BUNDLE_BANK_MAX_SIZE=2000 \
python -m qeu_bundling.cli run materialize-final
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
