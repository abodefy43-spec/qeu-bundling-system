# Sensitive Data Removal and Placeholders

The repository excludes raw customer exports and generated artifacts derived from them.

## Removed from Git

- `odoo_products.xlsx`
- `data/processed/**` generated artifacts (except `data/processed/README.md`)

## Kept Locally (not tracked)

- `data/raw/order_items.csv`
- `odoo_products.xlsx` (or `odoo_product.xlsx`)
- pipeline outputs under `data/processed/` and `output/`

## Safe Examples in Repo

- `data/raw/order_items.example.csv` (synthetic order-item schema sample)
- `data/raw/odoo_products.example.csv` (synthetic product price sample)

## How to regenerate processed artifacts

1. Place real input files locally (not in Git).
2. Run:

```bash
python -m qeu_bundling.cli run full
```

3. Generated files appear under `data/processed/` and `output/`.
