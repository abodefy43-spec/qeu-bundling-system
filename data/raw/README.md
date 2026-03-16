# data/raw

Raw source exports used by pipeline ingestion.

Expected primary file:

- `order_items.csv`

Safe templates (committed):

- `order_items.example.csv`
- `odoo_products.example.csv`

`order_items.csv` required columns:

- `order_id`
- `product_id`
- `product_name`
- `unit_price`
- `effective_price`
- `base_price`
- `discount_amount`
- `quantity`
- `created_at`
- `campaign_id`
- `campaign_type`
- `product_role`
- `product_picture`

Notes:

- Raw source exports are sensitive and are not committed to Git.
- Keep original format/columns from source systems.
- Extra columns like `store_name` and `city` are allowed.
