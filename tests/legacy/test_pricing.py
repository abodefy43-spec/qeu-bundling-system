import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qeu_bundling.core.pricing import (
    FIXED_MARGIN_DISCOUNT_PCT,
    ProductPriceRecord,
    load_odoo_price_lookup,
    margin_discounted_sale_price,
    price_paid_and_free_items,
    price_paid_and_free_items_fixed_margin,
    resolve_sale_and_purchase,
)


class PricingRulesTests(unittest.TestCase):
    def test_margin_discount_never_below_purchase_cost(self):
        final_price = margin_discounted_sale_price(selling_price=22.0, purchase_price=15.0, discount_pct=95.0)
        self.assertGreaterEqual(final_price, 15.0)
        self.assertEqual(final_price, 15.35)

    def test_discount_applies_to_margin_not_total_price(self):
        # selling=100, purchase=60, margin=40, discount=50% -> final=60 + 20 = 80
        final_price = margin_discounted_sale_price(selling_price=100.0, purchase_price=60.0, discount_pct=50.0)
        self.assertEqual(final_price, 80.0)

    def test_paid_item_is_higher_original_sale_price_and_other_is_free(self):
        priced = price_paid_and_free_items(
            sale_price_a=12.0,
            purchase_price_a=8.0,
            discount_pct_a=20.0,
            sale_price_b=20.0,
            purchase_price_b=10.0,
            discount_pct_b=30.0,
        )
        self.assertEqual(str(priced["paid_side"]), "b")
        self.assertEqual(str(priced["free_product"]), "product_a")
        self.assertEqual(float(priced["price_after_discount_a"]), 0.0)
        self.assertGreaterEqual(float(priced["price_after_discount_b"]), 10.0)

    def test_fixed_margin_bundle_pricing_uses_80pct_margin_discount(self):
        priced = price_paid_and_free_items_fixed_margin(
            sale_price_a=20.0,
            purchase_price_a=15.0,
            sale_price_b=30.0,
            purchase_price_b=10.0,
        )
        self.assertEqual(FIXED_MARGIN_DISCOUNT_PCT, 80.0)
        self.assertEqual(str(priced["paid_side"]), "b")
        self.assertEqual(float(priced["price_after_discount_b"]), 14.0)
        self.assertEqual(float(priced["price_after_discount_a"]), 0.0)

    def test_resolve_sale_and_purchase_missing_purchase_uses_safe_fallback(self):
        sale, purchase, missing = resolve_sale_and_purchase(
            product_id=999,
            fallback_sale_price=14.0,
            price_lookup={999: ProductPriceRecord(sale_price=14.0, purchase_price=0.0)},
        )
        self.assertTrue(missing)
        self.assertEqual(sale, 14.0)
        self.assertEqual(purchase, 14.0)

    def test_resolve_sale_and_purchase_invalid_cost_above_sale_falls_back_safely(self):
        sale, purchase, missing = resolve_sale_and_purchase(
            product_id=777,
            fallback_sale_price=10.0,
            price_lookup={777: ProductPriceRecord(sale_price=10.0, purchase_price=35.0)},
        )
        self.assertTrue(missing)
        self.assertEqual(sale, 10.0)
        self.assertEqual(purchase, 10.0)

    def test_load_odoo_price_lookup_reads_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "data" / "reference"
            ref.mkdir(parents=True, exist_ok=True)
            workbook = ref / "odoo_product.xlsx"
            frame = pd.DataFrame(
                {
                    "product_id": [1, 2],
                    "sale_price": [30.0, 12.5],
                    "purchase_price": [20.0, 10.0],
                }
            )
            frame.to_excel(workbook, index=False)

            lookup, meta = load_odoo_price_lookup(root)
            self.assertIn(1, lookup)
            self.assertIn(2, lookup)
            self.assertEqual(lookup[1].sale_price, 30.0)
            self.assertEqual(lookup[1].purchase_price, 20.0)
            self.assertEqual(meta.get("id_column"), "product_id")
            self.assertEqual(meta.get("sale_column"), "sale_price")
            self.assertEqual(meta.get("purchase_column"), "purchase_price")


if __name__ == "__main__":
    unittest.main()
