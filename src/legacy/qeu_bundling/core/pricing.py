"""Shared pricing helpers for margin-safe bundle discounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ODOO_WORKBOOK_CANDIDATES = (
    "odoo_product.xlsx",
    "odoo_products.xlsx",
)

ODOO_ID_COLUMNS = (
    "product_id",
    "id",
    "product_variant_id",
    "variant_id",
)

ODOO_SALE_COLUMNS = (
    "sale_price",
    "list_price",
    "public_price",
    "lst_price",
)

ODOO_PURCHASE_COLUMNS = (
    "purchase_price",
    "standard_price",
    "cost",
    "cost_price",
    "purchase_cost",
)

# Product truth: bundle paid-item pricing always uses a fixed 80% discount on margin.
FIXED_MARGIN_DISCOUNT_PCT = 80.0


@dataclass(frozen=True)
class ProductPriceRecord:
    sale_price: float
    purchase_price: float


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def _first_existing(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    available = {str(col).strip().lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in available:
            return available[candidate]
    return None


def resolve_odoo_workbook_path(project_root: Path) -> Path | None:
    root = Path(project_root).resolve()
    for filename in ODOO_WORKBOOK_CANDIDATES:
        for path in (root / filename, root / "data" / "reference" / filename):
            if path.exists():
                return path
    return None


def load_odoo_price_lookup(project_root: Path) -> tuple[dict[int, ProductPriceRecord], dict[str, str]]:
    """
    Load Odoo product prices keyed by product_id.

    Returns:
      - lookup: {product_id -> ProductPriceRecord}
      - metadata: workbook path and resolved column names
    """
    workbook_path = resolve_odoo_workbook_path(project_root)
    if workbook_path is None:
        return {}, {}

    try:
        df = pd.read_excel(workbook_path, sheet_name=0)
    except Exception:
        return {}, {"workbook_path": str(workbook_path)}

    if df.empty:
        return {}, {"workbook_path": str(workbook_path)}

    id_col = _first_existing(df.columns, ODOO_ID_COLUMNS)
    sale_col = _first_existing(df.columns, ODOO_SALE_COLUMNS)
    purchase_col = _first_existing(df.columns, ODOO_PURCHASE_COLUMNS)
    if id_col is None:
        return {}, {"workbook_path": str(workbook_path)}

    product_ids = pd.to_numeric(df[id_col], errors="coerce").fillna(-1).astype(int)
    sale_raw = (
        pd.to_numeric(df[sale_col], errors="coerce").fillna(0.0)
        if sale_col is not None
        else pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    )
    purchase_raw = (
        pd.to_numeric(df[purchase_col], errors="coerce").fillna(0.0)
        if purchase_col is not None
        else pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    )

    frame = pd.DataFrame({"product_id": product_ids, "sale_price": sale_raw, "purchase_price": purchase_raw})
    frame = frame[frame["product_id"] > 0].copy()
    if frame.empty:
        metadata = {
            "workbook_path": str(workbook_path),
            "id_column": str(id_col),
            "sale_column": str(sale_col or ""),
            "purchase_column": str(purchase_col or ""),
        }
        return {}, metadata

    agg = (
        frame.groupby("product_id", as_index=False)
        .agg({"sale_price": "max", "purchase_price": "max"})
        .reset_index(drop=True)
    )

    lookup: dict[int, ProductPriceRecord] = {}
    for row in agg.itertuples(index=False):
        pid = int(row.product_id)
        sale_price = max(0.0, _safe_float(row.sale_price, default=0.0))
        purchase_price = max(0.0, _safe_float(row.purchase_price, default=0.0))
        lookup[pid] = ProductPriceRecord(sale_price=sale_price, purchase_price=purchase_price)

    metadata = {
        "workbook_path": str(workbook_path),
        "id_column": str(id_col),
        "sale_column": str(sale_col or ""),
        "purchase_column": str(purchase_col or ""),
    }
    return lookup, metadata


def resolve_sale_and_purchase(
    product_id: int,
    fallback_sale_price: float,
    price_lookup: dict[int, ProductPriceRecord],
) -> tuple[float, float, bool]:
    """
    Resolve (sale_price, purchase_price, purchase_missing).

    If purchase cost is missing we use a conservative fallback:
    purchase_price := sale_price, so discounted price cannot go below the observed sale price.
    """
    fallback_sale = max(0.0, _safe_float(fallback_sale_price, default=0.0))
    odoo = price_lookup.get(int(product_id))
    sale_price = fallback_sale
    purchase_price = 0.0
    purchase_missing = True

    if odoo is not None:
        if odoo.sale_price > 0:
            sale_price = float(odoo.sale_price)
        if odoo.purchase_price > 0:
            purchase_price = float(odoo.purchase_price)
            purchase_missing = False

    if purchase_missing:
        purchase_price = max(0.0, sale_price)

    if sale_price <= 0 and purchase_price > 0:
        sale_price = float(purchase_price)

    # Guard against invalid catalog rows where reported purchase exceeds selling.
    # In that case, use a conservative no-discount fallback on sale price.
    if sale_price > 0 and purchase_price > sale_price:
        purchase_price = float(sale_price)
        purchase_missing = True

    return float(sale_price), float(purchase_price), bool(purchase_missing)


def margin_discounted_sale_price(selling_price: float, purchase_price: float, discount_pct: float) -> float:
    """Apply discount to margin only, then clamp to purchase floor."""
    selling = max(0.0, _safe_float(selling_price, default=0.0))
    purchase = max(0.0, _safe_float(purchase_price, default=0.0))
    discount = _safe_float(discount_pct, default=0.0)
    discount = min(100.0, max(0.0, discount))

    profit_margin = max(0.0, selling - purchase)
    discounted_profit_margin = profit_margin * (1.0 - discount / 100.0)
    discounted_sale = purchase + discounted_profit_margin
    final_sale = max(purchase, discounted_sale)
    return round(float(final_sale), 2)


def price_paid_and_free_items(
    sale_price_a: float,
    purchase_price_a: float,
    discount_pct_a: float,
    sale_price_b: float,
    purchase_price_b: float,
    discount_pct_b: float,
) -> dict[str, object]:
    """
    Return paid/free assignment and final prices.

    Paid item is selected by original selling price (higher one is paid).
    """
    sale_a = max(0.0, _safe_float(sale_price_a, default=0.0))
    sale_b = max(0.0, _safe_float(sale_price_b, default=0.0))
    purchase_a = max(0.0, _safe_float(purchase_price_a, default=0.0))
    purchase_b = max(0.0, _safe_float(purchase_price_b, default=0.0))

    paid_side = "a" if sale_a >= sale_b else "b"
    free_product = "product_b" if paid_side == "a" else "product_a"

    final_a = 0.0
    final_b = 0.0
    if paid_side == "a":
        final_a = margin_discounted_sale_price(sale_a, purchase_a, discount_pct_a)
        final_b = 0.0
    else:
        final_b = margin_discounted_sale_price(sale_b, purchase_b, discount_pct_b)
        final_a = 0.0

    return {
        "paid_side": paid_side,
        "free_product": free_product,
        "price_after_discount_a": round(float(final_a), 2),
        "price_after_discount_b": round(float(final_b), 2),
        "unit_profit_a": round(max(0.0, float(final_a) - purchase_a), 4),
        "unit_profit_b": round(max(0.0, float(final_b) - purchase_b), 4),
    }


def price_paid_and_free_items_fixed_margin(
    sale_price_a: float,
    purchase_price_a: float,
    sale_price_b: float,
    purchase_price_b: float,
) -> dict[str, object]:
    """Apply fixed 80%-of-margin discount pricing and return paid/free assignment."""
    return price_paid_and_free_items(
        sale_price_a=sale_price_a,
        purchase_price_a=purchase_price_a,
        discount_pct_a=FIXED_MARGIN_DISCOUNT_PCT,
        sale_price_b=sale_price_b,
        purchase_price_b=purchase_price_b,
        discount_pct_b=FIXED_MARGIN_DISCOUNT_PCT,
    )
