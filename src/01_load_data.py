"""Phase 1: Data loading and preprocessing for the QEU bundling system."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from product_name_translation import _is_arabic, translate_arabic_to_english

DEFAULT_START_DATE = "2026-01-26"
DEFAULT_END_DATE = "2026-02-25"

SAUDI_IMPORTANCE_WEIGHTS: dict[str, float] = {
    "critical": 1.0,
    "very_high": 0.9,
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}

REQUIRED_ORDER_COLUMNS = [
    "order_id",
    "product_id",
    "product_name",
    "unit_price",
    "effective_price",
    "base_price",
    "discount_amount",
    "quantity",
    "created_at",
    "campaign_id",
    "campaign_type",
    "product_role",
    "product_picture",
]


@dataclass
class LoadedData:
    orders: pd.DataFrame
    recipe_data: dict[str, Any]
    category_importance: pd.DataFrame
    ingredient_index: pd.DataFrame


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return _project_root() / "data"


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _parse_product_name(raw: str, cache_path: Path | None = None) -> str:
    """Extract a usable product name from the JSON-encoded column.

    Format is typically ``{"ar": "...", "en": "..."}``.
    Prefer the ``en`` value; fall back to ``ar``.
    """
    if not isinstance(raw, str):
        return str(raw) if raw is not None else ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            en = str(parsed.get("en", "")).strip()
            ar = str(parsed.get("ar", "")).strip()
            result = en if en else ar
            if cache_path is not None and _is_arabic(result):
                return translate_arabic_to_english(result, cache_path)
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    cleaned = re.sub(r'[{}":]', " ", raw).strip()
    if cache_path is not None and _is_arabic(cleaned):
        return translate_arabic_to_english(cleaned, cache_path)
    return cleaned


def load_order_data(
    data_dir: str | Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    force_rebuild: bool = False,
    save_cache: bool = True,
) -> pd.DataFrame:
    """Load order items filtered to the target date range.

    Uses ``data/filtered_orders.pkl`` when available (and it contains
    ``order_id``).  Falls back to CSV filtering otherwise.
    """
    base = _data_dir(data_dir)
    root = base.parent
    pickle_path = base / "filtered_orders.pkl"

    if pickle_path.exists() and not force_rebuild:
        orders = pd.read_pickle(pickle_path)
        if "order_id" in orders.columns:
            return _finalise_orders(orders, cache_path=base / "arabic_translations_cache.json")

    csv_path = _first_existing([
        base / "order_items.csv",
        root / "order_items.csv",
        root / "data first" / "order_items.csv",
    ])
    if csv_path is None:
        raise FileNotFoundError("Cannot locate order_items.csv (checked data/, project root, data first/)")

    orders = pd.read_csv(csv_path, low_memory=False)
    orders = _coerce_numerics(orders)

    if "created_at" in orders.columns:
        orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce", utc=True)
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = orders["created_at"].notna() & orders["created_at"].between(start, end)
        orders = orders.loc[mask].copy()

    orders = _finalise_orders(orders, cache_path=base / "arabic_translations_cache.json")

    if save_cache:
        base.mkdir(parents=True, exist_ok=True)
        orders.to_pickle(pickle_path)
        print(f"  Cached {len(orders):,} rows -> {pickle_path}")

    return orders


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["quantity", "unit_price", "total_price", "base_price",
                "discount_amount", "effective_price", "product_id", "order_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _finalise_orders(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    """Keep only required columns and parse product names."""
    keep = [c for c in REQUIRED_ORDER_COLUMNS if c in df.columns]
    df = df[keep].copy()
    if "product_name" in df.columns:
        df["product_name_raw"] = df["product_name"]
        df["product_name"] = df["product_name_raw"].apply(
            lambda raw: _parse_product_name(raw, cache_path=cache_path)
        )
    return df


# ------------------------------------------------------------------
# Recipe data
# ------------------------------------------------------------------

def load_recipe_data(data_dir: str | Path | None = None) -> dict[str, Any]:
    path = _data_dir(data_dir) / "recipe_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for section in ("saudi_consumption_data", "saudi_ramadan_dishes", "ingredients"):
        if section not in data:
            raise ValueError(f"recipe_data.json missing section: {section}")
    return data


# ------------------------------------------------------------------
# Ingredient index (for recipe scoring & category matching)
# ------------------------------------------------------------------

def _normalise_token(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def build_ingredient_index(recipe_data: dict[str, Any]) -> pd.DataFrame:
    """Build a flat lookup table: alias -> ingredient metadata."""
    ingredients = recipe_data.get("ingredients", {})
    rows: list[dict[str, Any]] = []

    for key, info in ingredients.items():
        info = info or {}
        importance = str(info.get("saudi_importance", "medium")).lower()
        weight = SAUDI_IMPORTANCE_WEIGHTS.get(importance, 0.6)
        qeu = float(info.get("qeu_relevance", 0))
        boosted = min(100.0, qeu * weight)

        aliases: set[str] = set()
        aliases.add(_normalise_token(key))
        if info.get("en"):
            aliases.add(_normalise_token(info["en"]))
        if info.get("ar"):
            aliases.add(_normalise_token(info["ar"]))
        for pt in info.get("product_types", []):
            aliases.add(_normalise_token(pt))
        aliases.discard("")

        for alias in aliases:
            rows.append({
                "ingredient_key": key,
                "alias": alias,
                "qeu_relevance": qeu,
                "saudi_importance": importance,
                "importance_weight": weight,
                "boosted_relevance": boosted,
                "category": info.get("category"),
                "subcategory": info.get("subcategory"),
            })

    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx = (
            idx.sort_values(["alias", "boosted_relevance"], ascending=[True, False])
            .drop_duplicates(subset=["alias", "ingredient_key"])
            .reset_index(drop=True)
        )
    return idx


# ------------------------------------------------------------------
# Category importance
# ------------------------------------------------------------------

def load_category_importance(data_dir: str | Path | None = None) -> pd.DataFrame:
    path = _data_dir(data_dir) / "category_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for col in ["purchase_count", "purchase_frequency_pct", "revenue_sar",
                "revenue_pct", "avg_price", "unique_products",
                "recipe_importance", "final_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "seasonal_weight" in df.columns:
        df["seasonal_weight"] = df["seasonal_weight"].astype(str).str.lower()
    return df


# ------------------------------------------------------------------
# Convenience loader
# ------------------------------------------------------------------

def load_all(
    data_dir: str | Path | None = None,
    force_rebuild: bool = False,
) -> LoadedData:
    orders = load_order_data(data_dir=data_dir, force_rebuild=force_rebuild)
    recipe = load_recipe_data(data_dir=data_dir)
    cats = load_category_importance(data_dir=data_dir)
    idx = build_ingredient_index(recipe)
    return LoadedData(orders=orders, recipe_data=recipe,
                      category_importance=cats, ingredient_index=idx)


if __name__ == "__main__":
    print("Phase 1: Loading data ...")
    data = load_all(force_rebuild=True)
    print(f"  Orders:      {len(data.orders):,} rows, columns={list(data.orders.columns)}")
    print(f"  Ingredients: {len(data.ingredient_index):,} alias rows")
    print(f"  Categories:  {len(data.category_importance):,} rows")
    print(f"  Sample name: {data.orders['product_name'].iloc[0]}")
    print("Phase 1 complete.")
