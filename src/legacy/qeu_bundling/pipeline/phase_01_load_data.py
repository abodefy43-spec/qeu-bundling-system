"""Phase 1: Data loading and preprocessing for the QEU bundling system."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.product_name_translation import _is_arabic, translate_arabic_to_english
from qeu_bundling.core.schema_validation import (
    NumericRangeRule,
    require_columns,
    require_no_nulls,
    validate_numeric_ranges,
)

DEFAULT_START_DATE = "2026-01-26"
DEFAULT_END_DATE = "2026-02-25"

SAUDI_IMPORTANCE_WEIGHTS: dict[str, float] = {
    "critical": 1.0,
    "very_high": 0.9,
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}
VALID_IMPORTANCE_LEVELS = frozenset(SAUDI_IMPORTANCE_WEIGHTS.keys())
REQUIRED_RECIPE_SECTIONS = ("saudi_consumption_data", "saudi_ramadan_dishes", "ingredients")

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
    return get_paths().project_root


def _data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return get_paths().data_processed_dir


def _reference_dir() -> Path:
    return get_paths().data_reference_dir


def _raw_dir() -> Path:
    return get_paths().data_raw_dir


def _first_existing(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _clean_lower(value: Any) -> str:
    return _clean_text(value).lower()


def _normalise_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9\u0600-\u06FF]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _normalise_seasonal_relevance(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        norm_key = _normalise_token(key)
        if not norm_key:
            continue
        out[norm_key] = _clip(_safe_float(value, 0.0), 0.0, 100.0)
    return out


def _normalise_str_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _clean_text(item)
        if not text:
            continue
        norm = _normalise_token(text)
        if norm in seen:
            continue
        seen.add(norm)
        out.append(text)
    return out


def _validate_required_recipe_sections(data: dict[str, Any]) -> None:
    for section in REQUIRED_RECIPE_SECTIONS:
        if section not in data:
            raise ValueError(f"recipe_data.json missing section: {section}")
        if not isinstance(data[section], dict):
            raise ValueError(f"recipe_data.json section `{section}` must be an object/dict")


def _normalise_recipe_data(raw_data: dict[str, Any]) -> dict[str, Any]:
    data = dict(raw_data)
    _validate_required_recipe_sections(data)

    normalized: dict[str, Any] = {
        "metadata": data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        "recipe_categories": data.get("recipe_categories", {}) if isinstance(data.get("recipe_categories"), dict) else {},
        "categories": data.get("categories", {}) if isinstance(data.get("categories"), dict) else {},
        "saudi_consumption_data": data.get("saudi_consumption_data", {}),
        "saudi_ramadan_dishes": {},
        "ingredients": {},
    }

    ingredients = data.get("ingredients", {})
    for raw_key, raw_info in ingredients.items():
        key = _clean_lower(raw_key)
        if not key:
            continue
        info = dict(raw_info) if isinstance(raw_info, dict) else {}
        recipes = _normalise_str_list(info.get("recipes", []))
        recipe_count = max(_safe_int(info.get("recipe_count", len(recipes)), len(recipes)), len(recipes))
        importance = _clean_lower(info.get("saudi_importance", "medium"))
        if importance not in VALID_IMPORTANCE_LEVELS:
            importance = "medium"
        qeu_relevance = _clip(_safe_float(info.get("qeu_relevance", 0.0), 0.0), 0.0, 100.0)
        seasonal = _normalise_seasonal_relevance(info.get("seasonal_relevance", {}))
        product_types = _normalise_str_list(info.get("product_types", []))

        normalized["ingredients"][key] = {
            **info,
            "en": _clean_text(info.get("en", "")),
            "ar": _clean_text(info.get("ar", "")),
            "category": _clean_lower(info.get("category", "other")) or "other",
            "subcategory": _clean_lower(info.get("subcategory", "")),
            "qeu_relevance": qeu_relevance,
            "saudi_importance": importance,
            "product_types": product_types,
            "recipes": recipes,
            "recipe_count": recipe_count,
            "seasonal_relevance": seasonal,
        }

    dishes = data.get("saudi_ramadan_dishes", {})
    if isinstance(dishes, dict):
        for raw_key, raw_info in dishes.items():
            key = _clean_lower(raw_key)
            if not key:
                continue
            info = dict(raw_info) if isinstance(raw_info, dict) else {}
            normalized["saudi_ramadan_dishes"][key] = {
                **info,
                "ar": _clean_text(info.get("ar", "")),
                "key_ingredients": _normalise_str_list(info.get("key_ingredients", [])),
            }

    return normalized


def _compute_ingredient_intensity_row(ingredient_key: str, info: dict[str, Any]) -> dict[str, Any]:
    qeu = _clip(_safe_float(info.get("qeu_relevance", 0.0), 0.0), 0.0, 100.0)
    importance = _clean_lower(info.get("saudi_importance", "medium"))
    importance_weight = SAUDI_IMPORTANCE_WEIGHTS.get(importance, 0.6)
    seasonal = _normalise_seasonal_relevance(info.get("seasonal_relevance", {}))
    seasonal_peak = max(seasonal.values()) if seasonal else 0.0
    recipe_count = max(_safe_int(info.get("recipe_count", len(info.get("recipes", []))), 0), 0)

    # Blend relevance + usage breadth + seasonal signal.
    intensity_score = _clip(
        qeu * importance_weight * 0.7
        + min(recipe_count * 6.0, 100.0) * 0.2
        + seasonal_peak * 0.1,
        0.0,
        100.0,
    )
    return {
        "ingredient_key": ingredient_key,
        "qeu_relevance": qeu,
        "saudi_importance": importance,
        "importance_weight": importance_weight,
        "recipe_count": recipe_count,
        "seasonal_peak": seasonal_peak,
        "intensity_score": round(float(intensity_score), 4),
    }


def _parse_product_name(raw: str, cache_path: Path | None = None) -> str:
    """Extract a usable product name from the JSON-encoded column."""
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
    """Load order items filtered to the target date range."""
    base = _data_dir(data_dir)
    root = _project_root()
    pickle_path = base / "filtered_orders.pkl"

    if pickle_path.exists() and not force_rebuild:
        orders = pd.read_pickle(pickle_path)
        if "order_id" in orders.columns:
            return _finalise_orders(orders, cache_path=base / "arabic_translations_cache.json")

    csv_path = _first_existing(
        [
            _raw_dir() / "order_items.csv",
            base / "order_items.csv",
            root / "order_items.csv",
            root / "data first" / "order_items.csv",
        ]
    )
    if csv_path is None:
        raise FileNotFoundError(
            "Cannot locate order_items.csv (checked data/raw, data/processed, project root, data first/)"
        )

    orders = pd.read_csv(csv_path, low_memory=False)
    orders = _coerce_numerics(orders)

    if "created_at" in orders.columns:
        orders["created_at"] = pd.to_datetime(orders["created_at"], errors="coerce", utc=True)
        start = pd.Timestamp(start_date, tz="UTC")
        end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = orders["created_at"].notna() & orders["created_at"].between(start, end)
        orders = orders.loc[mask].copy()

    orders = _finalise_orders(orders, cache_path=base / "arabic_translations_cache.json")
    _validate_orders_schema(orders, artifact_name="phase_01.filtered_orders")

    if save_cache:
        base.mkdir(parents=True, exist_ok=True)
        orders.to_pickle(pickle_path)
        print(f"  Cached {len(orders):,} rows -> {pickle_path}")

    return orders


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    for col in [
        "quantity",
        "unit_price",
        "total_price",
        "base_price",
        "discount_amount",
        "effective_price",
        "product_id",
        "order_id",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _finalise_orders(df: pd.DataFrame, cache_path: Path | None = None) -> pd.DataFrame:
    keep = [c for c in REQUIRED_ORDER_COLUMNS if c in df.columns]
    df = df[keep].copy()
    if "product_name" in df.columns:
        df["product_name_raw"] = df["product_name"]
        df["product_name"] = df["product_name_raw"].apply(
            lambda raw: _parse_product_name(raw, cache_path=cache_path)
        )
    return df


def _validate_orders_schema(df: pd.DataFrame, artifact_name: str) -> None:
    require_columns(df, ["order_id", "product_id", "product_name", "unit_price"], artifact_name=artifact_name)
    require_no_nulls(df, ["order_id", "product_id", "product_name"], artifact_name=artifact_name)
    validate_numeric_ranges(
        df,
        rules={
            "unit_price": NumericRangeRule(minimum=0.0, allow_null=False),
            "effective_price": NumericRangeRule(minimum=0.0, allow_null=True),
            "base_price": NumericRangeRule(minimum=0.0, allow_null=True),
            "discount_amount": NumericRangeRule(minimum=0.0, allow_null=True),
            "quantity": NumericRangeRule(minimum=0.0, allow_null=True),
        },
        artifact_name=artifact_name,
    )


def load_recipe_data(data_dir: str | Path | None = None) -> dict[str, Any]:
    """Load and normalize recipe data with strict schema hardening."""
    path = _reference_dir() / "recipe_data.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("recipe_data.json must be a JSON object")
    return _normalise_recipe_data(raw)


def build_ingredient_index(recipe_data: dict[str, Any]) -> pd.DataFrame:
    """Build alias-level ingredient index with intensity metadata."""
    ingredients = recipe_data.get("ingredients", {})
    rows: list[dict[str, Any]] = []

    for key, info in ingredients.items():
        info = info if isinstance(info, dict) else {}
        intensity = _compute_ingredient_intensity_row(key, info)

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
            rows.append(
                {
                    "ingredient_key": key,
                    "alias": alias,
                    "qeu_relevance": intensity["qeu_relevance"],
                    "saudi_importance": intensity["saudi_importance"],
                    "importance_weight": intensity["importance_weight"],
                    "recipe_count": intensity["recipe_count"],
                    "seasonal_peak": intensity["seasonal_peak"],
                    "recipe_intensity_score": intensity["intensity_score"],
                    "category": info.get("category", "other"),
                    "subcategory": info.get("subcategory", ""),
                }
            )

    idx = pd.DataFrame(rows)
    if not idx.empty:
        idx = (
            idx.sort_values(["alias", "recipe_intensity_score"], ascending=[True, False])
            .drop_duplicates(subset=["alias", "ingredient_key"])
            .reset_index(drop=True)
        )
    return idx


def build_recipe_links(recipe_data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ingredient<->recipe links and per-ingredient intensity table."""
    ingredients = recipe_data.get("ingredients", {})
    ingredient_to_recipe_rows: list[dict[str, Any]] = []
    recipe_to_ingredient_rows: list[dict[str, Any]] = []
    intensity_rows: list[dict[str, Any]] = []

    for ingredient_key, raw_info in ingredients.items():
        info = raw_info if isinstance(raw_info, dict) else {}
        intensity_rows.append(_compute_ingredient_intensity_row(ingredient_key, info))
        for raw_recipe in info.get("recipes", []):
            recipe_name = _clean_text(raw_recipe)
            if not recipe_name:
                continue
            recipe_key = _normalise_token(recipe_name)
            if not recipe_key:
                continue
            ingredient_to_recipe_rows.append(
                {
                    "ingredient_key": ingredient_key,
                    "recipe_key": recipe_key,
                    "recipe_name": recipe_name,
                    "source": "ingredient_recipes",
                }
            )
            recipe_to_ingredient_rows.append(
                {
                    "recipe_key": recipe_key,
                    "recipe_name": recipe_name,
                    "ingredient_key": ingredient_key,
                    "source": "ingredient_recipes",
                }
            )

    ingredient_lookup = {_normalise_token(k): k for k in ingredients.keys()}
    for dish_key, dish in recipe_data.get("saudi_ramadan_dishes", {}).items():
        if not isinstance(dish, dict):
            continue
        recipe_name = _clean_text(dish_key)
        recipe_key = _normalise_token(recipe_name)
        if not recipe_key:
            continue
        for raw_ing in dish.get("key_ingredients", []):
            ing_norm = _normalise_token(raw_ing)
            ingredient_key = ingredient_lookup.get(ing_norm, ing_norm)
            if not ingredient_key:
                continue
            ingredient_to_recipe_rows.append(
                {
                    "ingredient_key": ingredient_key,
                    "recipe_key": recipe_key,
                    "recipe_name": recipe_name,
                    "source": "ramadan_dish_key_ingredients",
                }
            )
            recipe_to_ingredient_rows.append(
                {
                    "recipe_key": recipe_key,
                    "recipe_name": recipe_name,
                    "ingredient_key": ingredient_key,
                    "source": "ramadan_dish_key_ingredients",
                }
            )

    ingredient_recipe_df = pd.DataFrame(ingredient_to_recipe_rows)
    if not ingredient_recipe_df.empty:
        ingredient_recipe_df = ingredient_recipe_df.drop_duplicates().reset_index(drop=True)

    recipe_ingredient_df = pd.DataFrame(recipe_to_ingredient_rows)
    if not recipe_ingredient_df.empty:
        recipe_ingredient_df = recipe_ingredient_df.drop_duplicates().reset_index(drop=True)

    intensity_df = pd.DataFrame(intensity_rows)
    if not intensity_df.empty:
        intensity_df = intensity_df.sort_values("intensity_score", ascending=False).reset_index(drop=True)

    return ingredient_recipe_df, recipe_ingredient_df, intensity_df


def save_recipe_artifacts(recipe_data: dict[str, Any], data_dir: str | Path | None = None) -> dict[str, int]:
    """Persist recipe-derived artifacts to data/processed for downstream phases."""
    base = _data_dir(data_dir)
    base.mkdir(parents=True, exist_ok=True)

    alias_index = build_ingredient_index(recipe_data)
    ingredient_recipe_df, recipe_ingredient_df, intensity_df = build_recipe_links(recipe_data)

    alias_path = base / "ingredient_alias_index.csv"
    ingredient_recipe_path = base / "ingredient_recipe_links.csv"
    recipe_ingredient_path = base / "recipe_ingredient_links.csv"
    intensity_path = base / "ingredient_recipe_intensity.csv"

    alias_index.to_csv(alias_path, index=False, encoding="utf-8-sig")
    ingredient_recipe_df.to_csv(ingredient_recipe_path, index=False, encoding="utf-8-sig")
    recipe_ingredient_df.to_csv(recipe_ingredient_path, index=False, encoding="utf-8-sig")
    intensity_df.to_csv(intensity_path, index=False, encoding="utf-8-sig")

    summary = {
        "alias_rows": int(len(alias_index)),
        "ingredient_recipe_links": int(len(ingredient_recipe_df)),
        "recipe_ingredient_links": int(len(recipe_ingredient_df)),
        "ingredient_intensity_rows": int(len(intensity_df)),
    }
    with (base / "recipe_artifact_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def load_category_importance(data_dir: str | Path | None = None) -> pd.DataFrame:
    path = _reference_dir() / "category_importance.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for col in [
        "purchase_count",
        "purchase_frequency_pct",
        "revenue_sar",
        "revenue_pct",
        "avg_price",
        "unique_products",
        "recipe_importance",
        "final_score",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "seasonal_weight" in df.columns:
        df["seasonal_weight"] = df["seasonal_weight"].astype(str).str.lower()
    return df


def load_all(
    data_dir: str | Path | None = None,
    force_rebuild: bool = False,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> LoadedData:
    orders = load_order_data(
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
        force_rebuild=force_rebuild,
    )
    recipe = load_recipe_data(data_dir=data_dir)
    cats = load_category_importance(data_dir=data_dir)
    idx = build_ingredient_index(recipe)
    save_recipe_artifacts(recipe, data_dir=data_dir)
    return LoadedData(
        orders=orders,
        recipe_data=recipe,
        category_importance=cats,
        ingredient_index=idx,
    )


def run(
    force_rebuild: bool = True,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> LoadedData:
    return load_all(force_rebuild=force_rebuild, start_date=start_date, end_date=end_date)


if __name__ == "__main__":
    print("Phase 1: Loading data ...")
    data = load_all(force_rebuild=True)
    print(f"  Orders:      {len(data.orders):,} rows, columns={list(data.orders.columns)}")
    print(f"  Ingredients: {len(data.ingredient_index):,} alias rows")
    print(f"  Categories:  {len(data.category_importance):,} rows")
    print(f"  Sample name: {data.orders['product_name'].iloc[0]}")
    print("Phase 1 complete.")
