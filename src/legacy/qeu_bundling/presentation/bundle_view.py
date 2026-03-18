"""View-model helpers for people-only recommendation presentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.pricing import FIXED_MARGIN_DISCOUNT_PCT
from qeu_bundling.core.run_manifest import resolve_latest_artifact


@dataclass
class BundleViewData:
    kpis: dict[str, str]
    top10_rows: list[dict[str, object]]
    all_rows: list[dict[str, object]]
    person_recommendations: list[dict[str, object]] | None = None
    data_warning: str = ""
    bundles_df: pd.DataFrame | None = None


def load_bundle_view(
    base_dir: Path,
    query: str = "",
    person_recommendations: list[dict[str, object]] | None = None,
) -> BundleViewData:
    paths = get_paths(project_root=base_dir)
    output_dir = paths.output_dir
    processed_dir = paths.data_processed_dir

    scored_path = resolve_latest_artifact(
        "person_candidates_scored",
        base_dir=base_dir,
        fallback=output_dir / "person_candidates_scored.csv",
    )
    fallback_path = resolve_latest_artifact(
        "person_candidate_pairs",
        base_dir=base_dir,
        fallback=paths.data_processed_candidates_dir / "person_candidate_pairs.csv",
    )

    source_path = scored_path if scored_path is not None and scored_path.exists() else fallback_path
    if source_path is None or not source_path.exists():
        return BundleViewData(
            kpis=_empty_kpis(),
            top10_rows=[],
            all_rows=[],
            person_recommendations=person_recommendations or [],
            data_warning="No people candidate data found yet. Run a refresh to generate files.",
            bundles_df=pd.DataFrame(),
        )

    df = _safe_read_csv(source_path)
    if df.empty:
        return BundleViewData(
            kpis=_empty_kpis(),
            top10_rows=[],
            all_rows=[],
            person_recommendations=person_recommendations or [],
            data_warning="People candidate data is present but empty.",
            bundles_df=pd.DataFrame(),
        )

    df = _normalise_bundle_columns(df)
    if query.strip():
        df = _filter_rows(df, query)

    kpis = _build_kpis(df, source_path)
    return BundleViewData(
        kpis=kpis,
        top10_rows=[],
        all_rows=[],
        person_recommendations=person_recommendations or [],
        data_warning="",
        bundles_df=df,
    )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def _normalise_bundle_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["product_a_name"] = _series_from_columns(data, ["product_a_name", "product_a"], "")
    data["product_b_name"] = _series_from_columns(data, ["product_b_name", "product_b"], "")

    data["product_a_price"] = pd.to_numeric(
        _series_from_columns(data, ["product_a_price", "price_a_sar"], 0.0), errors="coerce"
    ).fillna(0.0)
    data["product_b_price"] = pd.to_numeric(
        _series_from_columns(data, ["product_b_price", "price_b_sar"], 0.0), errors="coerce"
    ).fillna(0.0)
    purchase_a = pd.to_numeric(
        _series_from_columns(data, ["purchase_price_a", "product_a_purchase_price"], float("nan")),
        errors="coerce",
    )
    purchase_b = pd.to_numeric(
        _series_from_columns(data, ["purchase_price_b", "product_b_purchase_price"], float("nan")),
        errors="coerce",
    )
    data["purchase_price_a"] = purchase_a.where(purchase_a > 0, data["product_a_price"]).fillna(data["product_a_price"])
    data["purchase_price_b"] = purchase_b.where(purchase_b > 0, data["product_b_price"]).fillna(data["product_b_price"])

    data["margin_discount_pct"] = pd.to_numeric(
        _series_from_columns(data, ["margin_discount_pct"], float(FIXED_MARGIN_DISCOUNT_PCT)),
        errors="coerce",
    ).fillna(float(FIXED_MARGIN_DISCOUNT_PCT))

    free = _series_from_columns(data, ["free_product", "free_item"], "")
    data["free_product"] = free.astype(str).str.strip().str.lower().replace({"": "product_b"})

    data["price_after_discount_a"] = pd.to_numeric(
        _series_from_columns(data, ["price_after_discount_a"], 0.0), errors="coerce"
    ).fillna(0.0)
    data["price_after_discount_b"] = pd.to_numeric(
        _series_from_columns(data, ["price_after_discount_b"], 0.0), errors="coerce"
    ).fillna(0.0)

    data["price_a_sar"] = data["product_a_price"].map(lambda v: f"{float(v):,.2f}")
    data["price_b_sar"] = data["product_b_price"].map(lambda v: f"{float(v):,.2f}")
    data["purchase_price_a_sar"] = data["purchase_price_a"].map(lambda v: f"{float(v):,.2f}")
    data["purchase_price_b_sar"] = data["purchase_price_b"].map(lambda v: f"{float(v):,.2f}")
    data["price_after_a_sar"] = data["price_after_discount_a"].map(lambda v: f"{float(v):,.2f}")
    data["price_after_b_sar"] = data["price_after_discount_b"].map(lambda v: f"{float(v):,.2f}")

    data["product_a_picture"] = _series_from_columns(data, ["product_a_picture"], "")
    data["product_b_picture"] = _series_from_columns(data, ["product_b_picture"], "")

    data["is_triple"] = False
    data["product_c_name"] = ""
    data["price_c_sar"] = "0.00"
    data["price_after_c_sar"] = "0.00"

    if "final_score" not in data.columns:
        data["final_score"] = pd.to_numeric(_series_from_columns(data, ["new_final_score"], 0.0), errors="coerce").fillna(0.0)

    return data


def _build_kpis(df: pd.DataFrame, source_path: Path) -> dict[str, str]:
    if df.empty:
        return _empty_kpis()

    total = len(df)
    avg_discount = float(pd.to_numeric(df.get("margin_discount_pct", float(FIXED_MARGIN_DISCOUNT_PCT)), errors="coerce").mean())
    mtime = datetime.fromtimestamp(source_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    return {
        "total_bundles": f"{total:,}",
        "avg_discount": f"{avg_discount:,.1f}% margin",
        "ramadan_pct": "-",
        "triple_bundles": "0",
        "last_output_update": mtime,
    }


def _filter_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return df
    mask = (
        df["product_a_name"].astype(str).str.lower().str.contains(q, na=False)
        | df["product_b_name"].astype(str).str.lower().str.contains(q, na=False)
    )
    return df[mask].copy()


def row_to_record(row: pd.Series) -> dict[str, object]:
    rec = row.to_dict()
    rec["bundle_items"] = _items_in_order(rec)
    return rec


def _items_in_order(rec: dict[str, object]) -> list[dict[str, object]]:
    free = str(rec.get("free_product", "product_b")).strip().lower()
    slots = [
        (
            "product_a",
            "product_a_name",
            "price_a_sar",
            "price_after_a_sar",
            "purchase_price_a_sar",
            "product_a_picture",
        ),
        (
            "product_b",
            "product_b_name",
            "price_b_sar",
            "price_after_b_sar",
            "purchase_price_b_sar",
            "product_b_picture",
        ),
    ]

    paid: list[dict[str, object]] = []
    free_item: dict[str, object] | None = None
    for key, name_key, price_key, after_key, purchase_key, pic_key in slots:
        name = str(rec.get(name_key, "") or "").strip()
        if not name:
            continue
        is_free = key == free
        effective_after = "0.00" if is_free else str(rec.get(after_key, "0") or "0")
        item = {
            "name": name,
            "price_sar": str(rec.get(price_key, "0") or "0"),
            "original_price_sar": str(rec.get(price_key, "0") or "0"),
            "purchase_price_sar": str(rec.get(purchase_key, "0") or "0"),
            "price_after_sar": effective_after,
            "discount": "100%" if is_free else "",
            "is_free": is_free,
            "image_url": str(rec.get(pic_key, "") or "").strip(),
        }
        if is_free:
            free_item = item
        else:
            paid.append(item)
    if free_item is not None:
        paid.append(free_item)
    return paid


def _series_from_columns(df: pd.DataFrame, cols: list[str], default: object) -> pd.Series:
    for col in cols:
        if col in df.columns:
            return df[col]
    return pd.Series([default] * len(df), index=df.index)


def _empty_kpis() -> dict[str, str]:
    return {
        "total_bundles": "0",
        "avg_discount": "0.0% margin",
        "ramadan_pct": "0.0%",
        "triple_bundles": "0",
        "last_output_update": "-",
    }
