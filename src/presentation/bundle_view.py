"""View-model helpers for executive bundle presentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class BundleViewData:
    kpis: dict[str, str]
    top10_rows: list[dict[str, object]]
    all_rows: list[dict[str, object]]
    data_warning: str = ""


def load_bundle_view(base_dir: Path, query: str = "") -> BundleViewData:
    output_dir = base_dir / "output"
    final_path = output_dir / "final_bundles.csv"
    top10_path = output_dir / "top_10_bundles.csv"

    if not final_path.exists() and not top10_path.exists():
        return BundleViewData(
            kpis=_empty_kpis(),
            top10_rows=[],
            all_rows=[],
            data_warning="No bundle outputs found yet. Run a refresh to generate files.",
        )

    final_df = _safe_read_csv(final_path)
    top10_df = _safe_read_csv(top10_path)

    if final_df.empty and top10_df.empty:
        return BundleViewData(
            kpis=_empty_kpis(),
            top10_rows=[],
            all_rows=[],
            data_warning="Bundle outputs are present but empty.",
        )

    if top10_df.empty:
        top10_df = final_df.head(10).copy()

    final_df = _normalise_bundle_columns(final_df)
    top10_df = _normalise_bundle_columns(top10_df)

    if query.strip():
        final_df = _filter_rows(final_df, query)

    kpis = _build_kpis(final_df, final_path)
    return BundleViewData(
        kpis=kpis,
        top10_rows=_to_records(top10_df),
        all_rows=_to_records(final_df),
        data_warning="",
    )


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception:
        return pd.DataFrame()


def _normalise_bundle_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    data = df.copy()
    data["product_a_name"] = _series_from_columns(data, ["product_a_name", "product_a"], "")
    data["product_b_name"] = _series_from_columns(data, ["product_b_name", "product_b"], "")

    price_a = _series_from_columns(data, ["product_a_price", "price_a_sar"], 0)
    price_b = _series_from_columns(data, ["product_b_price", "price_b_sar"], 0)
    data["product_a_price"] = pd.to_numeric(price_a, errors="coerce").fillna(0.0)
    data["product_b_price"] = pd.to_numeric(price_b, errors="coerce").fillna(0.0)

    discount_raw = _series_from_columns(data, ["discount_amount", "discount_pred"], 0.0)
    data["discount_amount"] = pd.to_numeric(discount_raw, errors="coerce").fillna(0.0)
    data["discount_pred_a"] = pd.to_numeric(_series_from_columns(data, ["discount_pred_a", "discount_a"], 0.0), errors="coerce").fillna(0.0)
    data["discount_pred_b"] = pd.to_numeric(_series_from_columns(data, ["discount_pred_b", "discount_b"], 0.0), errors="coerce").fillna(0.0)
    data["discount_pred_c"] = pd.to_numeric(_series_from_columns(data, ["discount_pred_c", "discount_c"], 0.0), errors="coerce").fillna(0.0)
    if (data["discount_pred_a"].notna().any() or data["discount_pred_b"].notna().any()) and data["discount_amount"].isna().all():
        data["discount_amount"] = (data["discount_pred_a"].fillna(0) + data["discount_pred_b"].fillna(0)) / 2

    raw_free_item = _series_from_columns(data, ["free_product", "free_item"], 0)
    data["free_product_raw"] = raw_free_item.apply(_normalise_free_product)

    if "has_ramadan" in data.columns:
        has_ramadan_series = pd.to_numeric(data["has_ramadan"], errors="coerce").fillna(0)
    else:
        has_ramadan_series = pd.Series(0, index=data.index, dtype="float64")
    data["has_ramadan"] = has_ramadan_series.astype(int)

    data["price_a_sar"] = data["product_a_price"].map(lambda v: f"{v:,.2f}")
    data["price_b_sar"] = data["product_b_price"].map(lambda v: f"{v:,.2f}")
    data["discount_pct"] = data["discount_amount"].map(lambda v: f"{v:,.1f}%")

    pa_after = _series_from_columns(data, ["price_after_discount_a"], 0.0)
    pb_after = _series_from_columns(data, ["price_after_discount_b"], 0.0)
    data["price_after_discount_a"] = pd.to_numeric(pa_after, errors="coerce").fillna(0.0)
    data["price_after_discount_b"] = pd.to_numeric(pb_after, errors="coerce").fillna(0.0)
    data["price_after_a_sar"] = data["price_after_discount_a"].map(lambda v: f"{v:,.2f}")
    data["price_after_b_sar"] = data["price_after_discount_b"].map(lambda v: f"{v:,.2f}")

    data["product_c_name"] = _series_from_columns(data, ["product_c_name", "product_c"], "")
    pc_price = _series_from_columns(data, ["product_c_price"], 0.0)
    data["product_c_price"] = pd.to_numeric(pc_price, errors="coerce").fillna(0.0)
    data["price_c_sar"] = data["product_c_price"].map(lambda v: f"{v:,.2f}")

    pc_after = _series_from_columns(data, ["price_after_discount_c"], 0.0)
    data["price_after_discount_c"] = pd.to_numeric(pc_after, errors="coerce").fillna(0.0)
    data["price_after_c_sar"] = data["price_after_discount_c"].map(lambda v: f"{v:,.2f}")

    is_triple = _series_from_columns(data, ["is_triple_bundle"], False)
    data["is_triple"] = is_triple.apply(lambda v: str(v).strip().lower() in {"true", "1", "yes"})

    return data


def _build_kpis(df: pd.DataFrame, final_path: Path) -> dict[str, str]:
    if df.empty:
        return _empty_kpis()

    total = len(df)
    avg_discount = float(df["discount_amount"].mean()) if "discount_amount" in df.columns else 0.0
    ramadan_pct = float(df["has_ramadan"].mean() * 100.0) if "has_ramadan" in df.columns else 0.0
    triple_count = int(df["is_triple"].sum()) if "is_triple" in df.columns else 0
    mtime = datetime.fromtimestamp(final_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")

    return {
        "total_bundles": f"{total:,}",
        "avg_discount": f"{avg_discount:,.1f}%",
        "ramadan_pct": f"{ramadan_pct:.1f}%",
        "triple_bundles": f"{triple_count}",
        "last_output_update": mtime,
    }


def _filter_rows(df: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return df
    mask = (
        df["product_a_name"].astype(str).str.lower().str.contains(q, na=False)
        | df["product_b_name"].astype(str).str.lower().str.contains(q, na=False)
        | df["product_c_name"].astype(str).str.lower().str.contains(q, na=False)
    )
    return df[mask].copy()


def _to_records(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    cols = [
        "product_a_name", "product_b_name", "product_c_name",
        "price_a_sar", "price_b_sar", "price_c_sar",
        "price_after_a_sar", "price_after_b_sar", "price_after_c_sar",
        "free_product_raw", "discount_pct", "is_triple", "has_ramadan",
        "discount_pred_a", "discount_pred_b", "discount_pred_c",
    ]
    rename_map = {
        "product_a_name": "product_a",
        "product_b_name": "product_b",
        "product_c_name": "product_c",
        "free_product_raw": "free_product",
        "discount_pct": "discount",
        "discount_pred_a": "discount_a",
        "discount_pred_b": "discount_b",
        "discount_pred_c": "discount_c",
    }
    out = df[[c for c in cols if c in df.columns]].rename(columns=rename_map)
    records = []
    for _, row in out.iterrows():
        rec = row.to_dict()
        rec["bundle_items"] = _items_in_order(rec)
        records.append(rec)
    return records


def _items_in_order(rec: dict[str, object]) -> list[dict[str, object]]:
    """Return items in display order: paid first, free last."""
    free = str(rec.get("free_product", "product_b")).strip().lower()
    is_triple = rec.get("is_triple", False)
    discount_a = rec.get("discount_a", rec.get("discount_pred_a", rec.get("discount_amount", 0)))
    discount_b = rec.get("discount_b", rec.get("discount_pred_b", rec.get("discount_amount", 0)))
    discount_c = rec.get("discount_c", rec.get("discount_pred_c"))
    if discount_c is None and (discount_a is not None or discount_b is not None):
        try:
            discount_c = (float(discount_a or 0) + float(discount_b or 0)) / 2
        except (TypeError, ValueError):
            discount_c = 0
    slots = [
        ("product_a", "product_a", "price_a_sar", "price_after_a_sar", discount_a),
        ("product_b", "product_b", "price_b_sar", "price_after_b_sar", discount_b),
        ("product_c", "product_c", "price_c_sar", "price_after_c_sar", discount_c),
    ]
    paid, free_item = [], None
    for key, name_key, price_key, after_key, disc in slots:
        name = str(rec.get(name_key, "") or "").strip()
        if key == "product_c" and (not is_triple or not name):
            continue
        price_sar = str(rec.get(price_key, "0") or "0")
        after_sar = str(rec.get(after_key, "0") or "0")
        try:
            disc_str = f"{float(disc or 0):,.1f}%" if disc is not None else "0%"
        except (TypeError, ValueError):
            disc_str = "0%"
        item = {"name": name, "price_sar": price_sar, "price_after_sar": after_sar, "discount": disc_str, "is_free": key == free}
        if key == free:
            free_item = item
        else:
            paid.append(item)
    if free_item:
        paid.append(free_item)
    return paid


def _empty_kpis() -> dict[str, str]:
    return {
        "total_bundles": "0",
        "avg_discount": "0.0%",
        "ramadan_pct": "0.0%",
        "triple_bundles": "0",
        "last_output_update": "N/A",
    }


def _normalise_free_product(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"product_a", "a", "0"}:
        return "product_a"
    if text in {"product_b", "b", "1"}:
        return "product_b"
    if text in {"product_c", "c", "2"}:
        return "product_c"
    return "product_b"


def _series_from_columns(df: pd.DataFrame, candidates: list[str], default: object) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            s = df[col].fillna(default)
            if s.dtype == object or str(s.dtype) == "string":
                s = s.astype(str).str.strip()
                s = s.replace("nan", str(default))
            return s
    return pd.Series(default, index=df.index)
