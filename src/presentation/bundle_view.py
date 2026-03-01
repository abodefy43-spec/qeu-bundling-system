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
    person_recommendations: list[dict[str, object]] | None = None
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
        person_recommendations=_build_person_recommendations(base_dir, final_df, max_people=10),
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

    data["product_a_picture"] = _series_from_columns(data, ["product_a_picture"], "")
    data["product_b_picture"] = _series_from_columns(data, ["product_b_picture"], "")
    data["product_c_picture"] = _series_from_columns(data, ["product_c_picture"], "")

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
        "product_a_picture", "product_b_picture", "product_c_picture",
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
    return [_row_to_record(row) for _, row in out.iterrows()]


def _row_to_record(row: pd.Series) -> dict[str, object]:
    rec = row.to_dict()
    rec["bundle_items"] = _items_in_order(rec)
    return rec


def _build_person_recommendations(
    base_dir: Path, bundles_df: pd.DataFrame, max_people: int = 10
) -> list[dict[str, object]]:
    """Build sample shopper recommendations from order history.

    Data has no customer_id, so each "Person N" is represented by one order history.
    Recommendations use a hybrid score:
    - 35% purchase-history overlap
    - 50% bundle quality score (final_score)
    - 15% copurchase strength (purchase_score)
    """
    if bundles_df.empty or max_people <= 0:
        return []

    orders_path = base_dir / "data" / "filtered_orders.pkl"
    if not orders_path.exists():
        return []
    try:
        orders = pd.read_pickle(orders_path)
    except Exception:
        return []

    required = {"order_id", "product_id", "product_name"}
    if orders.empty or not required.issubset(set(orders.columns)):
        return []

    orders = orders.copy()
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    orders = orders.dropna(subset=["product_id"])
    if orders.empty:
        return []
    orders["product_id"] = orders["product_id"].astype(int)

    order_product_map = orders.groupby("order_id")["product_id"].apply(set).to_dict()
    order_name_map = orders.groupby("order_id").apply(
        lambda x: dict(zip(x["product_id"], x["product_name"]))
    ).to_dict()

    order_sizes = (
        orders.groupby("order_id")["product_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    candidate_order_ids = order_sizes[order_sizes >= 2].head(max_people * 4).index.tolist()
    if not candidate_order_ids:
        return []
    person_profiles: list[tuple[set[int], set[int], dict[int, str]]] = []
    for oid in candidate_order_ids:
        pids = order_product_map.get(oid, set())
        if len(pids) < 2:
            continue
        names = order_name_map.get(oid, {})
        person_profiles.append(({int(oid)}, set(pids), dict(names)))
        if len(person_profiles) >= max_people:
            break

    if not person_profiles:
        return []

    scored_base = bundles_df.copy()
    scored_base["product_a"] = pd.to_numeric(scored_base.get("product_a", -1), errors="coerce").fillna(-1).astype(int)
    scored_base["product_b"] = pd.to_numeric(scored_base.get("product_b", -1), errors="coerce").fillna(-1).astype(int)
    scored_base["product_c"] = pd.to_numeric(scored_base.get("product_c", -1), errors="coerce").fillna(-1).astype(int)
    scored_base["final_score"] = pd.to_numeric(scored_base.get("final_score", 0.0), errors="coerce").fillna(0.0)
    scored_base["purchase_score"] = pd.to_numeric(scored_base.get("purchase_score", 0.0), errors="coerce").fillna(0.0)

    final_min, final_max = float(scored_base["final_score"].min()), float(scored_base["final_score"].max())
    purchase_min, purchase_max = float(scored_base["purchase_score"].min()), float(scored_base["purchase_score"].max())
    final_span = max(final_max - final_min, 1e-9)
    purchase_span = max(purchase_max - purchase_min, 1e-9)
    scored_base["quality_norm"] = (scored_base["final_score"] - final_min) / final_span
    scored_base["purchase_norm"] = (scored_base["purchase_score"] - purchase_min) / purchase_span

    recommendations: list[dict[str, object]] = []
    used_bundle_keys: set[tuple[int, int, int]] = set()

    for idx, (order_ids, history_ids, name_map) in enumerate(person_profiles, start=1):
        if not history_ids:
            continue
        history_names = list(name_map.values())

        scored = scored_base.copy()
        scored["history_match_count"] = (
            scored["product_a"].isin(history_ids).astype(int)
            + scored["product_b"].isin(history_ids).astype(int)
            + scored["product_c"].isin(history_ids).astype(int)
        )
        scored["history_match_ratio"] = scored["history_match_count"] / 2.0
        scored["hybrid_reco_score"] = (
            scored["history_match_ratio"] * 0.35
            + scored["quality_norm"] * 0.50
            + scored["purchase_norm"] * 0.15
        )
        scored = scored.sort_values(
            ["hybrid_reco_score", "history_match_count", "final_score"],
            ascending=[False, False, False],
        )

        best_row = None
        for _, candidate in scored.iterrows():
            key = (
                int(candidate.get("product_a", -1)),
                int(candidate.get("product_b", -1)),
                int(candidate.get("product_c", -1)),
            )
            if key in used_bundle_keys:
                continue
            best_row = candidate
            used_bundle_keys.add(key)
            break

        if best_row is None:
            continue

        rec = _row_to_record(best_row)
        rec["person_label"] = f"Person {len(recommendations) + 1}"
        rec["source_order_ids"] = sorted(order_ids)
        rec["order_count"] = len(order_ids)
        rec["history_items"] = history_names
        rec["chosen_bundle_names"] = [
            str(rec.get("product_a_name", "") or "").strip(),
            str(rec.get("product_b_name", "") or "").strip(),
        ] + (
            [str(rec.get("product_c_name", "") or "").strip()]
            if bool(rec.get("is_triple")) and str(rec.get("product_c_name", "") or "").strip()
            else []
        )
        rec["history_match_count"] = int(best_row.get("history_match_count", 0))
        rec["hybrid_reco_score"] = round(float(best_row.get("hybrid_reco_score", 0.0)), 3)
        rec["recommendation_text"] = (
            "This recommendation blends purchase history with overall bundle quality "
            "and copurchase behavior (not history-only)."
        )
        recommendations.append(rec)
        if len(recommendations) >= max_people:
            break

    return recommendations


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
    pic_keys = {
        "product_a": "product_a_picture",
        "product_b": "product_b_picture",
        "product_c": "product_c_picture",
    }
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
        image_url = str(rec.get(pic_keys.get(key, ""), "") or "").strip()
        if image_url.lower() == "nan":
            image_url = ""
        try:
            disc_str = f"{float(disc or 0):,.1f}%" if disc is not None else "0%"
        except (TypeError, ValueError):
            disc_str = "0%"
        item = {"name": name, "price_sar": price_sar, "price_after_sar": after_sar, "discount": disc_str, "is_free": key == free, "image_url": image_url}
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
