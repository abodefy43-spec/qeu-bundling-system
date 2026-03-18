"""Phase 8: score internal person candidate pairs and write people-only output."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.pricing import (
    FIXED_MARGIN_DISCOUNT_PCT,
    load_odoo_price_lookup,
    price_paid_and_free_items_fixed_margin,
    resolve_sale_and_purchase,
)
from qeu_bundling.core.product_name_translation import translate_arabic_to_english

PERSON_CANDIDATES_INPUT = "person_candidate_pairs.csv"
PERSON_CANDIDATES_OUTPUT = "person_candidates_scored.csv"
RECIPE_ONLY_MIN_SCORE_NORM = 0.25
WEAK_FREE_BLOCKS_ARTIFACT = "weak_evidence_free_blocks.csv"
REFERENCE_PRODUCT_PICTURES = "product_pictures.csv"

NUMERIC_FEATURES = [
    "product_a_price",
    "product_b_price",
    "recipe_score_a",
    "recipe_score_b",
    "purchase_score",
    "embedding_score",
    "shared_categories_count",
    "shared_category_score",
    "category_match",
    "is_campaign_pair",
]
CATEGORICAL_FEATURES = [
    "category_a",
    "category_b",
    "importance_a",
    "importance_b",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _output_dir() -> Path:
    out = get_paths().output_dir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _diagnostics_dir() -> Path:
    path = get_paths().data_processed_diagnostics_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_models(output_dir: Path):
    with (output_dir / "free_item_model.pkl").open("rb") as fh:
        clf = pickle.load(fh)
    with (output_dir / "preprocessor.pkl").open("rb") as fh:
        preprocessor = pickle.load(fh)
    return clf, preprocessor


def _sample_ranked_pool(ranked: pd.DataFrame, max_rows: int = 120, run_seed: int | None = 42) -> pd.DataFrame:
    if ranked.empty:
        return ranked
    pool = ranked.head(min(max_rows, len(ranked)))
    if run_seed is None:
        return pool.sample(frac=1)
    return pool.sample(frac=1, random_state=int(run_seed))


def _ensure_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        source = out[col] if col in out.columns else pd.Series(["other"] * len(out), index=out.index, dtype="object")
        out[col] = source.fillna("other").astype(str)
    for col in NUMERIC_FEATURES:
        source = out[col] if col in out.columns else pd.Series([0.0] * len(out), index=out.index, dtype="float64")
        out[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    return out


def _join_pictures(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    out = df.copy()
    pic_path = base / "product_pictures.csv"
    ref_path = get_paths().data_reference_dir / REFERENCE_PRODUCT_PICTURES
    out["product_a_picture"] = ""
    out["product_b_picture"] = ""
    lookup: dict[int, str] = {}
    for path, prefer_override in ((pic_path, False), (ref_path, True)):
        if not path.exists():
            continue
        pics = pd.read_csv(path)
        if pics.empty or "product_id" not in pics.columns or "picture_url" not in pics.columns:
            continue
        product_ids = pd.to_numeric(pics["product_id"], errors="coerce").fillna(-1).astype(int)
        picture_urls = pics["picture_url"].fillna("").astype(str)
        mapped = {
            int(pid): str(url).strip()
            for pid, url in zip(product_ids, picture_urls)
            if int(pid) > 0 and str(url).strip()
        }
        if prefer_override:
            lookup.update(mapped)
        else:
            for product_id, picture_url in mapped.items():
                lookup.setdefault(product_id, picture_url)
    if not lookup:
        return out
    product_a_ids = pd.to_numeric(out["product_a"] if "product_a" in out.columns else pd.Series([-1] * len(out), index=out.index), errors="coerce").fillna(-1).astype(int)
    product_b_ids = pd.to_numeric(out["product_b"] if "product_b" in out.columns else pd.Series([-1] * len(out), index=out.index), errors="coerce").fillna(-1).astype(int)
    out["product_a_picture"] = product_a_ids.map(lookup).fillna("")
    out["product_b_picture"] = product_b_ids.map(lookup).fillna("")
    return out


def _translate_names(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    out = df.copy()
    cache_path = base / "arabic_translations_cache.json"
    if "product_a_name" in out.columns:
        out["product_a_name"] = out["product_a_name"].fillna("").astype(str).apply(lambda x: translate_arabic_to_english(x, cache_path))
    if "product_b_name" in out.columns:
        out["product_b_name"] = out["product_b_name"].fillna("").astype(str).apply(lambda x: translate_arabic_to_english(x, cache_path))
    return out


def _remove_legacy_outputs(out_dir: Path) -> None:
    legacy_names = [
        "final_bundles.csv",
        "top_10_bundles.csv",
        "top_10_bundles.txt",
        "results_final.csv",
        "results_100_final.csv",
    ]
    for name in legacy_names:
        path = out_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
    for path in out_dir.glob("top_10_bundles_*.csv"):
        try:
            path.unlink()
        except OSError:
            pass


def _is_weak_evidence_row(row: pd.Series) -> bool:
    cp_score = float(pd.to_numeric(row.get("purchase_score", 0.0), errors="coerce"))
    recipe_norm = float(pd.to_numeric(row.get("recipe_score_norm", 0.0), errors="coerce"))
    known_prior = int(float(pd.to_numeric(row.get("known_prior_flag", 0), errors="coerce")))
    return cp_score == 0.0 and recipe_norm < RECIPE_ONLY_MIN_SCORE_NORM and known_prior == 0


def _apply_margin_safe_pricing(
    bundles: pd.DataFrame,
    price_lookup: dict[int, object],
) -> pd.DataFrame:
    out = bundles.copy()

    def _row_pricing(row: pd.Series) -> pd.Series:
        pid_a_raw = pd.to_numeric(row.get("product_a", -1), errors="coerce")
        pid_b_raw = pd.to_numeric(row.get("product_b", -1), errors="coerce")
        pid_a = int(pid_a_raw) if pd.notna(pid_a_raw) else -1
        pid_b = int(pid_b_raw) if pd.notna(pid_b_raw) else -1

        fallback_sale_a_raw = pd.to_numeric(row.get("product_a_price", 0.0), errors="coerce")
        fallback_sale_b_raw = pd.to_numeric(row.get("product_b_price", 0.0), errors="coerce")
        fallback_sale_a = float(fallback_sale_a_raw) if pd.notna(fallback_sale_a_raw) else 0.0
        fallback_sale_b = float(fallback_sale_b_raw) if pd.notna(fallback_sale_b_raw) else 0.0

        sale_a, purchase_a, missing_purchase_a = resolve_sale_and_purchase(pid_a, fallback_sale_a, price_lookup)
        sale_b, purchase_b, missing_purchase_b = resolve_sale_and_purchase(pid_b, fallback_sale_b, price_lookup)

        priced = price_paid_and_free_items_fixed_margin(
            sale_price_a=sale_a,
            purchase_price_a=purchase_a,
            sale_price_b=sale_b,
            purchase_price_b=purchase_b,
        )
        paid_product = "product_a" if str(priced["paid_side"]) == "a" else "product_b"
        paid_final = float(
            priced["price_after_discount_a"] if paid_product == "product_a" else priced["price_after_discount_b"]
        )
        return pd.Series(
            {
                "product_a_price": round(float(sale_a), 2),
                "product_b_price": round(float(sale_b), 2),
                "purchase_price_a": round(float(purchase_a), 2),
                "purchase_price_b": round(float(purchase_b), 2),
                "purchase_price_missing_a": int(bool(missing_purchase_a)),
                "purchase_price_missing_b": int(bool(missing_purchase_b)),
                "free_product": str(priced["free_product"]),
                "paid_product": paid_product,
                "price_after_discount_a": round(float(priced["price_after_discount_a"]), 2),
                "price_after_discount_b": round(float(priced["price_after_discount_b"]), 2),
                "unit_profit_a": round(float(priced["unit_profit_a"]), 4),
                "unit_profit_b": round(float(priced["unit_profit_b"]), 4),
                "paid_item_final_price": round(float(paid_final), 2),
                "margin_discount_pct": float(FIXED_MARGIN_DISCOUNT_PCT),
            }
        )

    priced_df = out.apply(_row_pricing, axis=1)
    for col in priced_df.columns:
        out[col] = priced_df[col]
    return out


def predict_bundles(data_dir: Path | None = None, run_seed: int | None = 42) -> pd.DataFrame:
    """Score candidate pairs for people-only recommendation serving."""
    base = data_dir or _data_dir()
    paths = get_paths()
    out_dir = _output_dir()

    source_path = paths.data_processed_candidates_dir / PERSON_CANDIDATES_INPUT
    if not source_path.exists():
        source_path = base / PERSON_CANDIDATES_INPUT
    if not source_path.exists():
        legacy = base / "top_bundles.csv"
        if legacy.exists():
            source_path = legacy
        else:
            raise FileNotFoundError(
                f"Missing {PERSON_CANDIDATES_INPUT} in {base}. Run phase 6 first."
            )

    bundles = pd.read_csv(source_path)
    if bundles.empty:
        result = bundles.copy()
        result.to_csv(out_dir / PERSON_CANDIDATES_OUTPUT, index=False, encoding="utf-8-sig")
        return result

    bundles = _ensure_feature_columns(bundles)

    clf, preprocessor = _load_models(out_dir)
    X = preprocessor.transform(bundles[ALL_FEATURES])

    pred_free = clf.predict(X)

    odoo_price_lookup, odoo_meta = load_odoo_price_lookup(paths.project_root)
    if odoo_meta:
        workbook = str(odoo_meta.get("workbook_path", ""))
        id_col = str(odoo_meta.get("id_column", ""))
        sale_col = str(odoo_meta.get("sale_column", ""))
        purchase_col = str(odoo_meta.get("purchase_column", ""))
        if workbook:
            print(
                "  Odoo pricing lookup: "
                f"path={workbook} id_col={id_col or '-'} sale_col={sale_col or '-'} "
                f"purchase_col={purchase_col or '-'} rows={len(odoo_price_lookup):,}"
            )

    bundles = _apply_margin_safe_pricing(bundles, odoo_price_lookup)

    bundles["free_item_pred"] = pred_free
    if "recipe_score_norm" not in bundles.columns:
        bundles["recipe_score_norm"] = (
            pd.to_numeric(bundles.get("recipe_score", 0.0), errors="coerce").fillna(0.0) / 100.0
        )
    if "known_prior_flag" not in bundles.columns:
        bundles["known_prior_flag"] = 0
    bundles["weak_evidence_free_blocked"] = 0

    weak_mask = bundles.apply(
        lambda r: str(r.get("free_product", "")).strip().lower() == "product_b" and _is_weak_evidence_row(r),
        axis=1,
    )
    blocked_rows = bundles.loc[weak_mask].copy()
    diagnostics_path = _diagnostics_dir() / WEAK_FREE_BLOCKS_ARTIFACT
    if not blocked_rows.empty:
        blocked_rows["rule_id"] = "weak_evidence_free_block"
        blocked_rows["rule_type"] = "hard_block"
        blocked_rows["reason"] = "free_item_with_weak_evidence"
        blocked_rows["override_condition_met"] = 0
        blocked_rows.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["rule_id", "rule_type", "reason", "override_condition_met"]).to_csv(
            diagnostics_path, index=False, encoding="utf-8-sig"
        )
    bundles = bundles.loc[~weak_mask].copy()

    bundles = _join_pictures(bundles, base)
    bundles = _translate_names(bundles, base)

    if "final_score" not in bundles.columns:
        bundles["final_score"] = pd.to_numeric(bundles.get("new_final_score", 0.0), errors="coerce").fillna(0.0)

    sort_cols = [c for c in ["final_score", "new_final_score", "purchase_score"] if c in bundles.columns]
    bundles = bundles.sort_values(sort_cols, ascending=False).reset_index(drop=True)
    bundles.index.name = "rank"

    keep_cols = [
        "product_a",
        "product_b",
        "product_a_name",
        "product_b_name",
        "product_a_price",
        "product_b_price",
        "purchase_price_a",
        "purchase_price_b",
        "purchase_price_missing_a",
        "purchase_price_missing_b",
        "product_a_picture",
        "product_b_picture",
        "paid_product",
        "free_product",
        "price_after_discount_a",
        "price_after_discount_b",
        "paid_item_final_price",
        "unit_profit_a",
        "unit_profit_b",
        "purchase_score",
        "pair_count",
        "known_prior_flag",
        "gate_pass_reason",
        "recipe_score_norm",
        "recipe_overlap_tokens",
        "only_staples_overlap",
        "recipe_compat_score",
        "shared_categories_count",
        "shared_categories",
        "product_family_a",
        "product_family_b",
        "category_a",
        "category_b",
        "anchor_score",
        "complement_score",
        "final_score",
        "old_final_score",
        "new_final_score",
        "shadow_old_score",
        "shadow_new_score",
        "price_ratio_b_to_a",
        "pair_penalty_rule",
        "pair_penalty_multiplier",
        "tag_idf_overlap_score",
        "deal_signal",
        "deal_boost_applied",
        "utility_penalty_multiplier",
        "weak_evidence_free_blocked",
    ]
    keep = [c for c in keep_cols if c in bundles.columns]
    output = bundles[keep].copy()

    out_path = out_dir / PERSON_CANDIDATES_OUTPUT
    output.to_csv(out_path, index=False, encoding="utf-8-sig")
    _remove_legacy_outputs(out_dir)
    print(f"  Saved {len(output)} people candidate bundles -> {out_path}")
    return output


def predict_single(product_a_features: dict, product_b_features: dict) -> dict:
    """Predict bundle outcome for an arbitrary product pair."""
    out = _output_dir()
    clf, preprocessor = _load_models(out)

    row = {**product_a_features, **product_b_features}
    df = pd.DataFrame([row])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    X = preprocessor.transform(df[ALL_FEATURES])

    clf.predict(X)

    sale_a_raw = pd.to_numeric(row.get("product_a_price", 0.0), errors="coerce")
    sale_b_raw = pd.to_numeric(row.get("product_b_price", 0.0), errors="coerce")
    sale_a = float(sale_a_raw) if pd.notna(sale_a_raw) else 0.0
    sale_b = float(sale_b_raw) if pd.notna(sale_b_raw) else 0.0
    purchase_a_raw = pd.to_numeric(row.get("purchase_price_a", sale_a), errors="coerce")
    purchase_b_raw = pd.to_numeric(row.get("purchase_price_b", sale_b), errors="coerce")
    purchase_a = float(purchase_a_raw) if pd.notna(purchase_a_raw) and float(purchase_a_raw) > 0 else sale_a
    purchase_b = float(purchase_b_raw) if pd.notna(purchase_b_raw) and float(purchase_b_raw) > 0 else sale_b
    priced = price_paid_and_free_items_fixed_margin(
        sale_price_a=sale_a,
        purchase_price_a=purchase_a,
        sale_price_b=sale_b,
        purchase_price_b=purchase_b,
    )

    return {
        "free_product": str(priced["free_product"]),
        "paid_product": "product_a" if str(priced["paid_side"]) == "a" else "product_b",
        "discount_amount": round(float(FIXED_MARGIN_DISCOUNT_PCT), 2),
        "price_after_discount_a": float(priced["price_after_discount_a"]),
        "price_after_discount_b": float(priced["price_after_discount_b"]),
    }


def run(run_seed: int | None = 42):
    return predict_bundles(run_seed=run_seed)


if __name__ == "__main__":
    print("Phase 8: scoring people candidate bundles ...")
    result = predict_bundles()
    cols = [
        "product_a_name",
        "product_b_name",
        "free_product",
        "final_score",
    ]
    show = [c for c in cols if c in result.columns]
    print(f"\n  Top 15 rows:\n{result[show].head(15).to_string()}")
    print("Phase 8 complete.")
