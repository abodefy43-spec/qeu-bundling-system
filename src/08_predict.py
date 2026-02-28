"""Phase 8: Bundle prediction — apply trained models and produce final output."""

from __future__ import annotations

import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.product_name_translation import translate_arabic_to_english
except ImportError:
    from product_name_translation import translate_arabic_to_english

DISCOUNT_CAP_PCT = 30.0
ENABLE_TRIPLE_BUNDLES = False  # Set True to allow 3-item bundles


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _output_dir() -> Path:
    d = Path(__file__).resolve().parents[1] / "output"
    d.mkdir(exist_ok=True)
    return d


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
]
CATEGORICAL_FEATURES = [
    "category_a",
    "category_b",
    "importance_a",
    "importance_b",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SIZE_WORDS = {
    "kg", "كيلو", "كجم", "جم", "g", "gram", "grams",
    "l", "liter", "litre", "ml", "pack", "pcs", "pc",
    "قطعة", "قطع", "حبات", "حبة", "x",
}
NON_FOOD_KEYWORDS = frozenset(
    {
        "shampoo",
        "conditioner",
        "hair mask",
        "hair balm",
        "hair cream",
        "hair serum",
        "lotion",
        "body wash",
        "face wash",
        "soap",
        "toothpaste",
        "deodorant",
        "cosmetic",
        "beauty",
        "makeup",
        "skincare",
    }
)
SAUDI_DISH_TAGS = frozenset(
    {
        "kabsa",
        "jareesh",
        "saleeg",
        "samboosa",
        "quaker_soup",
        "vimto",
        "qamar_al_din",
        "laban",
        "saudi_coffee",
        "chocolate_biscuit_dessert",
    }
)


def _load_models(output_dir: Path):
    free_item_path = output_dir / "free_item_model.pkl"
    discount_path = output_dir / "discount_model.pkl"

    with free_item_path.open("rb") as f:
        clf = pickle.load(f)
    with discount_path.open("rb") as f:
        reg = pickle.load(f)
    with (output_dir / "preprocessor.pkl").open("rb") as f:
        preprocessor = pickle.load(f)
    return clf, reg, preprocessor


def predict_bundles(data_dir: Path | None = None) -> pd.DataFrame:
    """Load top-100 bundles, apply ML models, write final_bundles.csv."""
    base = data_dir or _data_dir()
    out = _output_dir()
    bundles = pd.read_csv(base / "top_bundles.csv")
    clf, reg, preprocessor = _load_models(out)

    for col in CATEGORICAL_FEATURES:
        bundles[col] = bundles[col].fillna("other").astype(str)
    for col in NUMERIC_FEATURES:
        bundles[col] = pd.to_numeric(bundles[col], errors="coerce").fillna(0)

    X = bundles[ALL_FEATURES]
    X_transformed = preprocessor.transform(X)

    bundles["free_item_pred"] = clf.predict(X_transformed)
    bundles["free_item"] = bundles["free_item_pred"].map({
        0: "product_a",
        1: "product_b",
    })
    discount_pred = reg.predict(X_transformed)
    if discount_pred.ndim == 1:
        discount_pred = np.column_stack([discount_pred, discount_pred])
    bundles["discount_pred_a"] = np.clip(discount_pred[:, 0], 0, DISCOUNT_CAP_PCT).round(2)
    bundles["discount_pred_b"] = np.clip(discount_pred[:, 1], 0, DISCOUNT_CAP_PCT).round(2)
    shared_tags = bundles.get("shared_categories", pd.Series([""] * len(bundles))).fillna("").astype(str)
    bundles["has_ramadan"] = shared_tags.apply(
        lambda x: "ramadan" in {tag for tag in x.split("|") if tag}
    ).astype(int)
    bundles["has_saudi_dish"] = shared_tags.apply(
        lambda x: len({tag for tag in x.split("|") if tag} & SAUDI_DISH_TAGS) > 0
    ).astype(int)
    if "ramadan_boost" not in bundles.columns:
        bundles["ramadan_boost"] = 1.0
    if "has_ramadan_tag" not in bundles.columns:
        bundles["has_ramadan_tag"] = bundles["has_ramadan"]
    if "has_saudi_dish_tag" not in bundles.columns:
        bundles["has_saudi_dish_tag"] = bundles["has_saudi_dish"]
    if "product_family_a" not in bundles.columns:
        bundles["product_family_a"] = ""
    if "product_family_b" not in bundles.columns:
        bundles["product_family_b"] = ""
    if "frequency_score" not in bundles.columns:
        bundles["frequency_score"] = 0.0
    bundles["product_family_a"] = bundles["product_family_a"].apply(_normalise_family_value)
    bundles["product_family_b"] = bundles["product_family_b"].apply(_normalise_family_value)
    bundles["frequency_score"] = pd.to_numeric(bundles["frequency_score"], errors="coerce").fillna(0.0)

    # Translate Arabic product names to English
    cache_path = base / "arabic_translations_cache.json"
    bundles["product_a_name"] = bundles["product_a_name"].apply(
        lambda name: translate_arabic_to_english(str(name), cache_path)
    )
    bundles["product_b_name"] = bundles["product_b_name"].apply(
        lambda name: translate_arabic_to_english(str(name), cache_path)
    )
    print(f"  Translated product names to English")

    output = bundles[[
        "product_a", "product_b",
        "product_a_name", "product_b_name",
        "product_a_price", "product_b_price",
        "free_item", "discount_pred_a", "discount_pred_b",
        "has_ramadan", "has_saudi_dish", "ramadan_boost",
        "shared_categories_count", "shared_category_score",
        "recipe_score", "purchase_score", "embedding_score",
        "frequency_score", "final_score",
        "recipe_score_a", "recipe_score_b",
        "category_a", "category_b",
        "product_family_a", "product_family_b",
        "importance_a", "importance_b",
        "category_match", "shared_categories", "has_ramadan_tag", "has_saudi_dish_tag",
    ]].rename(columns={
        "free_item": "free_product",
    }).copy()

    output = _attach_third_products(output, base, cache_path)

    output["price_after_discount_a"] = output.apply(
        lambda r: 0.0 if r["free_product"] == "product_a"
        else round(r["product_a_price"] * (1 - r["discount_pred_a"] / 100), 2),
        axis=1,
    )
    output["price_after_discount_b"] = output.apply(
        lambda r: 0.0 if r["free_product"] == "product_b"
        else round(r["product_b_price"] * (1 - r["discount_pred_b"] / 100), 2),
        axis=1,
    )
    if "product_c_price" in output.columns:
        output["discount_pred_c"] = ((output["discount_pred_a"] + output["discount_pred_b"]) / 2).round(2)
        output["price_after_discount_c"] = output.apply(
            lambda r: 0.0 if r["free_product"] == "product_c" or not r.get("is_triple_bundle", False)
            else round(float(r["product_c_price"]) * (1 - r["discount_pred_c"] / 100), 2),
            axis=1,
        )

    output = output.sort_values(
        ["has_ramadan", "has_saudi_dish", "shared_categories_count", "final_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    output.index.name = "rank"

    out_path = out / "final_bundles.csv"
    output.to_csv(out_path, encoding="utf-8-sig")
    print(f"  Saved {len(output)} bundles -> {out_path}")

    top10 = _select_top_compatible(output)

    simple_cols = [
        "product_a_name", "product_b_name",
        "product_a_price", "product_b_price",
        "free_product", "discount_pred_a", "discount_pred_b",
        "price_after_discount_a", "price_after_discount_b",
    ]
    simple_rename = {
        "product_a_name": "product_a",
        "product_b_name": "product_b",
        "discount_pred_a": "discount_a",
        "discount_pred_b": "discount_b",
    }
    if "product_c_name" in top10.columns:
        simple_cols.extend(["product_c_name", "product_c_price", "price_after_discount_c", "is_triple_bundle", "discount_pred_c"])
        simple_rename["product_c_name"] = "product_c"
        simple_rename["discount_pred_c"] = "discount_c"

    top10_simple = top10[[c for c in simple_cols if c in top10.columns]].rename(columns=simple_rename).copy()
    top10_path = out / "top_10_bundles.csv"
    top10_simple.to_csv(top10_path, encoding="utf-8-sig")
    print(f"  Saved {len(top10_simple)} compatible bundles (simple) -> {top10_path}")

    _write_top10_report(top10, out / "top_10_bundles.txt")
    print(f"  Saved human-readable report -> {out / 'top_10_bundles.txt'}")

    _archive_results(out, top10_simple, output)

    return output


def _attach_third_products(output: pd.DataFrame, base, cache_path) -> pd.DataFrame:
    """Try to find a complementary 3rd product for each bundle."""
    if not ENABLE_TRIPLE_BUNDLES:
        output["product_c"] = None
        output["product_c_name"] = ""
        output["product_c_price"] = 0.0
        output["is_triple_bundle"] = False
        print(f"  Triple bundles: disabled (2-item only)")
        return output

    cat_df_path = base / "product_categories.csv"
    if not cat_df_path.exists():
        output["product_c"] = None
        output["product_c_name"] = ""
        output["product_c_price"] = 0.0
        output["is_triple_bundle"] = False
        return output

    cat_df = pd.read_csv(cat_df_path)
    tag_lookup = {}
    for _, row in cat_df.iterrows():
        pid = int(row["product_id"])
        raw = str(row.get("category_tags", "other"))
        tag_lookup[pid] = {t for t in raw.split("|") if t} or {"other"}

    family_lookup = {}
    if "product_family" in cat_df.columns:
        for _, row in cat_df.iterrows():
            fam = str(row.get("product_family", "")).strip()
            if fam and fam.lower() != "nan":
                family_lookup[int(row["product_id"])] = fam

    orders_path = base / "filtered_orders.pkl"
    price_lookup: dict[int, float] = {}
    name_lookup: dict[int, str] = {}
    if orders_path.exists():
        orders = pd.read_pickle(orders_path)
        price_lookup = orders.groupby("product_id")["unit_price"].median().to_dict()
        name_lookup = orders.drop_duplicates("product_id").set_index("product_id")["product_name"].to_dict()

    all_product_ids = set(tag_lookup.keys())

    product_c_list = []
    product_c_name_list = []
    product_c_price_list = []
    is_triple_list = []

    rng = np.random.default_rng(42)
    n_bundles = len(output)
    make_triple_mask = rng.choice([True, False], size=n_bundles, p=[0.5, 0.5])

    for idx, (_, row) in enumerate(output.iterrows()):
        pid_a = int(row["product_a"])
        pid_b = int(row["product_b"])
        tags_a = tag_lookup.get(pid_a, {"other"})
        tags_b = tag_lookup.get(pid_b, {"other"})
        fam_a = family_lookup.get(pid_a, "")
        fam_b = family_lookup.get(pid_b, "")
        name_a = str(row.get("product_a_name", ""))
        name_b = str(row.get("product_b_name", ""))

        best_c = None
        best_score = -1

        candidates = all_product_ids - {pid_a, pid_b}
        for pid_c in candidates:
            tags_c = tag_lookup.get(pid_c, {"other"})
            shared_ac = len(tags_a & tags_c)
            shared_bc = len(tags_b & tags_c)
            if shared_ac < 2 or shared_bc < 2:
                continue
            fam_c = family_lookup.get(pid_c, "")
            if fam_c and (fam_c == fam_a or fam_c == fam_b):
                continue
            name_c = str(name_lookup.get(pid_c, ""))
            if not name_c:
                continue
            if _is_same_product_variant(name_a, name_c) or _is_same_product_variant(name_b, name_c):
                continue
            score = (shared_ac + shared_bc) / 2.0
            if score > best_score:
                best_score = score
                best_c = pid_c

        should_add_third = make_triple_mask[idx] if idx < len(make_triple_mask) else False
        if best_c is not None and best_score >= 3 and should_add_third:
            c_name = str(name_lookup.get(best_c, ""))
            c_name = translate_arabic_to_english(c_name, cache_path)
            product_c_list.append(best_c)
            product_c_name_list.append(c_name)
            product_c_price_list.append(round(float(price_lookup.get(best_c, 0)), 2))
            is_triple_list.append(True)
        else:
            product_c_list.append(None)
            product_c_name_list.append("")
            product_c_price_list.append(0.0)
            is_triple_list.append(False)

    output["product_c"] = product_c_list
    output["product_c_name"] = product_c_name_list
    output["product_c_price"] = product_c_price_list
    output["is_triple_bundle"] = is_triple_list

    triple_count = sum(is_triple_list)
    print(f"  Triple bundles found: {triple_count}/{len(output)}")

    return output


def _select_top_compatible(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Pick top-N compatible bundles: 50% 2-item, 50% 3-item."""
    theme_tokens = _load_theme_tokens(_data_dir() / "theme_tokens.json")

    ranked = df.copy()
    if "has_ramadan" not in ranked.columns or "has_saudi_dish" not in ranked.columns:
        shared = ranked.get("shared_categories", pd.Series([""] * len(ranked))).fillna("").astype(str)
        ranked["has_ramadan"] = shared.apply(
            lambda x: "ramadan" in {tag for tag in x.split("|") if tag}
        ).astype(int)
        ranked["has_saudi_dish"] = shared.apply(
            lambda x: len({tag for tag in x.split("|") if tag} & SAUDI_DISH_TAGS) > 0
        ).astype(int)
    ranked = ranked.sort_values(
        ["has_ramadan", "has_saudi_dish", "shared_categories_count", "final_score"],
        ascending=[False, False, False, False],
    )
    is_triple = ranked.get("is_triple_bundle", pd.Series(False, index=ranked.index))
    is_triple = is_triple.fillna(False).astype(bool)
    pool_double = ranked[~is_triple]
    pool_triple = ranked[is_triple]
    n_half = n // 2

    def _pick_from_pool(pool: pd.DataFrame, count: int, used_products: set, used_themes: set) -> list:
        out = []
        pool_sub = pool.head(min(40, len(pool))).sample(frac=1, random_state=42)
        for idx, row in pool_sub.iterrows():
            if len(out) >= count:
                break
            pid_a = int(row["product_a"])
            pid_b = int(row["product_b"])
            name_a = str(row.get("product_a_name", ""))
            name_b = str(row.get("product_b_name", ""))
            family_a = _normalise_family_value(row.get("product_family_a", ""))
            family_b = _normalise_family_value(row.get("product_family_b", ""))
            if pid_a == pid_b or float(row.get("shared_categories_count", 0)) < 2:
                continue
            if _is_non_food_product(name_a) or _is_non_food_product(name_b):
                continue
            if _is_same_product_variant(name_a, name_b):
                continue
            if family_a and family_b and family_a == family_b:
                continue
            themes = _extract_themes(name_a, name_b, theme_tokens)
            if themes & used_themes or pid_a in used_products or pid_b in used_products:
                continue
            out.append(idx)
            used_products.update([pid_a, pid_b])
            used_themes.update(themes)
        return out

    used_products: set[int] = set()
    used_themes: set[str] = set()
    selected_double = _pick_from_pool(pool_double, n_half, used_products, used_themes)
    selected_triple = _pick_from_pool(pool_triple, n_half, used_products, used_themes)

    selected: list[int] = []
    max_pairs = min(len(selected_double), len(selected_triple))
    for i in range(max_pairs):
        # Force display order to start with 3-item, then 2-item.
        selected.append(selected_triple[i])
        selected.append(selected_double[i])

    if len(selected) < n:
        already = set(selected) | set(selected_double) | set(selected_triple)
        remaining = ranked.index.difference(list(already))
        for idx in remaining:
            if len(selected) >= n:
                break
            row = ranked.loc[idx]
            pid_a, pid_b = int(row["product_a"]), int(row["product_b"])
            name_a = str(row.get("product_a_name", ""))
            name_b = str(row.get("product_b_name", ""))
            family_a = _normalise_family_value(row.get("product_family_a", ""))
            family_b = _normalise_family_value(row.get("product_family_b", ""))
            if pid_a == pid_b or float(row.get("shared_categories_count", 0)) < 2:
                continue
            if _is_non_food_product(name_a) or _is_non_food_product(name_b):
                continue
            if _is_same_product_variant(name_a, name_b):
                continue
            if family_a and family_b and family_a == family_b:
                continue
            themes = _extract_themes(name_a, name_b, theme_tokens)
            if themes & used_themes or pid_a in used_products or pid_b in used_products:
                continue
            selected.append(idx)
            used_products.update([pid_a, pid_b])
            used_themes.update(themes)

    result = df.loc[selected].reset_index(drop=True)
    result.index = result.index + 1
    result.index.name = "rank"
    return result


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(value).lower()).strip()


def _load_theme_tokens(config_path: Path) -> set[str]:
    if not config_path.exists():
        return set()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    raw_tokens = payload.get("tokens", []) if isinstance(payload, dict) else []
    return {
        _normalise_name(str(token))
        for token in raw_tokens
        if _normalise_name(str(token))
    }


def _extract_themes(name_a: str, name_b: str, tokens: set[str]) -> set[str]:
    if not tokens:
        return set()
    joined = f" {_normalise_name(name_a)} {_normalise_name(name_b)} "
    return {token for token in tokens if f" {token} " in joined}


def _normalise_family_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _is_non_food_product(name: str) -> bool:
    norm = _normalise_name(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


def _core_tokens(name: str) -> list[str]:
    tokens = _normalise_name(name).split()
    core = [t for t in tokens if t not in SIZE_WORDS and not t.isdigit()]
    return core if core else tokens


def _is_same_product_variant(name_a: str, name_b: str) -> bool:
    core_a = _core_tokens(name_a)
    core_b = _core_tokens(name_b)
    if not core_a or not core_b:
        return False
    text_a = " ".join(core_a)
    text_b = " ".join(core_b)
    if text_a == text_b:
        return True
    set_a, set_b = set(core_a), set(core_b)
    overlap = len(set_a & set_b)
    if overlap == 0:
        return False
    ratio = overlap / max(len(set_a), len(set_b))
    if ratio >= 0.8 and overlap >= 2:
        return True
    if (text_a in text_b or text_b in text_a) and ratio >= 0.66 and overlap >= 2:
        return True
    return False


def _archive_results(
    out: Path, top10_simple: pd.DataFrame, full_output: pd.DataFrame
) -> None:
    """Append current run results to cumulative archive files."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    top10_archive = out / "results_final.csv"
    top10_copy = top10_simple.copy()
    top10_copy["run_timestamp"] = ts
    if top10_archive.exists():
        top10_copy.to_csv(top10_archive, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        top10_copy.to_csv(top10_archive, index=False, encoding="utf-8-sig")
    print(f"  Appended top-10 to archive -> {top10_archive}")

    full_archive = out / "results_100_final.csv"
    full_copy = full_output.copy()
    full_copy["run_timestamp"] = ts
    if full_archive.exists():
        full_copy.to_csv(full_archive, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        full_copy.to_csv(full_archive, index=False, encoding="utf-8-sig")
    print(f"  Appended full bundles to archive -> {full_archive}")


def _write_top10_report(top10: pd.DataFrame, path) -> None:
    """Write a plain-text report you can copy-paste."""
    lines = [
        "QEU TOP 10 COMPATIBLE BUNDLE OFFERS",
        "=" * 50,
        "",
    ]
    for _, row in top10.iterrows():
        is_triple = bool(row.get("is_triple_bundle", False))
        disc = row["discount_amount"]
        score = row["final_score"]
        shared = int(row.get("shared_categories_count", 0))
        shared_list = str(row.get("shared_categories", "")).replace("|", ", ")

        lines.append(f"Bundle #{int(row.name)}" + (" [3-ITEM]" if is_triple else ""))

        items = [
            ("A", row["product_a_name"], row["product_a_price"], row.get("price_after_discount_a", 0)),
            ("B", row["product_b_name"], row["product_b_price"], row.get("price_after_discount_b", 0)),
        ]
        if is_triple and row.get("product_c_name"):
            items.append(("C", row["product_c_name"], row.get("product_c_price", 0), row.get("price_after_discount_c", 0)))

        for label, name, price, after_price in items:
            tag = f"product_{label.lower()}"
            if row["free_product"] == tag:
                lines.append(f"  {label}. FREE: {name}  (SAR {price:.2f})")
            else:
                lines.append(f"  {label}. BUY:  {name}  (SAR {price:.2f} -> SAR {after_price:.2f})")

        lines.append(f"  Discount: {disc:.1f}%  |  Score: {score:.1f}")
        lines.append(f"  Shared categories: {shared}")
        if shared_list:
            lines.append(f"  Shared tags: {shared_list}")
        lines.append(f"  Primary categories: {row['category_a']} + {row['category_b']}")
        lines.append("")

    lines.append("=" * 50)
    lines.append("Generated by QEU Bundling System")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def predict_single(product_a_features: dict, product_b_features: dict) -> dict:
    """Predict bundle outcome for an arbitrary product pair.

    Each dict must contain keys matching ALL_FEATURES.
    """
    out = _output_dir()
    clf, reg, preprocessor = _load_models(out)

    row = {**product_a_features, **product_b_features}
    df = pd.DataFrame([row])
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)
    X = preprocessor.transform(df[ALL_FEATURES])

    free_item = int(clf.predict(X)[0])
    disc = reg.predict(X)
    if disc.ndim == 1:
        disc_a = disc_b = float(np.clip(disc[0], 0, DISCOUNT_CAP_PCT))
    else:
        disc_a = float(np.clip(disc[0, 0], 0, DISCOUNT_CAP_PCT))
        disc_b = float(np.clip(disc[0, 1], 0, DISCOUNT_CAP_PCT))

    return {
        "free_product": "product_a" if free_item == 0 else "product_b",
        "discount_amount": round((disc_a + disc_b) / 2, 2),
        "discount_a": round(disc_a, 2),
        "discount_b": round(disc_b, 2),
    }


if __name__ == "__main__":
    print("Phase 8: Generating final bundle predictions ...")
    result = predict_bundles()
    cols = ["product_a_name", "product_b_name", "free_product", "discount_amount", "final_score"]
    print(f"\n  Top 15 final bundles:\n{result[cols].head(15).to_string()}")
    print("Phase 8 complete.")
