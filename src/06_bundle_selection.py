"""Phase 6: Bundle selection — Prompt 3 shared-category scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

try:
    from product_families import load_families
except ImportError:
    from src.product_families import load_families
import pandas as pd
from scipy import sparse

SHARED_CATEGORY_WEIGHT = 0.22
RECIPE_WEIGHT = 0.22
COPURCHASE_WEIGHT = 0.22
EMBEDDING_WEIGHT = 0.05
VERSATILITY_WEIGHT = 0.04
FREQUENCY_WEIGHT = 0.10
CAMPAIGN_WEIGHT = 0.15

MIN_ORDERS_PER_PRODUCT = 50
TOP_PRODUCTS_FOR_PAIRING = 4000
TOP_N_BUNDLES = 100

MIN_SHARED_CATEGORIES = 2
MAX_EMBEDDING_SIMILARITY = 55.0
MAX_BUNDLES_PER_THEME = 3
MAX_APPEARANCES_PER_PRODUCT = 3
VEGETABLE_PENALTY_ONE = 0.85
VEGETABLE_PENALTY_BOTH = 0.60
RAMADAN_BOOST_BASE = 1.08
RAMADAN_BOOST_DISH = 1.15
NON_FOOD_TAG = "non_food"
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

SIZE_WORDS = {
    "kg", "كيلو", "كجم", "جم", "g", "gram", "grams",
    "l", "liter", "litre", "ml", "pack", "pcs", "pc",
    "قطعة", "قطع", "حبات", "حبة", "x",
}


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(value).lower()).strip()


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


def _parse_tags(value: str | float | int | None, fallback: str = "other") -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {fallback}
    text = str(value).strip()
    if not text:
        return {fallback}
    tags = {x for x in text.split("|") if x}
    return tags if tags else {fallback}


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


def _load_artefacts(base: Path):
    """Load all intermediate data produced by phases 1-5."""
    orders = pd.read_pickle(base / "filtered_orders.pkl")
    copurchase = pd.read_csv(base / "copurchase_scores.csv")
    recipe_scores = pd.read_csv(base / "product_recipe_scores.csv")
    categories = pd.read_csv(base / "product_categories.csv")
    embeddings = np.load(base / "product_embeddings.npy")

    with (base / "embedding_mapping.json").open("r") as f:
        emb_map = {int(k): v for k, v in json.load(f).items()}

    return orders, copurchase, recipe_scores, categories, embeddings, emb_map


def _top_products_by_frequency(
    orders: pd.DataFrame, min_orders: int, top_n: int
) -> pd.DataFrame:
    """Top-N products by order frequency after minimum-order filter."""
    product_order_counts = (
        orders.groupby("product_id")["order_id"]
        .nunique()
        .reset_index()
        .rename(columns={"order_id": "order_count"})
    )
    eligible = product_order_counts[product_order_counts["order_count"] >= min_orders]
    eligible = eligible.sort_values("order_count", ascending=False).head(top_n)
    return eligible


def _load_similarity_for_top_products(
    base: Path,
    product_ids: list[int],
    embeddings: np.ndarray,
    emb_map: dict[int, int],
) -> np.ndarray:
    """Load precomputed sparse matrix when available; otherwise compute via matmul."""
    sim_path = base / "top_product_similarity.npz"
    ids_path = base / "top_product_ids.json"
    if sim_path.exists() and ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as f:
            stored_ids = [int(x) for x in json.load(f)]
        if stored_ids[: len(product_ids)] == product_ids:
            sim_sparse = sparse.load_npz(sim_path)
            return sim_sparse[: len(product_ids), : len(product_ids)].toarray()

    emb_idx = [emb_map.get(pid, -1) for pid in product_ids]
    emb_mat = np.vstack(
        [
            embeddings[idx] if idx >= 0 else np.zeros(embeddings.shape[1], dtype=np.float32)
            for idx in emb_idx
        ]
    )
    sim = np.matmul(emb_mat, emb_mat.T)
    return np.clip(sim, -1.0, 1.0)


def select_bundles(
    data_dir: Path | None = None,
    top_n: int = TOP_N_BUNDLES,
    min_orders: int = MIN_ORDERS_PER_PRODUCT,
    top_products: int = TOP_PRODUCTS_FOR_PAIRING,
) -> pd.DataFrame:
    """Score candidate pairs and select the top bundles.

    Prompt 3 logic:
    1. Score compatibility via shared categories.
    2. Block same-product variants (different sizes/packaging).
    3. Enforce minimum shared categories.
    """
    base = data_dir or _data_dir()
    orders, copurchase, recipe_df, cat_df, embeddings, emb_map = _load_artefacts(base)

    eligible = _top_products_by_frequency(orders, min_orders, top_products)
    product_ids = eligible["product_id"].astype(int).tolist()
    eligible_set = set(product_ids)
    print(
        f"  Eligible products (>={min_orders} orders): {len(eligible_set):,} "
        f"(top {top_products})"
    )

    recipe_lookup = dict(zip(
        recipe_df["product_id"].astype(int),
        recipe_df["recipe_score"].astype(float),
    ))
    importance_lookup = dict(zip(
        recipe_df["product_id"].astype(int),
        recipe_df["saudi_importance"].astype(str),
    ))
    cat_lookup = dict(zip(
        cat_df["product_id"].astype(int),
        cat_df["category"].astype(str),
    ))
    _count_col = cat_df["category_count"] if "category_count" in cat_df.columns else pd.Series([0.0] * len(cat_df), index=cat_df.index)
    count_lookup = dict(zip(
        cat_df["product_id"].astype(int),
        pd.to_numeric(_count_col, errors="coerce").fillna(0).astype(float),
    ))
    _freq_col = cat_df["frequency_score"] if "frequency_score" in cat_df.columns else pd.Series([0.0] * len(cat_df), index=cat_df.index)
    frequency_lookup = dict(zip(
        cat_df["product_id"].astype(int),
        pd.to_numeric(_freq_col, errors="coerce").fillna(0).astype(float),
    ))
    family_lookup = dict(
        zip(
            cat_df["product_id"].astype(int),
            cat_df.get("product_family", pd.Series([""] * len(cat_df))).fillna("").astype(str),
        )
    )
    tag_lookup = {
        int(pid): _parse_tags(tags, fallback=str(cat_lookup.get(int(pid), "other")))
        for pid, tags in zip(
            cat_df["product_id"].astype(int),
            cat_df.get("category_tags", pd.Series(["other"] * len(cat_df))),
        )
    }
    non_food_lookup = {
        int(pid): (NON_FOOD_TAG in tags)
        for pid, tags in tag_lookup.items()
    }

    price_lookup = (
        orders.groupby("product_id")["unit_price"]
        .median()
        .to_dict()
    )
    name_lookup = (
        orders.drop_duplicates("product_id")
        .set_index("product_id")["product_name"]
        .to_dict()
    )

    copurchase["product_a"] = copurchase["product_a"].astype(int)
    copurchase["product_b"] = copurchase["product_b"].astype(int)
    copurchase = copurchase[
        copurchase["product_a"].isin(eligible_set)
        & copurchase["product_b"].isin(eligible_set)
    ].copy()
    purchase_lookup = {
        (int(a), int(b)): float(s)
        for a, b, s in zip(copurchase["product_a"], copurchase["product_b"], copurchase["score"])
    }

    # Generate all product pairs among top-N products (Prompt 2 fix).
    n = len(product_ids)
    i_idx, j_idx = np.triu_indices(n, k=1)
    pairs = pd.DataFrame(
        {
            "product_a": np.array(product_ids, dtype=np.int64)[i_idx],
            "product_b": np.array(product_ids, dtype=np.int64)[j_idx],
        }
    )
    print(f"  Candidate pairs from top products: {len(pairs):,}")

    pairs["recipe_score_a"] = pairs["product_a"].map(recipe_lookup).fillna(0.0)
    pairs["recipe_score_b"] = pairs["product_b"].map(recipe_lookup).fillna(0.0)
    pairs["recipe_score"] = (pairs["recipe_score_a"] + pairs["recipe_score_b"]) / 2.0
    pairs["category_a"] = pairs["product_a"].map(cat_lookup).fillna("other")
    pairs["category_b"] = pairs["product_b"].map(cat_lookup).fillna("other")
    pairs["category_count_a"] = pairs["product_a"].map(count_lookup).fillna(0.0)
    pairs["category_count_b"] = pairs["product_b"].map(count_lookup).fillna(0.0)
    pairs["frequency_score_a"] = pairs["product_a"].map(frequency_lookup).fillna(0.0)
    pairs["frequency_score_b"] = pairs["product_b"].map(frequency_lookup).fillna(0.0)
    pairs["frequency_score"] = (pairs["frequency_score_a"] + pairs["frequency_score_b"]) / 2.0
    pairs["product_family_a"] = pairs["product_a"].map(family_lookup).fillna("")
    pairs["product_family_b"] = pairs["product_b"].map(family_lookup).fillna("")
    pairs["cross_category"] = (pairs["category_a"] != pairs["category_b"]).astype(int)
    pairs["is_non_food_a"] = pairs["product_a"].map(non_food_lookup).fillna(False).astype(bool)
    pairs["is_non_food_b"] = pairs["product_b"].map(non_food_lookup).fillna(False).astype(bool)

    sim_matrix = _load_similarity_for_top_products(base, product_ids, embeddings, emb_map)
    pairs["embedding_score"] = np.clip(sim_matrix[i_idx, j_idx] * 100.0, 0.0, 100.0).astype(float)
    pairs["purchase_score"] = [
        purchase_lookup.get((int(a), int(b)), purchase_lookup.get((int(b), int(a)), 0.0))
        for a, b in zip(pairs["product_a"], pairs["product_b"])
    ]

    # Campaign bundle co-occurrence boost
    campaign_bundle_path = base / "campaign_bundle_pairs.csv"
    campaign_lookup: dict[tuple[int, int], float] = {}
    if campaign_bundle_path.exists():
        cb = pd.read_csv(campaign_bundle_path)
        if not cb.empty:
            max_occ = float(cb["occurrences"].max()) if cb["occurrences"].max() > 0 else 1.0
            for _, r in cb.iterrows():
                score = float(r["occurrences"]) / max_occ * 100.0
                a, b = int(r["product_a"]), int(r["product_b"])
                campaign_lookup[(a, b)] = score
                campaign_lookup[(b, a)] = score
            print(f"  Loaded {len(cb):,} campaign bundle pairs for scoring")
    pairs["campaign_score"] = [
        campaign_lookup.get((int(a), int(b)), 0.0)
        for a, b in zip(pairs["product_a"], pairs["product_b"])
    ]

    product_a_ids = pairs["product_a"].to_numpy(dtype=np.int64)
    product_b_ids = pairs["product_b"].to_numpy(dtype=np.int64)
    shared_counts = np.fromiter(
        (
            len(tag_lookup.get(int(a), {"other"}) & tag_lookup.get(int(b), {"other"}))
            for a, b in zip(product_a_ids, product_b_ids)
        ),
        dtype=np.int16,
        count=len(pairs),
    )
    pairs["shared_categories_count"] = shared_counts.astype(int)
    pairs["shared_category_score"] = np.clip(pairs["shared_categories_count"] * 20.0, 0.0, 100.0)
    has_ramadan = np.fromiter(
        (
            "ramadan" in (tag_lookup.get(int(a), {"other"}) & tag_lookup.get(int(b), {"other"}))
            for a, b in zip(product_a_ids, product_b_ids)
        ),
        dtype=bool,
        count=len(pairs),
    )
    has_saudi_dish = np.fromiter(
        (
            len((tag_lookup.get(int(a), {"other"}) & tag_lookup.get(int(b), {"other"})) & SAUDI_DISH_TAGS) > 0
            for a, b in zip(product_a_ids, product_b_ids)
        ),
        dtype=bool,
        count=len(pairs),
    )
    pairs["has_ramadan_tag"] = has_ramadan.astype(int)
    pairs["has_saudi_dish_tag"] = has_saudi_dish.astype(int)
    pairs["versatility_score"] = np.clip(
        ((pairs["category_count_a"] + pairs["category_count_b"]) / 2.0) * 5.0,
        0.0,
        100.0,
    )

    pairs["final_score"] = (
        pairs["shared_category_score"] * SHARED_CATEGORY_WEIGHT
        + pairs["recipe_score"] * RECIPE_WEIGHT
        + pairs["purchase_score"] * COPURCHASE_WEIGHT
        + pairs["embedding_score"] * EMBEDDING_WEIGHT
        + pairs["versatility_score"] * VERSATILITY_WEIGHT
        + pairs["frequency_score"] * FREQUENCY_WEIGHT
        + pairs["campaign_score"] * CAMPAIGN_WEIGHT
    )
    pairs["ramadan_boost"] = np.where(
        has_ramadan & has_saudi_dish,
        RAMADAN_BOOST_DISH,
        np.where(has_ramadan, RAMADAN_BOOST_BASE, 1.0),
    )
    pairs["final_score"] = pairs["final_score"] * pairs["ramadan_boost"]

    veg_a = pairs["category_a"].eq("vegetables")
    veg_b = pairs["category_b"].eq("vegetables")
    pairs["vegetable_penalty"] = np.where(
        veg_a & veg_b,
        VEGETABLE_PENALTY_BOTH,
        np.where(veg_a | veg_b, VEGETABLE_PENALTY_ONE, 1.0),
    )
    pairs["final_score"] = (pairs["final_score"] * pairs["vegetable_penalty"]).round(2)

    filtered = pairs[pairs["shared_categories_count"] >= MIN_SHARED_CATEGORIES].copy()
    print(f"  Pairs passing shared-category minimum ({MIN_SHARED_CATEGORIES}+): {len(filtered):,}")
    filtered = filtered[filtered["embedding_score"] <= MAX_EMBEDDING_SIMILARITY].copy()
    print(f"  Pairs after embedding diversity filter (<={MAX_EMBEDDING_SIMILARITY:.0f}): {len(filtered):,}")
    before_non_food = len(filtered)
    filtered = filtered[(~filtered["is_non_food_a"]) & (~filtered["is_non_food_b"])].copy()
    print(f"  Blocked non-food product pairs: {before_non_food - len(filtered):,}")

    same_variant_mask = np.fromiter(
        (
            _is_same_product_variant(
                str(name_lookup.get(int(a), "")),
                str(name_lookup.get(int(b), "")),
            )
            for a, b in zip(filtered["product_a"], filtered["product_b"])
        ),
        dtype=bool,
        count=len(filtered),
    )
    blocked = int(same_variant_mask.sum())
    filtered = filtered.loc[~same_variant_mask].copy()
    print(f"  Blocked same-product variants: {blocked:,}")
    before_family = len(filtered)
    same_family_mask = (
        filtered["product_family_a"].astype(str).ne("")
        & filtered["product_family_b"].astype(str).ne("")
        & filtered["product_family_a"].astype(str).eq(filtered["product_family_b"].astype(str))
    )
    family_ids = set()
    try:
        families = load_families(base / "product_families.json")
        family_ids = {str(f.get("id", "")).strip() for f in families if f.get("id")}
    except (OSError, TypeError):
        pass
    if family_ids:
        same_family_via_tags = np.fromiter(
            (
                len(tag_lookup.get(int(a), set()) & tag_lookup.get(int(b), set()) & family_ids) > 0
                for a, b in zip(filtered["product_a"], filtered["product_b"])
            ),
            dtype=bool,
            count=len(filtered),
        )
        same_family_mask = same_family_mask | same_family_via_tags
    filtered = filtered.loc[~same_family_mask].copy()
    print(f"  Blocked same-family pairs: {before_family - len(filtered):,}")

    sorted_candidates = filtered.sort_values(
        ["cross_category", "versatility_score", "shared_categories_count", "final_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    pool_size = min(250, len(sorted_candidates))
    pool = sorted_candidates.head(pool_size)
    pool = pool.sample(frac=1, random_state=42)

    theme_tokens = _load_theme_tokens(base / "theme_tokens.json")
    theme_counts: dict[str, int] = {}
    product_counts: dict[int, int] = {}
    selected_idx: list[int] = []
    for idx, row in pool.iterrows():
        pid_a = int(row["product_a"])
        pid_b = int(row["product_b"])
        if product_counts.get(pid_a, 0) >= MAX_APPEARANCES_PER_PRODUCT:
            continue
        if product_counts.get(pid_b, 0) >= MAX_APPEARANCES_PER_PRODUCT:
            continue
        name_a = str(name_lookup.get(pid_a, ""))
        name_b = str(name_lookup.get(pid_b, ""))
        themes = _extract_themes(name_a, name_b, theme_tokens)
        if any(theme_counts.get(t, 0) >= MAX_BUNDLES_PER_THEME for t in themes):
            continue
        selected_idx.append(idx)
        product_counts[pid_a] = product_counts.get(pid_a, 0) + 1
        product_counts[pid_b] = product_counts.get(pid_b, 0) + 1
        for t in themes:
            theme_counts[t] = theme_counts.get(t, 0) + 1
        if len(selected_idx) >= top_n:
            break

    if len(selected_idx) < top_n:
        remaining = sorted_candidates.index.difference(selected_idx)
        for r_idx in remaining:
            if len(selected_idx) >= top_n:
                break
            r_row = sorted_candidates.loc[r_idx]
            r_a, r_b = int(r_row["product_a"]), int(r_row["product_b"])
            if product_counts.get(r_a, 0) >= MAX_APPEARANCES_PER_PRODUCT:
                continue
            if product_counts.get(r_b, 0) >= MAX_APPEARANCES_PER_PRODUCT:
                continue
            selected_idx.append(r_idx)
            product_counts[r_a] = product_counts.get(r_a, 0) + 1
            product_counts[r_b] = product_counts.get(r_b, 0) + 1

    top = sorted_candidates.loc[selected_idx].reset_index(drop=True)

    top["product_a_name"] = top["product_a"].map(name_lookup)
    top["product_b_name"] = top["product_b"].map(name_lookup)
    top["product_a_price"] = top["product_a"].map(price_lookup).round(2)
    top["product_b_price"] = top["product_b"].map(price_lookup).round(2)
    top["category_a"] = top["category_a"].fillna("other")
    top["category_b"] = top["category_b"].fillna("other")
    top["importance_a"] = top["product_a"].map(importance_lookup).fillna("low")
    top["importance_b"] = top["product_b"].map(importance_lookup).fillna("low")
    top["category_match"] = (top["category_a"] == top["category_b"]).astype(int)
    top["shared_categories"] = [
        "|".join(sorted(tag_lookup.get(int(a), {"other"}) & tag_lookup.get(int(b), {"other"})))
        for a, b in zip(top["product_a"], top["product_b"])
    ]

    out_path = base / "top_bundles.csv"
    top.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Selected top {len(top)} bundles -> {out_path}")
    return top


def load_top_bundles(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    return pd.read_csv(base / "top_bundles.csv")


if __name__ == "__main__":
    print("Phase 6: Selecting top bundles ...")
    bundles = select_bundles()
    cols = [
        "product_a_name", "product_b_name",
        "shared_categories_count", "shared_category_score",
        "recipe_score", "purchase_score", "embedding_score", "final_score",
    ]
    print(f"\n  Top 15 bundles:\n{bundles[cols].head(15).to_string()}")
    print("Phase 6 complete.")
