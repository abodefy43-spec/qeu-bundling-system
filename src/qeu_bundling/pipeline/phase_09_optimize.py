"""Phase 9: Validation, optimization, and artifact summary."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.product_families import assign_family, load_families
from qeu_bundling.core.run_manifest import resolve_latest_artifact

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


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _output_dir() -> Path:
    return get_paths().output_dir


def _reference_dir() -> Path:
    return get_paths().data_reference_dir


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(value).lower()).strip()


def _normalise_family_value(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _core_tokens(name: str) -> list[str]:
    tokens = _normalise_name(name).split()
    core = [t for t in tokens if t not in SIZE_WORDS and not t.isdigit()]
    return core if core else tokens


def _is_non_food_product(name: str) -> bool:
    norm = _normalise_name(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


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


def validate_bundles() -> bool:
    """Validate people-only candidate output quality."""
    out = _output_dir()
    family_rules = load_families(_reference_dir() / "product_families.json")
    passed = True

    for csv_name in ["person_candidates_scored.csv"]:
        path = resolve_latest_artifact(
            "person_candidates_scored",
            fallback=out / csv_name,
        )
        if path is None:
            continue
        if not path.exists():
            print(f"  SKIP {csv_name}: file not found")
            continue
        df = pd.read_csv(path)
        name_a_col = "product_a_name" if "product_a_name" in df.columns else "product_a"
        name_b_col = "product_b_name" if "product_b_name" in df.columns else "product_b"

        same_count = 0
        for _, row in df.iterrows():
            if _is_same_product_variant(str(row[name_a_col]), str(row[name_b_col])):
                same_count += 1
                print(f"    SAME-PRODUCT: {row[name_a_col]}  +  {row[name_b_col]}")

        if same_count == 0:
            print(f"  PASSED {csv_name}: 0 same-product pairs")
        else:
            print(f"  FAILED {csv_name}: {same_count} same-product pairs found")
            passed = False

        if "shared_categories_count" in df.columns:
            below_min = int((pd.to_numeric(df["shared_categories_count"], errors="coerce").fillna(0.0) < 2).sum())
            if below_min == 0:
                print(f"  PASSED {csv_name}: all bundles have shared_categories >= 2")
            else:
                print(f"  WARNING {csv_name}: {below_min} bundles with shared_categories < 2")

        if "shared_categories" in df.columns:
            tag_sets = df["shared_categories"].fillna("").astype(str).apply(
                lambda x: {t for t in x.split("|") if t}
            )
            ramadan_count = int(tag_sets.apply(lambda tags: "ramadan" in tags).sum())
            saudi_dish_count = int(tag_sets.apply(lambda tags: len(tags & SAUDI_DISH_TAGS) > 0).sum())
            print(f"  INFO {csv_name}: bundles with ramadan tag = {ramadan_count}/{len(df)}")
            print(f"  INFO {csv_name}: bundles with Saudi dish tags = {saudi_dish_count}/{len(df)}")

        family_pairs: list[str] = []
        same_family_count = 0
        for _, row in df.iterrows():
            family_a = _normalise_family_value(row.get("product_family_a", ""))
            family_b = _normalise_family_value(row.get("product_family_b", ""))
            if not family_a:
                family_a = assign_family(str(row.get(name_a_col, "")), family_rules)
            if not family_b:
                family_b = assign_family(str(row.get(name_b_col, "")), family_rules)
            if family_a and family_b:
                pair = "|".join(sorted([family_a, family_b]))
                family_pairs.append(pair)
            if family_a and family_b and family_a == family_b:
                same_family_count += 1

        if same_family_count == 0:
            print(f"  PASSED {csv_name}: 0 same-family bundles")
        else:
            print(f"  WARNING {csv_name}: {same_family_count} same-family bundles")
            passed = False

        if family_pairs:
            vc = pd.Series(family_pairs).value_counts().head(6)
            print(f"  INFO {csv_name}: top family pair distribution")
            for pair_name, count in vc.items():
                print(f"    - {pair_name}: {int(count)}")

        non_food_count = int(
            df.apply(
                lambda row: _is_non_food_product(str(row[name_a_col])) or _is_non_food_product(str(row[name_b_col])),
                axis=1,
            ).sum()
        )
        if non_food_count == 0:
            print(f"  PASSED {csv_name}: no non-food bundles detected")
        else:
            print(f"  WARNING {csv_name}: {non_food_count} bundles include non-food products")
            passed = False

    return passed


def summarize_artifact_sizes(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    paths = get_paths()
    targets = [
        "filtered_orders.pkl",
        "product_embeddings.npy",
        "top_product_similarity.npz",
        "copurchase_scores.csv",
        "training_data.csv",
        str(paths.data_processed_candidates_dir / "person_candidate_pairs.csv"),
        str(paths.data_processed_diagnostics_dir / "suspicious_pairs_audit.csv"),
    ]
    rows = []
    for name in targets:
        p = Path(name)
        if not p.is_absolute():
            p = base / name
        if p.exists():
            rows.append({"file": str(name), "size_mb": round(p.stat().st_size / (1024 * 1024), 2)})
    df = pd.DataFrame(rows).sort_values("size_mb", ascending=False).reset_index(drop=True)
    out_path = base / "artifact_sizes.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved artifact size summary -> {out_path}")
    return df


def make_sparse_similarity(data_dir: Path | None = None, threshold: float = 0.5) -> Path:
    base = data_dir or _data_dir()
    dense_path = base / "top_product_similarity.npz"
    if not dense_path.exists():
        raise FileNotFoundError("top_product_similarity.npz not found. Run phase 2 first.")
    sim = sparse.load_npz(dense_path).toarray()
    sim[np.abs(sim) < threshold] = 0.0
    sparse_mat = sparse.csr_matrix(sim)
    out = base / "top_product_similarity_thresholded.npz"
    sparse.save_npz(out, sparse_mat)
    print(f"  Saved thresholded sparse similarity -> {out}")
    return out


def run():
    ok = validate_bundles()
    summarize_artifact_sizes()
    try:
        make_sparse_similarity(threshold=0.6)
    except FileNotFoundError as exc:
        print(f"  Skip sparse thresholding: {exc}")
    return ok


if __name__ == "__main__":
    print("Phase 9: Validation & optimization ...")
    print()
    print("  === FINAL VALIDATION ===")
    ok = validate_bundles()
    print()
    summarize_artifact_sizes()
    try:
        make_sparse_similarity(threshold=0.6)
    except FileNotFoundError as exc:
        print(f"  Skip sparse thresholding: {exc}")
    print()
    if ok:
        print("  ALL VALIDATIONS PASSED")
    else:
        print("  VALIDATION ISSUES FOUND - check output above")
    print("Phase 9 complete.")
