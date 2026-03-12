"""Offline quality evaluation for people-only bundle recommendations."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.feedback_memory import build_pair_multiplier_lookup, pair_feedback_multiplier
from qeu_bundling.core.run_manifest import resolve_latest_artifact

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
UTILITY_KEYWORDS = frozenset(
    {
        "salt",
        "sugar",
        "water",
        "oil",
        "pepper",
        "cumin",
        "spice",
        "powder",
        "seasoning",
        "mint",
        "herb",
    }
)


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(value).lower()).strip()


def _is_non_food_name(name: str) -> bool:
    norm = _normalise(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


def _is_utility_name(name: str) -> bool:
    norm = _normalise(name)
    return any(token in norm for token in UTILITY_KEYWORDS)


def _safe_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0.0] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def evaluate_quality(base_dir: Path | None = None, save: bool = True) -> dict[str, object]:
    paths = get_paths(project_root=base_dir)
    candidates_path = resolve_latest_artifact(
        "person_candidates_scored",
        base_dir=base_dir,
        fallback=paths.output_dir / "person_candidates_scored.csv",
    )
    person_path = resolve_latest_artifact(
        "person_reco_quality",
        base_dir=base_dir,
        fallback=paths.output_dir / "person_reco_quality.json",
    )

    if candidates_path is None or not candidates_path.exists():
        raise FileNotFoundError(f"Missing {candidates_path}. Run pipeline first.")

    candidates_df = pd.read_csv(candidates_path)
    top10 = candidates_df.head(10).copy()
    top20 = candidates_df.head(20).copy()

    name_a_col = "product_a_name" if "product_a_name" in candidates_df.columns else "product_a"
    name_b_col = "product_b_name" if "product_b_name" in candidates_df.columns else "product_b"

    family_a = candidates_df.get("product_family_a", pd.Series([""] * len(candidates_df))).fillna("").astype(str).str.strip().str.lower()
    family_b = candidates_df.get("product_family_b", pd.Series([""] * len(candidates_df))).fillna("").astype(str).str.strip().str.lower()
    same_family_count = int(((family_a != "") & (family_b != "") & (family_a == family_b)).sum())

    non_food_count = int(
        candidates_df.apply(
            lambda r: _is_non_food_name(str(r.get(name_a_col, ""))) or _is_non_food_name(str(r.get(name_b_col, ""))),
            axis=1,
        ).sum()
    )

    top10_ids: set[int] = set()
    for col in ("product_a", "product_b"):
        if col in top10.columns:
            top10_ids.update(pd.to_numeric(top10[col], errors="coerce").dropna().astype(int).tolist())
    diversity_score_top10 = float(len(top10_ids) / max(1, 2 * len(top10)))

    utility_pair_count_top10 = int(
        top10.apply(
            lambda r: _is_utility_name(str(r.get(name_a_col, ""))) or _is_utility_name(str(r.get(name_b_col, ""))),
            axis=1,
        ).sum()
    )

    purchase = _safe_float_series(top20, "purchase_score")
    recipe_compat = _safe_float_series(top20, "recipe_compat_score")
    shared_count = _safe_float_series(top20, "shared_categories_count")
    bad_pair_mask = (purchase < 15.0) & (recipe_compat < 12.0) & (shared_count < 2.0)
    bad_pair_rate_top20 = float(bad_pair_mask.mean()) if len(top20) else 0.0

    recipe_anchor_alignment = float(_safe_float_series(top10, "anchor_score").mean())

    feedback_lookup = build_pair_multiplier_lookup(base_dir)
    feedback_conflict_count_top20 = 0
    for _, row in top20.iterrows():
        mult, is_conflict = pair_feedback_multiplier(
            str(row.get(name_a_col, "")),
            str(row.get(name_b_col, "")),
            feedback_lookup,
        )
        if is_conflict and mult < 1.0:
            feedback_conflict_count_top20 += 1

    anchor_in_history_rate = None
    if person_path is not None and person_path.exists():
        try:
            person_payload = json.loads(person_path.read_text(encoding="utf-8"))
            anchor_in_history_rate = float(person_payload.get("anchor_in_history_rate", 0.0))
        except Exception:
            anchor_in_history_rate = None

    critical_gates = {
        "same_family_zero": same_family_count == 0,
        "non_food_zero": non_food_count == 0,
        "person_anchor_history_ok": True if anchor_in_history_rate is None else anchor_in_history_rate >= 1.0,
    }
    critical_gates_passed = all(bool(v) for v in critical_gates.values())

    payload: dict[str, object] = {
        "same_family_count": same_family_count,
        "non_food_count": non_food_count,
        "utility_pair_count_top10": utility_pair_count_top10,
        "diversity_score_top10": round(diversity_score_top10, 4),
        "bad_pair_rate_top20": round(bad_pair_rate_top20, 4),
        "recipe_anchor_alignment": round(recipe_anchor_alignment, 4),
        "feedback_conflict_count_top20": int(feedback_conflict_count_top20),
        "anchor_in_history_rate": anchor_in_history_rate,
        "critical_gates": critical_gates,
        "critical_gates_passed": bool(critical_gates_passed),
    }

    if save:
        out_path = paths.output_dir / "bundle_quality_metrics.json"
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    payload = evaluate_quality(save=True)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if bool(payload.get("critical_gates_passed", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
