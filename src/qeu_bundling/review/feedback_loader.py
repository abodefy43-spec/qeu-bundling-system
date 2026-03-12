"""Load and normalize human bundle feedback for presentation-time scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

FEEDBACK_COLUMNS = (
    "feedback_id",
    "reviewed_at",
    "reviewer",
    "person_id",
    "lane",
    "anchor_product_id",
    "anchor_product_name",
    "complement_product_id",
    "complement_product_name",
    "source_reason",
    "source_origin",
    "confidence_score",
    "hybrid_reco_score",
    "rating",
    "decision",
    "feedback_class",
    "issue_type",
    "pair_fit_score",
    "lane_fit_score",
    "would_show_to_customer",
    "notes",
)

FEEDBACK_CLASS_STRONG = "strong"
FEEDBACK_CLASS_STAPLE = "staple"
FEEDBACK_CLASS_WEAK = "weak"
FEEDBACK_CLASS_TRASH = "trash"
ALLOWED_FEEDBACK_CLASSES = frozenset(
    {
        FEEDBACK_CLASS_STRONG,
        FEEDBACK_CLASS_STAPLE,
        FEEDBACK_CLASS_WEAK,
        FEEDBACK_CLASS_TRASH,
    }
)

DEFAULT_STRONG_BOOST = 0.22
DEFAULT_WEAK_PENALTY = 0.10
DEFAULT_TRASH_PENALTY = 0.55


def _empty_feedback_df() -> pd.DataFrame:
    return pd.DataFrame(columns=list(FEEDBACK_COLUMNS))


def _pair_key(pid_a: Any, pid_b: Any) -> tuple[int, int] | None:
    try:
        a = int(pid_a)
        b = int(pid_b)
    except Exception:
        return None
    if a <= 0 or b <= 0:
        return None
    return (a, b) if a <= b else (b, a)


def load_bundle_feedback(path: str | Path) -> pd.DataFrame:
    src = Path(path)
    if not src.exists():
        return _empty_feedback_df()
    try:
        df = pd.read_csv(src)
    except Exception:
        return _empty_feedback_df()
    return normalize_feedback(df)


def normalize_feedback(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return _empty_feedback_df()
    out = df.copy()
    for col in FEEDBACK_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out["anchor_product_id"] = pd.to_numeric(out["anchor_product_id"], errors="coerce").fillna(0).astype(int)
    out["complement_product_id"] = pd.to_numeric(out["complement_product_id"], errors="coerce").fillna(0).astype(int)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(0).astype(int)
    out["decision"] = out["decision"].astype(str).str.strip().str.lower()
    out["feedback_class"] = out["feedback_class"].astype(str).str.strip().str.lower()
    out.loc[~out["feedback_class"].isin(ALLOWED_FEEDBACK_CLASSES), "feedback_class"] = ""
    out["lane"] = out["lane"].astype(str).str.strip().str.lower()
    out["reviewed_at"] = out["reviewed_at"].astype(str).str.strip()
    out["feedback_id"] = out["feedback_id"].astype(str).str.strip()
    for score_col in ("confidence_score", "hybrid_reco_score", "pair_fit_score", "lane_fit_score"):
        out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    return out.loc[:, list(FEEDBACK_COLUMNS)]


def build_feedback_lookup(df: pd.DataFrame) -> dict[str, Any]:
    normalized = normalize_feedback(df)
    good_pairs: set[tuple[int, int]] = set()
    bad_pairs: set[tuple[int, int]] = set()
    strong_pairs: set[tuple[int, int]] = set()
    staple_pairs: set[tuple[int, int]] = set()
    weak_pairs: set[tuple[int, int]] = set()
    trash_pairs: set[tuple[int, int]] = set()
    pair_penalties: dict[tuple[int, int], float] = {}
    pair_boosts: dict[tuple[int, int], float] = {}
    pair_overrides: set[tuple[int, int]] = set()

    if normalized.empty:
        return {
            "good_pairs": good_pairs,
            "bad_pairs": bad_pairs,
            "strong_pairs": strong_pairs,
            "staple_pairs": staple_pairs,
            "weak_pairs": weak_pairs,
            "trash_pairs": trash_pairs,
            "pair_penalties": pair_penalties,
            "pair_boosts": pair_boosts,
            "pair_overrides": pair_overrides,
            "rows": 0,
        }

    for _, row in normalized.iterrows():
        key = _pair_key(row.get("anchor_product_id"), row.get("complement_product_id"))
        if key is None:
            continue
        decision = str(row.get("decision", "")).strip().lower()
        feedback_class = str(row.get("feedback_class", "")).strip().lower()
        rating = int(pd.to_numeric(row.get("rating", 0), errors="coerce") or 0)

        # Preferred path: explicit feedback class.
        if feedback_class == FEEDBACK_CLASS_STRONG:
            strong_pairs.add(key)
            good_pairs.add(key)
            pair_boosts[key] = max(float(pair_boosts.get(key, 0.0)), float(DEFAULT_STRONG_BOOST))
            if decision == "accept" and rating >= 4:
                pair_overrides.add(key)
            continue
        if feedback_class == FEEDBACK_CLASS_STAPLE:
            staple_pairs.add(key)
            continue
        if feedback_class == FEEDBACK_CLASS_WEAK:
            weak_pairs.add(key)
            pair_penalties[key] = max(float(pair_penalties.get(key, 0.0)), float(DEFAULT_WEAK_PENALTY))
            continue
        if feedback_class == FEEDBACK_CLASS_TRASH:
            trash_pairs.add(key)
            bad_pairs.add(key)
            pair_penalties[key] = max(float(pair_penalties.get(key, 0.0)), float(DEFAULT_TRASH_PENALTY))
            continue

        # Backward-compatible path for old feedback rows.
        is_good = rating >= 4 or decision == "accept"
        is_bad = rating <= 2 or decision == "reject"
        is_borderline = decision == "borderline" and not (is_good or is_bad)

        if is_good:
            strong_pairs.add(key)
            good_pairs.add(key)
            boost = 0.18 + 0.04 * max(0, min(5, rating) - 4)
            pair_boosts[key] = max(float(pair_boosts.get(key, 0.0)), float(boost))
            if decision == "accept" and rating >= 4:
                pair_overrides.add(key)
            continue
        if is_bad:
            trash_pairs.add(key)
            bad_pairs.add(key)
            penalty = 0.40 + 0.05 * max(0, 2 - max(0, rating))
            pair_penalties[key] = max(float(pair_penalties.get(key, 0.0)), float(penalty))
            continue
        if is_borderline:
            weak_pairs.add(key)
            pair_penalties[key] = max(float(pair_penalties.get(key, 0.0)), float(0.08))

    return {
        "good_pairs": good_pairs,
        "bad_pairs": bad_pairs,
        "strong_pairs": strong_pairs,
        "staple_pairs": staple_pairs,
        "weak_pairs": weak_pairs,
        "trash_pairs": trash_pairs,
        "pair_penalties": pair_penalties,
        "pair_boosts": pair_boosts,
        "pair_overrides": pair_overrides,
        "rows": int(len(normalized)),
    }
