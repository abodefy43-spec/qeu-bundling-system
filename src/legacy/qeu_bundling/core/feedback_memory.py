"""Feedback persistence and normalized pair-level preference memory."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

import pandas as pd

from qeu_bundling.config.paths import get_paths

VALID_FEEDBACK_TYPES = frozenset({"like", "dislike", "wrong_pair", "too_expensive"})
NEGATIVE_TYPES = frozenset({"dislike", "wrong_pair", "too_expensive"})
POSITIVE_TYPES = frozenset({"like"})
FEEDBACK_FILENAME = "person_feedback.csv"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalise_text(value: object) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(value).lower()).strip()


def _pair_key_from_names(name_a: str, name_b: str) -> tuple[str, str]:
    a = _normalise_text(name_a)
    b = _normalise_text(name_b)
    return tuple(sorted((a, b)))


def _feedback_path(base_dir: Path | None = None) -> Path:
    paths = get_paths(project_root=base_dir) if base_dir is not None else get_paths()
    return paths.data_processed_dir / FEEDBACK_FILENAME


def append_feedback_row(
    *,
    profile_id: str,
    anchor_product_id: int,
    complement_product_id: int,
    product_a_name: str,
    product_b_name: str,
    feedback_type: str,
    recommendation_origin: str = "",
    source: str = "",
    base_dir: Path | None = None,
) -> Path:
    ftype = str(feedback_type).strip().lower()
    if ftype not in VALID_FEEDBACK_TYPES:
        raise ValueError(f"Unsupported feedback type: {feedback_type}")

    out_path = _feedback_path(base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": _utc_now_iso(),
                "profile_id": str(profile_id).strip(),
                "anchor_product_id": int(anchor_product_id),
                "complement_product_id": int(complement_product_id),
                "product_a_name": str(product_a_name or "").strip(),
                "product_b_name": str(product_b_name or "").strip(),
                "pair_key": "|".join(_pair_key_from_names(product_a_name, product_b_name)),
                "feedback_type": ftype,
                "recommendation_origin": str(recommendation_origin or "").strip(),
                "source": str(source or "").strip(),
            }
        ]
    )
    if out_path.exists():
        row.to_csv(out_path, mode="a", header=False, index=False, encoding="utf-8-sig")
    else:
        row.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def load_feedback(base_dir: Path | None = None) -> pd.DataFrame:
    path = _feedback_path(base_dir)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "feedback_type" in df.columns:
        df["feedback_type"] = df["feedback_type"].astype(str).str.lower()
    return df


def build_pair_multiplier_lookup(base_dir: Path | None = None) -> dict[tuple[str, str], float]:
    """Return normalized pair multipliers from explicit feedback."""
    df = load_feedback(base_dir)
    if df.empty:
        return {}

    required = {"product_a_name", "product_b_name", "feedback_type"}
    if not required.issubset(set(df.columns)):
        return {}

    score_by_pair: dict[tuple[str, str], float] = {}
    for row in df.itertuples(index=False):
        key = _pair_key_from_names(getattr(row, "product_a_name", ""), getattr(row, "product_b_name", ""))
        if not key[0] or not key[1]:
            continue
        ftype = str(getattr(row, "feedback_type", "")).strip().lower()
        delta = 0.0
        if ftype in NEGATIVE_TYPES:
            delta = -1.0 if ftype != "too_expensive" else -0.6
        elif ftype in POSITIVE_TYPES:
            delta = 0.7
        if delta == 0.0:
            continue
        score_by_pair[key] = score_by_pair.get(key, 0.0) + delta

    multipliers: dict[tuple[str, str], float] = {}
    for key, score in score_by_pair.items():
        # bounded signal to avoid runaway boosts/penalties
        multiplier = 1.0 + max(-0.25, min(0.2, score * 0.08))
        multipliers[key] = float(max(0.75, min(1.2, multiplier)))
    return multipliers


def pair_feedback_multiplier(
    name_a: str,
    name_b: str,
    lookup: dict[tuple[str, str], float],
) -> tuple[float, bool]:
    key = _pair_key_from_names(name_a, name_b)
    value = float(lookup.get(key, 1.0))
    is_conflict = value < 1.0
    return value, is_conflict
