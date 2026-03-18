"""Batch materialization and runtime loading for final API recommendations."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.pricing import FIXED_MARGIN_DISCOUNT_PCT, margin_discounted_sale_price
from qeu_bundling.core.run_manifest import read_latest_manifest
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.person_predictions import (
    OrderPool,
    PersonProfile,
    build_recommendations_for_profiles,
    load_order_pool,
)

LOGGER = logging.getLogger("qeu_bundling.core.final_recommendations")

FINAL_RECOMMENDATIONS_ARTIFACT = "final_recommendations_by_user.json"
DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY = f"output/{FINAL_RECOMMENDATIONS_ARTIFACT}"
FINAL_RECOMMENDATIONS_MAX_USERS_ENV = "QEU_FINAL_RECOMMENDATIONS_MAX_USERS"
FINAL_RECOMMENDATIONS_USER_SELECTION_ENV = "QEU_FINAL_RECOMMENDATIONS_USER_SELECTION"
FINAL_RECOMMENDATIONS_RANDOM_SEED_ENV = "QEU_FINAL_RECOMMENDATIONS_RANDOM_SEED"
FINAL_RECOMMENDATIONS_MAX_USERS_DISABLED_TOKENS = {
    "0",
    "all",
    "off",
    "none",
    "null",
    "false",
    "disabled",
    "unlimited",
}
FALLBACK_BUNDLE_BANK_ARTIFACT = "fallback_bundle_bank.json"
DEFAULT_FALLBACK_BUNDLE_BANK_S3_KEY = f"output/{FALLBACK_BUNDLE_BANK_ARTIFACT}"
BUNDLE_IDS_ARTIFACT = "bundle_ids.csv"
DEFAULT_BUNDLE_IDS_S3_KEY = f"output/{BUNDLE_IDS_ARTIFACT}"
FALLBACK_BUNDLE_BANK_ENABLED_ENV = "QEU_FALLBACK_BUNDLE_BANK_ENABLED"
FALLBACK_BUNDLE_BANK_TARGET_SIZE_ENV = "QEU_FALLBACK_BUNDLE_BANK_TARGET_SIZE"
FALLBACK_BUNDLE_BANK_MAX_SIZE_ENV = "QEU_FALLBACK_BUNDLE_BANK_MAX_SIZE"
FALLBACK_BUNDLE_MIN_SCORE_ENV = "QEU_FALLBACK_BUNDLE_MIN_SCORE"
DEFAULT_FALLBACK_BUNDLE_BANK_TARGET_SIZE = 1000
DEFAULT_FALLBACK_BUNDLE_BANK_MAX_SIZE = 2000
BUNDLE_ID_PREFIX = "B"
BUNDLE_ID_WIDTH = 8
BUNDLE_ID_COLUMNS = (
    "bundle_id",
    "item_1_id",
    "item_2_id",
    "first_seen_at",
    "first_seen_run_id",
    "last_seen_at",
    "last_seen_run_id",
    "last_bundle_price",
    "seen_count",
)
USER_ID_COLUMNS = ("user_id", "customer_id", "customer_no", "partner_id")
MIN_HISTORY_PRODUCTS = 2
MAX_ORDER_IDS_PER_PROFILE = 6
FALLBACK_PREMIUM_POOL_MULTIPLIER = 6
FALLBACK_PREMIUM_POOL_MIN = 18
PERSONALIZED_HISTORY_PRIMARY_BOOST = 1.35
PERSONALIZED_HISTORY_SECONDARY_BOOST = 0.35
PERSONALIZED_UPSTREAM_SCORE_WEIGHT = 0.45
PERSONALIZED_CONFIDENCE_WEIGHT = 0.03
PERSONALIZED_UPSTREAM_RANK_PENALTY = 0.14
PERSONALIZED_GENERIC_THEME_PENALTY = 0.9
FALLBACK_HISTORY_PRIMARY_BOOST = 1.2
FALLBACK_HISTORY_SECONDARY_BOOST = 0.3
FALLBACK_SUPPORT_SIGNAL_BONUS = 1.2
FALLBACK_GENERIC_LOW_SIGNAL_PENALTY = 5.0
FALLBACK_GENERIC_SINGLE_SIGNAL_PENALTY = 2.0
FALLBACK_GENERIC_SAME_CATEGORY_PENALTY = 3.5
FALLBACK_GENERIC_STAPLE_OVERLAP_PENALTY = 7.5
FALLBACK_QUALITY_BAND_WIDTH = 12.0
FALLBACK_BANK_PREMIUM_WINDOW_MULTIPLIER = 4
FALLBACK_BANK_PREMIUM_WINDOW_MIN = 18


@dataclass(frozen=True)
class FinalRecommendationsWriteResult:
    path: Path
    run_id: str
    profile_count: int
    user_count: int
    generated_at: str


@dataclass(frozen=True)
class FinalRecommendationsLoadResult:
    path: Path
    run_id: str
    generated_at: str
    recommendations_by_user: dict[int, list[dict[str, object]]]


def final_recommendations_artifact_path(base_dir: Path | None = None) -> Path:
    paths = get_paths(project_root=base_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    return paths.output_dir / FINAL_RECOMMENDATIONS_ARTIFACT


def fallback_bundle_bank_artifact_path(base_dir: Path | None = None) -> Path:
    paths = get_paths(project_root=base_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    return paths.output_dir / FALLBACK_BUNDLE_BANK_ARTIFACT


def bundle_ids_artifact_path(base_dir: Path | None = None) -> Path:
    paths = get_paths(project_root=base_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    return paths.output_dir / BUNDLE_IDS_ARTIFACT


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def _safe_positive_int(value: object) -> int | None:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    if out <= 0:
        return None
    return int(out)


def _history_support_boost(overlap: int, *, first: float, second: float) -> float:
    overlap_int = max(0, int(overlap))
    if overlap_int <= 0:
        return 0.0
    return float(first + (max(0, overlap_int - 1) * second))


def _quality_band(score: object, *, width: float = FALLBACK_QUALITY_BAND_WIDTH) -> int:
    band_width = max(0.5, float(width))
    return int(float(_safe_float(score, 0.0)) // band_width)


def _bundle_origin_group(bundle: dict[str, object]) -> str:
    if not isinstance(bundle, dict):
        return "other"
    origin = str(bundle.get("recommendation_origin", bundle.get("recommendation_origin_raw", bundle.get("source", "")))).strip().lower()
    if origin in {"top_bundle", "copurchase_fallback", "fallback_food", "fallback_cleaning"}:
        return origin
    if origin.startswith("fallback_cleaning"):
        return "fallback_cleaning"
    if origin.startswith("fallback"):
        return "fallback_food"
    return "other"


def _bundle_origin_bonus(origin_group: str) -> float:
    return float(
        {
            "top_bundle": 1.6,
            "copurchase_fallback": 0.8,
            "fallback_food": 0.2,
            "fallback_cleaning": -0.4,
            "other": 0.0,
        }.get(str(origin_group).strip().lower(), 0.0)
    )


def _pair_strength_bonus(value: object) -> float:
    raw = str(value or "").strip().lower()
    return float(
        {
            "strong": 1.2,
            "medium": 0.35,
            "weak": -0.35,
        }.get(raw, 0.0)
    )


def _fallback_support_signal_count(row: dict[str, object]) -> int:
    pair_count = float(_safe_float(row.get("pair_count"), 0.0))
    recipe_compat = float(_safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0)))
    shared_categories_count = float(_safe_float(row.get("shared_categories_count"), 0.0))
    known_prior_flag = float(_safe_float(row.get("known_prior_flag"), 0.0))
    signals = 0
    if pair_count >= 3.0:
        signals += 1
    if recipe_compat >= 0.65:
        signals += 1
    if shared_categories_count >= 2.0:
        signals += 1
    if known_prior_flag > 0.0:
        signals += 1
    return int(signals)


def _fallback_genericity_penalty(row: dict[str, object]) -> float:
    pair_count = float(_safe_float(row.get("pair_count"), 0.0))
    recipe_compat = float(_safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0)))
    shared_categories_count = float(_safe_float(row.get("shared_categories_count"), 0.0))
    known_prior_flag = float(_safe_float(row.get("known_prior_flag"), 0.0))
    category_a = str(row.get("category_a", "") or "").strip().lower()
    category_b = str(row.get("category_b", "") or "").strip().lower()
    support_signal_count = _fallback_support_signal_count(row)

    penalty = 0.0
    if support_signal_count <= 1:
        penalty += FALLBACK_GENERIC_LOW_SIGNAL_PENALTY
    elif support_signal_count == 2 and recipe_compat < 0.65 and pair_count < 4.0:
        penalty += FALLBACK_GENERIC_SINGLE_SIGNAL_PENALTY
    if category_a and category_b and category_a == category_b and recipe_compat < 0.75 and pair_count < 5.0:
        penalty += FALLBACK_GENERIC_SAME_CATEGORY_PENALTY
    if _is_truthy_numeric(row.get("only_staples_overlap")):
        penalty += FALLBACK_GENERIC_STAPLE_OVERLAP_PENALTY
    if known_prior_flag <= 0.0 and pair_count < 3.0 and shared_categories_count >= 2.0 and recipe_compat < 0.65:
        penalty += 1.5
    return float(penalty)


def _bundle_use_case_reject_reason(row: dict[str, object]) -> str | None:
    pair_count = float(_safe_float(row.get("pair_count"), 0.0))
    recipe_compat = float(_safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0)))
    shared_categories_count = float(_safe_float(row.get("shared_categories_count"), 0.0))
    known_prior_flag = float(_safe_float(row.get("known_prior_flag"), 0.0))
    same_category = str(row.get("category_a", "") or "").strip().lower() == str(row.get("category_b", "") or "").strip().lower()
    support_signal_count = _fallback_support_signal_count(row)

    if support_signal_count <= 0:
        return "no_use_case_signal"
    if support_signal_count == 1 and pair_count < 4.0 and recipe_compat < 0.72:
        return "single_signal_weak_pair"
    if shared_categories_count >= 2.0 and pair_count < 3.0 and recipe_compat < 0.65 and known_prior_flag <= 0.0:
        return "category_only_pair"
    if same_category and pair_count < 5.0 and recipe_compat < 0.75 and known_prior_flag <= 0.0:
        return "same_category_generic"
    if _is_truthy_numeric(row.get("only_staples_overlap")) and pair_count < 6.0 and recipe_compat < 0.90:
        return "staple_collision"
    if pair_count < 2.0 and recipe_compat < 0.60 and shared_categories_count <= 1.0 and known_prior_flag <= 0.0:
        return "no_clear_use_case"
    return None


def _validate_bundle_use_case(row: dict[str, object]) -> bool:
    return _bundle_use_case_reject_reason(row) is None


def _resolve_bool_flag(value: object, *, env_name: str, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    raw = str(value).strip().lower()
    if not raw:
        return bool(default)
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{env_name} must be one of: 1,0,true,false,yes,no,on,off; got {value!r}")


def _latest_run_id(base_dir: Path) -> str:
    payload = read_latest_manifest(base_dir=base_dir)
    return str(payload.get("run_id", "") or "").strip()


def resolve_final_recommendations_max_users(value: object) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.lower() in FINAL_RECOMMENDATIONS_MAX_USERS_DISABLED_TOKENS:
        return None
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{FINAL_RECOMMENDATIONS_MAX_USERS_ENV} must be a positive integer or disabled token; got {raw!r}"
        ) from exc
    if parsed <= 0:
        raise ValueError(
            f"{FINAL_RECOMMENDATIONS_MAX_USERS_ENV} must be a positive integer or disabled token; got {raw!r}"
        )
    return int(parsed)


def resolve_final_recommendations_max_users_from_env() -> int | None:
    return resolve_final_recommendations_max_users(os.getenv(FINAL_RECOMMENDATIONS_MAX_USERS_ENV, ""))


def resolve_final_recommendations_user_selection_mode(value: object) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "sorted"
    if raw in {"sorted", "random"}:
        return raw
    raise ValueError(
        f"{FINAL_RECOMMENDATIONS_USER_SELECTION_ENV} must be one of: sorted, random; got {raw!r}"
    )


def resolve_final_recommendations_user_selection_mode_from_env() -> str:
    return resolve_final_recommendations_user_selection_mode(os.getenv(FINAL_RECOMMENDATIONS_USER_SELECTION_ENV, ""))


def resolve_final_recommendations_random_seed(value: object) -> int | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{FINAL_RECOMMENDATIONS_RANDOM_SEED_ENV} must be an integer; got {raw!r}"
        ) from exc


def resolve_final_recommendations_random_seed_from_env() -> int | None:
    return resolve_final_recommendations_random_seed(os.getenv(FINAL_RECOMMENDATIONS_RANDOM_SEED_ENV, ""))


def resolve_fallback_bundle_bank_enabled(value: object) -> bool:
    return _resolve_bool_flag(
        value,
        env_name=FALLBACK_BUNDLE_BANK_ENABLED_ENV,
        default=True,
    )


def resolve_fallback_bundle_bank_enabled_from_env() -> bool:
    return resolve_fallback_bundle_bank_enabled(os.getenv(FALLBACK_BUNDLE_BANK_ENABLED_ENV, "1"))


def resolve_fallback_bundle_bank_target_size(value: object) -> int:
    raw = str(value or "").strip()
    if not raw:
        return int(DEFAULT_FALLBACK_BUNDLE_BANK_TARGET_SIZE)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{FALLBACK_BUNDLE_BANK_TARGET_SIZE_ENV} must be a positive integer; got {raw!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{FALLBACK_BUNDLE_BANK_TARGET_SIZE_ENV} must be a positive integer; got {raw!r}")
    return int(parsed)


def resolve_fallback_bundle_bank_target_size_from_env() -> int:
    return resolve_fallback_bundle_bank_target_size(os.getenv(FALLBACK_BUNDLE_BANK_TARGET_SIZE_ENV, ""))


def resolve_fallback_bundle_bank_max_size(value: object) -> int:
    raw = str(value or "").strip()
    if not raw:
        return int(DEFAULT_FALLBACK_BUNDLE_BANK_MAX_SIZE)
    try:
        parsed = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{FALLBACK_BUNDLE_BANK_MAX_SIZE_ENV} must be a positive integer; got {raw!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{FALLBACK_BUNDLE_BANK_MAX_SIZE_ENV} must be a positive integer; got {raw!r}")
    return int(parsed)


def resolve_fallback_bundle_bank_max_size_from_env() -> int:
    return resolve_fallback_bundle_bank_max_size(os.getenv(FALLBACK_BUNDLE_BANK_MAX_SIZE_ENV, ""))


def resolve_fallback_bundle_min_score(value: object) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{FALLBACK_BUNDLE_MIN_SCORE_ENV} must be numeric; got {raw!r}") from exc
    if parsed != parsed:  # NaN
        raise ValueError(f"{FALLBACK_BUNDLE_MIN_SCORE_ENV} must be numeric; got {raw!r}")
    return float(parsed)


def resolve_fallback_bundle_min_score_from_env() -> float | None:
    return resolve_fallback_bundle_min_score(os.getenv(FALLBACK_BUNDLE_MIN_SCORE_ENV, ""))


def _select_user_ids(
    all_user_ids: list[int],
    *,
    max_users_limit: int | None,
    selection_mode: str,
    random_seed: int | None,
) -> tuple[list[int], str]:
    if max_users_limit is None:
        return list(all_user_ids), "sorted_unique_user_id_asc"

    limit = max(0, int(max_users_limit))
    if selection_mode == "sorted":
        return list(all_user_ids[:limit]), "sorted_unique_user_id_asc"

    if selection_mode == "random":
        if limit >= len(all_user_ids):
            return list(all_user_ids), "random_sample_from_sorted_user_ids"
        rng = random.Random(random_seed) if random_seed is not None else random.Random()
        return list(rng.sample(list(all_user_ids), limit)), "random_sample_from_sorted_user_ids"

    raise ValueError(f"Unsupported user selection mode: {selection_mode}")


def _load_orders_frame(base_dir: Path) -> pd.DataFrame:
    paths = get_paths(project_root=base_dir)
    orders_path = paths.data_processed_dir / "filtered_orders.pkl"
    if not orders_path.exists():
        return pd.DataFrame()
    try:
        orders = pd.read_pickle(orders_path)
    except Exception:
        return pd.DataFrame()

    if orders.empty or "order_id" not in orders.columns:
        return pd.DataFrame()

    user_cols = [column for column in USER_ID_COLUMNS if column in orders.columns]
    data = orders.loc[:, ["order_id", *user_cols]].copy()
    del orders

    data["order_id"] = pd.to_numeric(data["order_id"], errors="coerce")
    data = data.dropna(subset=["order_id"])
    if data.empty:
        return pd.DataFrame()
    data["order_id"] = data["order_id"].astype("int64")

    for column in user_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if user_cols:
        data = data.dropna(subset=user_cols, how="all")
    data = data.drop_duplicates(subset=["order_id", *user_cols], keep="last")
    return data


def _resolve_order_ids_for_user(user_id: int, orders_df: pd.DataFrame) -> list[int]:
    if orders_df.empty:
        return []

    user_order_ids: list[int] = []
    for col in USER_ID_COLUMNS:
        if col not in orders_df.columns:
            continue
        matched = orders_df.loc[orders_df[col] == int(user_id), "order_id"]
        if matched.empty:
            continue
        user_order_ids = [int(oid) for oid in matched.dropna().astype("int64").unique().tolist()]
        if user_order_ids:
            break

    # Fallback only when explicit user columns are unavailable.
    if not user_order_ids and "order_id" in orders_df.columns and not any(col in orders_df.columns for col in USER_ID_COLUMNS):
        if bool((orders_df["order_id"] == int(user_id)).any()):
            user_order_ids = [int(user_id)]

    unique_ids = sorted({int(oid) for oid in user_order_ids if int(oid) > 0})
    return unique_ids[-MAX_ORDER_IDS_PER_PROFILE:]


def _build_profile_from_orders(user_id: int, order_ids: list[int], order_pool: OrderPool) -> PersonProfile | None:
    order_ids_clean = sorted({int(oid) for oid in order_ids if int(oid) in order_pool.order_product_ids})
    if not order_ids_clean:
        return None

    history_ids: set[int] = set()
    history_counts: dict[int, int] = {}
    history_items: list[str] = []
    seen_names: set[str] = set()
    for oid in order_ids_clean:
        for pid in order_pool.order_product_ids.get(oid, ()):
            pid_int = int(pid)
            if pid_int <= 0:
                continue
            history_ids.add(pid_int)
            history_counts[pid_int] = int(history_counts.get(pid_int, 0)) + 1
        for name in order_pool.order_product_names.get(oid, ()):
            text = str(name).strip()
            if text and text not in seen_names:
                seen_names.add(text)
                history_items.append(text)

    if not history_ids:
        return None

    return PersonProfile(
        profile_id=f"api_user_{int(user_id)}",
        source="api_customer",
        order_ids=order_ids_clean,
        history_product_ids=sorted(history_ids),
        history_items=history_items,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        history_counts={int(k): int(v) for k, v in sorted(history_counts.items())},
    )


def _candidate_user_ids(orders_df: pd.DataFrame) -> list[int]:
    if orders_df.empty:
        return []

    user_ids: set[int] = set()
    user_cols = [column for column in USER_ID_COLUMNS if column in orders_df.columns]
    for column in user_cols:
        values = pd.to_numeric(orders_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        for value in values.astype("int64").tolist():
            if int(value) > 0:
                user_ids.add(int(value))

    if user_ids:
        return sorted(user_ids)

    # Fallback for legacy datasets with no explicit customer column.
    if "order_id" in orders_df.columns:
        return sorted({int(v) for v in orders_df["order_id"].dropna().astype("int64").tolist() if int(v) > 0})
    return []


def _bundle_to_api_record(bundle: dict[str, object]) -> dict[str, object] | None:
    item_1_id = _safe_positive_int(bundle.get("product_a"))
    item_2_id = _safe_positive_int(bundle.get("product_b"))
    if item_1_id is None or item_2_id is None or item_1_id == item_2_id:
        return None

    sale_a = max(0.0, _safe_float(bundle.get("product_a_price"), default=_safe_float(bundle.get("price_a_sar"), 0.0)))
    sale_b = max(0.0, _safe_float(bundle.get("product_b_price"), default=_safe_float(bundle.get("price_b_sar"), 0.0)))
    purchase_a = max(0.0, _safe_float(bundle.get("purchase_price_a"), default=sale_a))
    purchase_b = max(0.0, _safe_float(bundle.get("purchase_price_b"), default=sale_b))
    if sale_a >= sale_b:
        final_paid_price = margin_discounted_sale_price(sale_a, purchase_a, FIXED_MARGIN_DISCOUNT_PCT)
    else:
        final_paid_price = margin_discounted_sale_price(sale_b, purchase_b, FIXED_MARGIN_DISCOUNT_PCT)

    return {
        "item_1_id": int(item_1_id),
        "item_2_id": int(item_2_id),
        "bundle_price": float(round(final_paid_price, 2)),
    }


def _extract_api_user_id(profile_id: object) -> int | None:
    raw = str(profile_id or "").strip()
    prefix = "api_user_"
    if not raw.startswith(prefix):
        return None
    return _safe_positive_int(raw[len(prefix) :])


def _bundle_pair_key(item_1_id: int, item_2_id: int) -> tuple[int, int]:
    a = int(item_1_id)
    b = int(item_2_id)
    if a <= b:
        return (a, b)
    return (b, a)


def _stable_hash_int(value: object) -> int:
    raw = str(value or "").strip()
    if not raw:
        return 0
    digest = hashlib.sha1(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _run_rotation_hint(run_id: str) -> int:
    raw = str(run_id or "").strip()
    if not raw:
        return 0
    digits = re.findall(r"\d+", raw)
    if digits:
        try:
            return max(0, int(digits[-1]))
        except (TypeError, ValueError):
            pass
    return _stable_hash_int(raw)


def _rotated_index_order(length: int, start: int) -> list[int]:
    size = int(length)
    if size <= 0:
        return []
    offset = int(start) % size
    return list(range(offset, size)) + list(range(0, offset))


def _coprime_rotation_step(size: int) -> int:
    candidate = 2
    pool_size = max(1, int(size))
    while candidate < pool_size:
        left = candidate
        right = pool_size
        while right:
            left, right = right, left % right
        if left == 1:
            return int(candidate)
        candidate += 1
    return 1


def _bundle_id_sequence(bundle_id: str) -> int:
    raw = str(bundle_id or "").strip()
    if not raw:
        return 0
    match = re.fullmatch(rf"{re.escape(BUNDLE_ID_PREFIX)}(\d+)", raw)
    if match is None:
        return 0
    try:
        return int(match.group(1))
    except (TypeError, ValueError):
        return 0


def _bundle_id_from_sequence(sequence: int) -> str:
    return f"{BUNDLE_ID_PREFIX}{int(sequence):0{int(BUNDLE_ID_WIDTH)}d}"


def _bundle_price_as_text(value: object) -> str:
    return f"{float(round(max(0.0, _safe_float(value, 0.0)), 2)):.2f}"


def _load_bundle_id_registry(path: Path) -> tuple[dict[tuple[int, int], dict[str, str]], int]:
    registry: dict[tuple[int, int], dict[str, str]] = {}
    max_sequence = 0
    if not path.exists():
        return registry, max_sequence

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw_row in reader:
            if not isinstance(raw_row, dict):
                continue
            item_1_id = _safe_positive_int(raw_row.get("item_1_id"))
            item_2_id = _safe_positive_int(raw_row.get("item_2_id"))
            bundle_id = str(raw_row.get("bundle_id", "")).strip()
            if item_1_id is None or item_2_id is None or item_1_id == item_2_id or not bundle_id:
                continue
            pair_key = _bundle_pair_key(item_1_id, item_2_id)
            if pair_key in registry:
                continue

            seen_count_raw = str(raw_row.get("seen_count", "0")).strip()
            try:
                seen_count = max(0, int(seen_count_raw))
            except (TypeError, ValueError):
                seen_count = 0

            row = {
                "bundle_id": bundle_id,
                "item_1_id": str(int(pair_key[0])),
                "item_2_id": str(int(pair_key[1])),
                "first_seen_at": str(raw_row.get("first_seen_at", "") or "").strip(),
                "first_seen_run_id": str(raw_row.get("first_seen_run_id", "") or "").strip(),
                "last_seen_at": str(raw_row.get("last_seen_at", "") or "").strip(),
                "last_seen_run_id": str(raw_row.get("last_seen_run_id", "") or "").strip(),
                "last_bundle_price": _bundle_price_as_text(raw_row.get("last_bundle_price", 0.0)),
                "seen_count": str(int(seen_count)),
            }
            registry[pair_key] = row
            max_sequence = max(max_sequence, _bundle_id_sequence(bundle_id))
    return registry, max_sequence


def _write_bundle_id_registry(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            _bundle_id_sequence(str(row.get("bundle_id", ""))),
            str(row.get("bundle_id", "")),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(BUNDLE_ID_COLUMNS))
        writer.writeheader()
        writer.writerows(ordered_rows)


def _restore_bundle_id_registry_from_s3_if_configured(path: Path) -> None:
    if path.exists():
        return

    bucket = str(os.getenv("QEU_ARTIFACTS_S3_BUCKET", "") or "").strip()
    key = str(os.getenv("QEU_S3_BUNDLE_IDS_KEY", DEFAULT_BUNDLE_IDS_S3_KEY) or "").strip()
    if not bucket or not key:
        return

    try:
        import boto3  # type: ignore
    except Exception:
        LOGGER.info("[batch] boto3 unavailable; skipping bundle id registry restore from S3")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    started = time.perf_counter()
    try:
        boto3.client("s3").download_file(bucket, key, str(partial))
        partial.replace(path)
        LOGGER.info(
            "[batch] restored bundle id registry from s3://%s/%s bytes=%d duration_sec=%.3f",
            bucket,
            key,
            int(path.stat().st_size) if path.exists() else 0,
            time.perf_counter() - started,
        )
    except Exception:
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        LOGGER.info("[batch] no existing bundle id registry at s3://%s/%s; starting fresh", bucket, key)


def _collect_materialized_bundle_pairs(
    recommendations_by_user: dict[int, list[dict[str, object]]],
    fallback_bank: list[dict[str, object]],
) -> dict[tuple[int, int], float]:
    pairs: dict[tuple[int, int], float] = {}
    for user_id in sorted(recommendations_by_user):
        bundles = recommendations_by_user.get(int(user_id), [])
        if not isinstance(bundles, list):
            continue
        for bundle in bundles:
            if not isinstance(bundle, dict):
                continue
            clean = _sanitize_bundle_record(bundle)
            if clean is None:
                continue
            pair_key = _bundle_pair_key(int(clean["item_1_id"]), int(clean["item_2_id"]))
            pairs[pair_key] = float(clean["bundle_price"])

    for bundle in fallback_bank:
        if not isinstance(bundle, dict):
            continue
        clean = _sanitize_bundle_record(bundle)
        if clean is None:
            continue
        pair_key = _bundle_pair_key(int(clean["item_1_id"]), int(clean["item_2_id"]))
        pairs[pair_key] = float(clean["bundle_price"])
    return pairs


def _upsert_bundle_id_registry(
    *,
    base_dir: Path,
    run_id: str,
    generated_at: str,
    recommendations_by_user: dict[int, list[dict[str, object]]],
    fallback_bank: list[dict[str, object]],
) -> tuple[Path, int, int, int]:
    path = bundle_ids_artifact_path(base_dir)
    _restore_bundle_id_registry_from_s3_if_configured(path)
    registry, max_sequence = _load_bundle_id_registry(path)
    pairs_seen_this_run = _collect_materialized_bundle_pairs(recommendations_by_user, fallback_bank)

    new_count = 0
    for pair_key in sorted(pairs_seen_this_run):
        bundle_price = float(pairs_seen_this_run[pair_key])
        if pair_key in registry:
            existing = dict(registry[pair_key])
            seen_count_raw = str(existing.get("seen_count", "0")).strip()
            try:
                seen_count = max(0, int(seen_count_raw))
            except (TypeError, ValueError):
                seen_count = 0
            existing["last_seen_at"] = str(generated_at)
            existing["last_seen_run_id"] = str(run_id)
            existing["last_bundle_price"] = _bundle_price_as_text(bundle_price)
            existing["seen_count"] = str(int(seen_count + 1))
            registry[pair_key] = existing
            continue

        max_sequence += 1
        registry[pair_key] = {
            "bundle_id": _bundle_id_from_sequence(max_sequence),
            "item_1_id": str(int(pair_key[0])),
            "item_2_id": str(int(pair_key[1])),
            "first_seen_at": str(generated_at),
            "first_seen_run_id": str(run_id),
            "last_seen_at": str(generated_at),
            "last_seen_run_id": str(run_id),
            "last_bundle_price": _bundle_price_as_text(bundle_price),
            "seen_count": "1",
        }
        new_count += 1

    _write_bundle_id_registry(path, list(registry.values()))
    return path, int(new_count), int(len(registry)), int(len(pairs_seen_this_run))


def _fallback_quality_score(row: dict[str, object]) -> float:
    final_score = _safe_float(row.get("new_final_score"), default=_safe_float(row.get("final_score"), 0.0))
    model_score = _safe_float(row.get("model_score"), 0.0)
    recipe_compat = max(0.0, min(1.0, _safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0))))
    pair_count = max(0.0, _safe_float(row.get("pair_count"), 0.0))
    shared_categories_count = max(0.0, _safe_float(row.get("shared_categories_count"), 0.0))
    known_prior_flag = 1.0 if _safe_float(row.get("known_prior_flag"), 0.0) > 0 else 0.0
    deal_signal = max(0.0, _safe_float(row.get("deal_signal"), 0.0))
    return float(
        final_score
        + (0.15 * model_score)
        + (22.0 * recipe_compat)
        + (0.75 * min(pair_count, 20.0))
        + (0.65 * min(shared_categories_count, 12.0))
        + (3.0 * known_prior_flag)
        + (1.5 * deal_signal)
    )


def _is_truthy_numeric(value: object) -> bool:
    return _safe_float(value, 0.0) > 0.0


def _passes_fallback_quality_filters(row: dict[str, object], *, min_score: float | None) -> bool:
    if _is_truthy_numeric(row.get("weak_evidence_free_blocked")):
        return False
    if _safe_float(row.get("pair_penalty_multiplier"), 1.0) < 0.95:
        return False
    if _safe_float(row.get("utility_penalty_multiplier"), 1.0) < 0.95:
        return False
    if not _validate_bundle_use_case(row):
        return False

    quality_score = _fallback_quality_score(row)
    if min_score is not None and quality_score < float(min_score):
        return False

    recipe_compat = _safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0))
    pair_count = _safe_float(row.get("pair_count"), 0.0)
    shared_categories_count = _safe_float(row.get("shared_categories_count"), 0.0)
    known_prior_flag = _safe_float(row.get("known_prior_flag"), 0.0)
    has_evidence = bool(
        pair_count >= 1.0
        or known_prior_flag > 0.0
        or recipe_compat >= 0.60
        or shared_categories_count >= 2.0
    )
    if not has_evidence:
        return False

    if pair_count < 2.0 and recipe_compat < 0.50 and known_prior_flag <= 0.0:
        return False

    if pair_count < 3.0 and shared_categories_count <= 1.0 and recipe_compat < 0.55:
        return False

    if _is_truthy_numeric(row.get("only_staples_overlap")) and recipe_compat < 0.85 and pair_count < 2.0:
        return False

    if _is_truthy_numeric(row.get("only_staples_overlap")) and known_prior_flag <= 0.0 and pair_count < 4.0:
        return False
    return True


def _fallback_category_key(row: dict[str, object]) -> tuple[str, str]:
    a = str(row.get("category_a", "") or "").strip().lower() or "unknown"
    b = str(row.get("category_b", "") or "").strip().lower() or "unknown"
    if a <= b:
        return (a, b)
    return (b, a)


def _build_pair_quality_lookup(bundles_df: pd.DataFrame) -> dict[tuple[int, int], dict[str, object]]:
    if bundles_df.empty:
        return {}

    lookup: dict[tuple[int, int], dict[str, object]] = {}
    for row in bundles_df.to_dict(orient="records"):
        if not isinstance(row, dict):
            continue
        api_record = _bundle_to_api_record(row)
        if api_record is None:
            continue
        pair_key = _bundle_pair_key(int(api_record["item_1_id"]), int(api_record["item_2_id"]))
        category_a, category_b = _fallback_category_key(row)
        quality_row = {
            "quality_score": float(round(_fallback_quality_score(row), 6)),
            "final_score": float(
                _safe_float(
                    row.get("new_final_score"),
                    default=_safe_float(row.get("final_score"), 0.0),
                )
            ),
            "pair_count": float(_safe_float(row.get("pair_count"), 0.0)),
            "recipe_compat_score": float(
                _safe_float(
                    row.get("recipe_compat_score"),
                    default=_safe_float(row.get("recipe_score_norm"), 0.0),
                )
            ),
            "shared_categories_count": float(_safe_float(row.get("shared_categories_count"), 0.0)),
            "known_prior_flag": float(_safe_float(row.get("known_prior_flag"), 0.0)),
            "pair_penalty_multiplier": float(_safe_float(row.get("pair_penalty_multiplier"), 1.0)),
            "utility_penalty_multiplier": float(_safe_float(row.get("utility_penalty_multiplier"), 1.0)),
            "weak_evidence_free_blocked": int(1 if _is_truthy_numeric(row.get("weak_evidence_free_blocked")) else 0),
            "only_staples_overlap": int(1 if _is_truthy_numeric(row.get("only_staples_overlap")) else 0),
            "support_signal_count": int(_fallback_support_signal_count(row)),
            "genericity_penalty": float(_fallback_genericity_penalty(row)),
            "use_case_valid": int(1 if _validate_bundle_use_case(row) else 0),
            "use_case_reject_reason": str(_bundle_use_case_reject_reason(row) or ""),
            "category_key": f"{category_a}|{category_b}",
        }
        existing = lookup.get(pair_key)
        if existing is None:
            lookup[pair_key] = quality_row
            continue
        if (
            float(quality_row["quality_score"]),
            float(quality_row["final_score"]),
            float(quality_row["pair_count"]),
        ) > (
            float(existing.get("quality_score", 0.0)),
            float(existing.get("final_score", 0.0)),
            float(existing.get("pair_count", 0.0)),
        ):
            lookup[pair_key] = quality_row
    return lookup


def _history_overlap_count(item_1_id: int, item_2_id: int, history_product_ids: set[int]) -> int:
    if not history_product_ids:
        return 0
    overlap = 0
    if int(item_1_id) in history_product_ids:
        overlap += 1
    if int(item_2_id) in history_product_ids:
        overlap += 1
    return int(overlap)


def _personalized_candidate_score(
    *,
    bundle: dict[str, object],
    candidate_rank: int,
    item_1_id: int,
    item_2_id: int,
    metadata: dict[str, object] | None,
    history_product_ids: set[int],
) -> tuple[float, int, float]:
    meta = metadata or {}
    history_overlap = _history_overlap_count(item_1_id, item_2_id, history_product_ids)
    quality_score = float(_safe_float(meta.get("quality_score"), 0.0))
    pair_count = float(_safe_float(meta.get("pair_count"), 0.0))
    recipe_compat = float(_safe_float(meta.get("recipe_compat_score"), 0.0))
    known_prior = float(_safe_float(meta.get("known_prior_flag"), 0.0))
    support_signal_count = float(_safe_float(meta.get("support_signal_count"), 0.0))
    genericity_penalty = float(_safe_float(meta.get("genericity_penalty"), 0.0))
    upstream_score = float(_safe_float(bundle.get("hybrid_reco_score"), 0.0))
    confidence_score = float(_safe_float(bundle.get("confidence_score"), 0.0))
    theme = str(bundle.get("bundle_theme", bundle.get("theme", ""))).strip().lower()
    origin_group = _bundle_origin_group(bundle)
    score = quality_score
    score += PERSONALIZED_UPSTREAM_SCORE_WEIGHT * upstream_score
    score += PERSONALIZED_CONFIDENCE_WEIGHT * confidence_score
    score += _bundle_origin_bonus(origin_group)
    score += _pair_strength_bonus(bundle.get("pair_strength"))
    score += 0.45 * min(support_signal_count, 4.0)
    score += _history_support_boost(
        history_overlap,
        first=PERSONALIZED_HISTORY_PRIMARY_BOOST,
        second=PERSONALIZED_HISTORY_SECONDARY_BOOST,
    )
    score -= PERSONALIZED_UPSTREAM_RANK_PENALTY * float(max(0, int(candidate_rank)))
    score -= 0.55 * genericity_penalty
    if theme.endswith("_generic"):
        score -= PERSONALIZED_GENERIC_THEME_PENALTY

    if history_overlap <= 0 and pair_count < 2.0 and recipe_compat < 0.50 and known_prior <= 0.0:
        score -= 7.0
    if _safe_float(meta.get("pair_penalty_multiplier"), 1.0) < 0.95:
        score -= 4.5
    if _safe_float(meta.get("utility_penalty_multiplier"), 1.0) < 0.95:
        score -= 3.0
    if _is_truthy_numeric(meta.get("only_staples_overlap")) and history_overlap <= 0 and pair_count < 4.0:
        score -= 5.0
    return float(score), int(history_overlap), float(quality_score)


def _reject_personalized_candidate(metadata: dict[str, object] | None) -> bool:
    if metadata is None:
        return False
    if int(_safe_float(metadata.get("use_case_valid"), 1.0)) <= 0:
        return True
    if _is_truthy_numeric(metadata.get("weak_evidence_free_blocked")):
        return True
    if _safe_float(metadata.get("pair_penalty_multiplier"), 1.0) < 0.88:
        return True
    if _safe_float(metadata.get("utility_penalty_multiplier"), 1.0) < 0.88:
        return True
    pair_count = _safe_float(metadata.get("pair_count"), 0.0)
    recipe_compat = _safe_float(metadata.get("recipe_compat_score"), 0.0)
    shared_categories_count = _safe_float(metadata.get("shared_categories_count"), 0.0)
    known_prior = _safe_float(metadata.get("known_prior_flag"), 0.0)
    if pair_count < 1.0 and recipe_compat < 0.45 and shared_categories_count < 2.0 and known_prior <= 0.0:
        return True
    return False


def _select_personalized_bundles_for_user(
    *,
    bundles_raw: list[dict[str, object]],
    max_bundles_per_user: int,
    history_product_ids: set[int],
    pair_quality_lookup: dict[tuple[int, int], dict[str, object]],
) -> list[dict[str, object]]:
    target = max(1, int(max_bundles_per_user))
    if not bundles_raw:
        return []

    candidates: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for candidate_rank, bundle in enumerate(bundles_raw):
        if not isinstance(bundle, dict):
            continue
        api_record = _bundle_to_api_record(bundle)
        if api_record is None:
            continue
        item_1_id = int(api_record["item_1_id"])
        item_2_id = int(api_record["item_2_id"])
        pair_key = _bundle_pair_key(item_1_id, item_2_id)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        metadata = pair_quality_lookup.get(pair_key)
        if _reject_personalized_candidate(metadata):
            continue

        score, history_overlap, quality_score = _personalized_candidate_score(
            bundle=bundle,
            candidate_rank=int(candidate_rank),
            item_1_id=item_1_id,
            item_2_id=item_2_id,
            metadata=metadata,
            history_product_ids=history_product_ids,
        )
        candidates.append(
            {
                "record": api_record,
                "pair_key": pair_key,
                "category_key": str((metadata or {}).get("category_key", "")),
                "lane_key": str(bundle.get("lane", "")).strip().lower(),
                "origin_group": _bundle_origin_group(bundle),
                "candidate_rank": int(candidate_rank),
                "score": float(score),
                "history_overlap": int(history_overlap),
                "quality_score": float(quality_score),
            }
        )

    candidates.sort(
        key=lambda entry: (
            -float(entry.get("score", 0.0)),
            int(entry.get("candidate_rank", 0)),
            -float(entry.get("quality_score", 0.0)),
            str(entry.get("origin_group", "")),
            int(entry.get("pair_key", (0, 0))[0]),
            int(entry.get("pair_key", (0, 0))[1]),
        )
    )

    selected: list[dict[str, object]] = []
    pair_seen: set[tuple[int, int]] = set()
    item_seen: set[int] = set()
    category_seen: set[str] = set()
    lane_seen: set[str] = set()
    for strict_no_item_overlap, strict_lane_diversity, strict_category_diversity in (
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, False, True),
        (False, False, False),
    ):
        if len(selected) >= target:
            break
        for entry in candidates:
            if len(selected) >= target:
                break
            pair_key = tuple(entry.get("pair_key", (0, 0)))
            if pair_key in pair_seen:
                continue
            record = entry.get("record")
            if not isinstance(record, dict):
                continue
            item_1_id = _safe_positive_int(record.get("item_1_id"))
            item_2_id = _safe_positive_int(record.get("item_2_id"))
            if item_1_id is None or item_2_id is None or item_1_id == item_2_id:
                continue
            if strict_no_item_overlap and (int(item_1_id) in item_seen or int(item_2_id) in item_seen):
                continue
            lane_key = str(entry.get("lane_key", "")).strip().lower()
            if strict_lane_diversity and lane_key and lane_key in lane_seen and target >= 3:
                continue
            category_key = str(entry.get("category_key", ""))
            if strict_category_diversity and category_key and category_key in category_seen and int(
                entry.get("history_overlap", 0)
            ) <= 0:
                continue
            selected.append(
                {
                    "item_1_id": int(item_1_id),
                    "item_2_id": int(item_2_id),
                    "bundle_price": float(round(max(0.0, _safe_float(record.get("bundle_price"), 0.0)), 2)),
                }
            )
            pair_seen.add(_bundle_pair_key(int(item_1_id), int(item_2_id)))
            item_seen.add(int(item_1_id))
            item_seen.add(int(item_2_id))
            if category_key:
                category_seen.add(category_key)
            if lane_key:
                lane_seen.add(lane_key)
    return selected[:target]


def _order_fallback_bank_candidates_for_run(
    candidates: list[dict[str, object]],
    *,
    run_id: str,
    target_size: int,
) -> list[dict[str, object]]:
    if len(candidates) <= 1:
        return list(candidates)

    premium_window = min(
        len(candidates),
        max(int(target_size) * int(FALLBACK_BANK_PREMIUM_WINDOW_MULTIPLIER), int(FALLBACK_BANK_PREMIUM_WINDOW_MIN)),
    )
    premium_candidates = list(candidates[:premium_window])
    tail_candidates = list(candidates[premium_window:])
    run_hint = _run_rotation_hint(run_id)

    grouped: dict[int, list[dict[str, object]]] = {}
    for entry in premium_candidates:
        band_key = _quality_band(entry.get("adjusted_quality_score", entry.get("quality_score", 0.0)))
        grouped.setdefault(int(band_key), []).append(entry)

    rotated_premium: list[dict[str, object]] = []
    for band_key in sorted(grouped, reverse=True):
        band_entries = grouped[int(band_key)]
        if len(band_entries) <= 1:
            rotated_premium.extend(band_entries)
            continue
        offset = int(run_hint % len(band_entries))
        for idx in _rotated_index_order(len(band_entries), offset):
            rotated_premium.append(band_entries[int(idx)])
    return rotated_premium + tail_candidates


def _build_fallback_bundle_bank(
    bundles_df: pd.DataFrame,
    *,
    run_id: str,
    target_size: int,
    max_size: int,
    min_score: float | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if bundles_df.empty:
        return [], {
            "candidate_count": 0,
            "kept_after_filters": 0,
            "selected_count": 0,
            "category_cap": 0,
            "anchor_category_cap": 0,
        }

    requested_target_size = max(1, int(target_size))
    requested_max_size = max(1, int(max_size))
    effective_max_size = int(requested_max_size)
    if requested_target_size > effective_max_size:
        requested_target_size = int(effective_max_size)

    category_cap = max(2, min(6, int(requested_target_size // 4) + 1))
    anchor_category_cap = max(2, min(6, int(requested_target_size // 5) + 1))
    candidates: list[dict[str, object]] = []
    pair_seen: set[tuple[int, int]] = set()

    records = bundles_df.to_dict(orient="records")
    for row in records:
        if not isinstance(row, dict):
            continue
        if not _passes_fallback_quality_filters(row, min_score=min_score):
            continue

        api_record = _bundle_to_api_record(row)
        if api_record is None:
            continue

        pair_key = _bundle_pair_key(
            int(api_record["item_1_id"]),
            int(api_record["item_2_id"]),
        )
        if pair_key in pair_seen:
            continue
        pair_seen.add(pair_key)

        quality_score = _fallback_quality_score(row)
        support_signal_count = _fallback_support_signal_count(row)
        genericity_penalty = _fallback_genericity_penalty(row)
        adjusted_quality_score = float(
            quality_score + (FALLBACK_SUPPORT_SIGNAL_BONUS * float(support_signal_count)) - float(genericity_penalty)
        )
        category_key = _fallback_category_key(row)
        anchor_category = str(row.get("category_a", "") or "").strip().lower() or "unknown"
        complement_category = str(row.get("category_b", "") or "").strip().lower() or "unknown"
        candidates.append(
            {
                "item_1_id": int(api_record["item_1_id"]),
                "item_2_id": int(api_record["item_2_id"]),
                "bundle_price": float(api_record["bundle_price"]),
                "quality_score": float(round(quality_score, 6)),
                "adjusted_quality_score": float(round(adjusted_quality_score, 6)),
                "support_signal_count": int(support_signal_count),
                "genericity_penalty": float(round(genericity_penalty, 6)),
                "category_key": f"{category_key[0]}|{category_key[1]}",
                "anchor_category": str(anchor_category),
                "complement_category": str(complement_category),
                "final_score": float(_safe_float(row.get("new_final_score"), default=_safe_float(row.get("final_score"), 0.0))),
                "pair_count": float(_safe_float(row.get("pair_count"), 0.0)),
                "recipe_compat_score": float(_safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0))),
            }
        )

    candidates.sort(
        key=lambda entry: (
            -float(entry.get("adjusted_quality_score", entry.get("quality_score", 0.0))),
            -float(entry.get("quality_score", 0.0)),
            int(entry.get("support_signal_count", 0)) * -1,
            float(entry.get("genericity_penalty", 0.0)),
            -float(entry.get("final_score", 0.0)),
            -float(entry.get("pair_count", 0.0)),
            str(entry.get("category_key", "")),
            int(entry.get("item_1_id", 0)),
            int(entry.get("item_2_id", 0)),
        )
    )
    ordered_candidates = _order_fallback_bank_candidates_for_run(
        candidates,
        run_id=str(run_id),
        target_size=int(requested_target_size),
    )

    selected: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    anchor_category_counts: dict[str, int] = {}
    complement_category_counts: dict[str, int] = {}
    selected_pair_keys: set[tuple[int, int]] = set()
    for entry in ordered_candidates:
        category_key = str(entry.get("category_key", "unknown|unknown"))
        if int(category_counts.get(category_key, 0)) >= category_cap:
            continue
        anchor_category = str(entry.get("anchor_category", "unknown"))
        complement_category = str(entry.get("complement_category", "unknown"))
        if int(anchor_category_counts.get(anchor_category, 0)) >= anchor_category_cap:
            continue
        if int(complement_category_counts.get(complement_category, 0)) >= anchor_category_cap:
            continue
        category_counts[category_key] = int(category_counts.get(category_key, 0)) + 1
        anchor_category_counts[anchor_category] = int(anchor_category_counts.get(anchor_category, 0)) + 1
        complement_category_counts[complement_category] = int(complement_category_counts.get(complement_category, 0)) + 1
        selected.append(entry)
        selected_pair_keys.add(_bundle_pair_key(int(entry.get("item_1_id", 0)), int(entry.get("item_2_id", 0))))
        if len(selected) >= requested_target_size:
            break

    if len(selected) < min(requested_target_size, effective_max_size):
        for entry in ordered_candidates:
            pair_key = _bundle_pair_key(int(entry.get("item_1_id", 0)), int(entry.get("item_2_id", 0)))
            if pair_key in selected_pair_keys:
                continue
            selected.append(entry)
            selected_pair_keys.add(pair_key)
            if len(selected) >= min(requested_target_size, effective_max_size):
                break

    if len(selected) > effective_max_size:
        selected = selected[:effective_max_size]

    return selected, {
        "candidate_count": int(len(records)),
        "kept_after_filters": int(len(candidates)),
        "selected_count": int(len(selected)),
        "category_cap": int(category_cap),
        "anchor_category_cap": int(anchor_category_cap),
        "target_size": int(requested_target_size),
        "max_size": int(effective_max_size),
        "quality_band_width": float(FALLBACK_QUALITY_BAND_WIDTH),
        "premium_window_size": int(
            min(
                len(candidates),
                max(int(requested_target_size) * int(FALLBACK_BANK_PREMIUM_WINDOW_MULTIPLIER), int(FALLBACK_BANK_PREMIUM_WINDOW_MIN)),
            )
        ),
        "min_score": None if min_score is None else float(min_score),
    }


def _write_fallback_bundle_bank_artifact(
    *,
    base_dir: Path,
    run_id: str,
    generated_at: str,
    fallback_bank: list[dict[str, object]],
    bank_meta: dict[str, object],
) -> Path:
    path = fallback_bundle_bank_artifact_path(base_dir)
    payload = {
        "version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        **dict(bank_meta),
        "bundles": [
            {
                "item_1_id": int(entry["item_1_id"]),
                "item_2_id": int(entry["item_2_id"]),
                "bundle_price": float(entry["bundle_price"]),
                "quality_score": float(entry.get("quality_score", 0.0)),
                "category_key": str(entry.get("category_key", "")),
                "anchor_category": str(entry.get("anchor_category", "")),
                "complement_category": str(entry.get("complement_category", "")),
            }
            for entry in fallback_bank
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return path


def _fallback_rotation_start(*, user_id: int, run_id: str, pool_size: int, run_multiplier: int = 1) -> int:
    size = max(1, int(pool_size))
    run_hint = _run_rotation_hint(run_id)
    mixed = (int(user_id) * 2654435761) + (int(run_multiplier) * int(run_hint))
    return int(mixed % size)


def _ordered_fallback_indices(
    *,
    fallback_bank_size: int,
    run_id: str,
    user_id: int,
    target: int,
    run_multiplier: int = 1,
) -> list[int]:
    size = int(fallback_bank_size)
    if size <= 0:
        return []
    start = _fallback_rotation_start(
        user_id=int(user_id),
        run_id=str(run_id or ""),
        pool_size=size,
        run_multiplier=int(run_multiplier),
    )
    premium_size = min(size, max(int(target) * int(FALLBACK_PREMIUM_POOL_MULTIPLIER), int(FALLBACK_PREMIUM_POOL_MIN)))
    premium_order = _rotated_index_order(premium_size, start)
    if premium_size >= size:
        return premium_order
    tail_order = _rotated_index_order(size - premium_size, start)
    return premium_order + [int(premium_size + idx) for idx in tail_order]


def _prioritized_fallback_indices(
    *,
    fallback_bank: list[dict[str, object]],
    run_id: str,
    user_id: int,
    target: int,
    history_product_ids: set[int],
) -> list[int]:
    del history_product_ids
    selection_step = max(1, _coprime_rotation_step(len(fallback_bank)) - 1)
    return _ordered_fallback_indices(
        fallback_bank_size=len(fallback_bank),
        run_id=run_id,
        user_id=user_id,
        target=target,
        run_multiplier=selection_step,
    )


def _select_fallback_for_user(
    *,
    user_id: int,
    run_id: str,
    current_bundles: list[dict[str, object]],
    fallback_bank: list[dict[str, object]],
    max_bundles_per_user: int,
    history_product_ids: set[int],
) -> list[dict[str, object]]:
    target = max(1, int(max_bundles_per_user))
    current_selected = [dict(bundle) for bundle in current_bundles[:target] if isinstance(bundle, dict)]
    selected = list(current_selected)
    fallback_selected: list[dict[str, object]] = []
    if len(selected) >= target or not fallback_bank:
        return selected[:target]

    pair_seen = {
        _bundle_pair_key(int(bundle.get("item_1_id", 0)), int(bundle.get("item_2_id", 0)))
        for bundle in selected
        if _safe_positive_int(bundle.get("item_1_id")) is not None and _safe_positive_int(bundle.get("item_2_id")) is not None
    }
    item_seen = {
        int(pid)
        for bundle in selected
        for pid in (bundle.get("item_1_id"), bundle.get("item_2_id"))
        if _safe_positive_int(pid) is not None
    }
    category_seen = {
        str(bundle.get("category_key", "")).strip().lower()
        for bundle in selected
        if isinstance(bundle, dict) and str(bundle.get("category_key", "")).strip()
    }
    history_ids = {int(pid) for pid in history_product_ids if int(pid) > 0}
    ordered_indices = _prioritized_fallback_indices(
        fallback_bank=fallback_bank,
        run_id=run_id,
        user_id=int(user_id),
        target=target,
        history_product_ids=history_ids,
    )
    selection_passes = (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    )
    for strict_no_item_overlap, strict_category_diversity in selection_passes:
        if len(selected) >= target:
            break
        for idx in ordered_indices:
            if len(selected) >= target:
                break
            entry = fallback_bank[int(idx)]
            item_1_id = _safe_positive_int(entry.get("item_1_id"))
            item_2_id = _safe_positive_int(entry.get("item_2_id"))
            if item_1_id is None or item_2_id is None or item_1_id == item_2_id:
                continue
            pair_key = _bundle_pair_key(item_1_id, item_2_id)
            if pair_key in pair_seen:
                continue
            if strict_no_item_overlap and (item_1_id in item_seen or item_2_id in item_seen):
                continue
            history_overlap = _history_overlap_count(item_1_id, item_2_id, history_ids)
            category_key = str(entry.get("category_key", "")).strip().lower()
            if strict_category_diversity and category_key and category_key in category_seen and history_overlap <= 0:
                continue
            selection_score = float(_safe_float(entry.get("adjusted_quality_score"), default=_safe_float(entry.get("quality_score"), 0.0)))
            selection_score += _history_support_boost(
                history_overlap,
                first=FALLBACK_HISTORY_PRIMARY_BOOST,
                second=FALLBACK_HISTORY_SECONDARY_BOOST,
            )
            selected_entry = {
                "item_1_id": int(item_1_id),
                "item_2_id": int(item_2_id),
                "bundle_price": float(round(max(0.0, _safe_float(entry.get("bundle_price"), 0.0)), 2)),
                "_selection_score": float(selection_score),
            }
            selected.append(selected_entry)
            fallback_selected.append(selected_entry)
            pair_seen.add(pair_key)
            item_seen.add(int(item_1_id))
            item_seen.add(int(item_2_id))
            if category_key:
                category_seen.add(category_key)
    ordered_fallback = sorted(
        fallback_selected,
        key=lambda entry: (
            -float(entry.get("_selection_score", 0.0)),
            int(entry.get("item_1_id", 0)),
            int(entry.get("item_2_id", 0)),
        ),
    )
    combined = list(current_selected) + ordered_fallback
    return [
        {
            "item_1_id": int(entry["item_1_id"]),
            "item_2_id": int(entry["item_2_id"]),
            "bundle_price": float(round(max(0.0, _safe_float(entry.get("bundle_price"), 0.0)), 2)),
        }
        for entry in combined[:target]
    ]


def _upload_artifact_to_s3_if_configured(path: Path, *, key_env: str, default_key: str, artifact_label: str) -> None:
    bucket = str(os.getenv("QEU_ARTIFACTS_S3_BUCKET", "") or "").strip()
    key = str(os.getenv(key_env, default_key) or "").strip()
    if not bucket or not key:
        LOGGER.info("[batch] S3 upload skipped for %s; bucket/key not configured", artifact_label)
        return

    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for S3 upload but is unavailable") from exc

    started = time.perf_counter()
    LOGGER.info("[batch] uploading %s to s3://%s/%s", artifact_label, bucket, key)
    try:
        boto3.client("s3").upload_file(str(path), bucket, key)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to upload {artifact_label} to s3://{bucket}/{key}") from exc
    LOGGER.info(
        "[batch] uploaded %s to s3://%s/%s duration_sec=%.3f",
        artifact_label,
        bucket,
        key,
        time.perf_counter() - started,
    )


def _upload_final_artifact_to_s3_if_configured(path: Path) -> None:
    _upload_artifact_to_s3_if_configured(
        path,
        key_env="QEU_S3_FINAL_RECOMMENDATIONS_KEY",
        default_key=DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY,
        artifact_label="final recommendations artifact",
    )


def _upload_fallback_bank_artifact_to_s3_if_configured(path: Path) -> None:
    _upload_artifact_to_s3_if_configured(
        path,
        key_env="QEU_S3_FALLBACK_BUNDLE_BANK_KEY",
        default_key=DEFAULT_FALLBACK_BUNDLE_BANK_S3_KEY,
        artifact_label="fallback bundle bank artifact",
    )


def _upload_bundle_ids_artifact_to_s3_if_configured(path: Path) -> None:
    _upload_artifact_to_s3_if_configured(
        path,
        key_env="QEU_S3_BUNDLE_IDS_KEY",
        default_key=DEFAULT_BUNDLE_IDS_S3_KEY,
        artifact_label="bundle id registry artifact",
    )


def materialize_final_recommendations_by_user(
    base_dir: Path | None = None,
    *,
    max_bundles_per_user: int = 3,
    max_users: int | None = None,
    user_selection_mode: str | None = None,
    random_seed: int | None = None,
) -> FinalRecommendationsWriteResult:
    active_base_dir = (base_dir or get_paths().project_root).resolve()
    started = datetime.now(timezone.utc)
    run_id = _latest_run_id(active_base_dir)
    max_users_limit = resolve_final_recommendations_max_users(max_users)
    resolved_selection_mode = resolve_final_recommendations_user_selection_mode(
        "sorted" if user_selection_mode is None else user_selection_mode
    )
    resolved_random_seed = resolve_final_recommendations_random_seed(random_seed)
    fallback_enabled = resolve_fallback_bundle_bank_enabled_from_env()
    fallback_target_size = resolve_fallback_bundle_bank_target_size_from_env()
    fallback_max_size = resolve_fallback_bundle_bank_max_size_from_env()
    fallback_min_score = resolve_fallback_bundle_min_score_from_env()
    if fallback_target_size > fallback_max_size:
        LOGGER.warning(
            "[batch] %s (%d) exceeds %s (%d); clamping target to max",
            FALLBACK_BUNDLE_BANK_TARGET_SIZE_ENV,
            int(fallback_target_size),
            FALLBACK_BUNDLE_BANK_MAX_SIZE_ENV,
            int(fallback_max_size),
        )
        fallback_target_size = int(fallback_max_size)

    LOGGER.info("[batch] materializing final recommendations by user")

    order_pool = load_order_pool(active_base_dir)
    if not order_pool.order_product_ids:
        raise RuntimeError("Cannot materialize final recommendations: order pool is empty")

    orders_df = _load_orders_frame(active_base_dir)
    if orders_df.empty:
        raise RuntimeError("Cannot materialize final recommendations: no usable order rows")

    view = load_bundle_view(active_base_dir)
    bundles_df = view.bundles_df if view.bundles_df is not None else pd.DataFrame()
    if bundles_df.empty:
        raise RuntimeError("Cannot materialize final recommendations: no scored candidate bundles")

    # Candidate user IDs are always built as sorted unique IDs to keep baseline ordering stable.
    all_user_ids = _candidate_user_ids(orders_df)
    user_ids, selection_rule = _select_user_ids(
        all_user_ids,
        max_users_limit=max_users_limit,
        selection_mode=resolved_selection_mode,
        random_seed=resolved_random_seed,
    )
    LOGGER.info(
        "[batch] final recommendation user selection rule=%s mode=%s total=%d selected=%d max_users=%s random_seed=%s",
        selection_rule,
        resolved_selection_mode,
        int(len(all_user_ids)),
        int(len(user_ids)),
        "unlimited" if max_users_limit is None else str(int(max_users_limit)),
        "none" if resolved_random_seed is None else str(int(resolved_random_seed)),
    )

    fallback_bank: list[dict[str, object]] = []
    fallback_bank_meta: dict[str, object]
    if fallback_enabled:
        fallback_bank, fallback_bank_meta = _build_fallback_bundle_bank(
            bundles_df=bundles_df,
            run_id=str(run_id),
            target_size=int(fallback_target_size),
            max_size=int(fallback_max_size),
            min_score=fallback_min_score,
        )
    else:
        fallback_bank_meta = {
            "candidate_count": int(len(bundles_df)),
            "kept_after_filters": 0,
            "selected_count": 0,
            "category_cap": 0,
            "anchor_category_cap": 0,
            "target_size": int(fallback_target_size),
            "max_size": int(fallback_max_size),
            "min_score": None if fallback_min_score is None else float(fallback_min_score),
        }
    fallback_bank_meta["enabled"] = bool(fallback_enabled)
    LOGGER.info(
        "[batch] fallback bank enabled=%s selected=%d kept_after_filters=%d target=%d max=%d min_score=%s",
        bool(fallback_enabled),
        int(fallback_bank_meta.get("selected_count", 0)),
        int(fallback_bank_meta.get("kept_after_filters", 0)),
        int(fallback_bank_meta.get("target_size", 0)),
        int(fallback_bank_meta.get("max_size", 0)),
        "none" if fallback_min_score is None else str(float(fallback_min_score)),
    )
    pair_quality_lookup = _build_pair_quality_lookup(bundles_df)

    profiles: list[PersonProfile] = []
    for user_id in user_ids:
        order_ids = _resolve_order_ids_for_user(user_id, orders_df)
        if not order_ids:
            continue
        profile = _build_profile_from_orders(user_id=user_id, order_ids=order_ids, order_pool=order_pool)
        if profile is None or len(profile.history_product_ids) < MIN_HISTORY_PRODUCTS:
            continue
        profiles.append(profile)
    profile_history_by_user_id: dict[int, set[int]] = {}
    for profile in profiles:
        user_id = _extract_api_user_id(profile.profile_id)
        if user_id is None:
            continue
        profile_history_by_user_id[int(user_id)] = {
            int(pid) for pid in profile.history_product_ids if int(pid) > 0
        }

    recommendations_by_user: dict[int, list[dict[str, object]]] = {int(user_id): [] for user_id in user_ids}
    if profiles:
        raw_recommendations = build_recommendations_for_profiles(
            bundles_df=bundles_df,
            profiles=profiles,
            max_people=len(profiles),
            row_to_record=row_to_record,
            base_dir=active_base_dir,
            run_id=run_id,
            rng_salt="daily-precompute",
        )
    else:
        raw_recommendations = []

    for rec in raw_recommendations:
        if not isinstance(rec, dict):
            continue
        user_id = _extract_api_user_id(rec.get("profile_id"))
        if user_id is None:
            continue
        bundles_raw = rec.get("bundles", [])
        if not isinstance(bundles_raw, list):
            continue
        recommendations_by_user[int(user_id)] = _select_personalized_bundles_for_user(
            bundles_raw=[bundle for bundle in bundles_raw if isinstance(bundle, dict)],
            max_bundles_per_user=int(max_bundles_per_user),
            history_product_ids=profile_history_by_user_id.get(int(user_id), set()),
            pair_quality_lookup=pair_quality_lookup,
        )

    fallback_slots_filled = 0
    fallback_users_touched = 0
    if fallback_enabled:
        for user_id in user_ids:
            current = recommendations_by_user.get(int(user_id), [])
            before = int(len(current))
            filled = _select_fallback_for_user(
                user_id=int(user_id),
                run_id=str(run_id),
                current_bundles=current,
                fallback_bank=fallback_bank,
                max_bundles_per_user=int(max_bundles_per_user),
                history_product_ids=profile_history_by_user_id.get(int(user_id), set()),
            )
            after = int(len(filled))
            if after > before:
                fallback_slots_filled += int(after - before)
                fallback_users_touched += 1
            recommendations_by_user[int(user_id)] = filled

    recommendations_by_user = {
        int(user_id): bundles
        for user_id, bundles in recommendations_by_user.items()
        if isinstance(bundles, list) and bundles
    }

    generated_at = started.isoformat(timespec="seconds")
    fallback_bank_path = _write_fallback_bundle_bank_artifact(
        base_dir=active_base_dir,
        run_id=run_id,
        generated_at=generated_at,
        fallback_bank=fallback_bank,
        bank_meta=fallback_bank_meta,
    )
    _upload_fallback_bank_artifact_to_s3_if_configured(fallback_bank_path)
    bundle_ids_path, new_bundle_id_count, bundle_id_registry_count, bundles_seen_this_run = _upsert_bundle_id_registry(
        base_dir=active_base_dir,
        run_id=run_id,
        generated_at=generated_at,
        recommendations_by_user=recommendations_by_user,
        fallback_bank=fallback_bank,
    )
    _upload_bundle_ids_artifact_to_s3_if_configured(bundle_ids_path)

    payload = {
        "version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "max_bundles_per_user": int(max_bundles_per_user),
        "max_users": int(max_users_limit) if max_users_limit is not None else None,
        "candidate_user_count": int(len(all_user_ids)),
        "selected_user_count": int(len(user_ids)),
        "limited_mode": bool(max_users_limit is not None),
        "user_selection_mode": resolved_selection_mode,
        "user_selection_rule": selection_rule,
        "random_seed": int(resolved_random_seed) if resolved_random_seed is not None else None,
        "profile_count": int(len(profiles)),
        "fallback_bank_enabled": bool(fallback_enabled),
        "fallback_bank_size": int(len(fallback_bank)),
        "fallback_slots_filled": int(fallback_slots_filled),
        "fallback_users_touched": int(fallback_users_touched),
        "fallback_min_score": None if fallback_min_score is None else float(fallback_min_score),
        "fallback_bank_artifact_path": str(fallback_bank_path),
        "bundle_ids_artifact_path": str(bundle_ids_path),
        "bundle_ids_new_count": int(new_bundle_id_count),
        "bundle_ids_total_count": int(bundle_id_registry_count),
        "bundles_seen_this_run": int(bundles_seen_this_run),
        "user_count": int(len(recommendations_by_user)),
        "recommendations_by_user": {
            str(user_id): recommendations_by_user[user_id]
            for user_id in sorted(recommendations_by_user)
        },
    }

    out_path = final_recommendations_artifact_path(active_base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    _upload_final_artifact_to_s3_if_configured(out_path)

    LOGGER.info(
        "[batch] wrote final recommendations path=%s users=%d profiles=%d fallback_bank=%s bundle_ids=%s new_bundle_ids=%d fallback_slots_filled=%d",
        out_path,
        int(len(recommendations_by_user)),
        int(len(profiles)),
        fallback_bank_path,
        bundle_ids_path,
        int(new_bundle_id_count),
        int(fallback_slots_filled),
    )
    return FinalRecommendationsWriteResult(
        path=out_path,
        run_id=run_id,
        profile_count=int(len(profiles)),
        user_count=int(len(recommendations_by_user)),
        generated_at=generated_at,
    )


def _sanitize_bundle_record(bundle: dict[str, object]) -> dict[str, object] | None:
    item_1_id = _safe_positive_int(bundle.get("item_1_id"))
    item_2_id = _safe_positive_int(bundle.get("item_2_id"))
    bundle_price = _safe_float(bundle.get("bundle_price"), default=float("nan"))

    if item_1_id is None or item_2_id is None or item_1_id == item_2_id:
        return None
    if bundle_price != bundle_price:  # NaN
        return None

    return {
        "item_1_id": int(item_1_id),
        "item_2_id": int(item_2_id),
        "bundle_price": float(round(max(0.0, bundle_price), 2)),
    }


def load_final_recommendations_artifact(path: Path) -> FinalRecommendationsLoadResult:
    target = Path(path).resolve()
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Final recommendations artifact missing: {target}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Final recommendations artifact is invalid JSON: {target}") from exc

    if isinstance(payload, dict) and isinstance(payload.get("recommendations_by_user"), dict):
        mapping = payload.get("recommendations_by_user", {})
        run_id = str(payload.get("run_id", "") or "").strip()
        generated_at = str(payload.get("generated_at", "") or "").strip()
    elif isinstance(payload, dict):
        mapping = payload
        run_id = ""
        generated_at = ""
    else:
        raise ValueError("Final recommendations payload must be a JSON object")

    recommendations_by_user: dict[int, list[dict[str, object]]] = {}
    for raw_user_id, bundles_raw in mapping.items():
        user_id = _safe_positive_int(raw_user_id)
        if user_id is None or not isinstance(bundles_raw, list):
            continue
        bundles: list[dict[str, object]] = []
        for bundle in bundles_raw:
            if not isinstance(bundle, dict):
                continue
            clean = _sanitize_bundle_record(bundle)
            if clean is None:
                continue
            bundles.append(clean)
        if bundles:
            recommendations_by_user[user_id] = bundles

    if not recommendations_by_user and isinstance(mapping, dict) and mapping:
        raise ValueError("Final recommendations payload has no valid user recommendations")

    return FinalRecommendationsLoadResult(
        path=target,
        run_id=run_id,
        generated_at=generated_at,
        recommendations_by_user=recommendations_by_user,
    )
