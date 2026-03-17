"""Batch materialization and runtime loading for final API recommendations."""

from __future__ import annotations

import csv
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

    if _is_truthy_numeric(row.get("only_staples_overlap")) and recipe_compat < 0.85 and pair_count < 2.0:
        return False
    return True


def _fallback_category_key(row: dict[str, object]) -> tuple[str, str]:
    a = str(row.get("category_a", "") or "").strip().lower() or "unknown"
    b = str(row.get("category_b", "") or "").strip().lower() or "unknown"
    if a <= b:
        return (a, b)
    return (b, a)


def _build_fallback_bundle_bank(
    bundles_df: pd.DataFrame,
    *,
    target_size: int,
    max_size: int,
    min_score: float | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if bundles_df.empty:
        return [], {"candidate_count": 0, "kept_after_filters": 0, "selected_count": 0, "category_cap": 0}

    requested_target_size = max(1, int(target_size))
    requested_max_size = max(1, int(max_size))
    effective_max_size = int(requested_max_size)
    if requested_target_size > effective_max_size:
        requested_target_size = int(effective_max_size)

    category_cap = max(10, int(requested_target_size // 20))
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
        category_key = _fallback_category_key(row)
        candidates.append(
            {
                "item_1_id": int(api_record["item_1_id"]),
                "item_2_id": int(api_record["item_2_id"]),
                "bundle_price": float(api_record["bundle_price"]),
                "quality_score": float(round(quality_score, 6)),
                "category_key": f"{category_key[0]}|{category_key[1]}",
                "final_score": float(_safe_float(row.get("new_final_score"), default=_safe_float(row.get("final_score"), 0.0))),
                "pair_count": float(_safe_float(row.get("pair_count"), 0.0)),
                "recipe_compat_score": float(_safe_float(row.get("recipe_compat_score"), default=_safe_float(row.get("recipe_score_norm"), 0.0))),
            }
        )

    candidates.sort(
        key=lambda entry: (
            -float(entry.get("quality_score", 0.0)),
            -float(entry.get("final_score", 0.0)),
            -float(entry.get("pair_count", 0.0)),
            str(entry.get("category_key", "")),
            int(entry.get("item_1_id", 0)),
            int(entry.get("item_2_id", 0)),
        )
    )

    selected: list[dict[str, object]] = []
    category_counts: dict[str, int] = {}
    selected_pair_keys: set[tuple[int, int]] = set()
    for entry in candidates:
        category_key = str(entry.get("category_key", "unknown|unknown"))
        if int(category_counts.get(category_key, 0)) >= category_cap:
            continue
        category_counts[category_key] = int(category_counts.get(category_key, 0)) + 1
        selected.append(entry)
        selected_pair_keys.add(_bundle_pair_key(int(entry.get("item_1_id", 0)), int(entry.get("item_2_id", 0))))
        if len(selected) >= requested_target_size:
            break

    if len(selected) < min(requested_target_size, effective_max_size):
        for entry in candidates:
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
        "target_size": int(requested_target_size),
        "max_size": int(effective_max_size),
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
            }
            for entry in fallback_bank
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    return path


def _select_fallback_for_user(
    *,
    user_id: int,
    current_bundles: list[dict[str, object]],
    fallback_bank: list[dict[str, object]],
    max_bundles_per_user: int,
) -> list[dict[str, object]]:
    target = max(1, int(max_bundles_per_user))
    selected = [dict(bundle) for bundle in current_bundles[:target] if isinstance(bundle, dict)]
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

    start = int(user_id) % len(fallback_bank)
    for strict_no_item_overlap in (True, False):
        if len(selected) >= target:
            break
        for offset in range(len(fallback_bank)):
            entry = fallback_bank[(start + offset) % len(fallback_bank)]
            item_1_id = _safe_positive_int(entry.get("item_1_id"))
            item_2_id = _safe_positive_int(entry.get("item_2_id"))
            if item_1_id is None or item_2_id is None or item_1_id == item_2_id:
                continue
            pair_key = _bundle_pair_key(item_1_id, item_2_id)
            if pair_key in pair_seen:
                continue
            if strict_no_item_overlap and (item_1_id in item_seen or item_2_id in item_seen):
                continue

            selected.append(
                {
                    "item_1_id": int(item_1_id),
                    "item_2_id": int(item_2_id),
                    "bundle_price": float(round(max(0.0, _safe_float(entry.get("bundle_price"), 0.0)), 2)),
                }
            )
            pair_seen.add(pair_key)
            item_seen.add(int(item_1_id))
            item_seen.add(int(item_2_id))
            if len(selected) >= target:
                break
    return selected[:target]


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

    profiles: list[PersonProfile] = []
    for user_id in user_ids:
        order_ids = _resolve_order_ids_for_user(user_id, orders_df)
        if not order_ids:
            continue
        profile = _build_profile_from_orders(user_id=user_id, order_ids=order_ids, order_pool=order_pool)
        if profile is None or len(profile.history_product_ids) < MIN_HISTORY_PRODUCTS:
            continue
        profiles.append(profile)

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
        bundles: list[dict[str, object]] = []
        for bundle in bundles_raw:
            if not isinstance(bundle, dict):
                continue
            api_record = _bundle_to_api_record(bundle)
            if api_record is None:
                continue
            bundles.append(api_record)
            if len(bundles) >= int(max_bundles_per_user):
                break
        recommendations_by_user[int(user_id)] = bundles

    fallback_slots_filled = 0
    fallback_users_touched = 0
    if fallback_enabled:
        for user_id in user_ids:
            current = recommendations_by_user.get(int(user_id), [])
            before = int(len(current))
            filled = _select_fallback_for_user(
                user_id=int(user_id),
                current_bundles=current,
                fallback_bank=fallback_bank,
                max_bundles_per_user=int(max_bundles_per_user),
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
