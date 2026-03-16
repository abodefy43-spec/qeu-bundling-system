"""Batch materialization and runtime loading for final API recommendations."""

from __future__ import annotations

import json
import logging
import os
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


def _latest_run_id(base_dir: Path) -> str:
    payload = read_latest_manifest(base_dir=base_dir)
    return str(payload.get("run_id", "") or "").strip()


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


def _upload_final_artifact_to_s3_if_configured(path: Path) -> None:
    bucket = str(os.getenv("QEU_ARTIFACTS_S3_BUCKET", "") or "").strip()
    key = str(os.getenv("QEU_S3_FINAL_RECOMMENDATIONS_KEY", DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY) or "").strip()
    if not bucket or not key:
        LOGGER.info("[batch] QEU_ARTIFACTS_S3_BUCKET or QEU_S3_FINAL_RECOMMENDATIONS_KEY is unset; skipping S3 upload")
        return

    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("boto3 is required for S3 upload but is unavailable") from exc

    started = time.perf_counter()
    LOGGER.info("[batch] uploading final recommendations artifact to s3://%s/%s", bucket, key)
    try:
        boto3.client("s3").upload_file(str(path), bucket, key)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to upload final recommendations artifact to s3://{bucket}/{key}") from exc
    LOGGER.info(
        "[batch] uploaded final recommendations artifact to s3://%s/%s duration_sec=%.3f",
        bucket,
        key,
        time.perf_counter() - started,
    )


def materialize_final_recommendations_by_user(
    base_dir: Path | None = None,
    *,
    max_bundles_per_user: int = 3,
) -> FinalRecommendationsWriteResult:
    active_base_dir = (base_dir or get_paths().project_root).resolve()
    started = datetime.now(timezone.utc)
    run_id = _latest_run_id(active_base_dir)

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

    user_ids = _candidate_user_ids(orders_df)
    profiles: list[PersonProfile] = []
    for user_id in user_ids:
        order_ids = _resolve_order_ids_for_user(user_id, orders_df)
        if not order_ids:
            continue
        profile = _build_profile_from_orders(user_id=user_id, order_ids=order_ids, order_pool=order_pool)
        if profile is None or len(profile.history_product_ids) < MIN_HISTORY_PRODUCTS:
            continue
        profiles.append(profile)

    recommendations_by_user: dict[int, list[dict[str, object]]] = {}
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
        if bundles:
            recommendations_by_user[int(user_id)] = bundles

    generated_at = started.isoformat(timespec="seconds")
    payload = {
        "version": 1,
        "generated_at": generated_at,
        "run_id": run_id,
        "max_bundles_per_user": int(max_bundles_per_user),
        "profile_count": int(len(profiles)),
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
        "[batch] wrote final recommendations path=%s users=%d profiles=%d",
        out_path,
        int(len(recommendations_by_user)),
        int(len(profiles)),
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
