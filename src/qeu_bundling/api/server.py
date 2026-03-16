"""Minimal JSON-only API server for customer bundle recommendations."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock

import pandas as pd
from flask import Flask, jsonify, request

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.pricing import FIXED_MARGIN_DISCOUNT_PCT, margin_discounted_sale_price
from qeu_bundling.core.run_manifest import read_latest_manifest
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.person_predictions import (
    OrderPool,
    PersonProfile,
    build_recommendations_for_profiles,
    load_personalization_context,
    load_order_pool,
)

MAX_BUNDLES = 3
MIN_HISTORY_PRODUCTS = 2
MAX_ORDER_IDS_PER_PROFILE = 6
USER_ID_COLUMNS = ("user_id", "customer_id", "customer_no", "partner_id")

DEFAULT_S3_FILTERED_ORDERS_KEY = "processed/filtered_orders.pkl"
DEFAULT_S3_SCORED_CANDIDATES_KEY = "output/person_candidates_scored.csv"
DEFAULT_S3_CANDIDATE_PAIRS_KEY = "processed/candidates/person_candidate_pairs.csv"


LOG_LEVEL = str(os.getenv("QEU_API_LOG_LEVEL", "INFO") or "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
LOGGER = logging.getLogger("qeu_bundling.api.server")
LOGGER.setLevel(LOG_LEVEL)


class CustomerNotFoundError(RuntimeError):
    """Raised when no order history can be resolved for a requested user."""


class InsufficientHistoryError(RuntimeError):
    """Raised when a user has too little history to score bundles."""


class ServiceNotReadyError(RuntimeError):
    """Raised when recommendation serving state has not been initialized."""


@dataclass
class ServingRuntimeState:
    ready: bool = False
    error: str = ""
    initialized_at: str = ""
    base_dir: Path | None = None
    run_id: str = ""
    artifact_meta: dict[str, dict[str, object]] = field(default_factory=dict)
    order_pool: OrderPool | None = None
    orders_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    bundles_df: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class ServingAssets:
    base_dir: Path
    run_id: str
    order_pool: OrderPool
    orders_df: pd.DataFrame
    bundles_df: pd.DataFrame


app = Flask(__name__)
SERVING_STATE = ServingRuntimeState()
SERVING_STATE_LOCK = Lock()


def _project_root() -> Path:
    raw = str(os.environ.get("QEU_PROJECT_ROOT", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    return get_paths().project_root


def _env_str(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default).strip()


def _artifact_specs(base_dir: Path) -> list[tuple[str, Path, str]]:
    paths = get_paths(project_root=base_dir)
    return [
        (
            "filtered_orders",
            paths.data_processed_dir / "filtered_orders.pkl",
            _env_str("QEU_S3_FILTERED_ORDERS_KEY", DEFAULT_S3_FILTERED_ORDERS_KEY),
        ),
        (
            "person_candidates_scored",
            paths.output_dir / "person_candidates_scored.csv",
            _env_str("QEU_S3_SCORED_CANDIDATES_KEY", DEFAULT_S3_SCORED_CANDIDATES_KEY),
        ),
        (
            "person_candidate_pairs",
            paths.data_processed_candidates_dir / "person_candidate_pairs.csv",
            _env_str("QEU_S3_CANDIDATE_PAIRS_KEY", DEFAULT_S3_CANDIDATE_PAIRS_KEY),
        ),
    ]


def _download_file_from_s3(bucket: str, key: str, target: Path, artifact_name: str) -> bool:
    try:
        import boto3  # type: ignore
    except Exception as exc:  # pragma: no cover
        LOGGER.error(
            "[api] boto3 unavailable; cannot download artifact=%s from s3://%s/%s: %s",
            artifact_name,
            bucket,
            key,
            exc,
        )
        return False

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    started = time.perf_counter()
    LOGGER.info(
        "[api] downloading artifact=%s s3://%s/%s -> %s",
        artifact_name,
        bucket,
        key,
        target,
    )
    try:
        boto3.client("s3").download_file(bucket, key, str(partial))
        partial.replace(target)
        duration_sec = time.perf_counter() - started
        LOGGER.info(
            "[api] downloaded artifact=%s bytes=%d duration_sec=%.3f",
            artifact_name,
            int(target.stat().st_size) if target.exists() else 0,
            duration_sec,
        )
        return True
    except Exception as exc:  # pragma: no cover
        LOGGER.exception(
            "[api] failed to download artifact=%s from s3://%s/%s: %s",
            artifact_name,
            bucket,
            key,
            exc,
        )
        try:
            if partial.exists():
                partial.unlink()
        except OSError:
            pass
        return False


def _bootstrap_runtime_artifacts_from_s3_once(base_dir_str: str) -> None:
    base_dir = Path(base_dir_str).resolve()
    bucket = _env_str("QEU_ARTIFACTS_S3_BUCKET")
    if not bucket:
        LOGGER.info("[api] QEU_ARTIFACTS_S3_BUCKET is unset; skipping S3 bootstrap")
        return

    for artifact_name, target, key in _artifact_specs(base_dir):
        if target.exists() or not key:
            continue
        _download_file_from_s3(
            bucket=bucket,
            key=key,
            target=target,
            artifact_name=artifact_name,
        )


def _collect_artifact_meta(base_dir: Path) -> dict[str, dict[str, object]]:
    meta: dict[str, dict[str, object]] = {}
    for artifact_name, path, key in _artifact_specs(base_dir):
        exists = bool(path.exists())
        meta[artifact_name] = {
            "path": str(path),
            "s3_key": key,
            "exists": exists,
            "size_bytes": int(path.stat().st_size) if exists else 0,
        }
    return meta


def _assert_required_artifacts_present(artifact_meta: dict[str, dict[str, object]]) -> None:
    missing = [
        f"{artifact_name} ({meta.get('path', '')})"
        for artifact_name, meta in artifact_meta.items()
        if not bool(meta.get("exists"))
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required serving artifacts: {joined}")


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


def _parse_user_id(value: object) -> int | None:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return None
    user_id = int(num)
    if user_id <= 0:
        return None
    return user_id


@lru_cache(maxsize=4)
def _load_orders_frame_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns
    try:
        orders = pd.read_pickle(path_str)
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


def _load_orders_frame(base_dir: Path) -> pd.DataFrame:
    orders_path = get_paths(project_root=base_dir).data_processed_dir / "filtered_orders.pkl"
    if not orders_path.exists():
        return pd.DataFrame()
    return _load_orders_frame_cached(str(orders_path.resolve()), int(orders_path.stat().st_mtime_ns))


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

    # Fallback: if no explicit customer column exists, allow direct order_id lookup.
    if not user_order_ids and "order_id" in orders_df.columns:
        if bool((orders_df["order_id"] == int(user_id)).any()):
            user_order_ids = [int(user_id)]

    if not user_order_ids:
        return []

    # Keep profile size bounded for predictable API latency.
    unique_ids = sorted(set(int(oid) for oid in user_order_ids if int(oid) > 0))
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


def _latest_run_id(base_dir: Path) -> str:
    payload = read_latest_manifest(base_dir=base_dir)
    return str(payload.get("run_id", "") or "").strip()


def _probe_profile_for_warmup(order_pool: OrderPool) -> PersonProfile | None:
    candidate_order_ids = order_pool.preferred_order_ids or order_pool.fallback_order_ids
    for order_id in candidate_order_ids:
        profile = _build_profile_from_orders(
            user_id=0,
            order_ids=[int(order_id)],
            order_pool=order_pool,
        )
        if profile is not None and len(profile.history_product_ids) >= MIN_HISTORY_PRODUCTS:
            return profile
    return None


def _warm_recommendation_engine(base_dir: Path, run_id: str, order_pool: OrderPool, bundles_df: pd.DataFrame) -> None:
    LOGGER.info("[api] warming recommendation engine for readiness")
    load_personalization_context(base_dir)
    probe = _probe_profile_for_warmup(order_pool)
    if probe is None:
        LOGGER.warning("[api] warmup probe skipped because no suitable profile was found")
        return

    build_recommendations_for_profiles(
        bundles_df=bundles_df,
        profiles=[probe],
        max_people=1,
        row_to_record=row_to_record,
        base_dir=base_dir,
        run_id=run_id,
        rng_salt="api-readiness-warmup",
    )
    LOGGER.info("[api] recommendation engine warmup completed")


def _state_payload() -> dict[str, object]:
    with SERVING_STATE_LOCK:
        return {
            "ready": bool(SERVING_STATE.ready),
            "error": str(SERVING_STATE.error),
            "initialized_at": str(SERVING_STATE.initialized_at),
            "run_id": str(SERVING_STATE.run_id),
            "artifacts": {name: dict(meta) for name, meta in SERVING_STATE.artifact_meta.items()},
            "orders_rows": int(len(SERVING_STATE.orders_df.index)) if not SERVING_STATE.orders_df.empty else 0,
            "bundles_rows": int(len(SERVING_STATE.bundles_df.index)) if not SERVING_STATE.bundles_df.empty else 0,
        }


def _initialize_serving_state(force_reload: bool = False) -> None:
    base_dir = _project_root()
    with SERVING_STATE_LOCK:
        if (
            SERVING_STATE.ready
            and not force_reload
            and SERVING_STATE.base_dir is not None
            and SERVING_STATE.base_dir == base_dir
        ):
            return

        started = time.perf_counter()
        artifact_meta: dict[str, dict[str, object]] = {}
        try:
            LOGGER.info("[api] initializing serving state at base_dir=%s", base_dir)
            _bootstrap_runtime_artifacts_from_s3_once(str(base_dir))
            artifact_meta = _collect_artifact_meta(base_dir)
            _assert_required_artifacts_present(artifact_meta)

            order_pool = load_order_pool(base_dir)
            if not order_pool.order_product_ids:
                raise RuntimeError("Order pool is empty after loading filtered_orders.pkl")

            orders_df = _load_orders_frame(base_dir)
            if orders_df.empty:
                raise RuntimeError("No usable order rows found in filtered_orders.pkl")

            view = load_bundle_view(base_dir)
            bundles_df = view.bundles_df if view.bundles_df is not None else pd.DataFrame()
            if bundles_df.empty:
                raise RuntimeError("No bundle candidates found in person_candidates_scored.csv")

            run_id = _latest_run_id(base_dir)
            _warm_recommendation_engine(
                base_dir=base_dir,
                run_id=run_id,
                order_pool=order_pool,
                bundles_df=bundles_df,
            )

            SERVING_STATE.ready = True
            SERVING_STATE.error = ""
            SERVING_STATE.initialized_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            SERVING_STATE.base_dir = base_dir
            SERVING_STATE.run_id = run_id
            SERVING_STATE.artifact_meta = artifact_meta
            SERVING_STATE.order_pool = order_pool
            SERVING_STATE.orders_df = orders_df
            SERVING_STATE.bundles_df = bundles_df

            LOGGER.info(
                "[api] serving state ready duration_sec=%.3f orders_rows=%d bundles_rows=%d run_id=%s",
                time.perf_counter() - started,
                int(len(orders_df.index)),
                int(len(bundles_df.index)),
                SERVING_STATE.run_id,
            )
        except Exception as exc:
            SERVING_STATE.ready = False
            SERVING_STATE.error = str(exc)
            SERVING_STATE.initialized_at = ""
            SERVING_STATE.base_dir = base_dir
            SERVING_STATE.run_id = ""
            SERVING_STATE.artifact_meta = artifact_meta
            SERVING_STATE.order_pool = None
            SERVING_STATE.orders_df = pd.DataFrame()
            SERVING_STATE.bundles_df = pd.DataFrame()
            LOGGER.exception("[api] serving state initialization failed: %s", exc)
            raise


def _get_serving_assets() -> ServingAssets | None:
    with SERVING_STATE_LOCK:
        if not SERVING_STATE.ready:
            return None
        if SERVING_STATE.base_dir is None or SERVING_STATE.order_pool is None:
            return None
        return ServingAssets(
            base_dir=SERVING_STATE.base_dir,
            run_id=SERVING_STATE.run_id,
            order_pool=SERVING_STATE.order_pool,
            orders_df=SERVING_STATE.orders_df,
            bundles_df=SERVING_STATE.bundles_df,
        )


def _recommendation_records_for_user(user_id: int) -> list[dict[str, object]]:
    assets = _get_serving_assets()
    if assets is None:
        raise ServiceNotReadyError("serving_state_not_ready")

    order_ids = _resolve_order_ids_for_user(user_id, assets.orders_df)
    if not order_ids:
        raise CustomerNotFoundError

    profile = _build_profile_from_orders(
        user_id=user_id,
        order_ids=order_ids,
        order_pool=assets.order_pool,
    )
    if profile is None or len(profile.history_product_ids) < MIN_HISTORY_PRODUCTS:
        raise InsufficientHistoryError

    bundles_df = assets.bundles_df
    if bundles_df.empty:
        raise InsufficientHistoryError

    recommendations = build_recommendations_for_profiles(
        bundles_df=bundles_df,
        profiles=[profile],
        max_people=1,
        row_to_record=row_to_record,
        base_dir=assets.base_dir,
        run_id=assets.run_id,
    )
    if not recommendations:
        raise InsufficientHistoryError

    first = recommendations[0] if isinstance(recommendations[0], dict) else {}
    bundles = first.get("bundles", [])
    if not isinstance(bundles, list) or not bundles:
        raise InsufficientHistoryError

    return [bundle for bundle in bundles if isinstance(bundle, dict)]


def _bundle_to_api_record(bundle: dict[str, object]) -> dict[str, object] | None:
    item_1_id = int(pd.to_numeric(bundle.get("product_a", -1), errors="coerce") or -1)
    item_2_id = int(pd.to_numeric(bundle.get("product_b", -1), errors="coerce") or -1)
    if item_1_id <= 0 or item_2_id <= 0 or item_1_id == item_2_id:
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


def _error_payload(user_id: int | None, code: str) -> tuple[dict[str, object], int]:
    status = 500
    if code == "customer_not_found":
        status = 404
    elif code == "insufficient_history":
        status = 422
    elif code == "service_not_ready":
        status = 503
    return {"user_id": user_id, "bundles": [], "error": code}, status


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/readyz")
def readyz():
    try:
        _initialize_serving_state(force_reload=False)
    except Exception:
        payload = _state_payload()
        payload["status"] = "not_ready"
        return jsonify(payload), 503

    payload = _state_payload()
    payload["status"] = "ready" if bool(payload.get("ready")) else "not_ready"
    return jsonify(payload), 200 if payload["status"] == "ready" else 503


@app.post("/api/recommendations/by-customer")
def recommendations_by_customer():
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    user_id = _parse_user_id(payload.get("user_id"))
    if user_id is None:
        body, status = _error_payload(None, "customer_not_found")
        LOGGER.info(
            "[api] recommendations request rejected reason=invalid_user_id duration_sec=%.3f",
            time.perf_counter() - started,
        )
        return jsonify(body), status

    try:
        raw_bundles = _recommendation_records_for_user(user_id=user_id)
        bundles: list[dict[str, object]] = []
        for bundle in raw_bundles:
            api_bundle = _bundle_to_api_record(bundle)
            if api_bundle is None:
                continue
            bundles.append(api_bundle)
            if len(bundles) >= MAX_BUNDLES:
                break
        if not bundles:
            body, status = _error_payload(user_id, "insufficient_history")
            LOGGER.info(
                "[api] recommendations result user_id=%d status=%d bundles=0 duration_sec=%.3f",
                int(user_id),
                status,
                time.perf_counter() - started,
            )
            return jsonify(body), status
        LOGGER.info(
            "[api] recommendations result user_id=%d status=200 bundles=%d duration_sec=%.3f",
            int(user_id),
            len(bundles),
            time.perf_counter() - started,
        )
        return jsonify({"user_id": int(user_id), "bundles": bundles})
    except ServiceNotReadyError:
        body, status = _error_payload(user_id, "service_not_ready")
        LOGGER.warning(
            "[api] recommendations request blocked user_id=%d reason=service_not_ready duration_sec=%.3f",
            int(user_id),
            time.perf_counter() - started,
        )
        return jsonify(body), status
    except CustomerNotFoundError:
        body, status = _error_payload(user_id, "customer_not_found")
        LOGGER.info(
            "[api] recommendations result user_id=%d status=%d reason=customer_not_found duration_sec=%.3f",
            int(user_id),
            status,
            time.perf_counter() - started,
        )
        return jsonify(body), status
    except InsufficientHistoryError:
        body, status = _error_payload(user_id, "insufficient_history")
        LOGGER.info(
            "[api] recommendations result user_id=%d status=%d reason=insufficient_history duration_sec=%.3f",
            int(user_id),
            status,
            time.perf_counter() - started,
        )
        return jsonify(body), status
    except Exception:
        body, status = _error_payload(user_id, "recommendation_failed")
        LOGGER.exception(
            "[api] recommendations request failed user_id=%d duration_sec=%.3f",
            int(user_id),
            time.perf_counter() - started,
        )
        return jsonify(body), status


def _detect_lan_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            ip = str(sock.getsockname()[0]).strip()
            if ip and ip != "127.0.0.1":
                return ip
        finally:
            sock.close()
    except OSError:
        pass

    try:
        ip = str(socket.gethostbyname(socket.gethostname())).strip()
    except OSError:
        ip = ""
    return ip if ip and ip != "127.0.0.1" else ""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qeu-api-server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5001)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    host = str(args.host).strip() or "0.0.0.0"
    port = int(args.port)
    print("Starting QEU API server...")
    print(f"Listening on {host}:{port}")
    if host == "0.0.0.0":
        lan_ip = _detect_lan_ip()
        if lan_ip:
            print(f"LAN URL: http://{lan_ip}:{port}")
    app.run(host=host, port=port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
