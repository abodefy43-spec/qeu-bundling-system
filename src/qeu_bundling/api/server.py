"""Minimal JSON-only API server for customer bundle recommendations."""

from __future__ import annotations

import argparse
import os
import socket
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.run_manifest import read_latest_manifest
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.person_predictions import (
    OrderPool,
    PersonProfile,
    build_recommendations_for_profiles,
    load_order_pool,
)

MAX_BUNDLES = 3
MIN_HISTORY_PRODUCTS = 2
MAX_ORDER_IDS_PER_PROFILE = 6
USER_ID_COLUMNS = ("user_id", "customer_id", "customer_no", "partner_id")


class CustomerNotFoundError(RuntimeError):
    """Raised when no order history can be resolved for a requested user."""


class InsufficientHistoryError(RuntimeError):
    """Raised when a user has too little history to score bundles."""


app = Flask(__name__)


def _project_root() -> Path:
    raw = str(os.environ.get("QEU_PROJECT_ROOT", "") or "").strip()
    if raw:
        return Path(raw).resolve()
    return get_paths().project_root


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

    data = orders.copy()
    data["order_id"] = pd.to_numeric(data["order_id"], errors="coerce")
    data = data.dropna(subset=["order_id"])
    if data.empty:
        return pd.DataFrame()
    data["order_id"] = data["order_id"].astype("int64")

    for column in USER_ID_COLUMNS:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

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


def _recommendation_records_for_user(user_id: int) -> list[dict[str, object]]:
    base_dir = _project_root()
    order_pool = load_order_pool(base_dir)
    orders_df = _load_orders_frame(base_dir)
    order_ids = _resolve_order_ids_for_user(user_id, orders_df)
    if not order_ids:
        raise CustomerNotFoundError

    profile = _build_profile_from_orders(user_id=user_id, order_ids=order_ids, order_pool=order_pool)
    if profile is None or len(profile.history_product_ids) < MIN_HISTORY_PRODUCTS:
        raise InsufficientHistoryError

    view = load_bundle_view(base_dir)
    bundles_df = view.bundles_df if view.bundles_df is not None else pd.DataFrame()
    if bundles_df.empty:
        raise InsufficientHistoryError

    recommendations = build_recommendations_for_profiles(
        bundles_df=bundles_df,
        profiles=[profile],
        max_people=1,
        row_to_record=row_to_record,
        base_dir=base_dir,
        run_id=_latest_run_id(base_dir),
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
    if sale_a >= sale_b:
        free_item_id = int(item_2_id)
        original_paid_price = float(sale_a)
        final_paid_price = max(
            0.0,
            _safe_float(bundle.get("price_after_discount_a"), default=_safe_float(bundle.get("price_after_a_sar"), 0.0)),
        )
    else:
        free_item_id = int(item_1_id)
        original_paid_price = float(sale_b)
        final_paid_price = max(
            0.0,
            _safe_float(bundle.get("price_after_discount_b"), default=_safe_float(bundle.get("price_after_b_sar"), 0.0)),
        )

    discount_amount = round(max(0.0, float(original_paid_price - final_paid_price)), 2)
    return {
        "item_1_id": int(item_1_id),
        "item_2_id": int(item_2_id),
        "free_item_id": int(free_item_id),
        "paid_item_discount_amount": float(discount_amount),
    }


def _error_payload(user_id: int | None, code: str) -> tuple[dict[str, object], int]:
    status = 500
    if code == "customer_not_found":
        status = 404
    elif code == "insufficient_history":
        status = 422
    return {"user_id": user_id, "bundles": [], "error": code}, status


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.post("/api/recommendations/by-customer")
def recommendations_by_customer():
    payload = request.get_json(silent=True) or {}
    user_id = _parse_user_id(payload.get("user_id"))
    if user_id is None:
        body, status = _error_payload(None, "customer_not_found")
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
            return jsonify(body), status
        return jsonify({"user_id": int(user_id), "bundles": bundles})
    except CustomerNotFoundError:
        body, status = _error_payload(user_id, "customer_not_found")
        return jsonify(body), status
    except InsufficientHistoryError:
        body, status = _error_payload(user_id, "insufficient_history")
        return jsonify(body), status
    except Exception:
        body, status = _error_payload(user_id, "recommendation_failed")
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

