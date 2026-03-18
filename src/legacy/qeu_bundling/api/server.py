"""Minimal JSON-only API server for precomputed customer bundle recommendations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.final_recommendations import (
    BUNDLE_IDS_ARTIFACT,
    DEFAULT_BUNDLE_IDS_S3_KEY,
    DEFAULT_FALLBACK_BUNDLE_BANK_S3_KEY,
    DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY,
    FALLBACK_BUNDLE_BANK_ARTIFACT,
    FINAL_RECOMMENDATIONS_ARTIFACT,
    load_final_recommendations_artifact,
)

MAX_BUNDLES = 3
DEFAULT_S3_FINAL_RECOMMENDATIONS_KEY = DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY
RANDOM_USER_TOKENS = {"random", "rand", "any", "*"}

LOG_LEVEL = str(os.getenv("QEU_API_LOG_LEVEL", "INFO") or "INFO").upper()
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
LOGGER = logging.getLogger("qeu_bundling.api.server")
LOGGER.setLevel(LOG_LEVEL)


class CustomerNotFoundError(RuntimeError):
    """Raised when no precomputed recommendations exist for the user."""


class InsufficientHistoryError(RuntimeError):
    """Raised when a user has no usable precomputed bundles."""


class ServiceNotReadyError(RuntimeError):
    """Raised when recommendation serving state has not been initialized."""


class FallbackUnavailableError(RuntimeError):
    """Raised when no fallback bundle bank is available for degraded serving."""


@dataclass
class ServingRuntimeState:
    ready: bool = False
    error: str = ""
    initialized_at: str = ""
    base_dir: Path | None = None
    run_id: str = ""
    artifact_meta: dict[str, dict[str, object]] = field(default_factory=dict)
    recommendations_by_user: dict[int, list[dict[str, object]]] = field(default_factory=dict)
    fallback_bundle_bank: list[dict[str, object]] = field(default_factory=list)
    bundle_id_lookup: dict[tuple[int, int], str] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingAssets:
    base_dir: Path
    run_id: str
    recommendations_by_user: dict[int, list[dict[str, object]]]
    fallback_bundle_bank: list[dict[str, object]]
    bundle_id_lookup: dict[tuple[int, int], str] = field(default_factory=dict)


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


def _env_bool(name: str, default: bool) -> bool:
    raw = _env_str(name, "1" if default else "0").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _artifact_specs(base_dir: Path) -> list[tuple[str, Path, str]]:
    paths = get_paths(project_root=base_dir)
    return [
        (
            "final_recommendations_by_user",
            paths.output_dir / FINAL_RECOMMENDATIONS_ARTIFACT,
            _env_str("QEU_S3_FINAL_RECOMMENDATIONS_KEY", DEFAULT_S3_FINAL_RECOMMENDATIONS_KEY),
        ),
        (
            "fallback_bundle_bank",
            paths.output_dir / FALLBACK_BUNDLE_BANK_ARTIFACT,
            _env_str("QEU_S3_FALLBACK_BUNDLE_BANK_KEY", DEFAULT_FALLBACK_BUNDLE_BANK_S3_KEY),
        ),
        (
            "bundle_ids",
            paths.output_dir / BUNDLE_IDS_ARTIFACT,
            _env_str("QEU_S3_BUNDLE_IDS_KEY", DEFAULT_BUNDLE_IDS_S3_KEY),
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
    required_artifacts = {"final_recommendations_by_user"}
    missing = [
        f"{artifact_name} ({meta.get('path', '')})"
        for artifact_name, meta in artifact_meta.items()
        if artifact_name in required_artifacts
        if not bool(meta.get("exists"))
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing required serving artifacts: {joined}")


def _parse_user_id(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        value = raw
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or not numeric.is_integer():
        return None
    user_id = int(numeric)
    if user_id <= 0:
        return None
    return user_id


def _wants_random_user(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False

    random_user_flag = payload.get("random_user")
    if isinstance(random_user_flag, bool):
        if random_user_flag:
            return True
    elif isinstance(random_user_flag, str):
        if random_user_flag.strip().lower() in {"1", "true", "yes", "on"}:
            return True

    mode_raw = payload.get("mode")
    if isinstance(mode_raw, str) and mode_raw.strip().lower() in {"random", "any"}:
        return True

    user_id_raw = payload.get("user_id")
    if isinstance(user_id_raw, str) and user_id_raw.strip().lower() in RANDOM_USER_TOKENS:
        return True
    return False


def _state_payload() -> dict[str, object]:
    with SERVING_STATE_LOCK:
        return {
            "ready": bool(SERVING_STATE.ready),
            "error": str(SERVING_STATE.error),
            "initialized_at": str(SERVING_STATE.initialized_at),
            "run_id": str(SERVING_STATE.run_id),
            "artifacts": {name: dict(meta) for name, meta in SERVING_STATE.artifact_meta.items()},
            "loaded_user_count": int(len(SERVING_STATE.recommendations_by_user)),
            "fallback_bundle_count": int(len(SERVING_STATE.fallback_bundle_bank)),
            "bundle_id_count": int(len(SERVING_STATE.bundle_id_lookup)),
        }


def _safe_positive_int(value: object) -> int | None:
    try:
        out = int(float(value))
    except (TypeError, ValueError):
        return None
    if out <= 0:
        return None
    return int(out)


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:
        return None
    return float(out)


def _pair_key(item_1_id: int, item_2_id: int) -> tuple[int, int]:
    a = int(item_1_id)
    b = int(item_2_id)
    if a <= b:
        return (a, b)
    return (b, a)


def _with_bundle_id(bundle: dict[str, object], bundle_id_lookup: dict[tuple[int, int], str]) -> dict[str, object]:
    item_1_id = _safe_positive_int(bundle.get("item_1_id"))
    item_2_id = _safe_positive_int(bundle.get("item_2_id"))
    if item_1_id is None or item_2_id is None:
        return dict(bundle)
    out = dict(bundle)
    bundle_id = str(bundle_id_lookup.get(_pair_key(item_1_id, item_2_id), "")).strip()
    if bundle_id:
        out["bundle_id"] = bundle_id
    return out


def _load_bundle_id_lookup(path: Path) -> dict[tuple[int, int], str]:
    lookup: dict[tuple[int, int], str] = {}
    if not path.exists():
        return lookup
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                item_1_id = _safe_positive_int(row.get("item_1_id"))
                item_2_id = _safe_positive_int(row.get("item_2_id"))
                bundle_id = str(row.get("bundle_id", "")).strip()
                if item_1_id is None or item_2_id is None or item_1_id == item_2_id or not bundle_id:
                    continue
                lookup[_pair_key(item_1_id, item_2_id)] = bundle_id
    except Exception as exc:
        raise ValueError(f"Bundle ID artifact is invalid: {path}") from exc
    return lookup


def _load_personalized_bundles(
    bundles_raw: list[dict[str, object]] | None,
    bundle_id_lookup: dict[tuple[int, int], str],
) -> tuple[list[dict[str, object]], set[tuple[int, int]]]:
    selected: list[dict[str, object]] = []
    pair_seen: set[tuple[int, int]] = set()
    if not isinstance(bundles_raw, list):
        return selected, pair_seen

    for row in bundles_raw:
        if not isinstance(row, dict):
            continue
        item_1_id = _safe_positive_int(row.get("item_1_id"))
        item_2_id = _safe_positive_int(row.get("item_2_id"))
        bundle_price = _safe_float(row.get("bundle_price"))
        if item_1_id is None or item_2_id is None or item_1_id == item_2_id or bundle_price is None:
            continue
        pair = _pair_key(item_1_id, item_2_id)
        if pair in pair_seen:
            continue
        pair_seen.add(pair)
        selected.append(
            _with_bundle_id(
                {
                    "item_1_id": int(item_1_id),
                    "item_2_id": int(item_2_id),
                    "bundle_price": float(round(max(0.0, bundle_price), 2)),
                },
                bundle_id_lookup,
            )
        )
        if len(selected) >= MAX_BUNDLES:
            break
    return selected, pair_seen


def _load_fallback_bundle_bank(path: Path, bundle_id_lookup: dict[tuple[int, int], str]) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Fallback bundle bank artifact is invalid: {path}") from exc

    rows = payload.get("bundles") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []

    bundles: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item_1_id = _safe_positive_int(row.get("item_1_id"))
        item_2_id = _safe_positive_int(row.get("item_2_id"))
        bundle_price = _safe_float(row.get("bundle_price"))
        if item_1_id is None or item_2_id is None or item_1_id == item_2_id or bundle_price is None:
            continue
        bundles.append(
            _with_bundle_id(
                {
                    "item_1_id": int(item_1_id),
                    "item_2_id": int(item_2_id),
                    "bundle_price": float(round(max(0.0, bundle_price), 2)),
                },
                bundle_id_lookup,
            )
        )
    return bundles


def _fallback_records_for_user(
    user_id: int,
    fallback_bundle_bank: list[dict[str, object]],
    *,
    target_count: int = MAX_BUNDLES,
    existing_pairs: set[tuple[int, int]] | None = None,
    bundle_id_lookup: dict[tuple[int, int], str] | None = None,
) -> list[dict[str, object]]:
    if not fallback_bundle_bank:
        return []
    target = max(1, int(target_count))
    start = int(user_id) % len(fallback_bundle_bank)
    selected: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set(existing_pairs or set())
    lookup = bundle_id_lookup or {}
    for offset in range(len(fallback_bundle_bank)):
        row = fallback_bundle_bank[(start + offset) % len(fallback_bundle_bank)]
        item_1_id = _safe_positive_int(row.get("item_1_id"))
        item_2_id = _safe_positive_int(row.get("item_2_id"))
        bundle_price = _safe_float(row.get("bundle_price"))
        if item_1_id is None or item_2_id is None or item_1_id == item_2_id or bundle_price is None:
            continue
        pair = _pair_key(item_1_id, item_2_id)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        selected.append(
            _with_bundle_id(
                {
                    "item_1_id": int(item_1_id),
                    "item_2_id": int(item_2_id),
                    "bundle_price": float(round(max(0.0, bundle_price), 2)),
                },
                lookup,
            )
        )
        if len(selected) >= target:
            break
    return selected


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

            final_meta = artifact_meta.get("final_recommendations_by_user", {})
            final_path_raw = str(final_meta.get("path", "")).strip()
            if not final_path_raw:
                raise RuntimeError("Final recommendations artifact path is missing")

            loaded = load_final_recommendations_artifact(Path(final_path_raw))
            recommendations_by_user = loaded.recommendations_by_user
            if recommendations_by_user:
                sample_user = next(iter(recommendations_by_user))
                _ = recommendations_by_user.get(sample_user, [])
            bundle_id_lookup: dict[tuple[int, int], str] = {}
            bundle_ids_meta = artifact_meta.get("bundle_ids", {})
            bundle_ids_path_raw = str(bundle_ids_meta.get("path", "")).strip()
            bundle_ids_exists = bool(bundle_ids_meta.get("exists"))
            if bundle_ids_path_raw and bundle_ids_exists:
                try:
                    bundle_id_lookup = _load_bundle_id_lookup(Path(bundle_ids_path_raw))
                except Exception as exc:
                    LOGGER.warning(
                        "[api] bundle id registry load failed; continuing without bundle_id path=%s error=%s",
                        bundle_ids_path_raw,
                        exc,
                    )
            else:
                LOGGER.info("[api] bundle id artifact unavailable; continuing without bundle_id")
            fallback_bundle_bank: list[dict[str, object]] = []
            fallback_meta = artifact_meta.get("fallback_bundle_bank", {})
            fallback_path_raw = str(fallback_meta.get("path", "")).strip()
            fallback_exists = bool(fallback_meta.get("exists"))
            if fallback_path_raw and fallback_exists:
                try:
                    fallback_bundle_bank = _load_fallback_bundle_bank(Path(fallback_path_raw), bundle_id_lookup)
                except Exception as exc:
                    LOGGER.warning(
                        "[api] fallback bundle bank load failed; continuing without fallback path=%s error=%s",
                        fallback_path_raw,
                        exc,
                    )
            else:
                LOGGER.info("[api] fallback bundle bank artifact unavailable; continuing without fallback")

            SERVING_STATE.ready = True
            SERVING_STATE.error = ""
            SERVING_STATE.initialized_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            SERVING_STATE.base_dir = base_dir
            SERVING_STATE.run_id = str(loaded.run_id).strip()
            SERVING_STATE.artifact_meta = artifact_meta
            SERVING_STATE.recommendations_by_user = recommendations_by_user
            SERVING_STATE.fallback_bundle_bank = fallback_bundle_bank
            SERVING_STATE.bundle_id_lookup = bundle_id_lookup

            LOGGER.info(
                "[api] serving state ready duration_sec=%.3f loaded_user_count=%d fallback_bundle_count=%d bundle_id_count=%d run_id=%s",
                time.perf_counter() - started,
                int(len(recommendations_by_user)),
                int(len(fallback_bundle_bank)),
                int(len(bundle_id_lookup)),
                SERVING_STATE.run_id,
            )
        except Exception as exc:
            SERVING_STATE.ready = False
            SERVING_STATE.error = str(exc)
            SERVING_STATE.initialized_at = ""
            SERVING_STATE.base_dir = base_dir
            SERVING_STATE.run_id = ""
            SERVING_STATE.artifact_meta = artifact_meta
            SERVING_STATE.recommendations_by_user = {}
            SERVING_STATE.fallback_bundle_bank = []
            SERVING_STATE.bundle_id_lookup = {}
            LOGGER.exception("[api] serving state initialization failed: %s", exc)
            raise


def _eager_init_enabled() -> bool:
    return _env_bool("QEU_API_EAGER_INIT", default=True)


def _initialize_on_startup_or_exit() -> None:
    if not _eager_init_enabled():
        LOGGER.info("[api] startup serving-state initialization disabled via QEU_API_EAGER_INIT")
        return
    try:
        _initialize_serving_state(force_reload=False)
    except Exception as exc:
        LOGGER.critical("[api] fatal startup serving-state initialization failure; exiting: %s", exc)
        raise SystemExit(1) from exc


def _get_serving_assets() -> ServingAssets | None:
    with SERVING_STATE_LOCK:
        if not SERVING_STATE.ready:
            return None
        if SERVING_STATE.base_dir is None:
            return None
        return ServingAssets(
            base_dir=SERVING_STATE.base_dir,
            run_id=SERVING_STATE.run_id,
            recommendations_by_user=SERVING_STATE.recommendations_by_user,
            fallback_bundle_bank=SERVING_STATE.fallback_bundle_bank,
            bundle_id_lookup=SERVING_STATE.bundle_id_lookup,
        )


def _recommendation_records_for_user(user_id: int) -> tuple[list[dict[str, object]], str]:
    assets = _get_serving_assets()
    if assets is None:
        raise ServiceNotReadyError("serving_state_not_ready")

    personalized, pair_seen = _load_personalized_bundles(
        assets.recommendations_by_user.get(int(user_id)),
        assets.bundle_id_lookup,
    )
    had_personalized = bool(personalized)
    if len(personalized) < MAX_BUNDLES:
        fallback = _fallback_records_for_user(
            int(user_id),
            assets.fallback_bundle_bank,
            target_count=int(MAX_BUNDLES - len(personalized)),
            existing_pairs=pair_seen,
            bundle_id_lookup=assets.bundle_id_lookup,
        )
        if fallback:
            personalized.extend(fallback)
    if personalized:
        source = "personalized" if had_personalized else "fallback"
        return personalized[:MAX_BUNDLES], source
    raise FallbackUnavailableError("fallback_bundle_bank_unavailable")


def _pick_random_user_id() -> int:
    assets = _get_serving_assets()
    if assets is None:
        raise ServiceNotReadyError("serving_state_not_ready")

    user_ids = tuple(assets.recommendations_by_user.keys())
    if not user_ids:
        raise CustomerNotFoundError
    return int(random.choice(user_ids))


def _error_payload(user_id: int | None, code: str) -> tuple[dict[str, object], int]:
    status = 500
    if code == "customer_not_found":
        status = 404
    elif code == "insufficient_history":
        status = 422
    elif code == "service_not_ready":
        status = 503
    elif code == "fallback_unavailable":
        status = 503
    return {"user_id": user_id, "bundles": [], "error": code}, status


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/readyz")
def readyz():
    payload = _state_payload()
    payload["status"] = "ready" if bool(payload.get("ready")) else "not_ready"
    return jsonify(payload), 200 if payload["status"] == "ready" else 503


@app.post("/api/recommendations/by-customer")
def recommendations_by_customer():
    started = time.perf_counter()
    payload = request.get_json(silent=True) or {}
    wants_random_user = _wants_random_user(payload)
    user_id = _parse_user_id(payload.get("user_id"))
    if user_id is None and not wants_random_user:
        body, status = _error_payload(None, "customer_not_found")
        LOGGER.info(
            "[api] recommendations request rejected reason=invalid_user_id duration_sec=%.3f",
            time.perf_counter() - started,
        )
        return jsonify(body), status

    try:
        if wants_random_user:
            user_id = _pick_random_user_id()
            LOGGER.info(
                "[api] recommendations request random_user_selected user_id=%d duration_sec=%.3f",
                int(user_id),
                time.perf_counter() - started,
            )
        if user_id is None:
            raise CustomerNotFoundError
        bundles_raw, source = _recommendation_records_for_user(user_id=user_id)
        bundles = [bundle for bundle in bundles_raw if isinstance(bundle, dict)][:MAX_BUNDLES]
        if not bundles:
            raise FallbackUnavailableError("fallback_bundle_bank_unavailable")
        LOGGER.info(
            "[api] recommendations result user_id=%d status=200 source=%s bundles=%d duration_sec=%.3f",
            int(user_id),
            source,
            len(bundles),
            time.perf_counter() - started,
        )
        return jsonify({"user_id": int(user_id), "bundles": bundles, "source": source})
    except ServiceNotReadyError:
        body, status = _error_payload(user_id, "service_not_ready")
        LOGGER.warning(
            "[api] recommendations request blocked user_id=%d reason=service_not_ready duration_sec=%.3f",
            int(user_id),
            time.perf_counter() - started,
        )
        return jsonify(body), status
    except FallbackUnavailableError:
        body, status = _error_payload(user_id, "fallback_unavailable")
        LOGGER.info(
            "[api] recommendations result user_id=%d status=%d reason=fallback_unavailable duration_sec=%.3f",
            int(user_id),
            status,
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


_initialize_on_startup_or_exit()


if __name__ == "__main__":
    raise SystemExit(main())
