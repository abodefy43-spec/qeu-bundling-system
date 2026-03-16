"""Minimal JSON-only API server for precomputed customer bundle recommendations."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, request

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.final_recommendations import (
    DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY,
    FINAL_RECOMMENDATIONS_ARTIFACT,
    load_final_recommendations_artifact,
)

MAX_BUNDLES = 3
DEFAULT_S3_FINAL_RECOMMENDATIONS_KEY = DEFAULT_FINAL_RECOMMENDATIONS_S3_KEY

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


@dataclass
class ServingRuntimeState:
    ready: bool = False
    error: str = ""
    initialized_at: str = ""
    base_dir: Path | None = None
    run_id: str = ""
    artifact_meta: dict[str, dict[str, object]] = field(default_factory=dict)
    recommendations_by_user: dict[int, list[dict[str, object]]] = field(default_factory=dict)


@dataclass(frozen=True)
class ServingAssets:
    base_dir: Path
    run_id: str
    recommendations_by_user: dict[int, list[dict[str, object]]]


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
            "final_recommendations_by_user",
            paths.output_dir / FINAL_RECOMMENDATIONS_ARTIFACT,
            _env_str("QEU_S3_FINAL_RECOMMENDATIONS_KEY", DEFAULT_S3_FINAL_RECOMMENDATIONS_KEY),
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


def _state_payload() -> dict[str, object]:
    with SERVING_STATE_LOCK:
        return {
            "ready": bool(SERVING_STATE.ready),
            "error": str(SERVING_STATE.error),
            "initialized_at": str(SERVING_STATE.initialized_at),
            "run_id": str(SERVING_STATE.run_id),
            "artifacts": {name: dict(meta) for name, meta in SERVING_STATE.artifact_meta.items()},
            "loaded_user_count": int(len(SERVING_STATE.recommendations_by_user)),
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

            final_meta = artifact_meta.get("final_recommendations_by_user", {})
            final_path_raw = str(final_meta.get("path", "")).strip()
            if not final_path_raw:
                raise RuntimeError("Final recommendations artifact path is missing")

            loaded = load_final_recommendations_artifact(Path(final_path_raw))
            recommendations_by_user = loaded.recommendations_by_user
            if recommendations_by_user:
                sample_user = next(iter(recommendations_by_user))
                _ = recommendations_by_user.get(sample_user, [])

            SERVING_STATE.ready = True
            SERVING_STATE.error = ""
            SERVING_STATE.initialized_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            SERVING_STATE.base_dir = base_dir
            SERVING_STATE.run_id = str(loaded.run_id).strip()
            SERVING_STATE.artifact_meta = artifact_meta
            SERVING_STATE.recommendations_by_user = recommendations_by_user

            LOGGER.info(
                "[api] serving state ready duration_sec=%.3f loaded_user_count=%d run_id=%s",
                time.perf_counter() - started,
                int(len(recommendations_by_user)),
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
            LOGGER.exception("[api] serving state initialization failed: %s", exc)
            raise


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
        )


def _recommendation_records_for_user(user_id: int) -> list[dict[str, object]]:
    assets = _get_serving_assets()
    if assets is None:
        raise ServiceNotReadyError("serving_state_not_ready")

    bundles = assets.recommendations_by_user.get(int(user_id))
    if bundles is None:
        raise CustomerNotFoundError
    if not bundles:
        raise InsufficientHistoryError
    return bundles[:MAX_BUNDLES]


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
        bundles = [bundle for bundle in _recommendation_records_for_user(user_id=user_id) if isinstance(bundle, dict)][
            :MAX_BUNDLES
        ]
        if not bundles:
            raise InsufficientHistoryError
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
