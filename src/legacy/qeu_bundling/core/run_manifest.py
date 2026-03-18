"""Run manifest helpers for phase-level lineage and diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qeu_bundling.config.paths import ensure_layout, get_paths, latest_manifest_path, run_output_dir


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def file_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    stat = path.stat()
    payload: dict[str, object] = {
        "exists": True,
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    try:
        h = hashlib.sha1()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        payload["sha1"] = h.hexdigest()
    except OSError:
        payload["sha1"] = ""
    return payload


def write_run_manifest(
    *,
    mode: str,
    run_id: str,
    seed: int,
    started_at: str,
    finished_at: str,
    phases: list[dict[str, object]],
    artifact_paths: dict[str, Path],
    base_dir: Path | None = None,
) -> Path:
    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)
    run_dir = run_output_dir(run_id, project_root=base_dir)
    out_path = run_dir / "run_manifest.json"

    artifacts = {
        key: {"path": str(p), **file_metadata(p)}
        for key, p in artifact_paths.items()
    }
    payload = {
        "mode": mode,
        "run_id": run_id,
        "seed": int(seed),
        "started_at": started_at,
        "finished_at": finished_at,
        "phases": phases,
        "artifacts": artifacts,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    root_manifest = paths.output_dir / "run_manifest.json"
    with root_manifest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    latest_payload = {
        "run_id": run_id,
        "mode": mode,
        "seed": int(seed),
        "started_at": started_at,
        "finished_at": finished_at,
        "run_manifest_path": str(out_path.resolve()),
        "artifacts": {
            key: str(Path(meta.get("path", "")).resolve()) if str(meta.get("path", "")).strip() else ""
            for key, meta in artifacts.items()
        },
    }
    latest_path = latest_manifest_path(project_root=base_dir)
    latest_path.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def read_latest_manifest(base_dir: Path | None = None) -> dict[str, object]:
    path = latest_manifest_path(project_root=base_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_latest_artifact(
    key: str,
    base_dir: Path | None = None,
    fallback: Path | None = None,
) -> Path | None:
    payload = read_latest_manifest(base_dir=base_dir)
    latest_path: Path | None = None
    artifacts = payload.get("artifacts", {})
    if isinstance(artifacts, dict):
        raw = str(artifacts.get(str(key), "")).strip()
        if raw:
            path = Path(raw)
            if path.exists():
                latest_path = path
    if fallback is not None and fallback.exists():
        if latest_path is None:
            return fallback
        try:
            if fallback.resolve() == latest_path.resolve():
                return fallback
        except OSError:
            pass
        try:
            if int(fallback.stat().st_mtime_ns) >= int(latest_path.stat().st_mtime_ns):
                return fallback
        except OSError:
            return fallback
    if latest_path is not None:
        return latest_path
    if fallback is not None and fallback.exists():
        return fallback
    return fallback


def append_seed_history(
    *,
    mode: str,
    run_id: str,
    seed: int,
    started_at: str,
    finished_at: str,
    base_dir: Path | None = None,
) -> Path:
    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)
    out_path = paths.output_seeds_dir / "seeds_history.csv"
    row = {
        "run_id": str(run_id),
        "mode": str(mode),
        "seed": int(seed),
        "started_at": str(started_at),
        "finished_at": str(finished_at),
        "recorded_at": utc_now_iso(),
    }
    write_header = not out_path.exists()
    with out_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["run_id", "mode", "seed", "started_at", "finished_at", "recorded_at"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return out_path
