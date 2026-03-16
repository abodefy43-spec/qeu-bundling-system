"""Phase 0: Extract campaign data and product pictures from raw order_items.csv.

Reads the full raw file once and writes three artefacts to data/:
  - campaign_pairs.csv        (buy_x_get_y trigger/reward ground-truth labels)
  - campaign_bundle_pairs.csv (bundle-type campaign co-occurrence pairs)
  - product_pictures.csv      (product_id -> picture_url)
"""

from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from qeu_bundling.config.paths import get_paths

CHUNK_SIZE = 100_000
DEFAULT_RAW_ORDER_ITEMS_S3_KEYS = (
    "raw/order_items.csv",
    "data/raw/order_items.csv",
    "order_items.csv",
)


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _parse_s3_uri(s3_uri: str) -> tuple[str, str] | None:
    uri = str(s3_uri or "").strip()
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        return None
    return parsed.netloc.strip(), parsed.path.lstrip("/")


def _s3_raw_candidates() -> list[tuple[str, str]]:
    uri_candidate = _parse_s3_uri(os.environ.get("QEU_RAW_ORDER_ITEMS_S3_URI", ""))
    if uri_candidate is not None:
        return [uri_candidate]

    bucket = str(os.environ.get("QEU_ARTIFACTS_S3_BUCKET", "") or "").strip()
    if not bucket:
        return []

    env_key = str(os.environ.get("QEU_RAW_ORDER_ITEMS_S3_KEY", "") or "").strip()
    keys = [env_key] if env_key else list(DEFAULT_RAW_ORDER_ITEMS_S3_KEYS)
    return [(bucket, k) for k in keys if k]


def _download_raw_csv_from_s3(paths) -> Path | None:
    candidates = _s3_raw_candidates()
    if not candidates:
        return None

    try:
        import boto3  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "S3 raw bootstrap requested but boto3 is unavailable. "
            "Install batch dependencies (requirements.batch.txt)."
        ) from exc

    target = paths.data_raw_dir / "order_items.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(".csv.part")
    s3_client = boto3.client("s3")
    errors: list[str] = []

    for bucket, key in candidates:
        try:
            s3_client.download_file(bucket, key, str(temp_target))
            temp_target.replace(target)
            print(f"  Downloaded raw orders from s3://{bucket}/{key} -> {target}")
            return target
        except Exception as exc:  # pragma: no cover - network/service exceptions
            errors.append(f"s3://{bucket}/{key} ({exc})")
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except OSError:
                    pass

    print("  Unable to download raw order_items.csv from S3 candidates:")
    for err in errors:
        print(f"    - {err}")
    return None


def _raw_csv() -> Path:
    paths = get_paths()
    root = paths.project_root
    candidates = [
        paths.data_raw_dir / "order_items.csv",
        paths.data_processed_dir / "order_items.csv",
        root / "data first" / "order_items.csv",  # legacy fallback
        root / "order_items.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    downloaded = _download_raw_csv_from_s3(paths)
    if downloaded is not None and downloaded.exists():
        return downloaded

    raise FileNotFoundError(
        "Cannot locate raw order_items.csv. Checked local data paths and S3 bootstrap. "
        "Set QEU_RAW_ORDER_ITEMS_S3_URI (s3://bucket/key) or "
        "QEU_ARTIFACTS_S3_BUCKET (+ optional QEU_RAW_ORDER_ITEMS_S3_KEY)."
    )


def extract_campaigns(data_dir: Path | None = None) -> None:
    base = data_dir or _data_dir()
    base.mkdir(parents=True, exist_ok=True)
    csv_path = _raw_csv()
    reference_picture_path = get_paths().data_reference_dir / "product_pictures.csv"
    print(f"  Reading {csv_path.name} in chunks ...")

    bxy_rows: list[pd.DataFrame] = []
    bundle_rows: list[pd.DataFrame] = []
    pic_map: dict[int, str] = {}

    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False):
        chunk["product_id"] = pd.to_numeric(chunk["product_id"], errors="coerce")
        chunk["order_id"] = chunk["order_id"].astype(str)

        pics = chunk.loc[
            chunk["product_picture"].notna(),
            ["product_id", "product_picture"],
        ].drop_duplicates("product_id")
        for _, r in pics.iterrows():
            pid = int(r["product_id"])
            if pid not in pic_map:
                pic_map[pid] = str(r["product_picture"])

        bxy = chunk[chunk["campaign_type"] == "buy_x_get_y"].copy()
        if not bxy.empty:
            bxy_rows.append(bxy[["order_id", "product_id", "campaign_id",
                                  "campaign_slug", "product_role",
                                  "unit_price", "effective_price", "base_price"]])

        bnd = chunk[chunk["campaign_type"] == "bundle"].copy()
        if not bnd.empty:
            bundle_rows.append(bnd[["order_id", "product_id", "campaign_id"]])

    # --- Product pictures ---
    if reference_picture_path.exists():
        try:
            ref_df = pd.read_csv(reference_picture_path)
            if not ref_df.empty and {"product_id", "picture_url"}.issubset(set(ref_df.columns)):
                for _, row in ref_df.iterrows():
                    try:
                        pid = int(float(row.get("product_id", -1)))
                    except (TypeError, ValueError):
                        pid = -1
                    picture_url = str(row.get("picture_url", "") or "").strip()
                    if pid > 0 and picture_url:
                        pic_map[pid] = picture_url
        except Exception:
            pass
    pic_df = pd.DataFrame(
        [{"product_id": k, "picture_url": v} for k, v in pic_map.items()]
    )
    pic_df.to_csv(base / "product_pictures.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved {len(pic_df):,} product pictures -> product_pictures.csv")

    # --- buy_x_get_y campaign pairs ---
    if bxy_rows:
        bxy_all = pd.concat(bxy_rows, ignore_index=True)
        campaign_pairs = _build_campaign_pairs(bxy_all)
        campaign_pairs.to_csv(base / "campaign_pairs.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved {len(campaign_pairs):,} campaign pairs -> campaign_pairs.csv")
    else:
        pd.DataFrame({
            "product_a": pd.Series(dtype="int64"),
            "product_b": pd.Series(dtype="int64"),
            "campaign_id": pd.Series(dtype="object"),
            "campaign_slug": pd.Series(dtype="object"),
            "free_item": pd.Series(dtype="int64"),
            "occurrences": pd.Series(dtype="int64"),
        }).to_csv(base / "campaign_pairs.csv", index=False)
        print("  No buy_x_get_y campaigns found.")

    # --- bundle campaign co-occurrence ---
    if bundle_rows:
        bnd_all = pd.concat(bundle_rows, ignore_index=True)
        bundle_pairs = _build_bundle_copurchase(bnd_all)
        bundle_pairs.to_csv(base / "campaign_bundle_pairs.csv", index=False, encoding="utf-8-sig")
        print(f"  Saved {len(bundle_pairs):,} bundle co-occurrence pairs -> campaign_bundle_pairs.csv")
    else:
        pd.DataFrame({
            "product_a": pd.Series(dtype="int64"),
            "product_b": pd.Series(dtype="int64"),
            "occurrences": pd.Series(dtype="int64"),
        }).to_csv(base / "campaign_bundle_pairs.csv", index=False)
        print("  No bundle campaigns found.")


def run() -> None:
    extract_campaigns()


def _build_campaign_pairs(bxy: pd.DataFrame) -> pd.DataFrame:
    """Group buy_x_get_y items by (order, campaign) to find trigger->reward pairs."""
    bxy["product_id"] = bxy["product_id"].astype(int)
    bxy["effective_price"] = pd.to_numeric(bxy["effective_price"], errors="coerce").fillna(0)
    bxy["unit_price"] = pd.to_numeric(bxy["unit_price"], errors="coerce").fillna(0)

    rows: list[dict] = []
    for (oid, cid), grp in bxy.groupby(["order_id", "campaign_id"]):
        triggers = grp[grp["product_role"] == "trigger"]
        rewards = grp[grp["product_role"] == "reward"]
        if triggers.empty or rewards.empty:
            continue
        slug = str(grp["campaign_slug"].iloc[0])
        for _, t_row in triggers.iterrows():
            for _, r_row in rewards.iterrows():
                a_id = int(t_row["product_id"])
                b_id = int(r_row["product_id"])
                a_price = float(t_row["unit_price"])
                b_price = float(r_row["unit_price"])
                if b_price > a_price:
                    a_id, b_id = b_id, a_id
                    free_item = 0
                else:
                    free_item = 1
                rows.append({
                    "product_a": a_id,
                    "product_b": b_id,
                    "campaign_id": str(cid),
                    "campaign_slug": slug,
                    "free_item": free_item,
                })

    if not rows:
        return pd.DataFrame(columns=["product_a", "product_b", "campaign_id",
                                      "campaign_slug", "free_item", "occurrences"])

    df = pd.DataFrame(rows)
    agg = (
        df.groupby(["product_a", "product_b", "campaign_id", "campaign_slug", "free_item"])
        .size()
        .reset_index(name="occurrences")
    )
    return agg.sort_values("occurrences", ascending=False).reset_index(drop=True)


def _build_bundle_copurchase(bnd: pd.DataFrame) -> pd.DataFrame:
    """Build co-occurrence pairs from bundle-type campaign items in the same order."""
    bnd["product_id"] = bnd["product_id"].astype(int)
    pair_counts: dict[tuple[int, int], int] = {}

    for _, grp in bnd.groupby(["order_id", "campaign_id"]):
        pids = sorted(grp["product_id"].unique())
        if len(pids) < 2:
            continue
        for a, b in combinations(pids, 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    if not pair_counts:
        return pd.DataFrame(columns=["product_a", "product_b", "occurrences"])

    rows = [{"product_a": a, "product_b": b, "occurrences": c}
            for (a, b), c in pair_counts.items()]
    return (
        pd.DataFrame(rows)
        .sort_values("occurrences", ascending=False)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    print("Phase 0: Extracting campaign data & product pictures ...")
    extract_campaigns()
    print("Phase 0 complete.")
