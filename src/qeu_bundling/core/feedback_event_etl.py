"""Chunked ETL for raw behavioral feedback telemetry.

This module converts ``feedback/feedback.csv`` into clean event records and
derived feature tables that are suitable for Stage 2 personalized ranking.

The source file is raw implicit telemetry. It is not a bundle-feedback table
and is intentionally treated as behavioral evidence rather than explicit pair
supervision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qeu_bundling.config.paths import ensure_layout, get_paths

RAW_FEEDBACK_REL_PATH = Path("feedback") / "feedback.csv"
ALLOWED_FEEDBACK_TYPES = frozenset({"view", "add_to_cart", "purchase"})
DURABLE_ACTOR_TYPE = "durable_user_like"
SESSION_ACTOR_TYPE = "session_style"
EXPLICIT_SESSION_STRATEGY = "explicit_session_id"
ACTOR_DAY_SESSION_STRATEGY = "actor_day"
DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_CARRYOVER_ROWS = 5_000
DEFAULT_MAX_PAIR_ITEMS_PER_SESSION = 40
DEFAULT_MAX_VIEW_ONLY_NEGATIVES_PER_SESSION = 20
PARQUET_ARTIFACTS = {
    "clean_events": "feedback_events_clean.parquet",
    "actor_session_features": "feedback_actor_session_features.parquet",
    "actor_item_features": "feedback_actor_item_features.parquet",
    "session_item_features": "feedback_session_item_features.parquet",
    "proxy_pair_labels": "feedback_proxy_pair_labels.parquet",
}
TYPE_ORDER = {"view": 0, "add_to_cart": 1, "purchase": 2}
INTERNAL_SOURCE_ROW = "_source_row"


@dataclass(frozen=True)
class FeedbackETLArtifacts:
    input_path: Path
    clean_events_path: Path
    actor_session_features_path: Path
    actor_item_features_path: Path
    session_item_features_path: Path
    proxy_pair_labels_path: Path
    report: dict[str, object]


def default_feedback_source_path(base_dir: Path | None = None) -> Path:
    root = (base_dir or get_paths().project_root).resolve()
    return root / RAW_FEEDBACK_REL_PATH


def feedback_etl_artifact_paths(base_dir: Path | None = None) -> dict[str, Path]:
    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)
    return {
        name: paths.data_processed_dir / filename
        for name, filename in PARQUET_ARTIFACTS.items()
    }


def _require_parquet_support() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised at runtime
        raise RuntimeError(
            "Parquet support is required for feedback ETL. Install pyarrow in the batch environment."
        ) from exc
    return pa, pq


def _empty_clean_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            INTERNAL_SOURCE_ROW,
            "actor_id",
            "actor_type",
            "item_id",
            "feedback_type",
            "event_value_raw",
            "event_value_num",
            "comment",
            "event_ts",
            "updated_ts",
            "effective_ts",
            "event_date",
            "session_id",
            "session_strategy",
        ]
    )


def _empty_feature_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _normalize_string_series(series: pd.Series) -> pd.Series:
    out = series.fillna("").astype(str).str.strip()
    return out.replace({"nan": "", "None": ""})


def _classify_actor_type(actor_id: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(actor_id.str.startswith("sess_"), SESSION_ACTOR_TYPE, DURABLE_ACTOR_TYPE),
        index=actor_id.index,
        dtype="string",
    )


def _session_strategy(actor_type: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(actor_type == SESSION_ACTOR_TYPE, EXPLICIT_SESSION_STRATEGY, ACTOR_DAY_SESSION_STRATEGY),
        index=actor_type.index,
        dtype="string",
    )


def _build_session_id(actor_id: pd.Series, actor_type: pd.Series, effective_ts: pd.Series) -> pd.Series:
    day_stamp = effective_ts.dt.strftime("%Y%m%d")
    session_like = actor_id.astype("string") + "|" + day_stamp.astype("string")
    return pd.Series(
        np.where(actor_type == SESSION_ACTOR_TYPE, actor_id.astype("string"), session_like),
        index=actor_id.index,
        dtype="string",
    )


def normalize_feedback_event_chunk(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw telemetry chunk into clean event rows.

    The raw source can contain row updates for the same logical event. We keep
    the latest ``updated`` timestamp for the same actor/item/type/timestamp/value
    signature and preserve repeated interactions across different timestamps.
    """

    if raw_df is None or raw_df.empty:
        return _empty_clean_events()

    out = raw_df.copy()
    if INTERNAL_SOURCE_ROW not in out.columns:
        out[INTERNAL_SOURCE_ROW] = np.arange(len(out), dtype=np.int64)

    out["feedback_type"] = _normalize_string_series(out.get("feedback_type", pd.Series(index=out.index))).str.lower()
    out["actor_id"] = _normalize_string_series(out.get("user_id", pd.Series(index=out.index)))
    out["item_id"] = _normalize_string_series(out.get("item_id", pd.Series(index=out.index)))
    out["event_value_raw"] = _normalize_string_series(out.get("value", pd.Series(index=out.index)))
    out["event_value_num"] = pd.to_numeric(out["event_value_raw"], errors="coerce").astype("float32")
    out["comment"] = _normalize_string_series(out.get("comment", pd.Series(index=out.index)))

    out["event_ts"] = pd.to_datetime(
        _normalize_string_series(out.get("time_stamp", pd.Series(index=out.index))),
        errors="coerce",
        utc=True,
    )
    out["updated_ts"] = pd.to_datetime(
        _normalize_string_series(out.get("updated", pd.Series(index=out.index))),
        errors="coerce",
        utc=True,
    )
    out["event_ts"] = out["event_ts"].fillna(out["updated_ts"])
    out["updated_ts"] = out["updated_ts"].fillna(out["event_ts"])

    out = out.loc[
        out["feedback_type"].isin(ALLOWED_FEEDBACK_TYPES)
        & out["actor_id"].ne("")
        & out["item_id"].ne("")
        & out["event_ts"].notna()
        & out["updated_ts"].notna()
    ].copy()
    if out.empty:
        return _empty_clean_events()

    out["actor_type"] = _classify_actor_type(out["actor_id"])
    out["effective_ts"] = out[["event_ts", "updated_ts"]].max(axis=1)
    out["event_date"] = out["effective_ts"].dt.floor("D")
    out["session_strategy"] = _session_strategy(out["actor_type"])
    out["session_id"] = _build_session_id(out["actor_id"], out["actor_type"], out["effective_ts"])

    dedupe_subset = ["feedback_type", "actor_id", "item_id", "event_ts", "event_value_raw", "comment"]
    out = out.sort_values(
        dedupe_subset + ["updated_ts", INTERNAL_SOURCE_ROW],
        kind="mergesort",
    )
    out = out.drop_duplicates(subset=dedupe_subset, keep="last")
    out = out.sort_values(INTERNAL_SOURCE_ROW, kind="mergesort")

    return out.loc[
        :,
        [
            INTERNAL_SOURCE_ROW,
            "actor_id",
            "actor_type",
            "item_id",
            "feedback_type",
            "event_value_raw",
            "event_value_num",
            "comment",
            "event_ts",
            "updated_ts",
            "effective_ts",
            "event_date",
            "session_id",
            "session_strategy",
        ],
    ].reset_index(drop=True)


def _append_parquet_chunk(
    writer: Any,
    df: pd.DataFrame,
    *,
    path: Path,
    pa: Any,
    pq: Any,
) -> Any:
    if df.empty:
        return writer

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(path), table.schema)
    writer.write_table(table)
    return writer


def _update_event_report(report: dict[str, object], clean_chunk: pd.DataFrame) -> None:
    if clean_chunk.empty:
        return

    report["clean_rows"] = int(report.get("clean_rows", 0)) + int(len(clean_chunk))

    event_counts = report.setdefault("event_type_counts", {})
    actor_counts = report.setdefault("actor_type_counts", {})
    if isinstance(event_counts, dict):
        for name, count in clean_chunk["feedback_type"].value_counts().to_dict().items():
            event_counts[str(name)] = int(event_counts.get(str(name), 0)) + int(count)
    if isinstance(actor_counts, dict):
        for name, count in clean_chunk["actor_type"].value_counts().to_dict().items():
            actor_counts[str(name)] = int(actor_counts.get(str(name), 0)) + int(count)

    min_ts = clean_chunk["effective_ts"].min()
    max_ts = clean_chunk["effective_ts"].max()
    current_min = report.get("date_min")
    current_max = report.get("date_max")
    min_text = "" if pd.isna(min_ts) else pd.Timestamp(min_ts).isoformat()
    max_text = "" if pd.isna(max_ts) else pd.Timestamp(max_ts).isoformat()
    if min_text and (not current_min or min_text < str(current_min)):
        report["date_min"] = min_text
    if max_text and (not current_max or max_text > str(current_max)):
        report["date_max"] = max_text


def _update_actor_session_state(clean_chunk: pd.DataFrame, state: dict[tuple[str, str, str, str], dict[str, object]]) -> None:
    if clean_chunk.empty:
        return

    df = clean_chunk.loc[
        :, ["actor_id", "actor_type", "session_id", "session_strategy", "item_id", "feedback_type", "effective_ts"]
    ].copy()
    df["view_count"] = (df["feedback_type"] == "view").astype("int32")
    df["add_to_cart_count"] = (df["feedback_type"] == "add_to_cart").astype("int32")
    df["purchase_count"] = (df["feedback_type"] == "purchase").astype("int32")
    partial = (
        df.groupby(["actor_id", "actor_type", "session_id", "session_strategy"], observed=True, sort=False)
        .agg(
            event_count=("item_id", "size"),
            first_event_ts=("effective_ts", "min"),
            last_event_ts=("effective_ts", "max"),
            view_count=("view_count", "sum"),
            add_to_cart_count=("add_to_cart_count", "sum"),
            purchase_count=("purchase_count", "sum"),
        )
        .reset_index()
    )
    for row in partial.itertuples(index=False):
        key = (str(row.actor_id), str(row.actor_type), str(row.session_id), str(row.session_strategy))
        current = state.get(key)
        if current is None:
            state[key] = {
                "event_count": int(row.event_count),
                "first_event_ts": row.first_event_ts,
                "last_event_ts": row.last_event_ts,
                "view_count": int(row.view_count),
                "add_to_cart_count": int(row.add_to_cart_count),
                "purchase_count": int(row.purchase_count),
            }
            continue
        current["event_count"] = int(current["event_count"]) + int(row.event_count)
        current["view_count"] = int(current["view_count"]) + int(row.view_count)
        current["add_to_cart_count"] = int(current["add_to_cart_count"]) + int(row.add_to_cart_count)
        current["purchase_count"] = int(current["purchase_count"]) + int(row.purchase_count)
        if row.first_event_ts < current["first_event_ts"]:
            current["first_event_ts"] = row.first_event_ts
        if row.last_event_ts > current["last_event_ts"]:
            current["last_event_ts"] = row.last_event_ts


def _update_actor_item_state(clean_chunk: pd.DataFrame, state: dict[tuple[str, str, str], dict[str, object]]) -> None:
    if clean_chunk.empty:
        return

    df = clean_chunk.loc[:, ["actor_id", "actor_type", "item_id", "feedback_type", "effective_ts"]].copy()
    df["total_views"] = (df["feedback_type"] == "view").astype("int32")
    df["total_add_to_cart"] = (df["feedback_type"] == "add_to_cart").astype("int32")
    df["total_purchases"] = (df["feedback_type"] == "purchase").astype("int32")
    partial = (
        df.groupby(["actor_id", "actor_type", "item_id"], observed=True, sort=False)
        .agg(
            event_count=("item_id", "size"),
            total_views=("total_views", "sum"),
            total_add_to_cart=("total_add_to_cart", "sum"),
            total_purchases=("total_purchases", "sum"),
            first_seen_ts=("effective_ts", "min"),
            last_seen_ts=("effective_ts", "max"),
        )
        .reset_index()
    )
    for row in partial.itertuples(index=False):
        key = (str(row.actor_id), str(row.actor_type), str(row.item_id))
        current = state.get(key)
        if current is None:
            state[key] = {
                "event_count": int(row.event_count),
                "total_views": int(row.total_views),
                "total_add_to_cart": int(row.total_add_to_cart),
                "total_purchases": int(row.total_purchases),
                "first_seen_ts": row.first_seen_ts,
                "last_seen_ts": row.last_seen_ts,
            }
            continue
        current["event_count"] = int(current["event_count"]) + int(row.event_count)
        current["total_views"] = int(current["total_views"]) + int(row.total_views)
        current["total_add_to_cart"] = int(current["total_add_to_cart"]) + int(row.total_add_to_cart)
        current["total_purchases"] = int(current["total_purchases"]) + int(row.total_purchases)
        if row.first_seen_ts < current["first_seen_ts"]:
            current["first_seen_ts"] = row.first_seen_ts
        if row.last_seen_ts > current["last_seen_ts"]:
            current["last_seen_ts"] = row.last_seen_ts


def _pick_first_type(
    current_ts: pd.Timestamp | None,
    current_type: str | None,
    candidate_ts: pd.Timestamp,
    candidate_type: str,
) -> tuple[pd.Timestamp, str]:
    if current_ts is None or candidate_ts < current_ts:
        return candidate_ts, candidate_type
    if candidate_ts == current_ts and TYPE_ORDER.get(candidate_type, 99) < TYPE_ORDER.get(str(current_type), 99):
        return candidate_ts, candidate_type
    return current_ts, str(current_type or candidate_type)


def _pick_last_type(
    current_ts: pd.Timestamp | None,
    current_type: str | None,
    candidate_ts: pd.Timestamp,
    candidate_type: str,
) -> tuple[pd.Timestamp, str]:
    if current_ts is None or candidate_ts > current_ts:
        return candidate_ts, candidate_type
    if candidate_ts == current_ts and TYPE_ORDER.get(candidate_type, -1) > TYPE_ORDER.get(str(current_type), -1):
        return candidate_ts, candidate_type
    return current_ts, str(current_type or candidate_type)


def _update_session_item_state(
    clean_chunk: pd.DataFrame,
    state: dict[tuple[str, str, str, str, str], dict[str, object]],
) -> None:
    if clean_chunk.empty:
        return

    df = clean_chunk.loc[
        :,
        [
            "session_id",
            "actor_id",
            "actor_type",
            "session_strategy",
            "item_id",
            "feedback_type",
            "effective_ts",
        ],
    ].copy()
    df["feedback_type_order"] = df["feedback_type"].map(TYPE_ORDER).fillna(99).astype("int8")
    df = df.sort_values(["session_id", "item_id", "effective_ts", "feedback_type_order"], kind="mergesort")
    df["viewed_in_session"] = (df["feedback_type"] == "view").astype("int32")
    df["carted_in_session"] = (df["feedback_type"] == "add_to_cart").astype("int32")
    df["purchased_in_session"] = (df["feedback_type"] == "purchase").astype("int32")
    partial = (
        df.groupby(
            ["session_id", "actor_id", "actor_type", "session_strategy", "item_id"],
            observed=True,
            sort=False,
        )
        .agg(
            session_item_event_count=("item_id", "size"),
            viewed_in_session=("viewed_in_session", "max"),
            carted_in_session=("carted_in_session", "max"),
            purchased_in_session=("purchased_in_session", "max"),
            first_event_type=("feedback_type", "first"),
            last_event_type=("feedback_type", "last"),
            first_event_ts=("effective_ts", "min"),
            last_event_ts=("effective_ts", "max"),
        )
        .reset_index()
    )
    for row in partial.itertuples(index=False):
        key = (
            str(row.session_id),
            str(row.actor_id),
            str(row.actor_type),
            str(row.session_strategy),
            str(row.item_id),
        )
        current = state.get(key)
        if current is None:
            state[key] = {
                "session_item_event_count": int(row.session_item_event_count),
                "viewed_in_session": int(row.viewed_in_session),
                "carted_in_session": int(row.carted_in_session),
                "purchased_in_session": int(row.purchased_in_session),
                "first_event_type": str(row.first_event_type),
                "last_event_type": str(row.last_event_type),
                "first_event_ts": row.first_event_ts,
                "last_event_ts": row.last_event_ts,
            }
            continue
        current["session_item_event_count"] = int(current["session_item_event_count"]) + int(row.session_item_event_count)
        current["viewed_in_session"] = max(int(current["viewed_in_session"]), int(row.viewed_in_session))
        current["carted_in_session"] = max(int(current["carted_in_session"]), int(row.carted_in_session))
        current["purchased_in_session"] = max(int(current["purchased_in_session"]), int(row.purchased_in_session))
        first_ts, first_type = _pick_first_type(
            current["first_event_ts"],
            str(current["first_event_type"]),
            row.first_event_ts,
            str(row.first_event_type),
        )
        last_ts, last_type = _pick_last_type(
            current["last_event_ts"],
            str(current["last_event_type"]),
            row.last_event_ts,
            str(row.last_event_type),
        )
        current["first_event_ts"] = first_ts
        current["first_event_type"] = first_type
        current["last_event_ts"] = last_ts
        current["last_event_type"] = last_type


def _update_feature_states(clean_chunk: pd.DataFrame, feature_states: dict[str, dict[tuple[Any, ...], dict[str, object]]]) -> None:
    if not feature_states or clean_chunk.empty:
        return
    _update_actor_session_state(clean_chunk, feature_states.setdefault("actor_session", {}))
    _update_actor_item_state(clean_chunk, feature_states.setdefault("actor_item", {}))
    _update_session_item_state(clean_chunk, feature_states.setdefault("session_item", {}))


def write_clean_feedback_events(
    *,
    input_path: Path,
    output_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    carryover_rows: int = DEFAULT_CARRYOVER_ROWS,
    feature_states: dict[str, dict[tuple[Any, ...], dict[str, object]]] | None = None,
) -> dict[str, object]:
    """Normalize and write clean feedback events in a chunked, memory-safe pass."""

    pa, pq = _require_parquet_support()

    if output_path.exists():
        output_path.unlink()

    writer = None
    carryover_raw = pd.DataFrame()
    row_offset = 0
    report: dict[str, object] = {
        "input_rows": 0,
        "clean_rows": 0,
        "dropped_rows": 0,
        "event_type_counts": {},
        "actor_type_counts": {},
        "date_min": "",
        "date_max": "",
        "input_path": str(input_path),
        "output_path": str(output_path),
    }

    reader = pd.read_csv(
        input_path,
        dtype=str,
        chunksize=int(chunk_size),
        keep_default_na=False,
    )
    for raw_chunk in reader:
        raw_chunk = raw_chunk.copy()
        raw_chunk[INTERNAL_SOURCE_ROW] = np.arange(row_offset, row_offset + len(raw_chunk), dtype=np.int64)
        row_offset += len(raw_chunk)
        report["input_rows"] = int(row_offset)

        if not carryover_raw.empty:
            combined = pd.concat([carryover_raw, raw_chunk], ignore_index=True)
        else:
            combined = raw_chunk

        clean_combined = normalize_feedback_event_chunk(combined)
        if clean_combined.empty:
            carryover_raw = combined.tail(int(carryover_rows)).copy() if carryover_rows > 0 else pd.DataFrame()
            continue

        if carryover_rows > 0:
            cutoff = int(raw_chunk[INTERNAL_SOURCE_ROW].max()) - int(carryover_rows)
            write_chunk = clean_combined.loc[clean_combined[INTERNAL_SOURCE_ROW] <= cutoff].copy()
            carryover_raw = combined.loc[combined[INTERNAL_SOURCE_ROW] > cutoff].copy()
        else:
            write_chunk = clean_combined
            carryover_raw = pd.DataFrame()

        write_chunk = write_chunk.drop(columns=[INTERNAL_SOURCE_ROW], errors="ignore")
        writer = _append_parquet_chunk(writer, write_chunk, path=output_path, pa=pa, pq=pq)
        _update_event_report(report, write_chunk)
        _update_feature_states(write_chunk, feature_states or {})

    if not carryover_raw.empty:
        final_chunk = normalize_feedback_event_chunk(carryover_raw).drop(columns=[INTERNAL_SOURCE_ROW], errors="ignore")
        writer = _append_parquet_chunk(writer, final_chunk, path=output_path, pa=pa, pq=pq)
        _update_event_report(report, final_chunk)
        _update_feature_states(final_chunk, feature_states or {})

    if writer is not None:
        writer.close()

    report["dropped_rows"] = int(report["input_rows"]) - int(report["clean_rows"])
    return report


def _load_clean_events(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    _require_parquet_support()
    df = pd.read_parquet(path, columns=columns)
    if df.empty:
        return df
    for text_col in ("actor_id", "actor_type", "item_id", "feedback_type", "session_id", "session_strategy"):
        if text_col in df.columns:
            df[text_col] = df[text_col].astype("string")
    for cat_col in ("actor_type", "feedback_type", "session_strategy"):
        if cat_col in df.columns:
            df[cat_col] = df[cat_col].astype("category")
    return df


def build_actor_session_features(clean_events: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty:
        return _empty_feature_frame(
            [
                "actor_id",
                "actor_type",
                "session_id",
                "session_strategy",
                "event_count",
                "item_count",
                "unique_item_count",
                "first_event_ts",
                "last_event_ts",
                "session_duration_sec",
                "view_count",
                "add_to_cart_count",
                "purchase_count",
            ]
        )

    df = clean_events.loc[
        :, ["actor_id", "actor_type", "session_id", "session_strategy", "item_id", "feedback_type", "effective_ts"]
    ].copy()
    df["view_count"] = (df["feedback_type"] == "view").astype("int8")
    df["add_to_cart_count"] = (df["feedback_type"] == "add_to_cart").astype("int8")
    df["purchase_count"] = (df["feedback_type"] == "purchase").astype("int8")

    grouped = df.groupby(["actor_id", "actor_type", "session_id", "session_strategy"], observed=True, sort=False)
    out = grouped.agg(
        event_count=("item_id", "size"),
        item_count=("item_id", "size"),
        unique_item_count=("item_id", "nunique"),
        first_event_ts=("effective_ts", "min"),
        last_event_ts=("effective_ts", "max"),
        view_count=("view_count", "sum"),
        add_to_cart_count=("add_to_cart_count", "sum"),
        purchase_count=("purchase_count", "sum"),
    ).reset_index()
    out["session_duration_sec"] = (
        out["last_event_ts"] - out["first_event_ts"]
    ).dt.total_seconds().fillna(0.0).astype("float32")
    return out


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    out = pd.Series(np.zeros(len(numerator), dtype="float32"), index=numerator.index, dtype="float32")
    valid = denominator.fillna(0).astype("float64") > 0
    out.loc[valid] = (
        numerator.loc[valid].astype("float64") / denominator.loc[valid].astype("float64")
    ).astype("float32")
    return out


def build_actor_item_features(clean_events: pd.DataFrame) -> pd.DataFrame:
    if clean_events.empty:
        return _empty_feature_frame(
            [
                "actor_id",
                "actor_type",
                "item_id",
                "event_count",
                "session_count",
                "active_day_count",
                "total_views",
                "total_add_to_cart",
                "total_purchases",
                "first_seen_ts",
                "last_seen_ts",
                "activity_span_days",
                "days_since_last_seen",
                "events_per_active_day",
                "view_to_cart_rate",
                "add_to_purchase_rate",
                "purchase_after_view_rate",
            ]
        )

    df = clean_events.loc[
        :, ["actor_id", "actor_type", "item_id", "session_id", "feedback_type", "effective_ts", "event_date"]
    ].copy()
    df["total_views"] = (df["feedback_type"] == "view").astype("int8")
    df["total_add_to_cart"] = (df["feedback_type"] == "add_to_cart").astype("int8")
    df["total_purchases"] = (df["feedback_type"] == "purchase").astype("int8")

    grouped = df.groupby(["actor_id", "actor_type", "item_id"], observed=True, sort=False)
    out = grouped.agg(
        event_count=("item_id", "size"),
        session_count=("session_id", "nunique"),
        active_day_count=("event_date", "nunique"),
        total_views=("total_views", "sum"),
        total_add_to_cart=("total_add_to_cart", "sum"),
        total_purchases=("total_purchases", "sum"),
        first_seen_ts=("effective_ts", "min"),
        last_seen_ts=("effective_ts", "max"),
    ).reset_index()

    reference_ts = out["last_seen_ts"].max()
    if pd.isna(reference_ts):
        reference_ts = pd.Timestamp.utcnow(tz="UTC")
    out["activity_span_days"] = (
        out["last_seen_ts"] - out["first_seen_ts"]
    ).dt.total_seconds().div(86400.0).fillna(0.0).astype("float32")
    out["days_since_last_seen"] = (
        pd.Timestamp(reference_ts) - out["last_seen_ts"]
    ).dt.total_seconds().div(86400.0).fillna(0.0).astype("float32")
    out["events_per_active_day"] = _safe_divide(out["event_count"], out["active_day_count"])
    out["view_to_cart_rate"] = _safe_divide(out["total_add_to_cart"], out["total_views"])
    out["add_to_purchase_rate"] = _safe_divide(out["total_purchases"], out["total_add_to_cart"])
    out["purchase_after_view_rate"] = _safe_divide(out["total_purchases"], out["total_views"])
    return out


def build_session_item_features(
    clean_events: pd.DataFrame,
    actor_session_features: pd.DataFrame,
) -> pd.DataFrame:
    if clean_events.empty:
        return _empty_feature_frame(
            [
                "session_id",
                "actor_id",
                "actor_type",
                "session_strategy",
                "item_id",
                "session_item_event_count",
                "viewed_in_session",
                "carted_in_session",
                "purchased_in_session",
                "first_event_type",
                "last_event_type",
                "first_event_ts",
                "last_event_ts",
                "first_event_order",
                "last_event_order",
                "first_event_offset_sec",
                "last_event_recency_sec",
            ]
        )

    df = clean_events.loc[
        :,
        [
            "session_id",
            "actor_id",
            "actor_type",
            "session_strategy",
            "item_id",
            "feedback_type",
            "effective_ts",
        ],
    ].copy()
    df["feedback_type_order"] = df["feedback_type"].map(TYPE_ORDER).fillna(99).astype("int8")
    df = df.sort_values(["session_id", "effective_ts", "feedback_type_order", "item_id"], kind="mergesort")
    df["session_event_order"] = df.groupby("session_id", observed=True, sort=False).cumcount() + 1
    df["viewed_in_session"] = (df["feedback_type"] == "view").astype("int8")
    df["carted_in_session"] = (df["feedback_type"] == "add_to_cart").astype("int8")
    df["purchased_in_session"] = (df["feedback_type"] == "purchase").astype("int8")

    grouped = df.groupby(
        ["session_id", "actor_id", "actor_type", "session_strategy", "item_id"],
        observed=True,
        sort=False,
    )
    out = grouped.agg(
        session_item_event_count=("item_id", "size"),
        viewed_in_session=("viewed_in_session", "max"),
        carted_in_session=("carted_in_session", "max"),
        purchased_in_session=("purchased_in_session", "max"),
        first_event_type=("feedback_type", "first"),
        last_event_type=("feedback_type", "last"),
        first_event_ts=("effective_ts", "min"),
        last_event_ts=("effective_ts", "max"),
        first_event_order=("session_event_order", "min"),
        last_event_order=("session_event_order", "max"),
    ).reset_index()

    session_meta = actor_session_features.loc[
        :, ["session_id", "event_count", "first_event_ts", "last_event_ts"]
    ].rename(
        columns={
            "event_count": "session_event_count",
            "first_event_ts": "session_first_ts",
            "last_event_ts": "session_last_ts",
        }
    )
    out = out.merge(session_meta, on="session_id", how="left")
    out["first_event_offset_sec"] = (
        out["first_event_ts"] - out["session_first_ts"]
    ).dt.total_seconds().fillna(0.0).astype("float32")
    out["last_event_recency_sec"] = (
        out["session_last_ts"] - out["last_event_ts"]
    ).dt.total_seconds().fillna(0.0).astype("float32")
    out["first_event_order_norm"] = _safe_divide(out["first_event_order"], out["session_event_count"])
    out["last_event_order_norm"] = _safe_divide(out["last_event_order"], out["session_event_count"])
    return out


def build_proxy_pair_labels(
    session_item_features: pd.DataFrame,
    *,
    max_items_per_session: int = DEFAULT_MAX_PAIR_ITEMS_PER_SESSION,
    max_view_only_negatives_per_session: int = DEFAULT_MAX_VIEW_ONLY_NEGATIVES_PER_SESSION,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build weak pair labels from within-session co-behavior.

    This is weak supervision for later ranking, not a hard-coded bundle table.
    """

    if session_item_features.empty:
        empty = _empty_feature_frame(
            [
                "item_a",
                "item_b",
                "purchase_purchase_sessions",
                "cart_cart_sessions",
                "cart_purchase_sessions",
                "negative_view_only_sessions",
                "support_sessions",
                "label_score",
                "proxy_label",
            ]
        )
        return empty, {"sessions_considered": 0, "sessions_skipped_high_cardinality": 0}

    df = session_item_features.loc[
        :,
        [
            "session_id",
            "item_id",
            "viewed_in_session",
            "carted_in_session",
            "purchased_in_session",
            "first_event_order",
        ],
    ].copy()
    df["engaged_in_session"] = ((df["carted_in_session"] > 0) | (df["purchased_in_session"] > 0)).astype("int8")
    df["view_only_in_session"] = (
        (df["viewed_in_session"] > 0) & (df["carted_in_session"] == 0) & (df["purchased_in_session"] == 0)
    ).astype("int8")
    eligible_sessions = (
        df.groupby("session_id", observed=True, sort=False)
        .agg(
            item_count=("item_id", "size"),
            engaged_count=("engaged_in_session", "sum"),
            view_only_count=("view_only_in_session", "sum"),
        )
        .reset_index()
    )
    eligible_sessions = eligible_sessions.loc[
        (eligible_sessions["item_count"] >= 2)
        & (
            (eligible_sessions["engaged_count"] >= 2)
            | ((eligible_sessions["engaged_count"] >= 1) & (eligible_sessions["view_only_count"] >= 1))
        ),
        "session_id",
    ]
    if eligible_sessions.empty:
        empty = _empty_feature_frame(
            [
                "item_a",
                "item_b",
                "purchase_purchase_sessions",
                "cart_cart_sessions",
                "cart_purchase_sessions",
                "negative_view_only_sessions",
                "support_sessions",
                "label_score",
                "proxy_label",
            ]
        )
        return empty, {"sessions_considered": 0, "sessions_skipped_high_cardinality": 0}
    df = df.loc[df["session_id"].isin(eligible_sessions)].copy()
    df = df.sort_values(["session_id", "first_event_order", "item_id"], kind="mergesort")

    pair_counts: dict[tuple[str, str], dict[str, float]] = {}
    sessions_considered = 0
    sessions_skipped = 0

    for session_id, group in df.groupby("session_id", sort=False, observed=True):
        if group.empty:
            continue
        sessions_considered += 1
        group = group.sort_values(["first_event_order", "item_id"], kind="mergesort").reset_index(drop=True)
        if len(group) > int(max_items_per_session):
            group = group.head(int(max_items_per_session)).copy()
            sessions_skipped += 1

        engaged = group.loc[group["engaged_in_session"] > 0].copy()
        viewed_only = group.loc[group["view_only_in_session"] > 0].copy()

        if len(engaged) >= 2:
            engaged_rows = engaged.to_dict(orient="records")
            for left_idx in range(len(engaged_rows) - 1):
                left = engaged_rows[left_idx]
                for right_idx in range(left_idx + 1, len(engaged_rows)):
                    right = engaged_rows[right_idx]
                    key = tuple(sorted((str(left["item_id"]), str(right["item_id"]))))
                    payload = pair_counts.setdefault(
                        key,
                        {
                            "purchase_purchase_sessions": 0.0,
                            "cart_cart_sessions": 0.0,
                            "cart_purchase_sessions": 0.0,
                            "negative_view_only_sessions": 0.0,
                        },
                    )
                    left_purchase = bool(left["purchased_in_session"])
                    right_purchase = bool(right["purchased_in_session"])
                    left_cart = bool(left["carted_in_session"])
                    right_cart = bool(right["carted_in_session"])
                    if left_purchase and right_purchase:
                        payload["purchase_purchase_sessions"] += 1.0
                    elif left_cart and right_cart:
                        payload["cart_cart_sessions"] += 1.0
                    else:
                        payload["cart_purchase_sessions"] += 1.0

        if not engaged.empty and not viewed_only.empty:
            viewed_candidates = viewed_only.head(int(max_view_only_negatives_per_session))
            for _, engaged_row in engaged.iterrows():
                for _, negative_row in viewed_candidates.iterrows():
                    key = tuple(sorted((str(engaged_row["item_id"]), str(negative_row["item_id"]))))
                    payload = pair_counts.setdefault(
                        key,
                        {
                            "purchase_purchase_sessions": 0.0,
                            "cart_cart_sessions": 0.0,
                            "cart_purchase_sessions": 0.0,
                            "negative_view_only_sessions": 0.0,
                        },
                    )
                    payload["negative_view_only_sessions"] += 1.0

    rows: list[dict[str, object]] = []
    for (item_a, item_b), payload in pair_counts.items():
        support_sessions = int(
            payload["purchase_purchase_sessions"]
            + payload["cart_cart_sessions"]
            + payload["cart_purchase_sessions"]
            + payload["negative_view_only_sessions"]
        )
        label_score = (
            2.0 * payload["purchase_purchase_sessions"]
            + 1.5 * payload["cart_cart_sessions"]
            + 1.0 * payload["cart_purchase_sessions"]
            - 1.0 * payload["negative_view_only_sessions"]
        )
        rows.append(
            {
                "item_a": str(item_a),
                "item_b": str(item_b),
                "purchase_purchase_sessions": int(payload["purchase_purchase_sessions"]),
                "cart_cart_sessions": int(payload["cart_cart_sessions"]),
                "cart_purchase_sessions": int(payload["cart_purchase_sessions"]),
                "negative_view_only_sessions": int(payload["negative_view_only_sessions"]),
                "support_sessions": int(support_sessions),
                "label_score": float(label_score),
                "proxy_label": int(label_score > 0.0),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = _empty_feature_frame(
            [
                "item_a",
                "item_b",
                "purchase_purchase_sessions",
                "cart_cart_sessions",
                "cart_purchase_sessions",
                "negative_view_only_sessions",
                "support_sessions",
                "label_score",
                "proxy_label",
            ]
        )
    else:
        out = out.sort_values(["label_score", "support_sessions", "item_a", "item_b"], ascending=[False, False, True, True])
    return out.reset_index(drop=True), {
        "sessions_considered": int(sessions_considered),
        "sessions_skipped_high_cardinality": int(sessions_skipped),
    }


def _actor_session_state_to_frame(
    state: dict[tuple[Any, ...], dict[str, object]],
    session_item_features: pd.DataFrame,
) -> pd.DataFrame:
    if not state:
        return build_actor_session_features(pd.DataFrame())

    rows = []
    for key, payload in state.items():
        actor_id, actor_type, session_id, session_strategy = key
        rows.append(
            {
                "actor_id": str(actor_id),
                "actor_type": str(actor_type),
                "session_id": str(session_id),
                "session_strategy": str(session_strategy),
                "event_count": int(payload["event_count"]),
                "item_count": int(payload["event_count"]),
                "first_event_ts": payload["first_event_ts"],
                "last_event_ts": payload["last_event_ts"],
                "view_count": int(payload["view_count"]),
                "add_to_cart_count": int(payload["add_to_cart_count"]),
                "purchase_count": int(payload["purchase_count"]),
            }
        )
    out = pd.DataFrame(rows)
    unique_counts = (
        session_item_features.groupby(["actor_id", "actor_type", "session_id", "session_strategy"], observed=True, sort=False)
        .size()
        .rename("unique_item_count")
        .reset_index()
    )
    out = out.merge(unique_counts, on=["actor_id", "actor_type", "session_id", "session_strategy"], how="left")
    out["unique_item_count"] = out["unique_item_count"].fillna(0).astype("int32")
    out["session_duration_sec"] = (
        out["last_event_ts"] - out["first_event_ts"]
    ).dt.total_seconds().fillna(0.0).astype("float32")
    return out


def _actor_item_state_to_frame(
    state: dict[tuple[Any, ...], dict[str, object]],
    session_item_features: pd.DataFrame,
) -> pd.DataFrame:
    if not state:
        return build_actor_item_features(pd.DataFrame())

    rows = []
    for key, payload in state.items():
        actor_id, actor_type, item_id = key
        rows.append(
            {
                "actor_id": str(actor_id),
                "actor_type": str(actor_type),
                "item_id": str(item_id),
                "event_count": int(payload["event_count"]),
                "total_views": int(payload["total_views"]),
                "total_add_to_cart": int(payload["total_add_to_cart"]),
                "total_purchases": int(payload["total_purchases"]),
                "first_seen_ts": payload["first_seen_ts"],
                "last_seen_ts": payload["last_seen_ts"],
            }
        )
    out = pd.DataFrame(rows)
    session_counts = (
        session_item_features.groupby(["actor_id", "actor_type", "item_id"], observed=True, sort=False)
        .size()
        .rename("session_count")
        .reset_index()
    )
    out = out.merge(session_counts, on=["actor_id", "actor_type", "item_id"], how="left")
    out["session_count"] = out["session_count"].fillna(0).astype("int32")
    reference_ts = out["last_seen_ts"].max()
    if pd.isna(reference_ts):
        reference_ts = pd.Timestamp.utcnow(tz="UTC")
    out["activity_span_days"] = (
        out["last_seen_ts"] - out["first_seen_ts"]
    ).dt.total_seconds().div(86400.0).fillna(0.0).astype("float32")
    out["days_since_last_seen"] = (
        pd.Timestamp(reference_ts) - out["last_seen_ts"]
    ).dt.total_seconds().div(86400.0).fillna(0.0).astype("float32")
    out["events_per_session"] = _safe_divide(out["event_count"], out["session_count"])
    out["view_to_cart_rate"] = _safe_divide(out["total_add_to_cart"], out["total_views"])
    out["add_to_purchase_rate"] = _safe_divide(out["total_purchases"], out["total_add_to_cart"])
    out["purchase_after_view_rate"] = _safe_divide(out["total_purchases"], out["total_views"])
    return out


def _session_item_state_to_frame(
    state: dict[tuple[Any, ...], dict[str, object]],
    actor_session_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if not state:
        return build_session_item_features(pd.DataFrame(), pd.DataFrame())

    rows = []
    for key, payload in state.items():
        session_id, actor_id, actor_type, session_strategy, item_id = key
        rows.append(
            {
                "session_id": str(session_id),
                "actor_id": str(actor_id),
                "actor_type": str(actor_type),
                "session_strategy": str(session_strategy),
                "item_id": str(item_id),
                "session_item_event_count": int(payload["session_item_event_count"]),
                "viewed_in_session": int(payload["viewed_in_session"]),
                "carted_in_session": int(payload["carted_in_session"]),
                "purchased_in_session": int(payload["purchased_in_session"]),
                "first_event_type": str(payload["first_event_type"]),
                "last_event_type": str(payload["last_event_type"]),
                "first_event_ts": payload["first_event_ts"],
                "last_event_ts": payload["last_event_ts"],
            }
        )
    out = pd.DataFrame(rows)

    first_sort = out.assign(
        first_event_type_order=out["first_event_type"].map(TYPE_ORDER).fillna(99).astype("int8")
    ).sort_values(["session_id", "first_event_ts", "first_event_type_order", "item_id"], kind="mergesort")
    first_sort["first_event_order"] = first_sort.groupby("session_id", observed=True, sort=False).cumcount() + 1
    out = out.merge(
        first_sort.loc[:, ["session_id", "item_id", "first_event_order"]],
        on=["session_id", "item_id"],
        how="left",
    )

    last_sort = out.assign(
        last_event_type_order=out["last_event_type"].map(TYPE_ORDER).fillna(99).astype("int8")
    ).sort_values(["session_id", "last_event_ts", "last_event_type_order", "item_id"], kind="mergesort")
    last_sort["last_event_order"] = last_sort.groupby("session_id", observed=True, sort=False).cumcount() + 1
    out = out.merge(
        last_sort.loc[:, ["session_id", "item_id", "last_event_order"]],
        on=["session_id", "item_id"],
        how="left",
    )

    if actor_session_features is not None and not actor_session_features.empty:
        session_meta = actor_session_features.loc[
            :, ["session_id", "event_count", "first_event_ts", "last_event_ts"]
        ].rename(
            columns={
                "event_count": "session_event_count",
                "first_event_ts": "session_first_ts",
                "last_event_ts": "session_last_ts",
            }
        )
        out = out.merge(session_meta, on="session_id", how="left")
        out["first_event_offset_sec"] = (
            out["first_event_ts"] - out["session_first_ts"]
        ).dt.total_seconds().fillna(0.0).astype("float32")
        out["last_event_recency_sec"] = (
            out["session_last_ts"] - out["last_event_ts"]
        ).dt.total_seconds().fillna(0.0).astype("float32")
        out["first_event_order_norm"] = _safe_divide(out["first_event_order"], out["session_event_count"])
        out["last_event_order_norm"] = _safe_divide(out["last_event_order"], out["session_event_count"])
    else:
        out["first_event_offset_sec"] = 0.0
        out["last_event_recency_sec"] = 0.0
        out["first_event_order_norm"] = 0.0
        out["last_event_order_norm"] = 0.0
    return out


def write_dataframe_parquet(df: pd.DataFrame, path: Path) -> Path:
    pa, pq = _require_parquet_support()
    if path.exists():
        path.unlink()
    writer = _append_parquet_chunk(None, df, path=path, pa=pa, pq=pq)
    if writer is not None:
        writer.close()
    return path


def validate_feedback_etl_artifacts(base_dir: Path | None = None) -> dict[str, object]:
    artifacts = feedback_etl_artifact_paths(base_dir)
    clean_events = _load_clean_events(
        artifacts["clean_events"],
        columns=["feedback_type", "actor_type", "effective_ts"],
    )

    payload = {
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "clean_event_rows": int(len(clean_events)),
        "event_type_counts": (
            {str(k): int(v) for k, v in clean_events["feedback_type"].value_counts().to_dict().items()}
            if not clean_events.empty
            else {}
        ),
        "actor_type_counts": (
            {str(k): int(v) for k, v in clean_events["actor_type"].value_counts().to_dict().items()}
            if not clean_events.empty
            else {}
        ),
        "date_min": "" if clean_events.empty else pd.Timestamp(clean_events["effective_ts"].min()).isoformat(),
        "date_max": "" if clean_events.empty else pd.Timestamp(clean_events["effective_ts"].max()).isoformat(),
    }

    for artifact_name in (
        "actor_session_features",
        "actor_item_features",
        "session_item_features",
        "proxy_pair_labels",
    ):
        try:
            frame = pd.read_parquet(artifacts[artifact_name])
            payload[f"{artifact_name}_rows"] = int(len(frame))
        except Exception:
            payload[f"{artifact_name}_rows"] = 0
    print(json.dumps(payload, ensure_ascii=False))
    return payload


def build_feedback_feature_artifacts(
    *,
    base_dir: Path | None = None,
    input_path: Path | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    carryover_rows: int = DEFAULT_CARRYOVER_ROWS,
    max_pair_items_per_session: int = DEFAULT_MAX_PAIR_ITEMS_PER_SESSION,
    max_view_only_negatives_per_session: int = DEFAULT_MAX_VIEW_ONLY_NEGATIVES_PER_SESSION,
) -> FeedbackETLArtifacts:
    """Run the full feedback telemetry ETL and write processed artifacts."""

    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)
    artifacts = feedback_etl_artifact_paths(paths.project_root)
    source_path = (input_path or default_feedback_source_path(paths.project_root)).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Raw feedback telemetry not found: {source_path}")

    feature_states: dict[str, dict[tuple[Any, ...], dict[str, object]]] = {
        "actor_session": {},
        "actor_item": {},
        "session_item": {},
    }
    clean_report = write_clean_feedback_events(
        input_path=source_path,
        output_path=artifacts["clean_events"],
        chunk_size=int(chunk_size),
        carryover_rows=int(carryover_rows),
        feature_states=feature_states,
    )

    session_item_features = _session_item_state_to_frame(feature_states.get("session_item", {}))
    actor_session_features = _actor_session_state_to_frame(
        feature_states.get("actor_session", {}),
        session_item_features,
    )
    session_item_features = _session_item_state_to_frame(
        feature_states.get("session_item", {}),
        actor_session_features,
    )
    actor_item_features = _actor_item_state_to_frame(
        feature_states.get("actor_item", {}),
        session_item_features,
    )

    write_dataframe_parquet(actor_session_features, artifacts["actor_session_features"])
    write_dataframe_parquet(actor_item_features, artifacts["actor_item_features"])
    write_dataframe_parquet(session_item_features, artifacts["session_item_features"])

    proxy_pair_labels, pair_report = build_proxy_pair_labels(
        session_item_features,
        max_items_per_session=int(max_pair_items_per_session),
        max_view_only_negatives_per_session=int(max_view_only_negatives_per_session),
    )
    write_dataframe_parquet(proxy_pair_labels, artifacts["proxy_pair_labels"])

    validation = validate_feedback_etl_artifacts(paths.project_root)
    report = {
        "clean_report": clean_report,
        "pair_report": pair_report,
        "validation": validation,
    }
    return FeedbackETLArtifacts(
        input_path=source_path,
        clean_events_path=artifacts["clean_events"],
        actor_session_features_path=artifacts["actor_session_features"],
        actor_item_features_path=artifacts["actor_item_features"],
        session_item_features_path=artifacts["session_item_features"],
        proxy_pair_labels_path=artifacts["proxy_pair_labels"],
        report=report,
    )


def build_feedback_feature_artifacts_from_clean_events(
    *,
    base_dir: Path | None = None,
    clean_events_path: Path | None = None,
    batch_size: int = DEFAULT_CHUNK_SIZE,
    max_pair_items_per_session: int = DEFAULT_MAX_PAIR_ITEMS_PER_SESSION,
    max_view_only_negatives_per_session: int = DEFAULT_MAX_VIEW_ONLY_NEGATIVES_PER_SESSION,
) -> FeedbackETLArtifacts:
    """Derive ranking feature tables from an existing clean events parquet."""

    pa, pq = _require_parquet_support()
    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)
    artifacts = feedback_etl_artifact_paths(paths.project_root)
    clean_path = (clean_events_path or artifacts["clean_events"]).resolve()
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean feedback events parquet not found: {clean_path}")

    feature_states: dict[str, dict[tuple[Any, ...], dict[str, object]]] = {
        "actor_session": {},
        "actor_item": {},
        "session_item": {},
    }
    parquet_file = pq.ParquetFile(str(clean_path))
    for batch in parquet_file.iter_batches(
        batch_size=int(batch_size),
        columns=[
            "actor_id",
            "actor_type",
            "item_id",
            "feedback_type",
            "effective_ts",
            "session_id",
            "session_strategy",
        ],
    ):
        frame = batch.to_pandas()
        _update_feature_states(frame, feature_states)

    session_item_features = _session_item_state_to_frame(feature_states.get("session_item", {}))
    actor_session_features = _actor_session_state_to_frame(
        feature_states.get("actor_session", {}),
        session_item_features,
    )
    session_item_features = _session_item_state_to_frame(
        feature_states.get("session_item", {}),
        actor_session_features,
    )
    actor_item_features = _actor_item_state_to_frame(
        feature_states.get("actor_item", {}),
        session_item_features,
    )

    write_dataframe_parquet(actor_session_features, artifacts["actor_session_features"])
    write_dataframe_parquet(actor_item_features, artifacts["actor_item_features"])
    write_dataframe_parquet(session_item_features, artifacts["session_item_features"])

    proxy_pair_labels, pair_report = build_proxy_pair_labels(
        session_item_features,
        max_items_per_session=int(max_pair_items_per_session),
        max_view_only_negatives_per_session=int(max_view_only_negatives_per_session),
    )
    write_dataframe_parquet(proxy_pair_labels, artifacts["proxy_pair_labels"])

    validation = validate_feedback_etl_artifacts(paths.project_root)
    report = {
        "clean_report": {
            "clean_rows": int(parquet_file.metadata.num_rows),
            "input_path": str(clean_path),
        },
        "pair_report": pair_report,
        "validation": validation,
    }
    return FeedbackETLArtifacts(
        input_path=default_feedback_source_path(paths.project_root),
        clean_events_path=clean_path,
        actor_session_features_path=artifacts["actor_session_features"],
        actor_item_features_path=artifacts["actor_item_features"],
        session_item_features_path=artifacts["session_item_features"],
        proxy_pair_labels_path=artifacts["proxy_pair_labels"],
        report=report,
    )
