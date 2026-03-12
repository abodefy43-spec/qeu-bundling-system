"""Lightweight dataframe/artifact schema validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


class DataValidationError(ValueError):
    """Raised when a dataframe/artifact violates expected schema rules."""


@dataclass(frozen=True)
class NumericRangeRule:
    minimum: float | None = None
    maximum: float | None = None
    allow_null: bool = True


def require_columns(df: pd.DataFrame, required: Iterable[str], artifact_name: str) -> None:
    required_list = list(required)
    missing = [col for col in required_list if col not in df.columns]
    if missing:
        raise DataValidationError(
            f"{artifact_name}: missing required columns: {', '.join(missing)}"
        )


def require_not_empty(df: pd.DataFrame, artifact_name: str) -> None:
    if df.empty:
        raise DataValidationError(f"{artifact_name}: dataframe is empty")


def require_no_nulls(df: pd.DataFrame, columns: Iterable[str], artifact_name: str) -> None:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return
    null_counts = df[cols].isna().sum()
    offenders = [f"{c}={int(n)}" for c, n in null_counts.items() if int(n) > 0]
    if offenders:
        raise DataValidationError(
            f"{artifact_name}: null values found in required columns: {', '.join(offenders)}"
        )


def require_unique_rows(df: pd.DataFrame, subset: Iterable[str], artifact_name: str) -> None:
    subset_cols = [c for c in subset if c in df.columns]
    if not subset_cols:
        return
    dup_count = int(df.duplicated(subset=subset_cols).sum())
    if dup_count > 0:
        raise DataValidationError(
            f"{artifact_name}: found {dup_count} duplicated rows for keys {subset_cols}"
        )


def validate_numeric_ranges(
    df: pd.DataFrame,
    rules: dict[str, NumericRangeRule],
    artifact_name: str,
) -> None:
    for col, rule in rules.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if not rule.allow_null and series.isna().any():
            raise DataValidationError(f"{artifact_name}: column `{col}` contains null/non-numeric values")
        values = series.dropna()
        if rule.minimum is not None and (values < rule.minimum).any():
            min_found = float(values.min())
            raise DataValidationError(
                f"{artifact_name}: column `{col}` violates minimum {rule.minimum} (found {min_found})"
            )
        if rule.maximum is not None and (values > rule.maximum).any():
            max_found = float(values.max())
            raise DataValidationError(
                f"{artifact_name}: column `{col}` violates maximum {rule.maximum} (found {max_found})"
            )


def require_files_exist(base: Path, filenames: Iterable[str], artifact_group_name: str) -> None:
    missing = [name for name in filenames if not (base / name).exists()]
    if missing:
        raise DataValidationError(
            f"{artifact_group_name}: missing required files in {base}: {', '.join(missing)}"
        )
