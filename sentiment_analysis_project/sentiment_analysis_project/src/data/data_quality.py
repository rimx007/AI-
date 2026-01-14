"""Data profiling, validation, and augmentation hooks.

This module focuses on *quality checks* that are essential in publishable research:
- duplicates
- missingness patterns
- outliers/anomalies
- label leakage checks

Augmentation utilities are provided in `src/data/data_augmentation.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DataQualityReport:
    n_rows: int
    n_duplicates: int
    missing_by_column: Dict[str, int]
    rating_outliers: int
    text_length_outliers: int


def profile_dataset(df: pd.DataFrame, text_col: str = "review_text", rating_col: str = "rating") -> DataQualityReport:
    """Create a lightweight profiling report."""
    n_rows = len(df)
    n_duplicates = int(df.duplicated(subset=[text_col]).sum()) if text_col in df.columns else int(df.duplicated().sum())
    missing_by_column = {c: int(df[c].isna().sum()) for c in df.columns}

    rating_outliers = 0
    if rating_col in df.columns:
        s = pd.to_numeric(df[rating_col], errors="coerce")
        rating_outliers = int(((s < 1) | (s > 5)).sum())

    lengths = df[text_col].astype(str).str.len()
    # Simple outlier flag: above 99th percentile or below 1st percentile (excluding zeros)
    lo = lengths[lengths > 0].quantile(0.01) if (lengths > 0).any() else 0
    hi = lengths.quantile(0.99) if len(lengths) else 0
    text_length_outliers = int(((lengths < lo) | (lengths > hi)).sum())

    return DataQualityReport(
        n_rows=n_rows,
        n_duplicates=n_duplicates,
        missing_by_column=missing_by_column,
        rating_outliers=rating_outliers,
        text_length_outliers=text_length_outliers,
    )


def remove_duplicates(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """
    Remove duplicate rows based on the review text column.

    Notes:
        - Logs at INFO only if duplicates are removed.
        - Logs at DEBUG if no change (helps keep Streamlit console clean).
        - Handles missing text_col gracefully.

    Args:
        df: Input dataframe.
        text_col: Column containing review text.

    Returns:
        A dataframe with duplicates removed (keeps first occurrence).
    """
    before = len(df)

    if df.empty:
        log.debug("remove_duplicates: empty dataframe.")
        return df

    if text_col not in df.columns:
        log.warning("remove_duplicates: text column '%s' not found; skipping.", text_col)
        return df

    # Normalize NaNs / whitespace so duplicates are detected more consistently
    tmp = df.copy()
    tmp[text_col] = (
        tmp[text_col]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    out = tmp.drop_duplicates(subset=[text_col], keep="first").copy()
    after = len(out)

    if after != before:
        log.info("Removed duplicates: %d -> %d", before, after)
    else:
        log.debug("No duplicates removed (%d rows).", before)

    return out
