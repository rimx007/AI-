"""Dataset I/O and column normalization.

This module consolidates what used to live in the prototype Streamlit app.

Example:
    from src.data.data_loader import load_reviews_csv
    df = load_reviews_csv("data/raw/reviews.csv")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


DEFAULT_RENAME_MAP = {
    "reviews.text": "review_text",
    "review": "review_text",
    "text": "review_text",
    "content": "review_text",
    "body": "review_text",
    "summary": "summary",
    "rating": "rating",
    "stars": "rating",
    "verified_purchase": "verified_purchase",
    "verified": "verified_purchase",
    "helpful_vote": "helpful_votes",
    "helpful_votes": "helpful_votes",
    "total_votes": "total_votes",
    "date": "timestamp",
    "time": "timestamp",
    "parent_asin": "product_id",
    "asin": "product_id",
    "reviewerid": "reviewer_id",
}


def load_reviews_csv(path_or_buffer, sep: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV/TSV review file with robust fallbacks.

    Args:
        path_or_buffer: File path or file-like object.
        sep: Optional separator. If None, tries comma then tab.

    Returns:
        DataFrame with normalized column names where possible.
    """
    if sep is not None:
        df = pd.read_csv(path_or_buffer, sep=sep, engine="python", on_bad_lines="skip")
    else:
        try:
            df = pd.read_csv(path_or_buffer, sep=",", engine="python", quoting=3, on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(path_or_buffer, sep="\t", engine="python", on_bad_lines="skip")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Rename known columns (case-insensitive)
    renames: Dict[str, str] = {}
    for c in df.columns:
        key = c.strip().lower()
        if key in DEFAULT_RENAME_MAP:
            renames[c] = DEFAULT_RENAME_MAP[key]
    if renames:
        df = df.rename(columns=renames)

    # Ensure required column exists
    if "review_text" not in df.columns:
        raise ValueError("Missing required column 'review_text'. Provide or map it in your dataset.")

    df = df.dropna(subset=["review_text"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()

    return df
