"""Error analysis utilities for publishable reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def misclassified_examples(df: pd.DataFrame, y_true_col: str, y_pred_col: str, text_col: str = "review_text", k: int = 25) -> pd.DataFrame:
    """Return a dataframe of misclassified examples."""
    err = df[df[y_true_col] != df[y_pred_col]].copy()
    err["text_len"] = err[text_col].astype(str).str.len()
    return err.sort_values("text_len", ascending=False).head(k)


def breakdown_by_length(df: pd.DataFrame, y_true_col: str, y_pred_col: str, text_col: str = "review_text") -> pd.DataFrame:
    """Performance breakdown by text length bins."""
    tmp = df.copy()
    tmp["len"] = tmp[text_col].astype(str).str.len()
    tmp["bin"] = pd.qcut(tmp["len"], q=5, duplicates="drop")
    grp = tmp.groupby("bin").apply(lambda g: (g[y_true_col] == g[y_pred_col]).mean())
    return grp.reset_index(name="accuracy")
