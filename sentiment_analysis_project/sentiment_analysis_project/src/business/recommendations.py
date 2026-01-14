"""Product improvement recommendations from aspect-level signals."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd


def top_negative_aspects(df: pd.DataFrame, k: int = 5, prefix: str = "aspect_sent_") -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return pd.DataFrame(columns=["aspect", "mean_sentiment"])
    means = {c.replace(prefix, ""): float(df[c].fillna(0.0).mean()) for c in cols}
    items = sorted(means.items(), key=lambda x: x[1])
    return pd.DataFrame(items[:k], columns=["aspect", "mean_sentiment"])
