"""Revenue impact estimation based on aspect sentiment.

These methods are *models of impact*, not ground truth. In the thesis, state assumptions clearly.

Typical use:
- Estimate how much negative shipping sentiment reduces purchase recommendation probability.
- Prioritize fixes by estimated ROI.

Requires:
- aspect sentiment scores per review
- (optional) product price / margin info

TODO:
- Calibrate impact model with real conversion data if available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RevenueImpact:
    aspect: str
    avg_sentiment: float
    prevalence: float
    estimated_revenue_loss: float


def estimate_revenue_impact(
    df: pd.DataFrame,
    aspect_sent_cols_prefix: str = "aspect_sent_",
    revenue_per_recommend: float = 20.0,
) -> pd.DataFrame:
    """Estimate revenue loss per aspect using a simple prevalence-weighted model."""
    aspect_cols = [c for c in df.columns if c.startswith(aspect_sent_cols_prefix)]
    rows = []
    n = len(df)
    for c in aspect_cols:
        aspect = c.replace(aspect_sent_cols_prefix, "")
        s = df[c].fillna(0.0)
        prevalence = float((s != 0).mean())
        avg = float(s[s != 0].mean()) if (s != 0).any() else 0.0
        # Negative sentiment implies loss, scaled by prevalence
        loss = float(max(0.0, -avg) * prevalence * n * revenue_per_recommend)
        rows.append({"aspect": aspect, "avg_sentiment": avg, "prevalence": prevalence, "estimated_revenue_loss": loss})
    return pd.DataFrame(rows).sort_values("estimated_revenue_loss", ascending=False)
