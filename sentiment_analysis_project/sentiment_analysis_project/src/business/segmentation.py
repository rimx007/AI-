"""Customer segmentation based on sentiment/aspect patterns."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def segment_customers(df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 4, random_state: int = 42) -> Tuple[pd.Series, KMeans]:
    X = df[feature_cols].fillna(0.0).to_numpy()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xs)
    return pd.Series(labels, index=df.index, name="segment"), km
