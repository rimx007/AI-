"""Ensemble methods: stacking and weighted voting.

This module supports combining:
- Transformer probabilities (sentiment/recommend)
- BiLSTM probabilities
- Tree models (XGBoost/RandomForest) on engineered features

For publishable work, report:
- component performance
- ensemble performance
- calibration impact
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.utils.logger import get_logger

log = get_logger(__name__)


def weighted_voting(probs: List[np.ndarray], weights: List[float]) -> np.ndarray:
    """Weighted average of probability arrays."""
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    out = np.zeros_like(probs[0])
    for p, wi in zip(probs, w):
        out += wi * p
    return out


class StackingEnsemble:
    """Simple stacking using logistic regression as meta-learner."""

    def __init__(self) -> None:
        self.meta = LogisticRegression(max_iter=500)

    def fit(self, base_probs: List[np.ndarray], y: np.ndarray) -> None:
        X = np.hstack(base_probs)
        self.meta.fit(X, y)

    def predict_proba(self, base_probs: List[np.ndarray]) -> np.ndarray:
        X = np.hstack(base_probs)
        return self.meta.predict_proba(X)
