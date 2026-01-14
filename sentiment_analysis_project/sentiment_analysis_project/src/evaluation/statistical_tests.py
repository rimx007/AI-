"""Statistical tests for model comparison.

Includes:
- McNemar's test (paired classification)
- Bootstrap confidence intervals
- Friedman test + Nemenyi post-hoc (multi-model multi-dataset)

Note: Some tests assume independent folds; report assumptions in the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import chi2, friedmanchisquare

from src.utils.logger import get_logger

log = get_logger(__name__)


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """McNemar's test for paired nominal data.

    Returns chi-square statistic and p-value (with continuity correction).
    """
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)

    n01 = int(np.sum(a_correct & (~b_correct)))
    n10 = int(np.sum((~a_correct) & b_correct))

    # continuity correction
    chi2_stat = ((abs(n01 - n10) - 1) ** 2) / max(1, (n01 + n10))
    p = float(1 - chi2.cdf(chi2_stat, df=1))
    return {"chi2": float(chi2_stat), "p_value": p, "n01": float(n01), "n10": float(n10)}


def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, y_pred: np.ndarray, n: int = 2000, alpha: float = 0.05) -> Dict[str, float]:
    """Bootstrap CI for a metric."""
    rng = np.random.default_rng(42)
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        samp = rng.choice(idx, size=len(idx), replace=True)
        vals.append(metric_fn(y_true[samp], y_pred[samp]))
    vals = np.array(vals)
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return {"low": lo, "high": hi, "mean": float(vals.mean())}


def friedman_test(scores: np.ndarray) -> Dict[str, float]:
    """Friedman test.

    Args:
        scores: shape (n_datasets_or_folds, n_models)

    Returns:
        statistic and p-value.
    """
    stat, p = friedmanchisquare(*[scores[:, i] for i in range(scores.shape[1])])
    return {"statistic": float(stat), "p_value": float(p)}
