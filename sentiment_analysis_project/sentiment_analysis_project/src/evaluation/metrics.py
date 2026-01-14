"""Comprehensive evaluation metrics, including calibration and business metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    brier_score_loss,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) for binary or multiclass probs.

    For multiclass, uses max prob as confidence and argmax as prediction.
    """
    if probs.ndim == 2 and probs.shape[1] > 1:
        conf = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == y_true).astype(float)
    else:
        conf = probs.reshape(-1)
        preds = (conf >= 0.5).astype(int)
        correct = (preds == y_true).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.any():
            acc = correct[m].mean()
            c = conf[m].mean()
            ece += (m.mean()) * abs(acc - c)
    return float(ece)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    average: str = "weighted",
    cost_fn_weight: float = 2.0,
    top_k: int = 3,
) -> Dict[str, object]:
    """Compute standard + advanced + business metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)

    out: Dict[str, object] = {
        "accuracy": acc,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # business: cost-sensitive accuracy (binary focus)
    if set(np.unique(y_true)).issubset({0, 1}):
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        cost = cost_fn_weight * fn + 1.0 * fp
        out["cost"] = float(cost)
        out["cost_sensitive_accuracy"] = float((tp + tn) / max(1, (tp + tn + fp + fn)))

    if y_prob is not None:
        # ROC-AUC and PR-AUC
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] > 2:
                out["roc_auc_ovr"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
            elif y_prob.ndim == 2 and y_prob.shape[1] == 2:
                out["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
                out["pr_auc"] = float(average_precision_score(y_true, y_prob[:, 1]))
                out["brier"] = float(brier_score_loss(y_true, y_prob[:, 1]))
                out["ece"] = expected_calibration_error(y_prob[:, 1], y_true)
        except Exception as e:
            log.warning("AUC computation failed: %s", e)

        # top-k accuracy (multiclass)
        if y_prob.ndim == 2 and y_prob.shape[1] > 2:
            top = np.argsort(-y_prob, axis=1)[:, :top_k]
            out["top_k_accuracy"] = float(np.mean([y_true[i] in top[i] for i in range(len(y_true))]))

    return out
