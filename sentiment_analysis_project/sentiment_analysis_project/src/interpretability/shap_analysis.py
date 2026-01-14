"""SHAP utilities for trees and neural models.

- Tree SHAP for RF/XGBoost
- Deep SHAP for torch models (best effort)

In Streamlit, you can render SHAP plots via matplotlib.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import shap

from src.utils.logger import get_logger

log = get_logger(__name__)


def tree_shap_values(model: Any, X):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def explain_tree_global(model: Any, X, class_index: Optional[int] = None):
    shap_values = tree_shap_values(model, X)
    if isinstance(shap_values, list) and class_index is not None:
        vals = shap_values[class_index]
    elif isinstance(shap_values, list):
        vals = shap_values[0]
    else:
        vals = shap_values
    return vals


def deep_shap_values(model, X):
    """Best-effort Deep SHAP. For many transformers, Integrated Gradients may be more stable."""
    try:
        explainer = shap.DeepExplainer(model, X)
        return explainer.shap_values(X)
    except Exception as e:
        log.warning("Deep SHAP failed: %s", e)
        return None
