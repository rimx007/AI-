"""Counterfactual explanations.

Uses `alibi` when possible for tabular models. For text, counterfactual generation is non-trivial;
we provide a minimal tabular implementation and a stub for text.

TODO:
- Implement text counterfactuals via constrained paraphrasing + classifier queries.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def tabular_counterfactual(model_predict_proba, X: np.ndarray, target_class: int = 1) -> Any:
    try:
        from alibi.explainers import Counterfactual
        # `feature_range` must be set based on your dataset distribution
        # TODO: compute feature ranges from training data and pass here.
        cf = Counterfactual(model_predict_proba, shape=(1, X.shape[1]), target_proba=0.51, tol=0.01)
        explanation = cf.explain(X, target_class=[target_class])
        return explanation
    except Exception as e:
        log.warning("Counterfactual generation failed: %s", e)
        return None
