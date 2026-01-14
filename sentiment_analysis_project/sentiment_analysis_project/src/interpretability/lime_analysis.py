"""LIME explanations for text or tabular features."""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def lime_text_explainer(predict_proba_fn: Callable[[List[str]], np.ndarray], class_names: List[str]):
    from lime.lime_text import LimeTextExplainer
    return LimeTextExplainer(class_names=class_names)


def explain_instance_text(explainer, text: str, predict_proba_fn, num_features: int = 10):
    return explainer.explain_instance(text, predict_proba_fn, num_features=num_features)
