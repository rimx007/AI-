"""Attention visualization utilities.

Supports:
- Transformer attention rollout (if model returns attentions)
- Heatmap generation for Streamlit (plotly)

Note:
- Some HF models require `output_attentions=True`.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def attention_rollout(attentions: List[np.ndarray]) -> np.ndarray:
    """Compute attention rollout across layers (Abnar & Zuidema style).

    Args:
        attentions: list of attention tensors per layer (L, H, T, T)

    Returns:
        token importance vector (T,)
    """
    # Average heads, add identity, normalize, multiply
    A = None
    for layer in attentions:
        a = layer.mean(axis=0)  # (T,T)
        t = a.shape[-1]
        a = a + np.eye(t)
        a = a / a.sum(axis=-1, keepdims=True)
        A = a if A is None else A @ a
    # importance is CLS row
    return A[0]
