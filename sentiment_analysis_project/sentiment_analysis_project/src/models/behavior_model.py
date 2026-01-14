"""Behavior prediction model.

This module upgrades the prototype RandomForest+SMOTE approach into a more general framework
that can incorporate:
- engineered features
- aspect sentiment features
- calibration

It still provides a strong tree baseline and can be used inside ensembles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from imblearn.over_sampling import SMOTE

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class BehaviorModelResult:
    model: object
    metrics: Dict[str, object]


def train_behavior_model(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
    calibrate: bool = True,
) -> BehaviorModelResult:
    """Train a calibrated RandomForest with SMOTE and return rich metrics.

    Notes:
    - Splits before SMOTE to avoid leakage (critical for publishable evaluation).
    - Optional calibration (Platt scaling) improves probability quality for business decisions.

    Args:
        X: Feature matrix.
        y: Labels (0/1).
        test_size: Test split.
        random_state: Seed.
        calibrate: If True, wrap model in CalibratedClassifierCV.

    Returns:
        BehaviorModelResult(model, metrics)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    sm = SMOTE(random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    base = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    base.fit(X_train_res, y_train_res)

    model = base
    if calibrate:
        model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        model.fit(X_train_res, y_train_res)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)

    metrics = {
        "report": report,
        "test_size": int(len(X_test)),
        "train_size": int(len(X_train)),
    }

    return BehaviorModelResult(model=model, metrics=metrics)
