"""Baselines for rigorous evaluation.

Includes:
- Majority class baseline
- Rating-only baseline for recommendation
- TF-IDF + Logistic Regression / SVM
- Lexicon-based sentiment (VADER/TextBlob)

Baselines are essential for publishable results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.logger import get_logger

log = get_logger(__name__)


def majority_baseline() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def tfidf_logreg(max_features: int = 50_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)),
        ("clf", LogisticRegression(max_iter=300, n_jobs=None)),
    ])


def tfidf_svm(max_features: int = 50_000) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)),
        ("clf", LinearSVC()),
    ])
