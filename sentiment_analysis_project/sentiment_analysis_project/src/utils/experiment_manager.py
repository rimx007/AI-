"""Experiment manager: training, evaluation, and MLflow logging.

Usage:
    python -m src.utils.experiment_manager --config config/config.yaml --data data/raw/reviews.csv

This is intentionally modular: each model type can be added as a "runner" with standardized outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import mlflow

from src.data.data_loader import load_reviews_csv
from src.data.data_preprocessing import clean_dataframe
from src.data.data_quality import profile_dataset, remove_duplicates
from src.features.advanced_features import AdvancedFeatureEngineer
from src.features.aspect_extraction import AspectExtractor
from src.models.baseline_models import majority_baseline, tfidf_logreg, tfidf_svm
from src.models.sentiment_model import get_sentiment_pipeline, predict_sentiment
from src.models.behavior_model import train_behavior_model
from src.evaluation.metrics import compute_classification_metrics
from src.utils.config_loader import load_config
from src.utils.seed import set_global_seed
from src.utils.logger import get_logger

log = get_logger(__name__)


def _normalize_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0}).fillna(0).astype(int)


def run(config_path: str, data_path: str) -> Dict[str, object]:
    cfg = load_config(config_path)
    seed = int(cfg["data"]["random_seed"])
    set_global_seed(seed)

    df = load_reviews_csv(data_path)
    df = remove_duplicates(df)
    df = clean_dataframe(df, cfg["data"]["text_column"])

    # Optional task: behavior prediction if labels exist
    recommend_col = cfg["data"]["recommend_column"]
    has_behavior = recommend_col in df.columns

    # Sentiment (weak labels could be derived from rating if no gold)
    sent_pipe = get_sentiment_pipeline()
    labels, conf = predict_sentiment(sent_pipe, df[cfg["data"]["text_column"]].tolist())
    df["sentiment_label"] = labels
    df["sentiment_confidence"] = conf

    # ABSA extraction + aspect sentiment features
    absa = AspectExtractor()
    aspects = []
    for t in df[cfg["data"]["text_column"]].tolist()[:5000]:  # cap for CLI speed; remove cap for full training
        aspects.append(absa.extract(t, sentiment_pipe=sent_pipe))

    # convert aspect sentiment dicts to columns
    all_aspects = sorted({a for r in aspects for a in r.aspect_sentiment.keys()})
    for a in all_aspects:
        df[f"aspect_sent_{a}"] = [r.aspect_sentiment.get(a, 0.0) for r in aspects]
        df[f"aspect_conf_{a}"] = [r.aspect_confidence.get(a, 0.0) for r in aspects]

    # advanced features
    fe = AdvancedFeatureEngineer()
    feat_df = fe.transform(
        df,
        text_col=cfg["data"]["text_column"],
        rating_col=cfg["data"]["rating_column"],
        verified_col=cfg["data"]["verified_column"],
        helpful_col=cfg["data"]["helpful_votes_column"],
        total_votes_col=cfg["data"]["total_votes_column"],
        sentiment_col="sentiment_label",
        sentiment_conf_col="sentiment_confidence",
    )

    results: Dict[str, object] = {"data_quality": profile_dataset(df).__dict__}

    # MLflow
    tracking = cfg.get("tracking", {}).get("mlflow", {})
    mlflow_enabled = bool(tracking.get("enabled", True))
    if mlflow_enabled:
        mlflow.set_tracking_uri(tracking.get("tracking_uri", "mlruns"))
        mlflow.set_experiment(tracking.get("experiment_name", "absa_multitask"))

    # Behavior task
    if has_behavior:
        y = _normalize_bool_series(df[recommend_col])
        X = feat_df

        if mlflow_enabled:
            with mlflow.start_run(run_name="behavior_rf_features"):
                res = train_behavior_model(X, y, test_size=cfg["data"]["train_test_split"], random_state=seed)
                # compute metrics with probabilities if available
                y_pred = res.model.predict(X)
                met = compute_classification_metrics(y.to_numpy(), y_pred)
                mlflow.log_metrics({k: float(v) for k, v in met.items() if isinstance(v, (int, float))})
                results["behavior_rf"] = {"metrics": met, "report": res.metrics["report"]}
        else:
            res = train_behavior_model(X, y, test_size=cfg["data"]["train_test_split"], random_state=seed)
            results["behavior_rf"] = res.metrics

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    args = ap.parse_args()

    out = run(args.config, args.data)
    log.info("Experiment complete. Keys: %s", list(out.keys()))


if __name__ == "__main__":
    main()
