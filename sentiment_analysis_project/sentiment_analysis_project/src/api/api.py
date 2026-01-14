"""FastAPI service for predictions and explanations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.models.sentiment_model import get_sentiment_pipeline, predict_sentiment
from src.features.aspect_extraction import AspectExtractor
from src.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(title="Sentiment & Behavior API", version="2.0.0")

# Lazy-loaded resources
_SENT_PIPE = None
_ASPECT = None


class ReviewRequest(BaseModel):
    review_text: str


class BatchReviewRequest(BaseModel):
    reviews: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/sentiment")
def predict_sentiment_endpoint(req: ReviewRequest):
    global _SENT_PIPE, _ASPECT
    if _SENT_PIPE is None:
        _SENT_PIPE = get_sentiment_pipeline()
    if _ASPECT is None:
        _ASPECT = AspectExtractor()

    labels, scores = predict_sentiment(_SENT_PIPE, [req.review_text])
    absa = _ASPECT.extract(req.review_text, sentiment_pipe=_SENT_PIPE)

    return {
        "sentiment_label": labels[0],
        "sentiment_confidence": scores[0],
        "aspects": absa.aspects,
        "aspect_sentiment": absa.aspect_sentiment,
        "aspect_confidence": absa.aspect_confidence,
    }


@app.post("/predict/sentiment_batch")
def predict_sentiment_batch(req: BatchReviewRequest):
    global _SENT_PIPE
    if _SENT_PIPE is None:
        _SENT_PIPE = get_sentiment_pipeline()
    labels, scores = predict_sentiment(_SENT_PIPE, req.reviews)
    return {"labels": labels, "scores": scores}
