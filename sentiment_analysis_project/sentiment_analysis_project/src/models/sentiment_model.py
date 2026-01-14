"""
Transformer-based sentiment model (production-grade).

This module provides:
- Hugging Face sentiment pipeline creation with device management
- Robust, batched inference with optional progress callbacks
- Confidence scores and normalized labels
- Logging hooks to prove where long runs "stop"

Why batching matters:
- Running inference on very large datasets (e.g., 640k reviews) without batching
  can look "stuck" in Streamlit and may exhaust memory.

Example usage:
    from src.models.sentiment_model import get_sentiment_pipeline, predict_sentiment_batched

    pipe = get_sentiment_pipeline()
    labels, scores = predict_sentiment_batched(pipe, ["Great product!", "Terrible shipping..."])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import pipeline
from transformers.utils import logging as hf_logging

from src.utils.logger import get_logger

log = get_logger(__name__)


# Optional: silence noisy HF warnings globally for this module.
# If you still want warnings, change to hf_logging.set_verbosity_warning().
hf_logging.set_verbosity_error()


@dataclass(frozen=True)
class SentimentPrediction:
    """Single sentiment prediction output."""
    label: str
    score: float


ProgressCallback = Callable[[float], None]
# ProgressCallback receives a float in [0, 1], e.g. 0.42 = 42%.


def _auto_device_index(device: Optional[int] = None) -> int:
    """
    Resolve HF pipeline device index.

    Returns:
        0..N for CUDA device index, or -1 for CPU.
    """
    if device is not None:
        return device
    return 0 if torch.cuda.is_available() else -1


def get_sentiment_pipeline(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    device: Optional[int] = None,
    max_length: int = 256,
) :
    """
    Create a Hugging Face pipeline for sentiment analysis.

    Args:
        model_name: HF model id.
        device: GPU index (0,1,...) or -1 for CPU, or None for auto-detect.
        max_length: Token truncation length. 256 is faster; 512 is more expensive.

    Returns:
        Hugging Face pipeline object.

    Notes:
        - The warning "Some weights ... not used" is normal for some checkpoints.
        - If you want to fine-tune later, replace pipeline-based inference with
          a Trainer/Accelerate training script.
    """
    device_idx = _auto_device_index(device)

    log.info(
        "Loading sentiment pipeline. model=%s device=%s max_length=%d",
        model_name,
        "cuda:0" if device_idx >= 0 else "cpu",
        max_length,
    )

    try:
        sent_pipe = pipeline(
            task="sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=max_length,
            device=device_idx,
        )
    except Exception as e:
        # This makes failures explicit instead of looking like the app "stopped".
        log.exception("Failed to load sentiment pipeline: %s", str(e))
        raise

    log.info("Sentiment model loaded successfully.")
    return sent_pipe


def _normalize_label(label: str) -> str:
    """
    Normalize labels across sentiment checkpoints.

    Some models output: LABEL_0/LABEL_1/LABEL_2 or NEGATIVE/NEUTRAL/POSITIVE.
    This function keeps it safe and consistent.
    """
    if not isinstance(label, str):
        return "NEUTRAL"

    up = label.strip().upper()

    # Common Twitter-RoBERTa labels: negative/neutral/positive
    if up in {"NEGATIVE", "NEUTRAL", "POSITIVE"}:
        return up

    # Some models output "LABEL_0" style; leave as-is but uppercase
    return up


def predict_sentiment(
    sentiment_pipeline,
    texts: Sequence[str],
) -> Tuple[List[str], List[float]]:
    """
    Predict sentiment for a batch (no progress bar, no batching).

    Prefer `predict_sentiment_batched` for large datasets.

    Args:
        sentiment_pipeline: pipeline returned by get_sentiment_pipeline().
        texts: input texts.

    Returns:
        (labels, scores)
    """
    safe_texts = [(t if isinstance(t, str) else "") for t in texts]
    try:
        outputs = sentiment_pipeline(safe_texts)
    except Exception as e:
        log.exception("Sentiment inference failed: %s", str(e))
        raise

    labels = [_normalize_label(o.get("label", "NEUTRAL")) for o in outputs]
    scores = [float(o.get("score", 0.0)) for o in outputs]
    return labels, scores


def predict_sentiment_batched(
    sentiment_pipeline,
    texts: Sequence[str],
    batch_size: int = 32,
    progress_cb: Optional[ProgressCallback] = None,
    log_every: int = 20,
) -> Tuple[List[str], List[float]]:
    """
    Batched sentiment inference with optional progress callback.

    This is the function you want for Streamlit so the UI doesn't look frozen.

    Args:
        sentiment_pipeline: HF pipeline.
        texts: input texts.
        batch_size: number of texts per model call (tune for GPU/CPU memory).
        progress_cb: optional callback receiving progress fraction in [0, 1].
                     Example: progress_cb = lambda p: st.progress(p)
        log_every: log progress every N batches.

    Returns:
        (labels, scores) aligned with input order.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    n = len(texts)
    if n == 0:
        return [], []

    log.info("Starting sentiment inference on %d texts (batch_size=%d).", n, batch_size)

    all_labels: List[str] = []
    all_scores: List[float] = []

    # Ensure safe strings
    safe_texts = [(t if isinstance(t, str) else "") for t in texts]

    num_batches = (n + batch_size - 1) // batch_size

    for b_idx in range(num_batches):
        start = b_idx * batch_size
        end = min(start + batch_size, n)
        batch = safe_texts[start:end]

        try:
            outputs = sentiment_pipeline(batch)
        except Exception as e:
            log.exception(
                "Sentiment inference failed at batch %d/%d (rows %d..%d): %s",
                b_idx + 1,
                num_batches,
                start,
                end,
                str(e),
            )
            raise

        all_labels.extend(_normalize_label(o.get("label", "NEUTRAL")) for o in outputs)
        all_scores.extend(float(o.get("score", 0.0)) for o in outputs)

        # Progress update (Streamlit-friendly)
        if progress_cb is not None:
            progress_cb(end / n)

        # Periodic logs (helps you prove it's working)
        if (b_idx + 1) % max(1, log_every) == 0 or (b_idx + 1) == num_batches:
            log.info("Inference progress: %d/%d texts", end, n)

    log.info("Inference completed successfully.")
    return all_labels, all_scores


def predict_one(
    sentiment_pipeline,
    text: str,
) -> SentimentPrediction:
    """
    Convenience function: predict sentiment for a single text.

    Args:
        sentiment_pipeline: HF pipeline
        text: input string

    Returns:
        SentimentPrediction(label, score)
    """
    labels, scores = predict_sentiment(sentiment_pipeline, [text])
    return SentimentPrediction(label=labels[0], score=scores[0])
