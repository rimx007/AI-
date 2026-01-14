"""Text cleaning, optional translation, and normalization.

Design goals:
- Keep sentiment-carrying tokens (!!!, emojis, ALL CAPS) rather than over-normalizing.
- Provide safe, configurable cleaning functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

_HTML_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1F\x7F]")

# Emoji handling: keep emojis but isolate them to help tokenization.
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE,
)


def basic_clean(text: str) -> str:
    """Clean text while preserving sentiment indicators.

    - Removes HTML tags and control chars
    - Replaces URLs with a token
    - Collapses whitespace

    Example:
        basic_clean("Great!! <br> http://x.com") -> "Great!! [URL]"
    """
    if not isinstance(text, str):
        return ""
    t = _HTML_RE.sub(" ", text)
    t = _CONTROL_CHARS_RE.sub(" ", t)
    t = _URL_RE.sub(" [URL] ", t)
    t = _EMOJI_RE.sub(lambda m: f" {m.group(0)} ", t)
    t = _MULTI_SPACE_RE.sub(" ", t).strip()
    return t


def clean_dataframe(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Apply `basic_clean` to a dataframe."""
    out = df.copy()
    out[text_col] = out[text_col].astype(str).map(basic_clean)
    return out


def detect_language_fast(texts: Iterable[str], sample_size: int = 5) -> bool:
    """Heuristic: returns True if all sampled texts appear to be English.

    Uses langdetect if available; otherwise returns True (fail open).

    Note: language detection is noisy for short texts.
    """
    try:
        from langdetect import detect
        import random

        texts = [t for t in texts if isinstance(t, str) and t.strip()]
        if not texts:
            return True
        sample = random.sample(texts, k=min(sample_size, len(texts)))
        return all(detect(t) == "en" for t in sample)
    except Exception:
        return True


def translate_to_english(texts: List[str]) -> List[str]:
    """Translate a list of texts to English (optional).

    WARNING: `googletrans` is often rate-limited and can break without notice.
    For research reproducibility, prefer using an offline translation model or a paid API.

    TODO:
        - Replace with a robust translation provider in production.
        - Cache translations and log failures.

    Args:
        texts: List of input strings.

    Returns:
        List of translated strings (best effort).
    """
    try:
        from googletrans import Translator  # type: ignore
        from langdetect import detect  # type: ignore

        translator = Translator()
        out: List[str] = []
        for t in texts:
            try:
                if detect(t) != "en":
                    out.append(translator.translate(t, src="auto", dest="en").text)
                else:
                    out.append(t)
            except Exception:
                out.append(t)
        return out
    except Exception as e:
        log.warning("Translation dependencies not available or failed; returning original texts. Error=%s", e)
        return texts
