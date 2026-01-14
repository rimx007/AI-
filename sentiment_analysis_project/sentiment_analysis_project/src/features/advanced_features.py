"""Advanced feature engineering for reviews.

Features implemented:
- Linguistic: readability, lexical diversity, POS distribution, NER counts
- Emotional: VADER intensities, TextBlob polarity/subjectivity
- Length stats: chars/words/sentences
- Social proof: helpful ratio, verified purchase
- Meta: rating-sentiment discordance, sentiment confidence

All features are returned as a numeric pandas DataFrame suitable for ML models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

log = get_logger(__name__)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


class AdvancedFeatureEngineer:
    """Compute engineered features from review text and metadata."""

    def __init__(self, spacy_model: str = "en_core_web_sm") -> None:
        self.spacy_model = spacy_model
        self._nlp = None

    def _load_spacy(self):
        if self._nlp is None:
            import spacy
            self._nlp = spacy.load(self.spacy_model, disable=["lemmatizer"])
            if "sentencizer" not in self._nlp.pipe_names:
                self._nlp.add_pipe("sentencizer")

    def _readability(self, text: str) -> Dict[str, float]:
        try:
            import textstat
            return {
                "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
                "gunning_fog": float(textstat.gunning_fog(text)),
            }
        except Exception:
            return {"flesch_kincaid_grade": 0.0, "gunning_fog": 0.0}

    def _vader(self, text: str) -> Dict[str, float]:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            analyzer = SentimentIntensityAnalyzer()
            s = analyzer.polarity_scores(text)
            return {
                "vader_neg": float(s["neg"]),
                "vader_neu": float(s["neu"]),
                "vader_pos": float(s["pos"]),
                "vader_compound": float(s["compound"]),
            }
        except Exception:
            return {"vader_neg": 0.0, "vader_neu": 0.0, "vader_pos": 0.0, "vader_compound": 0.0}

    def _textblob(self, text: str) -> Dict[str, float]:
        try:
            from textblob import TextBlob
            tb = TextBlob(text)
            return {"tb_polarity": float(tb.sentiment.polarity), "tb_subjectivity": float(tb.sentiment.subjectivity)}
        except Exception:
            return {"tb_polarity": 0.0, "tb_subjectivity": 0.0}

    def _pos_ner(self, text: str) -> Dict[str, float]:
        self._load_spacy()
        doc = self._nlp(text)

        pos_counts: Dict[str, int] = {}
        for tok in doc:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1

        total = max(1, sum(pos_counts.values()))
        pos_feats = {f"pos_pct_{k.lower()}": v / total for k, v in pos_counts.items()}

        n_entities = len(doc.ents)
        n_numbers = sum(1 for t in doc if t.like_num)
        return {**pos_feats, "ner_count": float(n_entities), "num_token_count": float(n_numbers)}

    def _lexical_diversity(self, text: str) -> float:
        tokens = [t for t in text.lower().split() if t.isalpha()]
        if not tokens:
            return 0.0
        return float(len(set(tokens)) / max(1, len(tokens)))

    def transform(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
        rating_col: str = "rating",
        verified_col: str = "verified_purchase",
        helpful_col: str = "helpful_votes",
        total_votes_col: str = "total_votes",
        sentiment_col: str = "sentiment_label",
        sentiment_conf_col: str = "sentiment_confidence",
    ) -> pd.DataFrame:
        """Compute features for each row.

        Returns:
            features_df: numeric DataFrame indexed like df
        """
        feats = []
        for _, row in df.iterrows():
            text = str(row.get(text_col, ""))

            # length stats
            chars = len(text)
            words = len(text.split())
            # sentence count via naive split (fast) + spacy sentencizer later in pos_ner
            sentences = max(1, text.count(".") + text.count("!") + text.count("?"))

            f: Dict[str, float] = {
                "char_count": float(chars),
                "word_count": float(words),
                "sentence_count": float(sentences),
                "type_token_ratio": self._lexical_diversity(text),
            }

            f.update(self._readability(text))
            f.update(self._vader(text))
            f.update(self._textblob(text))

            # POS/NER (heavier)
            try:
                f.update(self._pos_ner(text))
            except Exception:
                # keep going if spacy isn't installed
                pass

            # social proof
            helpful = float(row.get(helpful_col, 0) or 0)
            total = float(row.get(total_votes_col, 0) or 0)
            f["helpful_ratio"] = _safe_div(helpful, total) if total else 0.0
            verified = row.get(verified_col, False)
            f["verified_purchase"] = float(bool(verified))

            # meta: rating-sentiment discordance (requires rating + sentiment label)
            rating = row.get(rating_col, None)
            rating_num = float(rating) if rating is not None and str(rating).strip() != "" else np.nan
            f["rating"] = float(rating_num) if not np.isnan(rating_num) else 0.0

            sent_label = str(row.get(sentiment_col, "")).upper()
            # map to signed sentiment
            sent_sign = 0.0
            if "NEG" in sent_label:
                sent_sign = -1.0
            elif "POS" in sent_label:
                sent_sign = 1.0
            f["rating_sentiment_discordance"] = abs((rating_num - 3.0) / 2.0 - sent_sign) if not np.isnan(rating_num) else 0.0

            conf = row.get(sentiment_conf_col, None)
            f["sentiment_confidence"] = float(conf) if conf is not None else 0.0

            feats.append(f)

        feat_df = pd.DataFrame(feats, index=df.index).fillna(0.0)
        return feat_df
