"""Aspect extraction + aspect sentiment (ABSA core).

This module implements a **hybrid ABSA** approach that is robust with limited labels:

1) **Candidate aspect mining**: noun phrases + entities using spaCy.
2) **Aspect mapping**: map candidates to a fixed aspect inventory (quality, price, shipping, ...).
   - Uses a seed lexicon + optional embedding similarity.
3) **Aspect sentiment scoring**:
   - Uses a transformer sentiment model on sentences mentioning each aspect.

Why hybrid?
- Pure supervised ABSA requires aspect-level labeled data, often unavailable in theses.
- This approach provides a publishable, strong baseline and can be extended with weak supervision
  or distant labels (e.g., aspect lexicons, star ratings).

Example:
    from src.features.aspect_extraction import AspectExtractor
    extractor = AspectExtractor()
    aspects = extractor.extract_aspects("The price is great but shipping was slow.")
    # -> {"price": ["price"], "shipping": ["shipping"]}
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger
import spacy
log = get_logger(__name__)


DEFAULT_ASPECTS: Dict[str, List[str]] = {
    "quality": ["quality", "build", "material", "defect", "broken", "damage", "craftsmanship"],
    "price": ["price", "cost", "value", "worth", "expensive", "cheap", "deal"],
    "shipping": ["shipping", "delivery", "arrived", "packaging", "box", "late", "carrier"],
    "durability": ["durable", "durability", "last", "sturdy", "wear", "tear"],
    "design": ["design", "look", "style", "color", "size", "fit", "appearance"],
    "customer_service": ["service", "support", "customer", "refund", "return", "replacement"],
}


@dataclass
class AspectResult:
    aspects: Dict[str, List[str]]
    aspect_sentiment: Dict[str, float]  # -1..+1
    aspect_confidence: Dict[str, float]  # 0..1


class AspectExtractor:
    """Extract aspects and compute aspect sentiment scores."""

    def __init__(
        self,
        aspects_lexicon: Optional[Dict[str, List[str]]] = None,
        spacy_model: str = "en_core_web_sm",
        use_embedding_similarity: bool = False,
        embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.aspects_lexicon = aspects_lexicon or DEFAULT_ASPECTS
        self.spacy_model = spacy_model
        self.use_embedding_similarity = use_embedding_similarity
        self.embed_model = embed_model

        self._nlp = None
        self._embedder = None
        self._aspect_embeds = None

    def _load_spacy(self):
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(self.spacy_model)

            # Ensure sentence splitting exists
                if "sentencizer" not in self._nlp.pipe_names:
                    self._nlp.add_pipe("sentencizer")

            except Exception as e:
                raise RuntimeError(
                    "spaCy model not available. Run: python -m spacy download en_core_web_sm"
                ) from e

    
    def _load_embedder(self) -> None:
        if not self.use_embedding_similarity:
            return
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._embedder = SentenceTransformer(self.embed_model)
                aspect_names = list(self.aspects_lexicon.keys())
                self._aspect_embeds = self._embedder.encode(
                aspect_names, normalize_embeddings=True
                )
            except Exception as e:
                log.warning("Embedding similarity disabled. Error=%s", e)
                self.use_embedding_similarity = False
                self._embedder = None
                self._aspect_embeds = None



    def mine_candidates(self, text: str) -> List[str]:
        """Extract candidate aspects using noun phrases + named entities."""
        self._load_spacy()
        doc = self._nlp(text)
        candidates: List[str] = []

        # noun chunks
        for nc in getattr(doc, "noun_chunks", []):
            cand = nc.text.strip().lower()
            if 2 <= len(cand) <= 40:
                candidates.append(cand)

        # named entities
        for ent in doc.ents:
            cand = ent.text.strip().lower()
            if 2 <= len(cand) <= 40:
                candidates.append(cand)

        # de-dup preserve order
        seen = set()
        out = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def map_to_aspects(self, candidates: List[str]) -> Dict[str, List[str]]:
        """Map candidates to the fixed aspect inventory."""
        # 1) lexicon match
        mapped: Dict[str, List[str]] = {k: [] for k in self.aspects_lexicon}
        for cand in candidates:
            for aspect, seeds in self.aspects_lexicon.items():
                if any(seed in cand for seed in seeds):
                    mapped[aspect].append(cand)

        # 2) embedding similarity fallback
        if self.use_embedding_similarity:
            self._load_embedder()
            if self._embedder is not None and self._aspect_embeds is not None:
                remaining = [c for c in candidates if not any(c in v for v in mapped.values())]
                if remaining:
                    c_emb = self._embedder.encode(remaining, normalize_embeddings=True)
                    sims = np.dot(c_emb, self._aspect_embeds.T)  # (n, A)
                    aspect_names = list(self.aspects_lexicon.keys())
                    for i, cand in enumerate(remaining):
                        j = int(np.argmax(sims[i]))
                        if float(sims[i, j]) >= 0.55:
                            mapped[aspect_names[j]].append(cand)

        # prune empty
        return {k: v for k, v in mapped.items() if v}

    def _sentences_with_candidate(self, text: str, candidate: str) -> List[str]:
        self._load_spacy()
        doc = self._nlp(text)
        out = []
        for sent in doc.sents:
            s = sent.text.strip()
            if candidate in s.lower():
                out.append(s)
        return out or [text]

    def aspect_sentiment_scores(
        self,
        text: str,
        aspects: Dict[str, List[str]],
        sentiment_pipe,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute aspect sentiment in [-1, 1] using a sentiment pipeline.

        We score each aspect by running sentiment over sentences where its candidates appear.
        Label mapping is handled for common HF sentiment outputs.

        Returns:
            aspect_score: mean signed score (-1..+1)
            aspect_conf: mean confidence (0..1)
        """
        score: Dict[str, float] = {}
        conf: Dict[str, float] = {}

        for aspect, cands in aspects.items():
            sent_scores = []
            sent_confs = []
            for c in cands[:3]:  # cap for speed
                for s in self._sentences_with_candidate(text, c)[:2]:
                    res = sentiment_pipe(s)[0]
                    label = str(res.get("label", "")).upper()
                    prob = float(res.get("score", 0.5))
                    if "NEG" in label or label in {"0", "NEGATIVE"}:
                        sent_scores.append(-prob)
                    elif "POS" in label or label in {"1", "POSITIVE"}:
                        sent_scores.append(prob)
                    else:
                        sent_scores.append(0.0)
                    sent_confs.append(prob)

            if sent_scores:
                score[aspect] = float(np.mean(sent_scores))
                conf[aspect] = float(np.mean(sent_confs))
            else:
                score[aspect] = 0.0
                conf[aspect] = 0.0

        return score, conf

    def extract(self, text: str, sentiment_pipe=None) -> AspectResult:
        """Main API: extract aspects and (optionally) aspect sentiment."""
        candidates = self.mine_candidates(text)
        aspects = self.map_to_aspects(candidates)

        aspect_sent, aspect_conf = {}, {}
        if sentiment_pipe is not None and aspects:
            aspect_sent, aspect_conf = self.aspect_sentiment_scores(text, aspects, sentiment_pipe)

        return AspectResult(aspects=aspects, aspect_sentiment=aspect_sent, aspect_confidence=aspect_conf)
