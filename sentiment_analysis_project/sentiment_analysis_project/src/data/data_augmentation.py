"""Data augmentation strategies.

Augmentation can be valuable for class imbalance, but it must be reported carefully in academic writing.
This module provides *optional* augmentation methods.

Implemented:
- back-translation via Hugging Face translation models (requires GPU/CPU resources)
- masked-LM token substitution (contextual augmentation)

Note:
- Paraphrasing with T5 / synthetic generation is provided as a stub with TODOs, since model choice depends on hardware.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def masked_lm_substitution(texts: List[str], model_name: str = "bert-base-uncased", p: float = 0.15) -> List[str]:
    """Contextual augmentation using a masked language model.

    This replaces a small fraction of tokens with predicted alternatives.

    Args:
        texts: Input texts.
        model_name: Masked LM name.
        p: Fraction of tokens to mask (small, e.g., 0.10-0.20).

    Returns:
        Augmented texts (best effort).

    Warning:
        This can change meaning/sentiment. Use conservatively and evaluate impact.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        model.eval()

        out = []
        for t in texts:
            tokens = tokenizer.tokenize(t)
            if len(tokens) < 5:
                out.append(t)
                continue
            n_mask = max(1, int(len(tokens) * p))
            mask_idx = np.random.choice(len(tokens), size=n_mask, replace=False)

            token_ids = tokenizer.encode(t, return_tensors="pt")
            # naive: mask some positions within token_ids (skip special tokens)
            ids = token_ids.clone()
            valid_positions = list(range(1, ids.size(1) - 1))
            chosen = [valid_positions[i % len(valid_positions)] for i in range(n_mask)]
            for pos in chosen:
                ids[0, pos] = tokenizer.mask_token_id

            with torch.no_grad():
                logits = model(ids).logits

            for pos in chosen:
                pred_id = int(logits[0, pos].argmax().item())
                ids[0, pos] = pred_id

            out.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        return out
    except Exception as e:
        log.warning("masked_lm_substitution failed: %s", e)
        return texts


def back_translation(texts: List[str], src_lang: str = "en", pivot_lang: str = "de") -> List[str]:
    """Back-translation augmentation using Hugging Face translation models.

    Example:
        back_translation(["I love this product."]) -> paraphrased text

    TODO:
        - Choose pivot languages based on dataset language distribution.
        - Cache intermediate translations for reproducibility.

    Note:
        Requires installing relevant translation models at runtime.
    """
    try:
        from transformers import pipeline

        en_to_pivot = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{src_lang}-{pivot_lang}")
        pivot_to_en = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{pivot_lang}-{src_lang}")

        pivoted = [en_to_pivot(t, max_length=256)[0]["translation_text"] for t in texts]
        back = [pivot_to_en(t, max_length=256)[0]["translation_text"] for t in pivoted]
        return back
    except Exception as e:
        log.warning("back_translation failed: %s", e)
        return texts
