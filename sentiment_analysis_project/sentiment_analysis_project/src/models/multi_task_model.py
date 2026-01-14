"""Multi-task Transformer model (research core).

Tasks:
1) Overall sentiment (sequence classification)
2) Purchase recommendation (classification) - can be treated as consumer behavior proxy
3) Aspect extraction (token classification) - weakly supervised from `AspectExtractor` outputs

Implementation strategy:
- Shared encoder (DeBERTa/RoBERTa)
- Heads per task
- Weighted multi-task loss for balancing

Note:
- For token-level aspect extraction without gold labels, we support *weak labels*:
  - Mark tokens belonging to mined aspect candidates as aspect tokens.
  - This is noisy but useful for multi-task inductive bias and publishable ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MultiTaskOutputs:
    sentiment_logits: torch.Tensor
    recommend_logits: torch.Tensor
    aspect_logits: torch.Tensor
    losses: Dict[str, torch.Tensor]


class MultiTaskTransformer(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_sentiment_labels: int = 3,
        num_recommend_labels: int = 2,
        num_aspect_labels: int = 2,  # 0=O, 1=ASPECT
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        cfg = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name, config=cfg)

        hidden = cfg.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.sentiment_head = nn.Linear(hidden, num_sentiment_labels)
        self.recommend_head = nn.Linear(hidden, num_recommend_labels)
        self.aspect_head = nn.Linear(hidden, num_aspect_labels)

        self.sentiment_loss = nn.CrossEntropyLoss()
        self.recommend_loss = nn.CrossEntropyLoss()
        self.aspect_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_labels: Optional[torch.Tensor] = None,
        recommend_labels: Optional[torch.Tensor] = None,
        aspect_labels: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> MultiTaskOutputs:
        loss_weights = loss_weights or {"sentiment": 1.0, "recommend": 1.0, "aspect": 0.7}

        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS pooling
        pooled = enc.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)

        sentiment_logits = self.sentiment_head(pooled)
        recommend_logits = self.recommend_head(pooled)

        # token logits for aspect extraction
        token_out = self.dropout(enc.last_hidden_state)
        aspect_logits = self.aspect_head(token_out)

        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=input_ids.device)

        if sentiment_labels is not None:
            ls = self.sentiment_loss(sentiment_logits, sentiment_labels)
            losses["sentiment"] = ls
            total = total + loss_weights["sentiment"] * ls
        if recommend_labels is not None:
            lr = self.recommend_loss(recommend_logits, recommend_labels)
            losses["recommend"] = lr
            total = total + loss_weights["recommend"] * lr
        if aspect_labels is not None:
            # reshape for CE: (B*T, C)
            la = self.aspect_loss(aspect_logits.view(-1, aspect_logits.size(-1)), aspect_labels.view(-1))
            losses["aspect"] = la
            total = total + loss_weights["aspect"] * la

        losses["total"] = total
        return MultiTaskOutputs(
            sentiment_logits=sentiment_logits,
            recommend_logits=recommend_logits,
            aspect_logits=aspect_logits,
            losses=losses,
        )
