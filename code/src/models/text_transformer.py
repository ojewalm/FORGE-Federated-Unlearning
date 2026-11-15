"""Small transformer for 20 Newsgroups (distilBERT-mini)."""
from __future__ import annotations

import torch
import torch.nn as nn


class TextTransformer(nn.Module):
    """Wraps a HuggingFace transformer with a classification head that we can split.

    Uses `distilbert-base-uncased` by default. The base model plays the role of
    `features`, and the linear projection over the [CLS] token is `head`.
    """

    def __init__(self, num_classes: int = 20, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        from transformers import AutoModel  # local import to keep the module optional
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, num_classes)
        self.num_classes = num_classes

    def _pool(self, out) -> torch.Tensor:
        return out.last_hidden_state[:, 0]  # [CLS]-style pooling

    def features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self._pool(out)

    def forward(self, batch) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            # Fallback if given only input_ids
            batch = {"input_ids": batch, "attention_mask": torch.ones_like(batch)}
        feats = self.features(batch)
        return self.head(feats)

    def head_parameters(self):
        return self.head.parameters()

    def feature_parameters(self):
        return self.encoder.parameters()

    def logits_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.head(feats)


def build_text_transformer(num_classes: int, model_name: str = "distilbert-base-uncased") -> TextTransformer:
    return TextTransformer(num_classes=num_classes, model_name=model_name)
