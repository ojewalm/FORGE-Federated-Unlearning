"""20 Newsgroups FCL loader.

Documents are tokenised with a HuggingFace tokenizer (default: distilbert-base-uncased)
and truncated/padded to a fixed max length. Split into 4 continual tasks of 5 topics
each; partitioned across clients by Dirichlet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import Dataset

from .base import FCLSplit
from .partition import dirichlet_partition, task_partition_by_class


class _TokenisedTextDataset(Dataset):
    def __init__(self, input_ids: np.ndarray, attn_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = input_ids.astype(np.int64)
        self.attn_mask = attn_mask.astype(np.int64)
        self.labels = labels.astype(np.int64)
        self.targets = self.labels.tolist()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "input_ids": torch.from_numpy(self.input_ids[idx]),
            "attention_mask": torch.from_numpy(self.attn_mask[idx]),
        }, int(self.labels[idx])


def _tokenise(texts: list[str], tokenizer_name: str, max_len: int) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    out = tok(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="np")
    return out["input_ids"], out["attention_mask"]


def build_newsgroups20_fcl(
    data_root: str = "./data",
    num_clients: int = 100,
    num_tasks: int = 4,
    dirichlet_alpha: float = 0.3,
    tokenizer_name: str = "distilbert-base-uncased",
    max_len: int = 128,
    seed: int = 0,
) -> FCLSplit:
    Path(data_root).mkdir(parents=True, exist_ok=True)
    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), data_home=data_root)
    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), data_home=data_root)

    train_ids, train_mask = _tokenise(list(train.data), tokenizer_name, max_len)
    test_ids, test_mask = _tokenise(list(test.data), tokenizer_name, max_len)

    train_ds = _TokenisedTextDataset(train_ids, train_mask, train.target)
    test_ds = _TokenisedTextDataset(test_ids, test_mask, test.target)

    labels = np.array(train.target)
    client_indices = dirichlet_partition(labels, num_clients, dirichlet_alpha, seed=seed)
    task_classes = task_partition_by_class(labels, num_tasks, seed=seed)

    return FCLSplit(
        train_set=train_ds,
        test_set=test_ds,
        client_indices=client_indices,
        task_classes=task_classes,
        modality="text",
        num_classes=20,
        input_shape=(max_len,),
        metadata={
            "name": "newsgroups20",
            "dirichlet_alpha": dirichlet_alpha,
            "tokenizer": tokenizer_name,
            "max_len": max_len,
        },
    )
