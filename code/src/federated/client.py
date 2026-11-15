"""Client-side: local training and update production."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .params import ParamDict, sub_state, clip_state


@dataclass
class ClientUpdate:
    client_id: int
    delta: ParamDict           # theta_local - theta_global, already clipped
    num_samples: int
    round_index: int


class Client:
    """Owns a slice of the training data and produces update deltas on demand."""

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,
        local_epochs: int = 2,
        batch_size: int = 32,
        lr: float = 0.01,
        clip_norm: float = 1.0,
        device: str | torch.device = "cpu",
        text_mode: bool = False,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.clip_norm = clip_norm
        self.device = torch.device(device)
        self.text_mode = text_mode

    def local_update(self, global_state: ParamDict, model: nn.Module, round_index: int) -> ClientUpdate:
        model.load_state_dict(global_state)
        model.to(self.device)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.0)
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.local_epochs):
            for batch in loader:
                x, y = batch
                x = _to_device(x, self.device)
                y = torch.as_tensor(y, device=self.device, dtype=torch.long)
                opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        local_state = OrderedDict((k, v.detach().cpu()) for k, v in model.state_dict().items())
        delta = sub_state(local_state, OrderedDict((k, v.cpu()) for k, v in global_state.items()))
        delta = clip_state(delta, self.clip_norm)
        return ClientUpdate(
            client_id=self.client_id,
            delta=delta,
            num_samples=len(self.dataset),
            round_index=round_index,
        )


def _to_device(x, device):
    if isinstance(x, dict):
        return {k: v.to(device) for k, v in x.items()}
    return x.to(device)


def local_train(client: Client, global_state: ParamDict, model: nn.Module, round_index: int) -> ClientUpdate:
    """Free-function alias, convenient for parallelism wrappers."""
    return client.local_update(global_state, model, round_index)
