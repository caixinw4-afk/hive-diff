"""Exposure-value (EV) token: a tiny MLP mapping a scalar EV to a CLIP-sized vector.

The output is *added* to the single CLIP image token that conditions the UNet and both
ControlNets, telling the model at which exposure the reference frame was rendered.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ExposureTokenMLP(nn.Module):
    def __init__(self, dim: int = 1024, init_std: float = 0.02):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        for layer in (self.net[0], self.net[2]):
            nn.init.normal_(layer.weight, std=init_std)
            nn.init.zeros_(layer.bias)

    @property
    def device(self) -> torch.device:
        return self.net[0].weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.net[0].weight.dtype

    def forward(self, ev: torch.Tensor) -> torch.Tensor:
        """``ev``: (N,) or (N, 1) float -> (N, dim)."""
        ev = torch.as_tensor(ev, device=self.device).reshape(-1, 1).to(self.dtype)
        return self.net(ev)

    @classmethod
    def load(cls, path: str, dim: int = 1024, map_location="cpu") -> "ExposureTokenMLP":
        model = cls(dim)
        model.load_state_dict(torch.load(path, map_location=map_location))
        return model
