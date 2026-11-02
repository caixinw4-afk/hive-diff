"""Auxiliary decoder used only during stage-1 training.

It mirrors :class:`EventConditioningEncoder` (320 -> 256 -> 128 -> 64 -> 32 -> C_ev, three 2x
transposed-conv upsamplings) and reconstructs the input event tensor from the encoder output,
which regularises the event encoder (loss weight 0.1). Not used at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventReconstructor(nn.Module):
    def __init__(self, in_channels: int = 320, event_channels: int = 4, hidden=(256, 128, 64, 32)):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden[0], kernel_size=3, padding=1)
        self.up_blocks = nn.ModuleList()
        for c_in, c_out in zip(hidden[:-1], hidden[1:]):
            self.up_blocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.SiLU(),
            ))
        self.conv_out = nn.Conv2d(hidden[-1], event_channels, kernel_size=3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.conv_in(features))
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(x)
