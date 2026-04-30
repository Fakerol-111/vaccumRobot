"""Network building blocks shared across model architectures."""

from __future__ import annotations

import torch.nn as nn


def make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(layer.weight, gain=1.41421)
    nn.init.zeros_(layer.bias)
    return layer
