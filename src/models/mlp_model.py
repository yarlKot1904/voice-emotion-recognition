"""MLP baseline on flattened MFCC (or mel)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import NUM_CLASSES


class MLPClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.LazyLinear(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.LazyLinear(num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
