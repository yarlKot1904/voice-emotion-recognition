"""CNN on log-mel spectrogram [B, 1, n_mels, T]."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import NUM_CLASSES

CNNVariant = str  # "base" | "large"


class MelCNN(nn.Module):
    """
    CNN on mel spectrogram.
    ``variant=base`` — исходная 3-слойная сеть (совместимость со старыми ``best.pt``).
    ``variant=large`` — больше каналов и блок conv для лучшего качества.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, *, variant: CNNVariant = "base") -> None:
        super().__init__()
        self.variant = variant
        if variant == "large":
            self.features = nn.Sequential(
                nn.Conv2d(1, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(96, 192, kernel_size=3, padding=1),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((8, 12)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(384),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(384, num_classes),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 8)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(x)
