"""CNN models on log-mel spectrograms [B, 1, n_mels, T]."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES

CNNVariant = Literal["base", "large", "resnet"]


def normalize_mel_batch(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-sample standardization over frequency/time bins."""
    mean = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps)


class SpecAugment(nn.Module):
    """Small SpecAugment regularizer for log-mel inputs."""

    def __init__(
        self,
        *,
        p: float = 0.5,
        freq_mask_param: int = 8,
        time_mask_param: int = 16,
        freq_masks: int = 1,
        time_masks: int = 1,
    ) -> None:
        super().__init__()
        self.p = p
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_masks = freq_masks
        self.time_masks = time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        if torch.rand((), device=x.device) > self.p:
            return x

        batch, _, n_freq, n_time = x.shape
        out = x.clone()
        fill = out.mean(dim=(2, 3), keepdim=True)

        for idx in range(batch):
            for _ in range(self.freq_masks):
                max_width = min(self.freq_mask_param, n_freq)
                width = int(
                    torch.randint(0, max_width + 1, (), device=x.device).item()
                )
                if width > 0:
                    start = int(
                        torch.randint(0, n_freq - width + 1, (), device=x.device).item()
                    )
                    out[idx, :, start : start + width, :] = fill[idx]
            for _ in range(self.time_masks):
                max_width = min(self.time_mask_param, n_time)
                width = int(
                    torch.randint(0, max_width + 1, (), device=x.device).item()
                )
                if width > 0:
                    start = int(
                        torch.randint(0, n_time - width + 1, (), device=x.device).item()
                    )
                    out[idx, :, :, start : start + width] = fill[idx]
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: tuple[int, int] = (1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != (1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + self.shortcut(x))


class StatsPool2d(nn.Module):
    """Mean + max pooling keeps a bit more signal than plain global average."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        maxv = F.adaptive_max_pool2d(x, output_size=1)
        return torch.cat((avg, maxv), dim=1).flatten(1)


class MelCNN(nn.Module):
    """
    CNN on mel spectrogram.

    ``variant=base`` keeps the original 3-layer network for old checkpoints.
    ``variant=large`` keeps the wider pre-existing CNN.
    ``variant=resnet`` is the recommended stronger default for short runs.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        *,
        variant: CNNVariant = "base",
        normalize_input: bool = False,
        spec_augment: bool = False,
        spec_augment_p: float = 0.5,
        spec_freq_mask: int = 8,
        spec_time_mask: int = 16,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.normalize_input = normalize_input
        self.spec_augment = (
            SpecAugment(
                p=spec_augment_p,
                freq_mask_param=spec_freq_mask,
                time_mask_param=spec_time_mask,
            )
            if spec_augment
            else nn.Identity()
        )
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
        elif variant == "base":
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
        elif variant == "resnet":
            self.features = nn.Sequential(
                nn.Conv2d(1, 48, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.SiLU(inplace=True),
                ResidualBlock(48, 64, dropout=0.05),
                ResidualBlock(64, 96, stride=(2, 2), dropout=0.05),
                ResidualBlock(96, 128, stride=(2, 2), dropout=0.10),
                ResidualBlock(128, 192, stride=(2, 2), dropout=0.10),
                ResidualBlock(192, 256, stride=(1, 2), dropout=0.10),
            )
            self.head = nn.Sequential(
                StatsPool2d(),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.SiLU(inplace=True),
                nn.Dropout(0.35),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unknown CNN variant: {variant}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize_input:
            x = normalize_mel_batch(x)
        x = self.spec_augment(x)
        x = self.features(x)
        return self.head(x)
