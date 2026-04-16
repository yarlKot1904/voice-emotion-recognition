"""Log-mel на GPU через torchaudio (батч [B, T] → [B, 1, n_mels, frames])."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from src.models.cnn_model import MelCNN


class LogMelFrontend(nn.Module):
    """
    Близко к librosa: power mel + нормировка как power_to_db(ref=global max).
    """

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_mels=n_mels,
            power=2.0,
            center=True,
            pad_mode="reflect",
            mel_scale="slaney",
            norm="slaney",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        spec = self.mel(x)
        ref = torch.amax(spec, dim=(1, 2), keepdim=True).clamp(min=1e-10)
        db = 10.0 * torch.log10(spec / ref + 1e-10)
        return db.unsqueeze(1)


class MelCNNWithFrontend(nn.Module):
    """Вход: волна [B, T]; внутри mel на GPU, затем MelCNN."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        *,
        cnn_variant: str = "large",
        normalize_input: bool = False,
        spec_augment: bool = False,
        spec_augment_p: float = 0.5,
        spec_freq_mask: int = 8,
        spec_time_mask: int = 16,
    ) -> None:
        super().__init__()
        self.frontend = LogMelFrontend(sample_rate, n_fft, hop_length, n_mels)
        self.backbone = MelCNN(
            variant=cnn_variant,
            normalize_input=normalize_input,
            spec_augment=spec_augment,
            spec_augment_p=spec_augment_p,
            spec_freq_mask=spec_freq_mask,
            spec_time_mask=spec_time_mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.frontend(x))


def build_waveform_cnn(
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
    cnn_variant: str,
    *,
    normalize_input: bool = False,
    spec_augment: bool = False,
    spec_augment_p: float = 0.5,
    spec_freq_mask: int = 8,
    spec_time_mask: int = 16,
) -> MelCNNWithFrontend:
    return MelCNNWithFrontend(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        cnn_variant=cnn_variant,
        normalize_input=normalize_input,
        spec_augment=spec_augment,
        spec_augment_p=spec_augment_p,
        spec_freq_mask=spec_freq_mask,
        spec_time_mask=spec_time_mask,
    )
