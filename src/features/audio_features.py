"""MFCC and log-mel spectrogram for SER."""

from __future__ import annotations

import numpy as np
import librosa
import torch


def waveform_to_mel(
    y: np.ndarray,
    sr: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Log-power mel [n_mels, time]."""
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def waveform_to_mfcc(
    y: np.ndarray,
    sr: int,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return mfcc.astype(np.float32)


def mel_tensor(log_mel: np.ndarray) -> torch.Tensor:
    """[1, n_mels, T] for CNN."""
    return torch.from_numpy(log_mel).unsqueeze(0)


def mfcc_flat_vector(mfcc: np.ndarray) -> torch.Tensor:
    """[n_mfcc * T] for MLP."""
    return torch.from_numpy(mfcc.flatten())
