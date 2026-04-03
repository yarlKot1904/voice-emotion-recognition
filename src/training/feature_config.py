"""Feature extraction settings (must match between train / evaluate / predict)."""

from __future__ import annotations

from typing import Any

from src.config import HOP_LENGTH, MAX_LENGTH_SEC, N_FFT, N_MFCC, N_MELS, SAMPLE_RATE


def defaults() -> dict[str, Any]:
    return {
        "sample_rate": SAMPLE_RATE,
        "max_length_sec": MAX_LENGTH_SEC,
        "hop_length": HOP_LENGTH,
        "n_fft": N_FFT,
        "n_mels": N_MELS,
        "n_mfcc": N_MFCC,
        "mel_backend": "librosa",
    }


def merge_with_checkpoint(ckpt: dict) -> dict[str, Any]:
    out = defaults()
    fc = ckpt.get("feature_config")
    if isinstance(fc, dict):
        out.update(fc)
    return out
