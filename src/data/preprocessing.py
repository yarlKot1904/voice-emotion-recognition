"""Audio loading, padding and lightweight waveform augmentations."""

from __future__ import annotations

import librosa
import numpy as np


def load_mono(path: str, sample_rate: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(y, dtype=np.float32)


def fix_length(y: np.ndarray, target_length: int, *, random_crop: bool = False) -> np.ndarray:
    if y.shape[0] > target_length:
        if random_crop:
            start = int(np.random.randint(0, y.shape[0] - target_length + 1))
        else:
            start = int((y.shape[0] - target_length) // 2)
        y = y[start : start + target_length]
    elif y.shape[0] < target_length:
        pad = target_length - y.shape[0]
        y = np.pad(y, (0, pad))
    return np.asarray(y, dtype=np.float32)


def _shift_with_silence(y: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return y

    out = np.zeros_like(y)
    if abs(shift) >= y.shape[0]:
        return out
    if shift > 0:
        out[shift:] = y[:-shift]
    else:
        out[:shift] = y[-shift:]
    return out


def augment_waveform(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y

    out = np.array(y, copy=True, dtype=np.float32)

    shift = int(np.random.randint(-1600, 1601))
    if shift:
        out = _shift_with_silence(out, shift)

    gain = float(np.random.uniform(0.9, 1.1))
    out *= gain

    noise_scale = float(np.random.uniform(0.0, 0.005))
    if noise_scale > 0:
        out += np.random.normal(0.0, noise_scale, size=out.shape).astype(np.float32)

    return np.clip(out, -1.0, 1.0)
