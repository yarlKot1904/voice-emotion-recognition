"""Inference on a single wav file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config import CLASS_NAMES, HOP_LENGTH, MAX_LENGTH_SEC, N_FFT, N_MFCC, N_MELS, SAMPLE_RATE
from src.data.preprocessing import fix_length, load_mono
from src.features.audio_features import (
    mfcc_flat_vector,
    waveform_to_mfcc,
    waveform_to_mel,
)
from src.models.cnn_model import MelCNN
from src.models.mel_frontend import build_waveform_cnn
from src.models.mlp_model import MLPClassifier
from src.runtime.device import choose_device
from src.training.feature_config import merge_with_checkpoint


def _dummy_mel_batch(model_type: str, device: torch.device, fc: dict[str, Any]) -> torch.Tensor:
    sr = int(fc.get("sample_rate", SAMPLE_RATE))
    max_sec = float(fc.get("max_length_sec", MAX_LENGTH_SEC))
    n_samples = int(max_sec * sr)
    y = np.zeros(n_samples, dtype=np.float32)
    hop = int(fc.get("hop_length", HOP_LENGTH))
    n_fft = int(fc.get("n_fft", N_FFT))
    n_mels = int(fc.get("n_mels", N_MELS))
    n_mfcc = int(fc.get("n_mfcc", N_MFCC))
    if model_type == "mlp":
        mfcc = waveform_to_mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
        return mfcc_flat_vector(mfcc).unsqueeze(0).to(device)
    log_mel = waveform_to_mel(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
    return torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)


def load_model(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, str, dict[str, Any]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fc = merge_with_checkpoint(ckpt)
    model_type = ckpt.get("model_type", "cnn")

    if model_type == "mlp":
        model = MLPClassifier()
        model.to(device)
        model(_dummy_mel_batch(model_type, device, fc))
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, model_type, fc

    variant = ckpt.get("cnn_variant", "base")
    if fc.get("mel_backend") == "torchaudio":
        model = build_waveform_cnn(
            int(fc["sample_rate"]),
            int(fc["n_fft"]),
            int(fc["hop_length"]),
            int(fc["n_mels"]),
            variant,
        )
        model.to(device)
        sr = int(fc["sample_rate"])
        t = int(float(fc["max_length_sec"]) * sr)
        model(torch.zeros(1, t, device=device, dtype=torch.float32))
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, model_type, fc

    model = MelCNN(variant=variant)
    model.to(device)
    model(_dummy_mel_batch(model_type, device, fc))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, model_type, fc


def predict_file(
    ckpt_path: Path, wav_path: Path, device: torch.device | None = None
) -> tuple[str, list[float]]:
    if device is None:
        device, device_note = choose_device()
        if device_note:
            print(device_note)
    model, model_type, fc = load_model(ckpt_path, device)

    sr = int(fc.get("sample_rate", SAMPLE_RATE))
    max_sec = float(fc.get("max_length_sec", MAX_LENGTH_SEC))
    hop = int(fc.get("hop_length", HOP_LENGTH))
    n_fft = int(fc.get("n_fft", N_FFT))
    n_mels = int(fc.get("n_mels", N_MELS))
    n_mfcc = int(fc.get("n_mfcc", N_MFCC))

    y = load_mono(str(wav_path), sr)
    y = fix_length(y, int(max_sec * sr))

    if model_type == "mlp":
        mfcc = waveform_to_mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop)
        x = mfcc_flat_vector(mfcc).unsqueeze(0).to(device)
    elif fc.get("mel_backend") == "torchaudio":
        x = torch.from_numpy(y).float().unsqueeze(0).to(device)
    else:
        log_mel = waveform_to_mel(y, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop)
        x = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    cls = int(prob.argmax())
    return CLASS_NAMES[cls], prob.tolist()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--wav", type=Path, required=True)
    args = p.parse_args()
    label, probs = predict_file(args.checkpoint, args.wav)
    print(f"Predicted: {label}")
    for name, pr in zip(CLASS_NAMES, probs, strict=True):
        print(f"  {name}: {pr:.4f}")


if __name__ == "__main__":
    main()
