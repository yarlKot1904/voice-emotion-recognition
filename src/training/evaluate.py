"""Evaluate checkpoint on a split (default: test)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import CLASS_NAMES, SPLITS_DIR
from src.data.dataset import (
    SERDataset,
    SERWaveformDataset,
    collate_mel,
    collate_mfcc,
    collate_waveform,
)
from src.models.cnn_model import MelCNN
from src.models.mel_frontend import build_waveform_cnn
from src.models.mlp_model import MLPClassifier
from src.training.feature_config import merge_with_checkpoint
from src.training.metrics import compute_metrics


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=Path, default=None, help="CSV manifest (default: splits/test.csv)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    split = args.split or (SPLITS_DIR / "test.csv")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_type = ckpt.get("model_type", "cnn")
    fc = merge_with_checkpoint(ckpt)

    use_torchaudio = model_type == "cnn" and fc.get("mel_backend") == "torchaudio"

    if use_torchaudio:
        ds = SERWaveformDataset(
            split,
            sample_rate=int(fc["sample_rate"]),
            max_length_sec=float(fc["max_length_sec"]),
            augment=False,
        )
        collate = collate_waveform
    elif model_type == "mlp":
        ds = SERDataset(
            split,
            feature_mode="mfcc",
            sample_rate=int(fc["sample_rate"]),
            max_length_sec=float(fc["max_length_sec"]),
            hop_length=int(fc["hop_length"]),
            n_fft=int(fc["n_fft"]),
            n_mels=int(fc["n_mels"]),
            n_mfcc=int(fc["n_mfcc"]),
        )
        collate = collate_mfcc
    else:
        ds = SERDataset(
            split,
            feature_mode="mel",
            sample_rate=int(fc["sample_rate"]),
            max_length_sec=float(fc["max_length_sec"]),
            hop_length=int(fc["hop_length"]),
            n_fft=int(fc["n_fft"]),
            n_mels=int(fc["n_mels"]),
            n_mfcc=int(fc["n_mfcc"]),
        )
        collate = collate_mel

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "mlp":
        model = MLPClassifier().to(device)
    elif use_torchaudio:
        model = build_waveform_cnn(
            int(fc["sample_rate"]),
            int(fc["n_fft"]),
            int(fc["hop_length"]),
            int(fc["n_mels"]),
            ckpt.get("cnn_variant", "large"),
        ).to(device)
    else:
        variant = ckpt.get("cnn_variant", "base")
        model = MelCNN(variant=variant).to(device)

    x0, _ = next(iter(loader))
    x0 = x0.to(device)
    model(x0)
    model.load_state_dict(ckpt["model"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(pred.tolist())

    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    m = compute_metrics(y_t, y_p, list(CLASS_NAMES))
    print(f"Accuracy: {m['accuracy']:.4f}")
    print(f"Macro F1: {m['macro_f1']:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    for row in m["confusion_matrix"]:
        print(row)
    print(m["report"])


if __name__ == "__main__":
    main()
