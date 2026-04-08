"""Train MLP or CNN on Dusha manifest splits."""

from __future__ import annotations

import argparse
import contextlib
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import CHECKPOINT_DIR, CLASS_NAMES, NUM_CLASSES, RANDOM_SEED, SPLITS_DIR
from src.config import HOP_LENGTH, MAX_LENGTH_SEC, N_FFT, N_MELS, N_MFCC, SAMPLE_RATE
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
from src.runtime.device import choose_device

_DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 2


def _amp_fwd_context(device: torch.device, use_amp: bool):
    """torch.amp.autocast on CUDA only; avoids deprecated torch.cuda.amp.autocast."""
    if use_amp and device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights_from_csv(csv_path: Path, num_classes: int) -> torch.Tensor:
    labels: list[int] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["label"]))
    arr = np.array(labels, dtype=np.int64)
    counts = np.bincount(arr, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w * num_classes / w.sum()
    return torch.tensor(w, dtype=torch.float32)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    *,
    train: bool,
    epoch: int,
    grad_clip: float = 0.0,
    use_amp: bool = False,
    scaler: object | None = None,
) -> tuple[float, float]:
    if train:
        np.random.seed(RANDOM_SEED + epoch * 10009)
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in tqdm(loader, leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with _amp_fwd_context(device, use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        if train:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    use_amp: bool = False,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        with _amp_fwd_context(device, use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def _make_grad_scaler(use_amp: bool) -> object | None:
    if not use_amp:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("mlp", "cnn"), default="cnn")
    p.add_argument("--cnn-variant", choices=("base", "large"), default="large", help="CNN width/depth (default: large for better accuracy).")
    p.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default {_DEFAULT_NUM_WORKERS} on Windows — часто стабильнее; Linux можно 4–8).",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    p.add_argument("--exp-name", type=str, default="ser")
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument(
        "--no-augment",
        dest="augment",
        action="store_false",
        help="Disable waveform augment (default: augment on for CNN).",
    )
    p.set_defaults(augment=True)
    p.add_argument("--class-weights", action="store_true", help="Inverse-frequency class weights in loss.")
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="e.g. 0.05 for regularization; 0 disables.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0, help="0 disables.")
    p.add_argument("--scheduler", action="store_true", help="ReduceLROnPlateau on val acc.")
    p.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
        help="Disable mixed float16 training on CUDA (default: AMP on).",
    )
    p.set_defaults(amp=True)
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile(model) on PyTorch 2+ (may speed up GPU step).",
    )
    p.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    p.add_argument("--max-length-sec", type=float, default=MAX_LENGTH_SEC, help="Crop/pad length (smaller=faster, e.g. 2.0).")
    p.add_argument("--hop-length", type=int, default=HOP_LENGTH, help="Larger=fewer frames (e.g. 640 or 1024).")
    p.add_argument("--n-fft", type=int, default=N_FFT)
    p.add_argument("--n-mels", type=int, default=N_MELS)
    p.add_argument("--n-mfcc", type=int, default=N_MFCC)
    p.add_argument(
        "--mel-on-gpu",
        action="store_true",
        help="CNN: mel через torchaudio на GPU, в Dataset только wav (сильно быстрее на CUDA).",
    )
    args = p.parse_args()

    set_seed(RANDOM_SEED)
    device, device_note = choose_device()
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass

    if args.mel_on_gpu and args.model != "cnn":
        raise SystemExit("--mel-on-gpu только вместе с --model cnn")

    feature_cfg = {
        "sample_rate": args.sample_rate,
        "max_length_sec": args.max_length_sec,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "n_mels": args.n_mels,
        "n_mfcc": args.n_mfcc,
        "mel_backend": "torchaudio" if args.mel_on_gpu else "librosa",
    }

    train_csv = args.splits_dir / "train.csv"
    if args.mel_on_gpu:
        train_ds = SERWaveformDataset(
            train_csv,
            sample_rate=args.sample_rate,
            max_length_sec=args.max_length_sec,
            augment=args.augment,
        )
        val_ds = SERWaveformDataset(
            args.splits_dir / "val.csv",
            sample_rate=args.sample_rate,
            max_length_sec=args.max_length_sec,
            augment=False,
        )
        collate = collate_waveform
    else:
        feature_mode = "mfcc" if args.model == "mlp" else "mel"
        train_ds = SERDataset(
            train_csv,
            feature_mode=feature_mode,
            sample_rate=args.sample_rate,
            max_length_sec=args.max_length_sec,
            augment=args.augment and args.model == "cnn",
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
        )
        val_ds = SERDataset(
            args.splits_dir / "val.csv",
            feature_mode=feature_mode,
            sample_rate=args.sample_rate,
            max_length_sec=args.max_length_sec,
            augment=False,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
        )
        collate = collate_mfcc if args.model == "mlp" else collate_mel

    loader_kw: dict = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        **loader_kw,
    )

    if args.model == "mlp":
        model = MLPClassifier().to(device)
        cnn_variant = None
    elif args.mel_on_gpu:
        model = build_waveform_cnn(
            args.sample_rate,
            args.n_fft,
            args.hop_length,
            args.n_mels,
            args.cnn_variant,
        ).to(device)
        cnn_variant = args.cnn_variant
    else:
        model = MelCNN(variant=args.cnn_variant).to(device)
        cnn_variant = args.cnn_variant

    x0, y0 = next(iter(train_loader))
    x0 = x0.to(device)
    model(x0)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    scaler = _make_grad_scaler(use_amp)

    weight = None
    if args.class_weights:
        weight = class_weights_from_csv(train_csv, NUM_CLASSES).to(device)

    ls = args.label_smoothing if args.label_smoothing > 0 else 0.0
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=ls)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )

    ckpt_dir = args.checkpoint_dir / args.exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb")) if args.tensorboard else None

    best_val = -1.0
    stale = 0
    grad_clip = args.grad_clip if args.grad_clip > 0 else 0.0

    if device_note:
        print(device_note)

    if use_amp:
        print("Mixed precision (AMP) enabled on CUDA.")
    elif device.type == "cpu":
        print(
            "Обучение на CPU — AMP недоступен. Установите PyTorch с поддержкой CUDA для GPU и ускорения."
        )
    else:
        print("AMP выключен (--no-amp).")

    if args.mel_on_gpu:
        print(
            "Режим --mel-on-gpu: mel считается на GPU (torchaudio). "
            "Старые чекпоинты с librosa без переобучения не подмешивать."
        )

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            criterion,
            train=True,
            epoch=epoch,
            grad_clip=grad_clip,
            use_amp=use_amp,
            scaler=scaler,
        )
        va_loss, va_acc = evaluate(
            model, val_loader, device, criterion, use_amp=use_amp
        )
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} lr={lr_now:.2e}"
        )
        if scheduler:
            scheduler.step(va_acc)
        if writer:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("loss/val", va_loss, epoch)
            writer.add_scalar("acc/train", tr_acc, epoch)
            writer.add_scalar("acc/val", va_acc, epoch)
            writer.add_scalar("lr", lr_now, epoch)

        if va_acc > best_val:
            best_val = va_acc
            stale = 0
            to_save = model
            if hasattr(to_save, "_orig_mod"):
                to_save = to_save._orig_mod
            payload: dict = {
                "model": to_save.state_dict(),
                "model_type": args.model,
                "class_names": list(CLASS_NAMES),
                "feature_config": feature_cfg,
            }
            if cnn_variant is not None:
                payload["cnn_variant"] = cnn_variant
            torch.save(payload, ckpt_dir / "best.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping.")
                break

    if writer:
        writer.close()
    print(f"Best val acc={best_val:.4f}, checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
