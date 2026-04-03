"""Central configuration: paths from environment variables with safe defaults."""

from __future__ import annotations

import os
from pathlib import Path

# Root of extracted Dusha archive (contains crowd_train/, crowd_test/, ...)
DUSHA_ROOT: Path = Path(os.environ.get("DUSHA_ROOT", "data/raw/dusha")).resolve()

# Project data outputs
PROCESSED_DIR: Path = Path(os.environ.get("PROCESSED_DIR", "data/processed")).resolve()
MANIFEST_PATH: Path = Path(os.environ.get("MANIFEST_PATH", str(PROCESSED_DIR / "manifest.csv")))
SPLITS_DIR: Path = Path(os.environ.get("SPLITS_DIR", str(PROCESSED_DIR / "splits"))).resolve()

# Precomputed features for Podcast domain (optional extension)
FEATURES_ROOT: Path = Path(os.environ.get("FEATURES_ROOT", "data/raw/dusha/features")).resolve()

# Audio / features
SAMPLE_RATE: int = int(os.environ.get("SAMPLE_RATE", "16000"))
MAX_LENGTH_SEC: float = float(os.environ.get("MAX_LENGTH_SEC", "3.0"))
N_MELS: int = int(os.environ.get("N_MELS", "64"))
N_MFCC: int = int(os.environ.get("N_MFCC", "40"))
HOP_LENGTH: int = int(os.environ.get("HOP_LENGTH", "512"))
N_FFT: int = int(os.environ.get("N_FFT", "1024"))

# Training
RANDOM_SEED: int = int(os.environ.get("RANDOM_SEED", "42"))
CHECKPOINT_DIR: Path = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints")).resolve()

# Class names (order matches label indices)
CLASS_NAMES: tuple[str, ...] = ("angry", "sad", "neutral", "positive")
NUM_CLASSES: int = len(CLASS_NAMES)
