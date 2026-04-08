"""Datasets for feature-based and waveform-based SER training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.preprocessing import augment_waveform, fix_length, load_mono
from src.features.audio_features import (
    mel_tensor,
    mfcc_flat_vector,
    waveform_to_mel,
    waveform_to_mfcc,
)


@dataclass(frozen=True)
class SampleRow:
    path: Path
    label: int


def _read_rows(csv_path: Path) -> list[SampleRow]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            SampleRow(path=Path(row["path"]), label=int(row["label"]))
            for row in reader
        ]


class SERDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        csv_path: Path,
        *,
        feature_mode: str,
        sample_rate: int,
        max_length_sec: float,
        augment: bool = False,
        hop_length: int,
        n_fft: int,
        n_mels: int,
        n_mfcc: int,
    ) -> None:
        self.rows = _read_rows(csv_path)
        self.feature_mode = feature_mode
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_length_sec)
        self.augment = augment
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        wav = load_mono(str(row.path), self.sample_rate)
        wav = fix_length(wav, self.max_length, random_crop=self.augment)
        if self.augment:
            wav = augment_waveform(wav)

        if self.feature_mode == "mfcc":
            mfcc = waveform_to_mfcc(
                wav,
                self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            x = mfcc_flat_vector(mfcc)
        else:
            mel = waveform_to_mel(
                wav,
                self.sample_rate,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
            x = mel_tensor(mel)
        return x, row.label


class SERWaveformDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        csv_path: Path,
        *,
        sample_rate: int,
        max_length_sec: float,
        augment: bool = False,
    ) -> None:
        self.rows = _read_rows(csv_path)
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_length_sec)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        wav = load_mono(str(row.path), self.sample_rate)
        wav = fix_length(wav, self.max_length, random_crop=self.augment)
        if self.augment:
            wav = augment_waveform(wav)
        return torch.from_numpy(wav), row.label


def collate_mel(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch, strict=True)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def collate_mfcc(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch, strict=True)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)


def collate_waveform(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch, strict=True)
    return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

