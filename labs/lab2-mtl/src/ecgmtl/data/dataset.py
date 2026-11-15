"""PyTorch dataset and patient-level splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ..config import SplitConfig
from .labels import SUPERCLASSES


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def make_split(manifest: pd.DataFrame, cfg: SplitConfig) -> Split:
    """Rotate through the patient-level folds shipped with PTB-XL."""
    fold = manifest["strat_fold"].to_numpy()
    test_fold = cfg.test_folds[cfg.run]
    val_fold = cfg.val_folds[cfg.run]
    test = np.flatnonzero(fold == test_fold)
    val = np.flatnonzero(fold == val_fold)
    train = np.flatnonzero((fold != test_fold) & (fold != val_fold))
    patients = {
        "train": set(manifest["patient_id"].to_numpy()[train]),
        "val": set(manifest["patient_id"].to_numpy()[val]),
        "test": set(manifest["patient_id"].to_numpy()[test]),
    }
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patients[a] & patients[b]
        if overlap:
            raise RuntimeError(f"patient leakage between {a} and {b}: {len(overlap)} ids")
    return Split(train=train, val=val, test=test)


class EcgDataset(Dataset[dict[str, Tensor]]):
    """Returns one record with targets and a per-task mask."""

    def __init__(
        self,
        signals: np.ndarray,
        labels: pd.DataFrame,
        indices: np.ndarray,
    ) -> None:
        self.signals = signals
        self.labels = labels
        self.indices = indices
        self.diag = labels[[f"diag_{name}" for name in SUPERCLASSES]].to_numpy(dtype=np.float32)
        self.rhythm = labels["rhythm_idx"].to_numpy(dtype=np.int64)
        self.age = labels["age"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        row = int(self.indices[i])
        rhythm_ok = bool(self.rhythm[row] >= 0)
        age_ok = bool(np.isfinite(self.age[row]))
        return {
            "signal": torch.from_numpy(np.asarray(self.signals[row], dtype=np.float32)),
            "diag": torch.from_numpy(self.diag[row]),
            "rhythm": torch.tensor(max(int(self.rhythm[row]), 0), dtype=torch.long),
            "age": torch.tensor(float(self.age[row]) if age_ok else 0.0, dtype=torch.float32),
            "mask": torch.tensor([True, rhythm_ok, age_ok], dtype=torch.bool),
        }


def collate(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "signal": torch.stack([item["signal"] for item in batch]),  # (B, leads, T)
        "diag": torch.stack([item["diag"] for item in batch]),  # (B, 5)
        "rhythm": torch.stack([item["rhythm"] for item in batch]),  # (B,)
        "age": torch.stack([item["age"] for item in batch]),  # (B,)
        "mask": torch.stack([item["mask"] for item in batch]),  # (B, 3): diag, rhythm, age
    }


def make_loader(
    dataset: EcgDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[dict[str, Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
