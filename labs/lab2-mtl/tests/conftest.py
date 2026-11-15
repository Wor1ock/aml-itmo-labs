"""Fixtures with synthetic data: unit tests never touch the network or the real dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecgmtl.config import (
    Config,
    DataConfig,
    ExperimentsConfig,
    LossConfig,
    ModelConfig,
    OutputConfig,
    SplitConfig,
    TrainConfig,
)
from ecgmtl.data.labels import SUPERCLASSES
from ecgmtl.data.prepare import MANIFEST_NAME, META_NAME, SIGNALS_NAME

RHYTHM = ["SR", "AFIB", "STACH"]
N_RECORDS = 60


def make_manifest(n: int = N_RECORDS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    folds = (np.arange(n) % 10) + 1
    patients = np.arange(n) // 2 + 1000
    diag = rng.integers(0, 2, size=(n, len(SUPERCLASSES))).astype(np.int8)
    rhythm = rng.integers(-1, len(RHYTHM), size=n)
    age = rng.normal(60, 12, size=n).round(0)
    age[rng.random(n) < 0.1] = np.nan
    manifest = pd.DataFrame(
        {
            "ecg_id": np.arange(1, n + 1),
            "patient_id": patients,
            "strat_fold": folds,
            "age": age,
            "sex": rng.integers(0, 2, size=n),
            "rhythm_idx": rhythm,
            "filename": [f"records100/{i // 1000:05d}/{i:05d}_lr" for i in range(n)],
            "n_diag_positive": diag.sum(axis=1),
        }
    )
    for i, name in enumerate(SUPERCLASSES):
        manifest[f"diag_{name}"] = diag[:, i]
    manifest["rhythm_code"] = manifest["rhythm_idx"].map(
        lambda idx: RHYTHM[idx] if idx >= 0 else "none"
    )
    return manifest


def write_dataset(root: Path, manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    """Write processed artefacts in the same layout as ``ecgmtl.data.prepare``."""
    manifest = make_manifest() if manifest is None else manifest
    processed = root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    signals = rng.normal(size=(len(manifest), 12, 1000)).astype(np.float16)
    np.save(processed / SIGNALS_NAME, signals)
    manifest.to_parquet(processed / MANIFEST_NAME, index=False)
    meta = {
        "version": "test",
        "hz": 100,
        "signal_len": 1000,
        "n_leads": 12,
        "superclasses": list(SUPERCLASSES),
        "rhythm_classes": RHYTHM,
        "n_records": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "skipped": 0,
    }
    (processed / META_NAME).write_text(json.dumps(meta), encoding="utf-8")
    return manifest


@pytest.fixture
def tiny_config(tmp_path: Path) -> Config:
    root = tmp_path / "data"
    write_dataset(root)
    return Config(
        data=DataConfig(root=root),
        split=SplitConfig(run=0),
        model=ModelConfig(channels=(8, 16), kernels=(7, 5), hidden=32, head_hidden=16, dropout=0.0),
        loss=LossConfig(),
        train=TrainConfig(
            epochs=1,
            batch_size=8,
            limit_train=32,
            limit_val=16,
            patience=1,
            log_every=100,
            conflict_every=100,
        ),
        output=OutputConfig(
            out_dir=tmp_path / "outputs",
            report_dir=tmp_path / "report",
        ),
        experiments=ExperimentsConfig(folds=1, ablation_folds=1, main=("mtl_equal",), ablations=()),
    )
