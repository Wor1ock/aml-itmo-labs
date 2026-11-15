"""Static preprocessing: raw WFDB records -> one memmap of signals + a label manifest.

This step does not depend on any hyperparameter, so it belongs to the image build.
Per-record, per-lead z-scoring is deterministic and uses no train statistics, which keeps
the split leak-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from ..config import DataConfig
from ..logging_utils import get_logger
from .download import ensure_raw
from .labels import SUPERCLASSES, build_labels, load_statements

META_NAME = "meta.json"
MANIFEST_NAME = "manifest.parquet"
SIGNALS_NAME = "signals.npy"


def _read_signal(path: Path, cfg: DataConfig) -> np.ndarray | None:
    """Read one record, return (leads, length) float32 or None if the record is unusable."""
    try:
        record = wfdb.rdrecord(str(path))
    except Exception as exc:  # noqa: BLE001 - a broken record must not stop the build
        get_logger().warning("cannot read %s: %s", path.name, exc)
        return None
    signal = np.asarray(record.p_signal, dtype=np.float32)  # (T, leads)
    if signal.shape != (cfg.signal_len, cfg.n_leads):
        get_logger().warning("unexpected shape %s in %s", signal.shape, path.name)
        return None
    if not np.isfinite(signal).all():
        get_logger().warning("non-finite samples in %s", path.name)
        return None
    signal = signal.T  # (leads, T)
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    return (signal - mean) / (std + 1e-6)


def prepare(cfg: DataConfig) -> Path:
    """Build ``data/processed`` and return its path."""
    log = get_logger()
    raw = ensure_raw(cfg)
    db = pd.read_csv(raw / "ptbxl_database.csv")
    statements = load_statements(raw / "scp_statements.csv")
    labels = build_labels(db, statements, cfg)
    rhythm_classes = list(dict.fromkeys(labels["rhythm_code"]))
    rhythm_classes = [name for name in rhythm_classes if name != "none"]

    out = cfg.root / "processed"
    out.mkdir(parents=True, exist_ok=True)
    tmp_signals = out / "signals.tmp.npy"
    signals = np.lib.format.open_memmap(
        tmp_signals,
        mode="w+",
        dtype=np.float16,
        shape=(len(labels), cfg.n_leads, cfg.signal_len),
    )

    kept = []
    for i, row in enumerate(labels.itertuples(index=False)):
        signal = _read_signal(raw / row.filename, cfg)
        if signal is None:
            continue
        signals[len(kept)] = signal.astype(np.float16)
        kept.append(i)
        if (len(kept) + 1) % 2000 == 0:
            log.info("processed %d / %d records", len(kept) + 1, len(labels))

    signals.flush()
    del signals
    full = np.load(tmp_signals, mmap_mode="r")
    np.save(out / SIGNALS_NAME, np.asarray(full[: len(kept)]), allow_pickle=False)
    del full
    tmp_signals.unlink()

    manifest = labels.iloc[kept].reset_index(drop=True)
    manifest.to_parquet(out / MANIFEST_NAME, index=False)
    meta = {
        "version": cfg.version,
        "hz": cfg.hz,
        "signal_len": cfg.signal_len,
        "n_leads": cfg.n_leads,
        "superclasses": list(SUPERCLASSES),
        "rhythm_classes": rhythm_classes,
        "n_records": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "skipped": int(len(labels) - len(manifest)),
    }
    (out / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info(
        "processed %d records from %d patients into %s",
        meta["n_records"],
        meta["n_patients"],
        out,
    )
    return out


def load_processed(cfg: DataConfig) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    """Load manifest, memmap and metadata written by :func:`prepare`."""
    root = cfg.root / "processed"
    manifest = pd.read_parquet(root / MANIFEST_NAME)
    signals = np.load(root / SIGNALS_NAME, mmap_mode="r")
    meta = json.loads((root / META_NAME).read_text(encoding="utf-8"))
    return manifest, signals, meta
