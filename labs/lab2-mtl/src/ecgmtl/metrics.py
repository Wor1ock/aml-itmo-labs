"""Per-task metrics.

Every task keeps its own metric family: macro-AUC and macro-F1 for multi-label diagnosis,
macro-F1 and accuracy for single-label rhythm, MAE/RMSE/R2 for age regression.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)


def diag_metrics(y_true: np.ndarray, y_score: np.ndarray, names: list[str]) -> dict[str, object]:
    """y_true: (N, C) binary, y_score: (N, C) probabilities."""
    out: dict[str, object] = {}
    aucs: list[float] = []
    for i, name in enumerate(names):
        if y_true[:, i].min() == y_true[:, i].max():
            continue
        auc = float(roc_auc_score(y_true[:, i], y_score[:, i]))
        out[f"auc_{name}"] = auc
        aucs.append(auc)
        out[f"ap_{name}"] = float(average_precision_score(y_true[:, i], y_score[:, i]))
    out["macro_auc"] = float(np.mean(aucs)) if aucs else float("nan")
    y_pred = (y_score >= 0.5).astype(int)
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    for i, name in enumerate(names):
        out[f"f1_{name}"] = float(f1[i])
        out[f"precision_{name}"] = float(precision[i])
        out[f"recall_{name}"] = float(recall[i])
        out[f"support_{name}"] = int(support[i])
    return out


def rhythm_metrics(y_true: np.ndarray, y_score: np.ndarray, names: list[str]) -> dict[str, object]:
    """y_true: (N,) int, y_score: (N, K) probabilities."""
    out: dict[str, object] = {}
    y_pred = y_score.argmax(axis=1)
    present = np.zeros(len(names), dtype=bool)
    present[y_true] = True
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(names)), average=None, zero_division=0
    )
    for i, name in enumerate(names):
        out[f"f1_{name}"] = float(f1[i])
        out[f"precision_{name}"] = float(precision[i])
        out[f"recall_{name}"] = float(recall[i])
        out[f"support_{name}"] = int(support[i])
    return out


def age_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    out: dict[str, object] = {"n": int(len(y_true))}
    if len(y_true) < 2:
        return out
    out["mae"] = float(mean_absolute_error(y_true, y_pred))
    out["rmse"] = float(float(np.sqrt(mean_squared_error(y_true, y_pred))))
    out["r2"] = float(r2_score(y_true, y_pred))
    out["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    return out


def task_metrics(
    pred: dict[str, np.ndarray],
    meta: dict[str, object],
) -> dict[str, object]:
    """Metrics for all tasks from one prediction bundle."""
    names = list(meta["superclasses"])  # type: ignore[arg-type]
    rhythm_names = list(meta["rhythm_classes"])  # type: ignore[arg-type]
    mask = pred["mask"]
    out: dict[str, object] = {}
    diag_rows = mask[:, 0]
    if diag_rows.any():
        out["diag"] = diag_metrics(pred["diag_true"][diag_rows], pred["diag_score"][diag_rows], names)
    rhythm_rows = mask[:, 1]
    if rhythm_rows.any():
        out["rhythm"] = rhythm_metrics(
            pred["rhythm_true"][rhythm_rows], pred["rhythm_score"][rhythm_rows], rhythm_names
        )
    age_rows = mask[:, 2]
    if age_rows.any():
        out["age"] = age_metrics(pred["age_true"][age_rows], pred["age_pred"][age_rows])
    out["n"] = {"diag": int(diag_rows.sum()), "rhythm": int(rhythm_rows.sum()), "age": int(age_rows.sum())}
    return out


def aggregate(folds: list[dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    """Mean, std and 95% CI of a metric across fold runs."""
    out: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.array([f[key] for f in folds if key in f and np.isfinite(f[key])], dtype=float)
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        margin = 1.96 * std / np.sqrt(values.size) if values.size > 1 else 0.0
        out[key] = {
            "mean": float(values.mean()),
            "std": std,
            "ci_low": float(values.mean() - margin),
            "ci_high": float(values.mean() + margin),
            "n": int(values.size),
        }
    return out


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
