"""Metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from ecgmtl.metrics import age_metrics, aggregate, diag_metrics, rhythm_metrics, task_metrics


def test_diag_metrics_perfect() -> None:
    y_true = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
    y_score = y_true.astype(float)
    metrics = diag_metrics(y_true, y_score, ["NORM", "MI", "STTC", "CD", "HYP"])
    assert metrics["macro_auc"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["support_NORM"] == 2


def test_diag_metrics_single_class_has_no_auc() -> None:
    y_true = np.zeros((4, 5))
    y_true[:, 0] = 1
    metrics = diag_metrics(y_true, np.full((4, 5), 0.5), ["A", "B", "C", "D", "E"])
    assert "auc_B" not in metrics
    # every class is constant in this fold, so neither AUC nor macro-F1 is defined
    assert np.isnan(metrics["macro_auc"])
    assert np.isnan(metrics["macro_f1"])


def test_rhythm_metrics() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_score = np.eye(3)[y_true]
    metrics = rhythm_metrics(y_true, y_score, ["SR", "AFIB", "STACH"])
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)


def test_age_metrics() -> None:
    metrics = age_metrics(np.array([40.0, 50.0]), np.array([42.0, 48.0]))
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["rmse"] == pytest.approx(2.0)


def test_aggregate_reports_ci() -> None:
    folds = [{"m": 0.5}, {"m": 0.6}, {"m": 0.7}]
    summary = aggregate(folds, ["m"])["m"]
    assert summary["mean"] == pytest.approx(0.6)
    assert summary["n"] == 3
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]


def test_task_metrics_uses_masks() -> None:
    pred = {
        "diag_true": np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        "diag_score": np.array([[0.9, 0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2, 0.2]]),
        "rhythm_true": np.array([0, 0]),
        "rhythm_score": np.eye(2)[[0, 1]],
        "age_true": np.array([50.0, 0.0]),
        "age_pred": np.array([52.0, 0.0]),
        "mask": np.array([[True, False, True], [True, False, False]]),
    }
    meta = {"superclasses": ["A", "B", "C", "D", "E"], "rhythm_classes": ["SR", "AFIB"]}
    metrics = task_metrics(pred, meta)
    assert "diag" in metrics and "rhythm" not in metrics and "age" in metrics
    assert metrics["n"] == {"diag": 2, "rhythm": 0, "age": 1}
