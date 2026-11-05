from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def _as_int_array(values: Sequence[int] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size == 0:
        raise ValueError("metric inputs must not be empty")
    return array


def confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Rows are true classes, columns are predictions."""
    true = _as_int_array(y_true)
    pred = _as_int_array(y_pred)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.min() < 0 or pred.min() < 0 or true.max() >= n_classes or pred.max() >= n_classes:
        raise ValueError("class index is outside [0, n_classes)")
    return np.bincount(true * n_classes + pred, minlength=n_classes * n_classes).reshape(
        n_classes, n_classes
    )


def per_class_scores(confusion: np.ndarray) -> dict[str, np.ndarray]:
    diagonal = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    precision = np.divide(diagonal, predicted, out=np.zeros_like(diagonal), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros_like(diagonal), where=support > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def _mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    if values.size == 0:
        return 0.0
    if weights is None:
        return float(values.mean())
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float((values * weights).sum() / total)


def head_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    n_classes: int,
    class_names: Sequence[str],
) -> dict[str, Any]:
    true = _as_int_array(y_true)
    pred = _as_int_array(y_pred)
    confusion = confusion_matrix(true, pred, n_classes)
    scores = per_class_scores(confusion)
    present = scores["support"] > 0
    return {
        "accuracy": float((true == pred).mean()),
        "macro_f1": _mean(scores["f1"][present]),
        "weighted_f1": _mean(scores["f1"], scores["support"]),
        "n": int(true.size),
        "per_class": {
            name: {
                "precision": float(scores["precision"][index]),
                "recall": float(scores["recall"][index]),
                "f1": float(scores["f1"][index]),
                "support": int(scores["support"][index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion": confusion.tolist(),
    }


def joint_labels(labels: Sequence[int] | np.ndarray, dims: Sequence[int]) -> np.ndarray:
    array = np.asarray(labels, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("labels must have shape [n, 3]")
    return (array[:, 0] * dims[1] + array[:, 1]) * dims[2] + array[:, 2]


def overall_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    dims: Sequence[int],
    class_names: Sequence[Sequence[str]],
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.int64)
    pred = np.asarray(y_pred, dtype=np.int64)
    if true.ndim != 2 or true.shape != pred.shape or true.shape[1] != 3:
        raise ValueError("predictions and labels must have shape [n, 3]")
    heads = {
        name: head_metrics(true[:, index], pred[:, index], dims[index], class_names[index])
        for index, name in enumerate(("action", "object", "location"))
    }
    correct = (true == pred).all(axis=1)
    joint_true = joint_labels(true, dims)
    joint_pred = joint_labels(pred, dims)
    return {
        "n": int(true.shape[0]),
        "exact_match": float(correct.mean()),
        "mean_head_accuracy": float(np.mean([heads[name]["accuracy"] for name in heads])),
        "mean_weighted_f1": float(np.mean([heads[name]["weighted_f1"] for name in heads])),
        "joint_intent_accuracy": float((joint_true == joint_pred).mean()),
        "heads": heads,
    }


def expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    targets: Sequence[int] | np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    confidence = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    pred = _as_int_array(predictions)
    true = _as_int_array(targets)
    if not (confidence.shape == pred.shape == true.shape):
        raise ValueError("probabilities, predictions and targets must have the same shape")
    confidence = np.clip(confidence, 0.0, 1.0)
    correct = (pred == true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            mask = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "lo": float(edges[index]),
                    "hi": float(edges[index + 1]),
                    "count": 0,
                    "accuracy": 0.0,
                    "confidence": 0.0,
                }
            )
            continue
        accuracy = float(correct[mask].mean())
        mean_confidence = float(confidence[mask].mean())
        ece += count / confidence.size * abs(accuracy - mean_confidence)
        bins.append(
            {
                "lo": float(edges[index]),
                "hi": float(edges[index + 1]),
                "count": count,
                "accuracy": accuracy,
                "confidence": mean_confidence,
            }
        )
    return {"ece": float(ece), "bins": bins, "mean_confidence": float(confidence.mean())}


def top_confusions(
    confusion: np.ndarray,
    labels: Sequence[str],
    k: int = 10,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for true_index in range(confusion.shape[0]):
        for pred_index in range(confusion.shape[1]):
            if true_index == pred_index:
                continue
            count = int(confusion[true_index, pred_index])
            if count > 0:
                result.append(
                    {
                        "true": labels[true_index],
                        "predicted": labels[pred_index],
                        "count": count,
                    }
                )
    result.sort(key=lambda item: (-int(item["count"]), str(item["true"]), str(item["predicted"])))
    return result[:k]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, centre - margin), min(1.0, centre + margin)
