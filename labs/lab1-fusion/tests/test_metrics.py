from __future__ import annotations

import numpy as np
import pytest

from fusion_slu.metrics import (
    confusion_matrix,
    expected_calibration_error,
    head_metrics,
    joint_labels,
    overall_metrics,
    per_class_scores,
    top_confusions,
    wilson_interval,
)


def test_confusion_matrix_layout() -> None:
    matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], 2)
    assert matrix.tolist() == [[1, 1], [1, 1]]
    assert per_class_scores(matrix)["f1"].tolist() == pytest.approx([0.5, 0.5])


def test_head_metrics_hand_computed() -> None:
    metrics = head_metrics([0, 0, 1, 1], [0, 1, 0, 1], 2, ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["weighted_f1"] == pytest.approx(0.5)
    assert metrics["per_class"]["a"]["support"] == 2


def test_overall_metrics_exact_match_and_joint() -> None:
    true = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 1]])
    pred = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    metrics = overall_metrics(true, pred, [2, 2, 2], [["a", "b"]] * 3)
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert metrics["joint_intent_accuracy"] == pytest.approx(2 / 3)
    assert joint_labels(true, [2, 2, 2]).tolist() == [0, 7, 5]


def test_ece_is_zero_for_perfect_confident_predictions() -> None:
    result = expected_calibration_error([1.0, 1.0, 1.0], [0, 1, 2], [0, 1, 2])
    assert result["ece"] == pytest.approx(0.0, abs=1e-6)
    assert sum(bin_["count"] for bin_ in result["bins"]) == 3


def test_top_confusions_ignores_diagonal() -> None:
    matrix = np.array([[5, 2], [1, 4]])
    pairs = top_confusions(matrix, ["a", "b"], k=5)
    assert {(pair["true"], pair["predicted"], pair["count"]) for pair in pairs} == {
        ("a", "b", 2),
        ("b", "a", 1),
    }


def test_wilson_interval_contains_estimate() -> None:
    low, high = wilson_interval(80, 100)
    assert low < 0.8 < high
