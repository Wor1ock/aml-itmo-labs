"""Plotting helpers used by the report and by the training/evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fusion_slu.config import HEAD_NAMES

COLORS = ("#2f6f9f", "#d9822b", "#4c9f70", "#b04a5a", "#7b6aa8", "#8c8c8c")


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _empty(ax: Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def plot_training_curves(
    history: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "",
) -> None:
    _style()
    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", ms=3, color=COLORS[0], label="train")
    axes[0].plot(epochs, [row["valid_loss"] for row in history], marker="o", ms=3, color=COLORS[1], label="valid")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    for index, head in enumerate(HEAD_NAMES):
        axes[1].plot(
            epochs,
            [row["valid"][f"{head}_accuracy"] for row in history],
            marker="o",
            ms=3,
            color=COLORS[index],
            label=f"valid {head}",
        )
        axes[1].plot(
            epochs,
            [row["train"][f"{head}_accuracy"] for row in history],
            ls="--",
            lw=1.0,
            color=COLORS[index],
            alpha=0.55,
        )
    axes[1].plot(
        epochs,
        [row["valid"]["exact_match"] for row in history],
        marker="s",
        ms=3,
        color="#222222",
        label="valid exact match",
    )
    axes[1].set_title("Accuracy (dashed: train)")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=7, ncol=2)
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_dataset_overview(
    label_counts: Mapping[str, Mapping[str, int]],
    durations: Sequence[float],
    path: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), constrained_layout=True)
    for axis, head in zip(axes.flat[:3], HEAD_NAMES, strict=True):
        counts = label_counts[head]
        names = list(counts)
        values = [counts[name] for name in names]
        axis.bar(names, values, color=COLORS[0])
        axis.set_title(f"{head}: train samples per class")
        axis.tick_params(axis="x", rotation=60)
        axis.set_ylabel("count")
    axes.flat[3].hist(np.asarray(durations), bins=40, color=COLORS[1], edgecolor="white")
    axes.flat[3].set_title("Utterance duration, train split")
    axes.flat[3].set_xlabel("seconds")
    axes.flat[3].set_ylabel("count")
    _save(fig, path)


def plot_intent_distribution(counts: Mapping[str, int], path: Path) -> None:
    _style()
    items = sorted(counts.items(), key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(8.5, 7.4), constrained_layout=True)
    ax.barh([name for name, _ in items], [value for _, value in items], color=COLORS[0])
    ax.set_title("Train samples per resolved intent (action + object + location)")
    ax.set_xlabel("count")
    ax.tick_params(axis="y", labelsize=7)
    _save(fig, path)


def plot_spectrogram_grid(
    examples: Sequence[tuple[np.ndarray, str]],
    path: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(1, len(examples), figsize=(4.0 * len(examples), 3.2), constrained_layout=True)
    if len(examples) == 1:
        axes = np.asarray([axes], dtype=object)
    for axis, (feature, title) in zip(axes, examples, strict=True):
        image = axis.imshow(feature.T, origin="lower", aspect="auto", cmap="magma")
        axis.set_title(title, fontsize=8)
        axis.set_xlabel("frame (10 ms)")
        axis.set_ylabel("mel bin")
        axis.grid(False)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    _save(fig, path)


def plot_model_comparison(summary: Mapping[str, Mapping[str, tuple[float, float]]], path: Path) -> None:
    _style()
    models = list(summary)
    metrics = [("exact_match", "exact match")]
    metrics.extend((f"{head}_f1", f"{head} weighted F1") for head in HEAD_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
    x = np.arange(len(models))
    axes[0].bar(
        x,
        [summary[model].get("exact_match", (0.0, 0.0))[0] for model in models],
        yerr=[summary[model].get("exact_match", (0.0, 0.0))[1] for model in models],
        capsize=3,
        color=COLORS[2],
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=25, ha="right")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_ylabel("test exact match")
    axes[0].set_title("All three slots correct (mean ± std over seeds)")
    width = 0.8 / max(len(models), 1)
    for model_index, model in enumerate(models):
        values = [summary[model].get(f"{head}_f1", (0.0, 0.0))[0] for head in HEAD_NAMES]
        errors = [summary[model].get(f"{head}_f1", (0.0, 0.0))[1] for head in HEAD_NAMES]
        axes[1].bar(
            np.arange(len(HEAD_NAMES)) + model_index * width,
            values,
            width,
            yerr=errors,
            capsize=2,
            label=model,
            color=COLORS[model_index % len(COLORS)],
        )
    axes[1].set_xticks(np.arange(len(HEAD_NAMES)) + width * (len(models) - 1) / 2)
    axes[1].set_xticklabels(HEAD_NAMES)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("weighted F1")
    axes[1].set_title("Per-head quality")
    axes[1].legend(fontsize=7, ncol=2)
    _save(fig, path)


def plot_robustness(
    text_series: Mapping[str, Sequence[tuple[str, float, float]]],
    audio_series: Mapping[str, Sequence[tuple[str, float, float]]],
    path: Path,
) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), constrained_layout=True)
    for panel, series, title in (
        (axes[0], text_series, "Degraded transcript"),
        (axes[1], audio_series, "Degraded audio"),
    ):
        for index, (model, points) in enumerate(series.items()):
            if not points:
                continue
            labels = [point[0] for point in points]
            means = np.asarray([point[1] for point in points])
            errors = np.asarray([point[2] for point in points])
            x = np.arange(len(points))
            panel.plot(x, means, marker="o", ms=3.5, color=COLORS[index % len(COLORS)], label=model)
            panel.fill_between(x, means - errors, means + errors, color=COLORS[index % len(COLORS)], alpha=0.13)
        if panel.lines:
            first = next(iter(series.values()))
            panel.set_xticks(np.arange(len(first)))
            panel.set_xticklabels([point[0] for point in first], rotation=25, ha="right")
        else:
            _empty(panel, "no applicable conditions")
        panel.set_ylim(0, 1.02)
        panel.set_ylabel("exact match")
        panel.set_title(title)
        panel.legend(fontsize=7)
    _save(fig, path)


def plot_confusion_matrices(
    metrics: Mapping[str, Any],
    class_names: Sequence[Sequence[str]],
    path: Path,
    *,
    title: str = "",
) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), constrained_layout=True)
    for index, (axis, head) in enumerate(zip(axes, HEAD_NAMES, strict=True)):
        confusion = np.asarray(metrics["heads"][head]["confusion"], dtype=np.float64)
        row_sums = confusion.sum(axis=1, keepdims=True)
        normalized = confusion / np.maximum(row_sums, 1.0)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        names = list(class_names[index])
        axis.set_xticks(np.arange(len(names)))
        axis.set_xticklabels(names, rotation=90, fontsize=6.5)
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names, fontsize=6.5)
        axis.set_title(f"{head} (acc {metrics['heads'][head]['accuracy']:.3f})")
        axis.set_xlabel("predicted")
        axis.set_ylabel("true")
        axis.grid(False)
        if len(names) <= 8:
            for row in range(len(names)):
                for column in range(len(names)):
                    if normalized[row, column] >= 0.01:
                        axis.text(
                            column,
                            row,
                            f"{normalized[row, column]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color="white" if normalized[row, column] > 0.55 else "black",
                        )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_per_class_f1(
    metrics: Mapping[str, Any],
    class_names: Sequence[Sequence[str]],
    path: Path,
    *,
    title: str = "",
) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.6), constrained_layout=True)
    for index, (axis, head) in enumerate(zip(axes, HEAD_NAMES, strict=True)):
        per_class = metrics["heads"][head]["per_class"]
        names = list(class_names[index])
        values = [per_class[name]["f1"] for name in names]
        order = np.argsort(values)
        axis.barh([names[i] for i in order], [values[i] for i in order], color=COLORS[index])
        axis.set_xlim(0, 1.02)
        axis.set_title(f"{head}: per-class F1")
        axis.set_xlabel("F1")
        axis.tick_params(axis="y", labelsize=6.5)
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_calibration(
    calibration: Mapping[str, Mapping[str, Any]],
    path: Path,
    *,
    title: str = "",
) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    for index, (axis, head) in enumerate(zip(axes, HEAD_NAMES, strict=True)):
        payload = calibration[head]
        bins = [item for item in payload["bins"] if item["count"] > 0]
        confidence = [float(item["confidence"]) for item in bins]
        accuracy = [float(item["accuracy"]) for item in bins]
        sizes = [max(8.0, float(item["count"]) ** 0.5) for item in bins]
        axis.plot([0, 1], [0, 1], ls="--", color="#999999", lw=1)
        axis.scatter(confidence, accuracy, s=sizes, color=COLORS[index], alpha=0.8)
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("confidence")
        axis.set_ylabel("accuracy")
        axis.set_title(f"{head}, ECE={payload['ece']:.3f}")
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_attention_examples(examples: Sequence[Mapping[str, Any]], path: Path) -> None:
    _style()
    if not examples:
        fig, axis = plt.subplots(figsize=(6, 2))
        _empty(axis, "no attention examples")
        _save(fig, path)
        return
    rows = (len(examples) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13.0, 2.6 * rows), constrained_layout=True, squeeze=False)
    for axis, example in zip(axes.flat, examples, strict=False):
        attention = np.asarray(example["attention"], dtype=np.float64)
        tokens = list(example["tokens"])[: attention.shape[0]]
        frames = int(example["audio_frames"])
        matrix = attention[: len(tokens), :frames]
        if matrix.size == 0:
            _empty(axis, "empty attention")
            continue
        image = axis.imshow(matrix, aspect="auto", origin="upper", cmap="viridis")
        axis.set_yticks(np.arange(len(tokens)))
        axis.set_yticklabels(tokens, fontsize=6.5)
        axis.set_xlabel("audio time, seconds")
        positions = np.linspace(0, frames, 4)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{position * 0.08:.1f}" for position in positions])
        axis.grid(False)
        true_text = " / ".join(str(item) for item in example["true"])
        pred_text = " / ".join(str(item) for item in example["pred"])
        axis.set_title(
            f"“{example['transcription']}”\ntrue: {true_text} | pred: {pred_text}",
            fontsize=7.5,
        )
        fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02)
    for axis in list(axes.flat)[len(examples) :]:
        _empty(axis, "")
    _save(fig, path)


def plot_error_panels(analysis: Mapping[str, Any], path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    duration_rows = analysis.get("duration_buckets", [])
    axes[0].bar(
        [str(item["label"]) for item in duration_rows],
        [float(item["error_rate"]) for item in duration_rows],
        color=COLORS[1],
    )
    axes[0].set_title("Error rate by duration")
    axes[0].set_ylabel("fraction of utterances")
    axes[0].tick_params(axis="x", rotation=20)

    distribution = analysis.get("confidence", {})
    correct = distribution.get("correct", [])
    wrong = distribution.get("incorrect", [])
    bins = np.linspace(0.0, 1.0, 21)
    if correct:
        axes[1].hist(correct, bins=bins, alpha=0.65, label="correct", color=COLORS[2], density=True)
    if wrong:
        axes[1].hist(wrong, bins=bins, alpha=0.65, label="incorrect", color=COLORS[3], density=True)
    axes[1].set_title("Mean confidence per utterance")
    axes[1].set_xlabel("confidence")
    axes[1].set_ylabel("density")
    axes[1].legend(fontsize=7)
    if not correct and not wrong:
        axes[1].text(0.5, 0.5, "no data", ha="center", va="center", transform=axes[1].transAxes)

    speakers = analysis.get("speakers", [])[:10]
    axes[2].barh(
        [str(item["speaker"]) for item in speakers][::-1],
        [float(item["error_rate"]) for item in speakers][::-1],
        color=COLORS[3],
    )
    axes[2].set_title("Speakers with most errors")
    axes[2].set_xlabel("error rate")
    axes[2].tick_params(axis="y", labelsize=6.5)
    _save(fig, path)
