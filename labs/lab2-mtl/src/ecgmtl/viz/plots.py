"""Plots for EDA and for the final report. Every figure is written as PNG and embedded."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
COLORS = ["#2f6fb3", "#d2693a", "#4c9a52", "#8a5fa8", "#c2a33a", "#4a4a4a", "#b34a6f", "#3aa8a8"]


def save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def embed(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def class_distribution(manifest: pd.DataFrame, superclasses: Sequence[str], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    counts = [float(manifest[f"diag_{name}"].mean()) for name in superclasses]
    axes[0].bar(list(superclasses), counts, color=COLORS[: len(superclasses)])
    axes[0].set_title("доля записей с суперклассом")
    axes[0].set_ylabel("доля")
    axes[0].tick_params(axis="x", rotation=20)
    rhythm = manifest["rhythm_code"].value_counts()
    rhythm_labels = [str(name) for name in rhythm.index]
    axes[1].bar(rhythm_labels, rhythm.to_numpy(), color="#4a4a4a")
    axes[1].set_title("распределение класса ритма")
    axes[1].tick_params(axis="x", rotation=45)
    return save(fig, path)


def label_corr(matrix: pd.DataFrame, path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(matrix), 1.2 + 0.5 * len(matrix)))
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.grid(False)
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return save(fig, path)


def age_distribution(manifest: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    age = manifest["age"].dropna()
    ax.hist(age, bins=40, color=COLORS[0])
    ax.set_title(f"возраст: {len(age)} записей, {int(manifest['age'].isna().sum())} без метки")
    ax.set_xlabel("лет")
    return save(fig, path)


def split_summary(manifest: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3))
    counts = manifest["strat_fold"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.to_numpy(), color=COLORS[2])
    ax.set_title("размер фолда strat_fold (пациент-уровневый split)")
    ax.set_xlabel("фолд")
    return save(fig, path)


def ecg_examples(signals: np.ndarray, manifest: pd.DataFrame, path: Path, n: int = 4) -> Path:
    idx = np.linspace(0, len(manifest) - 1, n).astype(int)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        signal = np.asarray(signals[i], dtype=np.float32)
        offset = 0.0
        for lead in range(signal.shape[0]):
            ax.plot(signal[lead] + offset, lw=0.6, color="#1f4e79")
            offset += 5.0
        row = manifest.iloc[i]
        labels = [name for name in manifest.columns if name.startswith("diag_") and row[name] == 1]
        title = f"ecg {int(row['ecg_id'])}, возраст {row['age']:.0f}, ритм {row['rhythm_code']}"
        if labels:
            title += ", " + ", ".join(name.replace("diag_", "") for name in labels)
        ax.set_title(title, fontsize=8)
        ax.set_yticks([])
        ax.grid(False)
    axes[-1].set_xlabel("отсчёты (100 Гц)")
    return save(fig, path)


def history_curves(history: list[dict[str, Any]], path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    epochs = [entry["epoch"] for entry in history]
    tasks = list(history[0]["train"]["losses"]) if history else []
    for i, task in enumerate(tasks):
        axes[0].plot(epochs, [entry["train"]["losses"][task] for entry in history], marker="o", color=COLORS[i], label=task)
    axes[0].set_title("лоссы по задачам (train)")
    axes[0].set_xlabel("эпоха")
    axes[0].legend()
    for i, task in enumerate(tasks):
        if task == "diag":
            key = "macro_auc"
        elif task == "rhythm":
            key = "macro_f1"
        else:
            key = "mae"
        values = [entry["val"].get(task, {}).get(key, np.nan) for entry in history]
        axes[1].plot(epochs, values, marker="o", color=COLORS[i], label=f"{task}:{key}")
    axes[1].set_title("валидационные метрики")
    axes[1].set_xlabel("эпоха")
    axes[1].legend()
    return save(fig, path)


def conflict_curve(history: list[dict[str, Any]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    epochs = [entry["epoch"] for entry in history]
    keys = sorted({key for entry in history for key in entry["train"].get("conflict", {})})
    for i, key in enumerate(keys):
        values = [entry["train"].get("conflict", {}).get(key, np.nan) for entry in history]
        ax.plot(epochs, values, marker="o", color=COLORS[i % len(COLORS)], label=key.replace("__", " vs "))
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_title("косинус между градиентами задач на общем энкодере")
    ax.set_xlabel("эпоха")
    ax.set_ylabel("cos")
    ax.legend(fontsize=7)
    return save(fig, path)


def age_scatter(pred: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 4))
    rows = pred[pred["has_age"].astype(bool)]
    ax.scatter(rows["age_true"], rows["age_pred"], s=4, alpha=0.25, color=COLORS[0])
    lim = [float(rows["age_true"].min()), float(rows["age_true"].max())]
    ax.plot(lim, lim, color="black", lw=0.8, ls="--")
    ax.set_xlabel("истинный возраст")
    ax.set_ylabel("предсказанный")
    return save(fig, path)


def rhythm_confusion(pred: pd.DataFrame, names: Sequence[str], path: Path) -> Path:
    rows = pred[pred["has_rhythm"].astype(bool)]
    labels = list(range(len(names)))
    matrix = confusion_matrix(rows["rhythm_true"], rows["rhythm_pred"], labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(1.0 + 0.5 * len(names), 1.2 + 0.45 * len(names)))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(labels)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(labels)
    ax.set_yticklabels(names)
    ax.grid(False)
    ax.set_xlabel("предсказано")
    ax.set_ylabel("истина")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return save(fig, path)
