from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils import NAMES


def save(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_curves(history: list[dict[str, float]], path: str) -> None:
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in ("train", "valid"):
        ax[0].plot(ep, [h[f"{s}_loss"] for h in history], label=s)
    ax[0].set_title("loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    for i, name in enumerate(NAMES):
        ax[1].plot(ep, [h[f"train_{name}_acc"] for h in history], f"C{i}--")
        ax[1].plot(ep, [h[f"valid_{name}_acc"] for h in history], f"C{i}-", label=name)
    ax[1].set_title("accuracy (dashed: train, solid: valid)")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    save(fig, path)


def plot_confusions(confs: list[torch.Tensor], labels: list[list[str]], path: str) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, c, names, title in zip(ax, confs, labels, NAMES):
        norm = c.float() / c.sum(1, keepdim=True).clamp(min=1)
        a.imshow(norm.numpy(), vmin=0, vmax=1, cmap="Blues")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=90)
        a.set_yticks(range(len(names)))
        a.set_yticklabels(names)
        a.set_title(title)
        a.set_xlabel("predicted")
        a.set_ylabel("true")
    save(fig, path)


def plot_attention(items: list[tuple[torch.Tensor, list[str]]], path: str) -> None:
    # each item: attention [n_tokens, n_chunks] of the last cross-attention layer and the token strings
    fig, ax = plt.subplots(len(items), 1, figsize=(10, 2.5 * len(items)), squeeze=False)
    for a, (att, tokens) in zip(ax[:, 0], items):
        im = a.imshow(att.numpy(), aspect="auto", cmap="viridis")
        a.set_yticks(range(len(tokens)))
        a.set_yticklabels(tokens)
        a.set_xlabel("audio chunk")
        fig.colorbar(im, ax=a)
    save(fig, path)


def plot_data(df: dict[str, list[str]], lens: list[int], path: str) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    axes = ax.flatten()
    for a, name in zip(axes[:3], NAMES):
        c = Counter(df[name])
        a.bar(list(c), list(c.values()))
        a.tick_params(axis="x", labelrotation=90)
        a.set_title(f"{name}: samples per class")
    axes[3].hist(lens, bins=40)
    axes[3].set_title("audio length, samples")
    save(fig, path)


def plot_summary(summary: dict[str, dict[str, tuple[float, float]]], path: str) -> None:
    mods = list(summary)
    w = 0.8 / len(mods)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, mod in enumerate(mods):
        mean = [summary[mod][f"{n}_f1"][0] for n in NAMES]
        std = [summary[mod][f"{n}_f1"][1] for n in NAMES]
        ax.bar([i + j * w for i in range(len(NAMES))], mean, w, yerr=std, capsize=3, label=mod)
    ax.set_xticks([i + w * (len(mods) - 1) / 2 for i in range(len(NAMES))])
    ax.set_xticklabels(NAMES)
    ax.set_ylabel("test weighted f1 (mean, std over seeds)")
    ax.legend()
    save(fig, path)
