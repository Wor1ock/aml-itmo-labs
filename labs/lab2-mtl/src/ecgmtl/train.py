"""One training run: shared encoder, task heads, loss aggregation, artefacts."""

from __future__ import annotations

import json
import os
import platform
import random
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import Config, save_config, to_dict
from .data.dataset import EcgDataset, make_loader, make_split
from .data.labels import SUPERCLASSES
from .data.prepare import load_processed
from .logging_utils import get_logger
from .losses.equal import weighted_sum
from .losses.masking import TaskLoss, compute_task_losses
from .losses.pcgrad import conflict_matrix, pcgrad_backward, task_gradients
from .losses.uncertainty import UncertaintyWeighting
from .metrics import task_metrics, to_device
from .models import build_model
from .models.mtl import MtlEcg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def pos_weight_tensor(
    manifest: pd.DataFrame, train_idx: np.ndarray, enabled: bool
) -> torch.Tensor | None:
    """Class imbalance weights computed on the training split only."""
    if not enabled:
        return None
    target = manifest[[f"diag_{name}" for name in SUPERCLASSES]].to_numpy(dtype=np.float32)[train_idx]
    pos = target.sum(axis=0)
    neg = len(target) - pos
    weights = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict(model: MtlEcg, loader: Any, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    diag_score, diag_true = [], []
    rhythm_score, rhythm_true = [], []
    age_pred, age_true = [], []
    mask_all = []
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(batch["signal"])
        diag_score.append(torch.sigmoid(outputs["diag"]).cpu().numpy())
        rhythm_score.append(torch.softmax(outputs["rhythm"], dim=-1).cpu().numpy())
        age_pred.append(outputs["age"].squeeze(-1).cpu().numpy())
        diag_true.append(batch["diag"].cpu().numpy())
        rhythm_true.append(batch["rhythm"].cpu().numpy())
        age_true.append(batch["age"].cpu().numpy())
        mask_all.append(batch["mask"].cpu().numpy())
    return {
        "diag_score": np.concatenate(diag_score),
        "diag_true": np.concatenate(diag_true),
        "rhythm_score": np.concatenate(rhythm_score),
        "rhythm_true": np.concatenate(rhythm_true),
        "age_pred": np.concatenate(age_pred),
        "age_true": np.concatenate(age_true),
        "mask": np.concatenate(mask_all),
    }


def _log_conflict(
    model: MtlEcg, losses: dict[str, TaskLoss], acc: dict[str, list[float]]
) -> None:
    active = {name: task.loss for name, task in losses.items() if task.n_valid > 0}
    if len(active) < 2:
        return
    grads, _ = task_gradients(active, model.shared_parameters())
    for key, value in conflict_matrix(grads).items():
        acc[key].append(value)


def train_epoch(
    model: MtlEcg,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    uncertainty: UncertaintyWeighting | None,
    generator: torch.Generator,
) -> dict[str, Any]:
    model.train()
    log = get_logger()
    loss_sum: dict[str, float] = defaultdict(float)
    loss_count: dict[str, int] = defaultdict(int)
    conflict_acc: dict[str, list[float]] = defaultdict(list)
    projections = 0
    steps = 0
    for step, batch in enumerate(loader):
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["signal"])
        losses = compute_task_losses(outputs, batch, cfg.loss.tasks, pos_weight)
        active = {name: task.loss for name, task in losses.items() if task.n_valid > 0}
        if not active:
            continue

        if step % cfg.train.conflict_every == 0:
            _log_conflict(model, losses, conflict_acc)

        if cfg.loss.strategy == "pcgrad":
            n_proj, total = pcgrad_backward(
                active, model.shared_parameters(), model.head_parameters_all(), generator
            )
            projections += n_proj
        elif cfg.loss.strategy == "uncertainty":
            if uncertainty is None:
                raise RuntimeError("uncertainty weighting is not initialised")
            total = uncertainty(losses)
            total.backward()
        else:
            total = weighted_sum(losses, cfg.loss.weights)
            total.backward()

        params = list(model.parameters()) + ([uncertainty.log_var] if uncertainty is not None else [])
        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        optimizer.step()

        for name, task in losses.items():
            if task.n_valid == 0:
                continue
            loss_sum[name] += float(task.loss.detach())
            loss_count[name] += 1
        steps += 1
        if step % cfg.train.log_every == 0:
            log.info(
                "  step %d/%d total=%.4f %s",
                step,
                len(loader),
                float(total.detach()),
                " ".join(f"{n}={float(t.loss.detach()):.3f}" for n, t in losses.items()),
            )

    out: dict[str, Any] = {
        "losses": {name: loss_sum[name] / max(loss_count[name], 1) for name in loss_sum},
        "steps": steps,
        "projections": projections,
        "conflict": {key: float(np.mean(values)) for key, values in conflict_acc.items()},
    }
    if uncertainty is not None:
        out["weights"] = uncertainty.weights()
        out["log_var"] = [float(v) for v in uncertainty.log_var.detach()]
    return out


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _monitor_score(metrics: dict[str, Any], tasks: tuple[str, ...]) -> float:
    """Early stopping signal: main task first, then rhythm, then negative MAE."""
    if "diag" in tasks and "diag" in metrics:
        return float(metrics["diag"]["macro_auc"])  # type: ignore[index]
    if "rhythm" in tasks and "rhythm" in metrics:
        return float(metrics["rhythm"]["macro_f1"])  # type: ignore[index]
    if "age" in tasks and "age" in metrics:
        return -float(metrics["age"]["mae"])  # type: ignore[index]
    return 0.0


def _run_meta(device: torch.device, seconds: float, epochs: int) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "seconds": round(seconds, 1),
        "epochs_run": epochs,
    }


def _build_predictions(
    pred: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    split_name: str,
    rhythm_names: Sequence[str],
) -> pd.DataFrame:
    rhythm_columns = [f"rhythm_prob_{name}" for name in rhythm_names]
    frame = pd.DataFrame(
        {
            "ecg_id": manifest["ecg_id"].to_numpy(),
            "patient_id": manifest["patient_id"].to_numpy(),
            "strat_fold": manifest["strat_fold"].to_numpy(),
            "split": split_name,
            "rhythm_true": pred["rhythm_true"],
            "rhythm_pred": pred["rhythm_score"].argmax(axis=1),
            "age_true": pred["age_true"],
            "age_pred": pred["age_pred"],
            "has_rhythm": pred["mask"][:, 1],
            "has_age": pred["mask"][:, 2],
        }
    )
    for i, name in enumerate(SUPERCLASSES):
        frame[f"diag_true_{name}"] = pred["diag_true"][:, i].astype(int)
        frame[f"diag_prob_{name}"] = pred["diag_score"][:, i]
    frame = pd.concat([frame, pd.DataFrame(pred["rhythm_score"], columns=rhythm_columns)], axis=1)
    return frame


def run_training(cfg: Config, run_dir: Path) -> dict[str, Any]:
    """Train one configuration on one fold run and write all artefacts into ``run_dir``."""
    log = get_logger()
    started = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    manifest, signals, meta = load_processed(cfg.data)
    split = make_split(manifest, cfg.split)
    log.info(
        "fold run %d: train=%d val=%d test=%d (test fold %d)",
        cfg.split.run,
        len(split.train),
        len(split.val),
        len(split.test),
        cfg.split.test_folds[cfg.split.run],
    )
    rng = np.random.default_rng(cfg.train.seed)
    train_idx = split.train
    val_idx = split.val
    if cfg.train.limit_train:
        train_idx = rng.choice(train_idx, size=min(cfg.train.limit_train, len(train_idx)), replace=False)
    if cfg.train.limit_val:
        val_idx = rng.choice(val_idx, size=min(cfg.train.limit_val, len(val_idx)), replace=False)

    train_ds = EcgDataset(signals, manifest, np.sort(train_idx))
    val_ds = EcgDataset(signals, manifest, np.sort(val_idx))
    test_ds = EcgDataset(signals, manifest, split.test)
    train_loader = make_loader(train_ds, cfg.train.batch_size, True, cfg.train.num_workers, cfg.train.seed)
    val_loader = make_loader(val_ds, cfg.train.batch_size * 2, False, cfg.train.num_workers, cfg.train.seed)
    test_loader = make_loader(test_ds, cfg.train.batch_size * 2, False, cfg.train.num_workers, cfg.train.seed)

    n_rhythm = len(meta["rhythm_classes"])
    model = build_model(cfg.model, cfg.data, len(SUPERCLASSES), n_rhythm, cfg.loss.tasks).to(device)
    log.info("parameters: %s", model.count_parameters())
    pos_weight = pos_weight_tensor(manifest, np.sort(train_idx), cfg.loss.diag_pos_weight)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        log.info("diag pos_weight: %s", [round(float(v), 2) for v in pos_weight])

    uncertainty = (
        UncertaintyWeighting(cfg.loss.tasks).to(device)
        if cfg.loss.strategy == "uncertainty"
        else None
    )
    params: list[nn.Parameter] = list(model.parameters())
    if uncertainty is not None:
        params += list(uncertainty.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    # PCGrad randomises the order of projections; a CPU generator keeps it reproducible
    generator = torch.Generator()
    generator.manual_seed(cfg.train.seed)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    patience = 0
    monitor = "macro_auc" if "diag" in cfg.loss.tasks else "macro_f1"
    for epoch in range(cfg.train.epochs):
        log.info("epoch %d/%d", epoch + 1, cfg.train.epochs)
        train_stats = train_epoch(
            model, train_loader, optimizer, cfg, device, pos_weight, uncertainty, generator
        )
        scheduler.step()
        val_pred = predict(model, val_loader, device)
        val_metrics = task_metrics(val_pred, meta)
        score = _monitor_score(val_metrics, cfg.loss.tasks)
        history.append(
            {
                "epoch": epoch + 1,
                "train": train_stats,
                "val": val_metrics,
                "val_score": score,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        log.info(
            "epoch %d: %s score=%.4f",
            epoch + 1,
            " ".join(f"{k}={v:.4f}" for k, v in train_stats["losses"].items()),
            score,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train.patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model": model.state_dict(),
            "config": to_dict(cfg),
            "meta": meta,
            "params": model.count_parameters(),
            "best_epoch": best_epoch,
        },
        run_dir / "model.pt",
    )
    val_pred = predict(model, val_loader, device)
    test_pred = predict(model, test_loader, device)
    rhythm_names = [str(name) for name in meta["rhythm_classes"]]
    metrics = {
        "val": task_metrics(val_pred, meta),
        "test": task_metrics(test_pred, meta),
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "monitor": monitor,
        "params": model.count_parameters(),
        "split": {
            "test_fold": cfg.split.test_folds[cfg.split.run],
            "val_fold": cfg.split.val_folds[cfg.split.run],
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(split.test)),
        },
        "tasks": list(cfg.loss.tasks),
        "strategy": cfg.loss.strategy,
        "weights": cfg.loss.weights,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    test_frame = _build_predictions(
        test_pred, manifest.iloc[split.test].reset_index(drop=True), "test", rhythm_names
    )
    val_frame = _build_predictions(
        val_pred, manifest.iloc[np.sort(val_idx)].reset_index(drop=True), "val", rhythm_names
    )
    pd.concat([test_frame, val_frame]).to_csv(run_dir / "predictions.csv", index=False)
    save_config(cfg, run_dir / "config.yaml")
    meta_info = _run_meta(device, time.time() - started, len(history))
    (run_dir / "run_meta.json").write_text(json.dumps(meta_info, indent=2), encoding="utf-8")
    log.info("run finished in %.1f s, test metrics written to %s", meta_info["seconds"], run_dir)
    return metrics
