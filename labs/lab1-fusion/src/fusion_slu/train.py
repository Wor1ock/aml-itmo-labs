from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from fusion_slu.config import AppConfig, HEAD_NAMES
from fusion_slu.data.dataset import FeatureDataset, make_loader
from fusion_slu.data.prepare import PreparedData
from fusion_slu.engine import Prediction, run_inference
from fusion_slu.metrics import overall_metrics
from fusion_slu.models.fusion import FusionSLU, build_model, count_parameters
from fusion_slu.paths import history_path, model_path, prepare_run_directory
from fusion_slu.utils import (
    configure_torch,
    format_seconds,
    get_logger,
    human_int,
    resolve_device,
    save_json,
    set_seed,
)


@dataclass
class TrainResult:
    modality: str
    seed: int
    run_dir: Path
    best_epoch: int
    best_valid_loss: float
    best_valid_exact: float
    epochs_ran: int
    seconds: float
    parameters: int
    history: list[dict[str, Any]]


def class_names(prepared: PreparedData) -> list[list[str]]:
    return [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _lr_factor(epoch: int, total: int, warmup: int) -> float:
    if warmup > 0 and epoch < warmup:
        return max(0.05, (epoch + 1) / warmup)
    span = max(1, total - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / span))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _train_epoch(
    model: FusionSLU,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    clip: float,
) -> tuple[float, Prediction]:
    model.train()
    total_loss = 0.0
    total_rows = 0
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for batch in loader:
        moved = _move_batch(batch, device)
        output = model(
            moved["audio"],
            moved["audio_mask"],
            moved["text"],
            moved["labels"],
        )
        if output.loss is None:
            raise RuntimeError("model did not return a loss")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        rows = int(moved["labels"].shape[0])
        total_loss += float(output.loss.detach().item()) * rows
        total_rows += rows
        predictions.append(output.pred.detach().cpu())
        targets.append(moved["labels"].detach().cpu())
        confidences.append(
            torch.stack([prob.max(dim=-1).values for prob in output.probs], dim=1).detach().cpu()
        )
        indices.append(moved["index"].detach().cpu())
    if total_rows == 0:
        raise RuntimeError("training loader produced no batches")
    prediction = Prediction(
        loss=total_loss / total_rows,
        pred=torch.cat(predictions).numpy(),
        true=torch.cat(targets).numpy(),
        confidence=torch.cat(confidences).numpy(),
        indices=torch.cat(indices).numpy(),
    )
    return total_loss / total_rows, prediction


def _summary(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "exact_match": float(metrics["exact_match"]),
        "mean_head_accuracy": float(metrics["mean_head_accuracy"]),
        "mean_weighted_f1": float(metrics["mean_weighted_f1"]),
        "joint_intent_accuracy": float(metrics["joint_intent_accuracy"]),
        **{
            f"{head}_accuracy": float(metrics["heads"][head]["accuracy"])
            for head in HEAD_NAMES
        },
        **{
            f"{head}_f1": float(metrics["heads"][head]["weighted_f1"])
            for head in HEAD_NAMES
        },
    }


def train_one(
    config: AppConfig,
    prepared: PreparedData,
    modality: str,
    seed: int,
    *,
    force: bool = False,
) -> TrainResult:
    run_config = config.for_run(modality, seed)
    directory = prepare_run_directory(run_config, modality, seed)
    checkpoint = model_path(run_config, modality, seed)
    history_file = history_path(run_config, modality, seed)
    if not force and checkpoint.is_file() and (directory / "run.json").is_file():
        payload = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        return TrainResult(
            modality=modality,
            seed=seed,
            run_dir=directory,
            best_epoch=int(payload.get("best_epoch", 0)),
            best_valid_loss=float(payload.get("best_valid_loss", 0.0)),
            best_valid_exact=float(payload.get("best_valid_exact", 0.0)),
            epochs_ran=int(payload.get("epochs_ran", 0)),
            seconds=float(payload.get("seconds", 0.0)),
            parameters=int(payload.get("parameters", 0)),
            history=json.loads(history_file.read_text(encoding="utf-8")) if history_file.is_file() else [],
        )

    set_seed(seed)
    logger = get_logger(f"train:{modality}:{seed}", directory / "train.log")
    device = resolve_device(config.device)
    configure_torch(config.training.num_threads)
    logger.info(
        "modality=%s seed=%d device=%s",
        modality,
        seed,
        device,
    )
    dims = prepared.dims
    names = class_names(prepared)
    pad_id = prepared.vocab["<pad>"]
    train_store = prepared.split("train")
    valid_store = prepared.split("valid")
    train_dataset = FeatureDataset(
        train_store,
        prepared.vocab,
        config.data,
        config.audio,
        run_config.model,
        train=True,
        seed=seed,
    )
    valid_dataset = FeatureDataset(
        valid_store,
        prepared.vocab,
        config.data,
        config.audio,
        run_config.model,
        train=False,
        seed=seed,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        pad_id=pad_id,
        num_workers=config.data.num_workers,
        seed=seed,
    )
    valid_loader = make_loader(
        valid_dataset,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        pad_id=pad_id,
        num_workers=config.data.num_workers,
        seed=seed,
    )
    model = build_model(
        run_config.model,
        config.audio,
        vocab_size=len(prepared.vocab),
        pad_id=pad_id,
        dims=dims,
        head_weights=config.training.head_weights,
    ).to(device)
    parameters = count_parameters(model)
    logger.info("train=%s valid=%s parameters=%s", human_int(len(train_store)), human_int(len(valid_store)), human_int(parameters))
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    scheduler = LambdaLR(
        optimizer,
        lambda epoch: _lr_factor(epoch, config.training.epochs, config.training.warmup_epochs),
    )
    history: list[dict[str, Any]] = []
    best_score = (-1.0, float("-inf"))
    best_epoch = 0
    best_loss = float("inf")
    best_exact = 0.0
    epochs_without_improvement = 0
    started = time.perf_counter()
    epochs_ran = 0
    try:
        for epoch in range(config.training.epochs):
            epoch_started = time.perf_counter()
            train_dataset.set_epoch(epoch)
            train_loss, train_pred = _train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                config.training.grad_clip,
            )
            valid_pred = run_inference(model, valid_loader, device, seed=seed)
            train_metrics = overall_metrics(train_pred.pred, train_pred.true, dims, names)
            valid_metrics = overall_metrics(valid_pred.pred, valid_pred.true, dims, names)
            lr = float(optimizer.param_groups[0]["lr"])
            row = {
                "epoch": epoch + 1,
                "lr": lr,
                "train_loss": train_loss,
                "valid_loss": valid_pred.loss,
                "seconds": time.perf_counter() - epoch_started,
                "train": _summary(train_metrics),
                "valid": _summary(valid_metrics),
            }
            history.append(row)
            logger.info(
                "epoch %02d/%02d | %5.1fs | train loss %.4f | valid loss %.4f | "
                "valid exact %.4f | action %.4f object %.4f location %.4f",
                epoch + 1,
                config.training.epochs,
                row["seconds"],
                train_loss,
                valid_pred.loss,
                valid_metrics["exact_match"],
                valid_metrics["heads"]["action"]["accuracy"],
                valid_metrics["heads"]["object"]["accuracy"],
                valid_metrics["heads"]["location"]["accuracy"],
            )
            epochs_ran = epoch + 1
            score = (float(valid_metrics["exact_match"]), -float(valid_pred.loss))
            if score > best_score:
                best_score = score
                best_epoch = epoch + 1
                best_loss = float(valid_pred.loss)
                best_exact = float(valid_metrics["exact_match"])
                epochs_without_improvement = 0
                temporary = checkpoint.with_suffix(".tmp")
                torch.save(model.state_dict(), temporary)
                os.replace(temporary, checkpoint)
                logger.info("saved new best checkpoint to %s", checkpoint.name)
            else:
                epochs_without_improvement += 1
            scheduler.step()
            if epochs_without_improvement >= config.training.patience:
                logger.info("early stopping after %d epochs without improvement", epochs_without_improvement)
                break
    finally:
        seconds = time.perf_counter() - started
    save_json(history_file, history)
    payload = {
        "modality": modality,
        "seed": seed,
        "run_dir": str(directory),
        "best_epoch": best_epoch,
        "best_valid_loss": best_loss,
        "best_valid_exact": best_exact,
        "epochs_ran": epochs_ran,
        "seconds": seconds,
        "parameters": parameters,
        "device": str(device),
        "train_size": len(train_store),
        "valid_size": len(valid_store),
        "vocab_size": len(prepared.vocab),
        "class_names": names,
        "config": run_config.to_dict(),
    }
    save_json(directory / "run.json", payload)
    try:
        from fusion_slu.viz.plots import plot_training_curves

        plot_training_curves(history, directory / "curves.png", title=f"{modality}, seed {seed}")
    except Exception as error:  # plotting must never break training
        logger.warning("could not draw training curves: %s", error)
    logger.info(
        "done in %s, best epoch %d, valid exact %.4f",
        format_seconds(seconds),
        best_epoch,
        best_exact,
    )
    return TrainResult(
        modality=modality,
        seed=seed,
        run_dir=directory,
        best_epoch=best_epoch,
        best_valid_loss=best_loss,
        best_valid_exact=best_exact,
        epochs_ran=epochs_ran,
        seconds=seconds,
        parameters=parameters,
        history=history,
    )


def train_all(
    config: AppConfig,
    prepared: PreparedData,
    *,
    force: bool = False,
    modalities: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[TrainResult]:
    selected_modalities = tuple(modalities or config.training.modalities)
    selected_seeds = tuple(seeds or config.training.seeds)
    results: list[TrainResult] = []
    for modality in selected_modalities:
        for seed in selected_seeds:
            results.append(train_one(config, prepared, modality, seed, force=force))
    return results
