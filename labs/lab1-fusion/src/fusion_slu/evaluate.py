from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from fusion_slu.config import AppConfig, Condition, HEAD_NAMES, parse_condition
from fusion_slu.data.dataset import FeatureDataset, make_loader
from fusion_slu.data.prepare import PreparedData
from fusion_slu.engine import Prediction, collect_attention, run_inference
from fusion_slu.metrics import expected_calibration_error, overall_metrics
from fusion_slu.models.fusion import FusionSLU, build_model
from fusion_slu.paths import (
    attention_path,
    metrics_path,
    model_path,
    predictions_path,
    prepare_run_directory,
)
from fusion_slu.utils import (
    configure_torch,
    format_seconds,
    get_logger,
    resolve_device,
    save_json,
    set_seed,
)
from fusion_slu.utils import timer


def load_model(
    config: AppConfig,
    prepared: PreparedData,
    modality: str,
    seed: int,
    device: torch.device,
) -> FusionSLU:
    run_config = config.for_run(modality, seed)
    checkpoint = model_path(run_config, modality, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}; run training first")
    model = build_model(
        run_config.model,
        config.audio,
        vocab_size=len(prepared.vocab),
        pad_id=prepared.vocab["<pad>"],
        dims=prepared.dims,
        head_weights=config.training.head_weights,
    )
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def applicable_conditions(config: AppConfig, modality: str) -> list[Condition]:
    conditions: list[Condition] = []
    for name in config.eval.conditions:
        condition = parse_condition(name)
        if modality == "audio" and condition.kind.startswith("text_"):
            continue
        if modality == "text" and condition.kind.startswith("audio_"):
            continue
        conditions.append(condition)
    if not any(condition.kind == "clean" for condition in conditions):
        conditions.insert(0, parse_condition("clean"))
    return conditions


def calibration(prediction: Prediction, dims: Sequence[int]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for head_index, head in enumerate(HEAD_NAMES):
        result[head] = expected_calibration_error(
            prediction.confidence[:, head_index],
            prediction.pred[:, head_index],
            prediction.true[:, head_index],
            n_bins=15,
        )
    return result


def _prediction_rows(
    prediction: Prediction,
    prepared: PreparedData,
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]
    rows: list[dict[str, Any]] = []
    for row_index in range(prediction.pred.shape[0]):
        index = int(prediction.indices[row_index])
        item = manifest[index]
        true = prediction.true[row_index]
        pred = prediction.pred[row_index]
        record: dict[str, Any] = {
            "index": index,
            "path": item.get("path", ""),
            "speaker_id": item.get("speaker_id", ""),
            "transcription": item.get("transcription", ""),
            "duration": item.get("duration", 0.0),
            "exact_match": bool(np.array_equal(true, pred)),
        }
        for head_index, head in enumerate(HEAD_NAMES):
            record[f"true_{head}"] = names[head_index][int(true[head_index])]
            record[f"pred_{head}"] = names[head_index][int(pred[head_index])]
            record[f"confidence_{head}"] = round(float(prediction.confidence[row_index, head_index]), 5)
        rows.append(record)
    return rows


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _attention_examples(
    prediction: Prediction,
    prepared: PreparedData,
    dataset: FeatureDataset,
    model: FusionSLU,
    device: torch.device,
    *,
    max_examples: int = 6,
) -> list[dict[str, Any]]:
    correct = (prediction.pred == prediction.true).all(axis=1)
    confidence = prediction.confidence.mean(axis=1)
    wrong_indices = list(np.where(~correct)[0])
    right_indices = list(np.where(correct)[0])
    wrong_indices.sort(key=lambda index: -float(confidence[index]))
    right_indices.sort(key=lambda index: -float(confidence[index]))
    chosen: list[int] = []
    for pool in (wrong_indices, right_indices):
        for position in pool:
            if int(prediction.indices[position]) not in chosen:
                chosen.append(int(prediction.indices[position]))
            if len(chosen) >= max_examples:
                break
        if len(chosen) >= max_examples:
            break
    raw = collect_attention(model, dataset, chosen, device, pad_id=prepared.vocab["<pad>"])
    names = [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]
    inverse = {index: token for token, index in prepared.vocab.items()}
    examples: list[dict[str, Any]] = []
    for item in raw:
        index = int(item["index"])
        manifest_item = dataset.store.manifest[index]
        token_ids = [int(token) for token in item["tokens"] if int(token) != prepared.vocab["<pad>"]]
        token_strings = [inverse.get(token, "<unk>") for token in token_ids]
        token_strings = ["⟨sos⟩" if token == "<sos>" else token for token in token_strings]
        examples.append(
            {
                "index": index,
                "transcription": manifest_item.get("transcription", ""),
                "token_ids": token_ids,
                "tokens": token_strings,
                "attention": item["attention"],
                "audio_frames": item["audio_frames"],
                "speaker_id": manifest_item.get("speaker_id", ""),
                "true": [names[head][int(item["true"][head])] for head in range(3)],
                "pred": [names[head][int(item["pred"][head])] for head in range(3)],
                "duration": manifest_item.get("duration", 0.0),
            }
        )
    return examples


def evaluate_one(
    config: AppConfig,
    prepared: PreparedData,
    modality: str,
    seed: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    run_config = config.for_run(modality, seed)
    directory = prepare_run_directory(run_config, modality, seed)
    output_path = metrics_path(run_config, modality, seed)
    if not force and output_path.is_file():
        import json

        return dict(json.loads(output_path.read_text(encoding="utf-8")))
    set_seed(seed)
    logger = get_logger(f"evaluate:{modality}:{seed}", directory / "eval.log")
    device = resolve_device(config.device)
    configure_torch(config.training.num_threads)
    model = load_model(config, prepared, modality, seed, device)
    pad_id = prepared.vocab["<pad>"]
    test_store = prepared.split("test")
    clean_dataset = FeatureDataset(
        test_store,
        prepared.vocab,
        config.data,
        config.audio,
        run_config.model,
        train=False,
        seed=seed,
    )
    conditions = applicable_conditions(config, modality)
    logger.info("evaluating %d conditions on %d utterances", len(conditions), len(test_store))
    names = [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]
    results: dict[str, Any] = {
        "modality": modality,
        "seed": seed,
        "n_test": len(test_store),
        "conditions": {},
    }
    clean_prediction: Prediction | None = None
    started = timer()
    with started as clock:
        for condition_index, condition in enumerate(conditions):
            if condition.kind == "audio_noise":
                dataset = FeatureDataset(
                    test_store,
                    prepared.vocab,
                    config.data,
                    config.audio,
                    run_config.model,
                    train=False,
                    seed=seed,
                    audio_condition=condition,
                )
            elif condition.kind == "audio_missing":
                dataset = FeatureDataset(
                    test_store,
                    prepared.vocab,
                    config.data,
                    config.audio,
                    run_config.model,
                    train=False,
                    seed=seed,
                    audio_condition=condition,
                )
            else:
                dataset = clean_dataset
            loader = make_loader(
                dataset,
                batch_size=config.training.eval_batch_size,
                shuffle=False,
                pad_id=pad_id,
                num_workers=config.data.num_workers,
                seed=seed,
            )
            prediction = run_inference(
                model,
                loader,
                device,
                condition=condition,
                vocab=prepared.vocab,
                seed=seed * 1000 + condition_index,
            )
            metrics = overall_metrics(prediction.pred, prediction.true, prepared.dims, names)
            payload: dict[str, Any] = {"loss": prediction.loss, "metrics": metrics}
            if condition.kind == "clean":
                clean_prediction = prediction
                payload["calibration"] = calibration(prediction, prepared.dims)
            results["conditions"][condition.label] = payload
            logger.info(
                "%s: loss %.4f | exact %.4f | action %.4f object %.4f location %.4f",
                condition.label,
                prediction.loss,
                metrics["exact_match"],
                metrics["heads"]["action"]["accuracy"],
                metrics["heads"]["object"]["accuracy"],
                metrics["heads"]["location"]["accuracy"],
            )
    elapsed = clock["elapsed"]
    if clean_prediction is None:
        raise RuntimeError("clean condition was not evaluated")
    rows = _prediction_rows(clean_prediction, prepared, test_store.manifest)
    _write_predictions(predictions_path(run_config, modality, seed), rows)
    results["predictions"] = {
        "correct": int(sum(1 for row in rows if row["exact_match"])),
        "total": len(rows),
        "csv": predictions_path(run_config, modality, seed).name,
    }
    results["elapsed_seconds"] = elapsed
    save_json(output_path, results)
    logger.info("metrics saved to %s in %s", output_path, format_seconds(elapsed))

    try:
        from fusion_slu.viz.plots import (
            plot_attention_examples,
            plot_calibration,
            plot_confusion_matrices,
        )

        clean_metrics = results["conditions"]["clean"]["metrics"]
        plot_confusion_matrices(
            clean_metrics,
            names,
            directory / "confusion.png",
            title=f"{modality}, seed {seed}",
        )
        plot_calibration(
            results["conditions"]["clean"]["calibration"],
            directory / "calibration.png",
            title=f"{modality}, seed {seed}",
        )
        if modality == "fusion":
            examples = _attention_examples(
                clean_prediction,
                prepared,
                clean_dataset,
                model,
                device,
                max_examples=6,
            )
            save_json(attention_path(run_config, modality, seed), examples)
            plot_attention_examples(examples, directory / "attention.png")
    except Exception as error:
        logger.warning("could not draw evaluation plots: %s", error)
    return results


def evaluate_all(
    config: AppConfig,
    prepared: PreparedData,
    *,
    force: bool = False,
    modalities: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    selected_modalities = tuple(modalities or config.training.modalities)
    selected_seeds = tuple(seeds or config.training.seeds)
    results: list[dict[str, Any]] = []
    for modality in selected_modalities:
        for seed in selected_seeds:
            results.append(evaluate_one(config, prepared, modality, seed, force=force))
    return results
