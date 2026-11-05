from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from fusion_slu.config import AppConfig, HEAD_NAMES, MODALITIES
from fusion_slu.data.prepare import PreparedData
from fusion_slu.data.text import tokenize
from fusion_slu.metrics import overall_metrics, top_confusions
from fusion_slu.paths import attention_path, history_path, predictions_path
from fusion_slu.utils import load_json, mean_std, save_json
from fusion_slu.viz.plots import (
    plot_attention_examples,
    plot_calibration,
    plot_confusion_matrices,
    plot_dataset_overview,
    plot_error_panels,
    plot_intent_distribution,
    plot_model_comparison,
    plot_per_class_f1,
    plot_robustness,
    plot_spectrogram_grid,
    plot_training_curves,
)

TEXT_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("clean", "clean"),
    ("text_drop_10", "drop 10%"),
    ("text_drop_25", "drop 25%"),
    ("text_drop_50", "drop 50%"),
    ("text_missing", "missing"),
    ("text_shuffle", "shuffled"),
)
AUDIO_CONDITIONS: tuple[tuple[str, str], ...] = (
    ("clean", "clean"),
    ("audio_noise_5db", "5 dB SNR"),
    ("audio_noise_0db", "0 dB SNR"),
    ("audio_missing", "missing"),
)


@dataclass
class AnalysisBundle:
    summary: dict[str, Any]
    robustness: dict[str, Any]
    analysis: dict[str, Any]
    figures: dict[str, Path]
    best: dict[str, Any]


def _discover_runs(config: AppConfig) -> dict[str, list[tuple[int, dict[str, Any]]]]:
    runs: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for path in sorted(config.results_dir.glob("*_seed*/metrics.json")):
        name = path.parent.name
        if "_seed" not in name:
            continue
        modality, seed_text = name.rsplit("_seed", 1)
        if modality not in MODALITIES:
            continue
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        runs[modality].append((seed, dict(load_json(path))))
    for modality in runs:
        runs[modality].sort(key=lambda item: item[0])
    return dict(runs)


def _clean_metrics(run: Mapping[str, Any]) -> Mapping[str, Any]:
    conditions = run.get("conditions", {})
    clean = conditions.get("clean")
    if not isinstance(clean, Mapping):
        raise ValueError("run has no clean condition")
    metrics = clean.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("run has no clean metrics")
    return metrics


def _flat_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "exact_match": float(metrics["exact_match"]),
        "mean_head_accuracy": float(metrics["mean_head_accuracy"]),
        "mean_weighted_f1": float(metrics["mean_weighted_f1"]),
        "joint_intent_accuracy": float(metrics["joint_intent_accuracy"]),
    }
    for head in HEAD_NAMES:
        result[f"{head}_accuracy"] = float(metrics["heads"][head]["accuracy"])
        result[f"{head}_f1"] = float(metrics["heads"][head]["weighted_f1"])
    return result


def _summary_from_runs(runs: Mapping[str, list[tuple[int, dict[str, Any]]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for modality, items in runs.items():
        if not items:
            continue
        flattened = [_flat_metrics(_clean_metrics(run)) for _, run in items]
        entry: dict[str, Any] = {
            "n_seeds": len(items),
            "seeds": [seed for seed, _ in items],
        }
        for key in flattened[0]:
            mean, std = mean_std([row[key] for row in flattened])
            entry[key] = {"mean": mean, "std": std}
        summary[modality] = entry
    return summary


def _majority_summary(config: AppConfig, prepared: PreparedData) -> dict[str, Any]:
    test_store = prepared.split("test")
    true = np.asarray(test_store.labels, dtype=np.int64)
    prediction = np.zeros_like(true)
    for head_index, head in enumerate(HEAD_NAMES):
        counts = prepared.stats["labels"][head]["counts_train"]
        best_label = max(counts.items(), key=lambda item: item[1])[0]
        prediction[:, head_index] = prepared.label_vocabs[head][best_label]
    names = [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]
    metrics = overall_metrics(true, prediction, prepared.dims, names)
    flattened = _flat_metrics(metrics)
    entry: dict[str, Any] = {"n_seeds": 1, "seeds": [0]}
    for key, value in flattened.items():
        entry[key] = {"mean": value, "std": 0.0}
    return entry


def _plot_pairs(summary: Mapping[str, Any]) -> dict[str, dict[str, tuple[float, float]]]:
    result: dict[str, dict[str, tuple[float, float]]] = {}
    order = ("majority", "audio", "text", "late", "fusion")
    for model in order:
        entry = summary.get(model)
        if not isinstance(entry, Mapping):
            continue
        result[model] = {
            key: (float(value["mean"]), float(value["std"]))
            for key, value in entry.items()
            if isinstance(value, Mapping) and "mean" in value
        }
    return result


def _condition_stats(
    runs: Sequence[tuple[int, Mapping[str, Any]]],
    condition: str,
    field: str = "exact_match",
) -> tuple[float, float, int]:
    values: list[float] = []
    for _, run in runs:
        conditions = run.get("conditions", {})
        payload = conditions.get(condition)
        if not isinstance(payload, Mapping):
            continue
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        if field == "exact_match":
            values.append(float(metrics["exact_match"]))
        elif field.endswith("_f1"):
            head = field.removesuffix("_f1")
            values.append(float(metrics["heads"][head]["weighted_f1"]))
        else:
            values.append(float(metrics[field]))
    if not values:
        return 0.0, 0.0, 0
    mean, std = mean_std(values)
    return mean, std, len(values)


def _robustness(
    runs: Mapping[str, list[tuple[int, dict[str, Any]]]],
) -> dict[str, Any]:
    text_models = [model for model in ("text", "fusion", "late") if model in runs]
    audio_models = [model for model in ("audio", "fusion", "late") if model in runs]
    text_series: dict[str, list[tuple[str, float, float]]] = {}
    audio_series: dict[str, list[tuple[str, float, float]]] = {}
    for model in text_models:
        points: list[tuple[str, float, float]] = []
        for condition, label in TEXT_CONDITIONS:
            mean, std, count = _condition_stats(runs[model], condition)
            if count:
                points.append((label, mean, std))
        text_series[model] = points
    for model in audio_models:
        points = []
        for condition, label in AUDIO_CONDITIONS:
            mean, std, count = _condition_stats(runs[model], condition)
            if count:
                points.append((label, mean, std))
        audio_series[model] = points
    return {"text": text_series, "audio": audio_series}


def _best_run(
    runs: Mapping[str, list[tuple[int, dict[str, Any]]]],
) -> tuple[str, int, dict[str, Any]] | None:
    candidates: list[tuple[float, str, int, dict[str, Any]]] = []
    priority = {"fusion": 0, "late": 1, "text": 2, "audio": 3}
    for modality, items in runs.items():
        for seed, run in items:
            exact = float(_clean_metrics(run)["exact_match"])
            candidates.append((exact, modality, seed, run))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], priority.get(item[1], 9), item[2]))
    exact, modality, seed, run = candidates[0]
    return modality, seed, run


def _load_predictions(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _duration_buckets(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    edges = [(0.0, 1.5, "≤ 1.5 s"), (1.5, 2.2, "1.5–2.2 s"), (2.2, 3.2, "2.2–3.2 s"), (3.2, 99.0, "> 3.2 s")]
    result: list[dict[str, Any]] = []
    for low, high, label in edges:
        subset = [row for row in rows if low <= float(row["duration"]) < high]
        if not subset:
            continue
        errors = sum(1 for row in subset if row["exact_match"] != "True")
        result.append(
            {
                "label": label,
                "count": len(subset),
                "error_rate": errors / len(subset),
            }
        )
    return result


def _speaker_errors(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    totals: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for row in rows:
        speaker = row["speaker_id"]
        totals[speaker] += 1
        if row["exact_match"] != "True":
            errors[speaker] += 1
    result: list[dict[str, Any]] = [
        {
            "speaker": speaker,
            "count": totals[speaker],
            "errors": errors[speaker],
            "error_rate": errors[speaker] / totals[speaker],
        }
        for speaker in totals
        if totals[speaker] >= 10
    ]
    result.sort(key=lambda item: (-float(item["error_rate"]), -int(item["errors"])))
    return result


def _text_length_buckets(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        length = len(tokenize(row["transcription"]))
        if length <= 3:
            buckets["≤ 3 words"].append(row)
        elif length == 4:
            buckets["4 words"].append(row)
        elif length == 5:
            buckets["5 words"].append(row)
        else:
            buckets["> 5 words"].append(row)
    result: list[dict[str, Any]] = []
    for label in ("≤ 3 words", "4 words", "5 words", "> 5 words"):
        subset = buckets.get(label, [])
        if not subset:
            continue
        errors = sum(1 for row in subset if row["exact_match"] != "True")
        result.append({"label": label, "count": len(subset), "error_rate": errors / len(subset)})
    return result


def _error_analysis(
    config: AppConfig,
    prepared: PreparedData,
    modality: str,
    seed: int,
    run: Mapping[str, Any],
) -> dict[str, Any]:
    rows = _load_predictions(predictions_path(config, modality, seed))
    names = [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]
    clean_metrics = _clean_metrics(run)
    confusion_pairs: dict[str, list[dict[str, Any]]] = {}
    for head_index, head in enumerate(HEAD_NAMES):
        confusion = np.asarray(clean_metrics["heads"][head]["confusion"], dtype=np.int64)
        confusion_pairs[head] = top_confusions(confusion, names[head_index], config.report.top_confusions)
    correct = [row for row in rows if row["exact_match"] == "True"]
    incorrect = [row for row in rows if row["exact_match"] != "True"]
    confidence_key = f"confidence_{HEAD_NAMES[0]}"
    correct_confidence = [float(row[confidence_key]) for row in correct]
    incorrect_confidence = [float(row[confidence_key]) for row in incorrect]
    hard_examples = sorted(
        incorrect,
        key=lambda row: -float(np.mean([float(row[f"confidence_{head}"]) for head in HEAD_NAMES])),
    )[: config.report.max_error_examples]
    return {
        "best_modality": modality,
        "best_seed": seed,
        "errors": len(incorrect),
        "correct": len(correct),
        "error_rate": (len(incorrect) / len(rows)) if rows else 0.0,
        "confusion_pairs": confusion_pairs,
        "duration_buckets": _duration_buckets(rows),
        "text_length_buckets": _text_length_buckets(rows),
        "speakers": _speaker_errors(rows),
        "confidence": {
            "correct": correct_confidence,
            "incorrect": incorrect_confidence,
            "mean_correct": float(np.mean(correct_confidence)) if correct_confidence else 0.0,
            "mean_incorrect": float(np.mean(incorrect_confidence)) if incorrect_confidence else 0.0,
        },
        "hard_examples": hard_examples,
    }


def _intent_counts(manifest: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter(
        f"{row['action']} / {row['object']} / {row['location']}" for row in manifest
    )
    return dict(counter)


def _spectrogram_examples(
    prepared: PreparedData,
) -> list[tuple[np.ndarray, str]]:
    store = prepared.split("train")
    durations = [float(row.get("duration", 0.0)) for row in store.manifest]
    if not durations:
        return []
    chosen: list[int] = []
    for quantile in (0.1, 0.5, 0.9):
        target = float(np.quantile(durations, quantile))
        index = min(range(len(durations)), key=lambda i: abs(durations[i] - target))
        if index not in chosen:
            chosen.append(index)
    examples: list[tuple[np.ndarray, str]] = []
    for index in chosen:
        row = store.manifest[index]
        examples.append(
            (
                store.feature(index),
                f"{row['transcription']} ({row['duration']:.2f} s)",
            )
        )
    return examples


def build_analysis(config: AppConfig, prepared: PreparedData) -> AnalysisBundle:
    runs = _discover_runs(config)
    if not runs:
        raise RuntimeError(f"no completed runs found in {config.results_dir}; run training and evaluation first")
    summary = _summary_from_runs(runs)
    summary["majority"] = _majority_summary(config, prepared)
    robustness = _robustness(runs)
    best = _best_run(runs)
    if best is None:
        raise RuntimeError("no run could be selected for error analysis")
    modality, seed, run = best
    analysis = _error_analysis(config, prepared, modality, seed, run)
    analysis["dataset"] = {
        "intents": _intent_counts(prepared.split("train").manifest),
        "splits": prepared.stats.get("splits", {}),
        "text": prepared.stats.get("text", {}),
        "vocab_size": len(prepared.vocab),
    }

    figures_dir = config.report.output.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}
    train_store = prepared.split("train")
    durations = [float(row.get("duration", 0.0)) for row in train_store.manifest]
    label_counts = {
        head: {
            str(name): int(value)
            for name, value in prepared.stats["labels"][head]["counts_train"].items()
        }
        for head in HEAD_NAMES
    }
    figures["dataset"] = figures_dir / "dataset_overview.png"
    plot_dataset_overview(label_counts, durations, figures["dataset"])
    figures["intents"] = figures_dir / "intent_distribution.png"
    plot_intent_distribution(_intent_counts(train_store.manifest), figures["intents"])
    spectrograms = _spectrogram_examples(prepared)
    if spectrograms:
        figures["spectrograms"] = figures_dir / "spectrograms.png"
        plot_spectrogram_grid(spectrograms, figures["spectrograms"])

    figures["comparison"] = figures_dir / "model_comparison.png"
    plot_model_comparison(_plot_pairs(summary), figures["comparison"])
    figures["robustness"] = figures_dir / "robustness.png"
    plot_robustness(robustness["text"], robustness["audio"], figures["robustness"])

    clean_metrics = _clean_metrics(run)
    names = [list(prepared.label_vocabs[head]) for head in HEAD_NAMES]
    figures["confusion"] = figures_dir / "confusion_best.png"
    plot_confusion_matrices(
        clean_metrics,
        names,
        figures["confusion"],
        title=f"{modality}, seed {seed}",
    )
    figures["per_class"] = figures_dir / "per_class_f1.png"
    plot_per_class_f1(clean_metrics, names, figures["per_class"], title=f"{modality}, seed {seed}")
    calibration = run.get("conditions", {}).get("clean", {}).get("calibration")
    if isinstance(calibration, Mapping):
        figures["calibration"] = figures_dir / "calibration_best.png"
        plot_calibration(calibration, figures["calibration"], title=f"{modality}, seed {seed}")
    figures["errors"] = figures_dir / "error_analysis.png"
    plot_error_panels(analysis, figures["errors"])

    history_file = history_path(config, modality, seed)
    if history_file.is_file():
        history = list(load_json(history_file))
        if history:
            figures["curves"] = figures_dir / "training_curves.png"
            plot_training_curves(history, figures["curves"], title=f"{modality}, seed {seed}")

    attention_file = attention_path(config, modality, seed)
    if attention_file.is_file():
        examples = list(load_json(attention_file))
        if examples:
            figures["attention"] = figures_dir / "attention_examples.png"
            plot_attention_examples(examples, figures["attention"])
            analysis["attention"] = [
                {
                    "transcription": example["transcription"],
                    "true": example["true"],
                    "pred": example["pred"],
                    "duration": example["duration"],
                }
                for example in examples
            ]

    bundle = AnalysisBundle(
        summary=summary,
        robustness=robustness,
        analysis=analysis,
        figures=figures,
        best={"modality": modality, "seed": seed, "metrics": clean_metrics},
    )
    save_json(config.results_dir / "summary.json", summary)
    save_json(
        config.results_dir / "analysis.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best": {"modality": modality, "seed": seed},
            "analysis": analysis,
            "robustness": robustness,
        },
    )
    return bundle
