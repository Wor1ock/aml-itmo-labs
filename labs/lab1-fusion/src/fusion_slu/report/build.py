from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from jinja2 import Environment, FileSystemLoader, select_autoescape

from fusion_slu.analysis import AnalysisBundle
from fusion_slu.config import AppConfig, HEAD_NAMES
from fusion_slu.data.prepare import PreparedData
from fusion_slu.paths import metrics_path, run_directory
from fusion_slu.utils import load_json

MODEL_ORDER: tuple[str, ...] = ("majority", "audio", "text", "late", "fusion")
MODEL_LABELS: dict[str, str] = {
    "majority": "Majority class",
    "audio": "Audio only",
    "text": "Text only",
    "late": "Late fusion",
    "fusion": "Cross-attention fusion",
}


def _image_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _number(value: float) -> str:
    return f"{value:.3f}"


def _mean_std(payload: Mapping[str, Any]) -> tuple[float, float]:
    return float(payload["mean"]), float(payload["std"])


def _metric(entry: Mapping[str, Any], key: str) -> tuple[float, float]:
    value = entry.get(key)
    if not isinstance(value, Mapping):
        return 0.0, 0.0
    return _mean_std(value)


def _model_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in MODEL_ORDER:
        entry = summary.get(name)
        if not isinstance(entry, Mapping):
            continue
        rows.append(
            {
                "key": name,
                "name": MODEL_LABELS.get(name, name),
                "seeds": int(entry.get("n_seeds", 0)),
                "exact": _metric(entry, "exact_match"),
                "head_accuracy": _metric(entry, "mean_head_accuracy"),
                "joint": _metric(entry, "joint_intent_accuracy"),
                "f1": {head: _metric(entry, f"{head}_f1") for head in HEAD_NAMES},
                "accuracy": {head: _metric(entry, f"{head}_accuracy") for head in HEAD_NAMES},
            }
        )
    return rows


def _robustness_rows(series: Mapping[str, Sequence[tuple[str, float, float]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, points in series.items():
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model),
                "points": [
                    {"condition": condition, "mean": mean, "std": std}
                    for condition, mean, std in points
                ],
            }
        )
    return rows


def _run_metadata(config: AppConfig, summary: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for model, entry in summary.items():
        if model == "majority":
            continue
        seeds = entry.get("seeds", [])
        parameters: list[float] = []
        seconds: list[float] = []
        best_epochs: list[float] = []
        for seed in seeds:
            payload_path = run_directory(config, model, int(seed)) / "run.json"
            if not payload_path.is_file():
                continue
            payload = load_json(payload_path)
            parameters.append(float(payload.get("parameters", 0)))
            seconds.append(float(payload.get("seconds", 0.0)))
            best_epochs.append(float(payload.get("best_epoch", 0)))
        if parameters:
            result[model] = {
                "parameters": sum(parameters) / len(parameters),
                "seconds": sum(seconds) / len(seconds),
                "best_epoch": sum(best_epochs) / len(best_epochs),
            }
    return result


def _dataset_payload(config: AppConfig, prepared: PreparedData, bundle: AnalysisBundle) -> dict[str, Any]:
    train_store = prepared.split("train")
    durations = [float(row.get("duration", 0.0)) for row in train_store.manifest]
    phrases: dict[str, set[tuple[str, str, str]]] = {}
    for row in train_store.manifest:
        phrases.setdefault(str(row["transcription"]), set()).add(
            (str(row["action"]), str(row["object"]), str(row["location"]))
        )
    ambiguous = sum(1 for values in phrases.values() if len(values) > 1)
    splits = prepared.stats.get("splits", {})
    split_rows = [
        {
            "name": name,
            "utterances": int(payload.get("utterances", 0)),
            "speakers": int(payload.get("speakers", 0)),
            "frames": int(payload.get("frames", 0)),
            "hours": float(payload.get("frames", 0)) * config.audio.hop_length / config.data.sample_rate / 3600.0,
        }
        for name, payload in splits.items()
    ]
    labels = {
        head: {
            "classes": len(prepared.label_vocabs[head]),
            "counts": prepared.stats["labels"][head]["counts_train"],
        }
        for head in HEAD_NAMES
    }
    return {
        "splits": split_rows,
        "phrases": len(phrases),
        "ambiguous_phrases": ambiguous,
        "intents": len(bundle.analysis["dataset"]["intents"]),
        "vocab_size": len(prepared.vocab),
        "duration": {
            "min": min(durations) if durations else 0.0,
            "median": float(sorted(durations)[len(durations) // 2]) if durations else 0.0,
            "mean": float(sum(durations) / len(durations)) if durations else 0.0,
            "max": max(durations) if durations else 0.0,
        },
        "labels": labels,
        "total_hours": sum(row["hours"] for row in split_rows),
        "dataset_root": str(config.data.root),
    }


def _confusion_payload(bundle: AnalysisBundle) -> dict[str, list[dict[str, Any]]]:
    pairs = bundle.analysis.get("confusion_pairs", {})
    return {head: list(pairs.get(head, [])) for head in HEAD_NAMES}


def _calibration_payload(best_run: Mapping[str, Any]) -> dict[str, Any]:
    calibration = best_run.get("conditions", {}).get("clean", {}).get("calibration", {})
    result: dict[str, Any] = {}
    for head in HEAD_NAMES:
        payload = calibration.get(head, {})
        result[head] = {"ece": float(payload.get("ece", 0.0))}
    return result


def _build_conclusions(
    summary: Mapping[str, Any],
    robustness: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> dict[str, Any]:
    def exact(model: str) -> float:
        entry = summary.get(model, {})
        return _metric(entry, "exact_match")[0] if isinstance(entry, Mapping) else 0.0

    def robust(model: str, condition: str) -> float:
        for name, mean, _ in robustness.get("text", {}).get(model, []):
            if name == condition:
                return float(mean)
        return 0.0

    fusion = exact("fusion")
    text = exact("text")
    audio = exact("audio")
    late = exact("late")
    missing = robust("fusion", "missing")
    drop50 = robust("fusion", "drop 50%")
    text_drop50 = robust("text", "drop 50%")
    best_unimodal = max(text, audio)
    gap = fusion - best_unimodal
    return {
        "fusion": fusion,
        "text": text,
        "audio": audio,
        "late": late,
        "gap": gap,
        "missing": missing,
        "drop50": drop50,
        "text_drop50": text_drop50,
        "errors": int(analysis.get("errors", 0)),
        "total": int(analysis.get("correct", 0)) + int(analysis.get("errors", 0)),
        "error_rate": float(analysis.get("error_rate", 0.0)),
        "mean_conf_correct": float(analysis.get("confidence", {}).get("mean_correct", 0.0)),
        "mean_conf_incorrect": float(analysis.get("confidence", {}).get("mean_incorrect", 0.0)),
    }


def build_report(config: AppConfig, prepared: PreparedData, bundle: AnalysisBundle) -> Path:
    template_dir = Path(__file__).parent
    environment = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["pct"] = _pct
    environment.filters["num"] = _number
    template = environment.get_template("template.html.j2")

    figures = {name: _image_uri(path) for name, path in bundle.figures.items() if path.is_file()}
    model_rows = _model_rows(bundle.summary)
    metadata = _run_metadata(config, bundle.summary)
    dataset = _dataset_payload(config, prepared, bundle)
    conclusions = _build_conclusions(bundle.summary, bundle.robustness, bundle.analysis)
    best_run = bundle.best["metrics"]
    best_payload = load_json(metrics_path(config, bundle.best["modality"], bundle.best["seed"]))
    calibration = _calibration_payload(best_payload)
    per_class = {
        head: best_run["heads"][head]["per_class"]
        for head in HEAD_NAMES
    }
    context = {
        "title": config.report.title,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": config,
        "dataset": dataset,
        "figures": figures,
        "summary": bundle.summary,
        "model_rows": model_rows,
        "metadata": metadata,
        "text_rows": _robustness_rows(bundle.robustness["text"]),
        "audio_rows": _robustness_rows(bundle.robustness["audio"]),
        "analysis": bundle.analysis,
        "confusions": _confusion_payload(bundle),
        "calibration": calibration,
        "per_class": per_class,
        "best": bundle.best,
        "conclusions": conclusions,
        "model_labels": MODEL_LABELS,
        "heads": HEAD_NAMES,
        "best_payload": best_payload,
    }
    output = config.report.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.render(**context), encoding="utf-8")
    return output
