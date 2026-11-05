from __future__ import annotations

from pathlib import Path

from fusion_slu.config import AppConfig
from fusion_slu.utils import ensure_dir


def run_name(modality: str, seed: int) -> str:
    return f"{modality}_seed{seed}"


def run_directory(config: AppConfig, modality: str, seed: int) -> Path:
    return config.results_dir / run_name(modality, seed)


def model_path(config: AppConfig, modality: str, seed: int) -> Path:
    return run_directory(config, modality, seed) / "model.pt"


def history_path(config: AppConfig, modality: str, seed: int) -> Path:
    return run_directory(config, modality, seed) / "history.json"


def metrics_path(config: AppConfig, modality: str, seed: int) -> Path:
    return run_directory(config, modality, seed) / "metrics.json"


def predictions_path(config: AppConfig, modality: str, seed: int) -> Path:
    return run_directory(config, modality, seed) / "predictions.csv"


def attention_path(config: AppConfig, modality: str, seed: int) -> Path:
    return run_directory(config, modality, seed) / "attention.json"


def prepare_run_directory(config: AppConfig, modality: str, seed: int) -> Path:
    directory = run_directory(config, modality, seed)
    ensure_dir(directory)
    return directory
