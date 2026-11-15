"""Typed configuration.

One source of truth: a YAML file validated into frozen dataclasses, plus explicit
CLI overrides (CLI > file > defaults). Hyperparameters never come from env vars.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

TASKS: tuple[str, ...] = ("diag", "rhythm", "age")
STRATEGIES: tuple[str, ...] = ("equal", "uncertainty", "pcgrad")


class ConfigError(ValueError):
    """Raised when a config file does not match the expected schema."""


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("data")
    url: str = "https://physionet.org/files/ptb-xl/1.0.3"
    mirror_url: str = "https://huggingface.co/datasets/longisland3/ptb-xl/resolve/main"
    version: str = "1.0.3"
    prefer_mirror: bool = False
    hz: int = 100
    signal_len: int = 1000
    n_leads: int = 12
    rhythm_topk: int = 8
    min_age: int = 0
    max_age: int = 100

    def __post_init__(self) -> None:
        if self.hz == 100 and self.signal_len != 1000:
            raise ConfigError("100 Hz records are 10 s = 1000 samples")
        if self.hz == 500 and self.signal_len != 5000:
            raise ConfigError("500 Hz records are 10 s = 5000 samples")


@dataclass(frozen=True)
class SplitConfig:
    # five rotating runs, patient-level folds provided by PTB-XL (strat_fold 1..10)
    test_folds: tuple[int, ...] = (10, 8, 6, 4, 2)
    val_folds: tuple[int, ...] = (9, 7, 5, 3, 1)
    run: int = 0

    def __post_init__(self) -> None:
        if len(self.test_folds) != len(self.val_folds):
            raise ConfigError("test_folds and val_folds must have the same length")
        if set(self.test_folds) & set(self.val_folds):
            raise ConfigError("test and validation folds overlap")
        if not 0 <= self.run < len(self.test_folds):
            raise ConfigError(f"split.run must be in [0, {len(self.test_folds) - 1}]")


@dataclass(frozen=True)
class ModelConfig:
    channels: tuple[int, ...] = (64, 128, 256)
    kernels: tuple[int, ...] = (7, 5, 3)
    dropout: float = 0.2
    hidden: int = 256
    head_hidden: int = 128
    pool: str = "attention"

    def __post_init__(self) -> None:
        if len(self.channels) != len(self.kernels):
            raise ConfigError("channels and kernels must have the same length")
        if self.pool not in ("attention", "mean"):
            raise ConfigError("pool must be 'attention' or 'mean'")


@dataclass(frozen=True)
class LossConfig:
    strategy: str = "equal"
    tasks: tuple[str, ...] = TASKS
    weights: dict[str, float] = field(
        default_factory=lambda: {"diag": 1.0, "rhythm": 1.0, "age": 1.0}
    )
    diag_pos_weight: bool = True

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ConfigError(f"loss.strategy must be one of {STRATEGIES}")
        unknown = set(self.tasks) - set(TASKS)
        if unknown:
            raise ConfigError(f"unknown tasks: {sorted(unknown)}")
        if len(self.tasks) < 2 and self.strategy != "equal":
            raise ConfigError("multi-task losses require at least two tasks")
        missing = set(self.tasks) - set(self.weights)
        if missing:
            raise ConfigError(f"missing weights for tasks: {sorted(missing)}")


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0
    patience: int = 5
    grad_clip: float = 1.0
    log_every: int = 50
    conflict_every: int = 100
    limit_train: int = 0  # >0 keeps only N training records, used by smoke runs
    limit_val: int = 0


@dataclass(frozen=True)
class OutputConfig:
    out_dir: Path = Path("outputs")
    report_dir: Path = Path("report")
    run_name: str = "run"


@dataclass(frozen=True)
class ExperimentsConfig:
    """Which conditions the grid runs and on how many fold runs."""

    folds: int = 5
    ablation_folds: int = 2
    main: tuple[str, ...] = (
        "single_diag",
        "single_rhythm",
        "single_age",
        "mtl_equal",
        "mtl_uncertainty",
        "mtl_pcgrad",
    )
    ablations: tuple[str, ...] = ("loto_no_rhythm", "loto_no_age", "mtl_age_weight_0.1")


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    experiments: ExperimentsConfig = field(default_factory=ExperimentsConfig)


_SECTIONS: dict[str, type[Any]] = {
    "data": DataConfig,
    "split": SplitConfig,
    "model": ModelConfig,
    "loss": LossConfig,
    "train": TrainConfig,
    "output": OutputConfig,
    "experiments": ExperimentsConfig,
}


def _check_unknown(raw: dict[str, Any], known: Any, where: str) -> None:
    allowed = {f.name for f in dataclasses.fields(known)}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"unknown keys in {where}: {sorted(unknown)}")


def _coerce(value: Any, default: Any) -> Any:
    """Convert CLI strings into the type of the field default (``--set train.epochs=3``)."""
    if default is dataclasses.MISSING:
        return value
    if isinstance(value, str) and isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, str) and isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(value, str) and isinstance(default, float):
        return float(value)
    if isinstance(default, tuple):
        items = [item.strip() for item in value.split(",") if item.strip()] if isinstance(value, str) else value
        if isinstance(items, (list, tuple)):
            if default and all(isinstance(item, int) for item in default):
                return tuple(int(item) for item in items)
            return tuple(str(item) for item in items)
    return value


def _build_section(cls: Any, raw: dict[str, Any], where: str) -> Any:
    _check_unknown(raw, cls, where)
    kwargs: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.name not in raw:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:  # type: ignore[misc]
                raise ConfigError(f"missing required key {where}.{f.name}")
            continue
        value = _coerce(raw[f.name], f.default)
        if isinstance(f.default, Path):
            value = Path(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    section, _, key = dotted.partition(".")
    if not key:
        raise ConfigError(f"override must look like section.key, got {dotted!r}")
    cfg.setdefault(section, {})[key] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Read a YAML config and validate it; overrides are dotted keys from the CLI."""
    with Path(path).open(encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    _check_unknown(raw, Config, "config")
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        _set_path(raw, dotted, value)
    sections = {
        name: _build_section(cls, raw.get(name) or {}, name) for name, cls in _SECTIONS.items()
    }
    return Config(**sections)


def to_dict(obj: Any) -> Any:
    """Recursively convert a config into plain YAML-serialisable types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False, allow_unicode=True)
