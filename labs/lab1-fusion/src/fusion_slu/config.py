from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import yaml

MODALITIES: tuple[str, ...] = ("fusion", "text", "audio", "late")
HEAD_NAMES: tuple[str, ...] = ("action", "object", "location")


class ConfigError(ValueError):
    """Raised when a configuration file is inconsistent."""


def find_project_root(start: Path) -> Path:
    """Walk upwards until a directory with pyproject.toml is found."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd().resolve()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_raw(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    parent_name = raw.pop("extends", None)
    if parent_name is None:
        return dict(raw)
    if not isinstance(parent_name, str):
        raise ConfigError(f"{path}: extends must be a file name")
    parent_path = (path.parent / parent_name).resolve()
    parent = _load_raw(parent_path)
    return _deep_merge(parent, raw)


def _set_dotted(raw: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        child = node.get(part)
        if child is None:
            child = {}
            node[part] = child
        if not isinstance(child, dict):
            raise ConfigError(f"cannot override {key}: {part} is not a mapping")
        node = child
    node[parts[-1]] = value


def _check_keys(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ConfigError(f"unknown keys in {section}: {names}")


def _mapping(value: Any, section: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{section} must be a mapping")
    return value


def _int(value: Any, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _float(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return result


def _parse_number(text: str, name: str) -> float:
    try:
        return float(text)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number") from error


def _optional_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    return _int(value, name, minimum=1)


def _str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    return tuple(str(item) for item in value)


def _int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    return tuple(_int(item, name) for item in value)


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ConfigError(f"{name} must be a non-empty list")
    return tuple(_float(item, name) for item in value)


@dataclass
class DataConfig:
    root: Path
    csv_dir: Path
    cache_dir: Path
    train_csv: str = "train_data.csv"
    valid_csv: str = "valid_data.csv"
    test_csv: str = "test_data.csv"
    cache_tag: str = "full"
    sample_rate: int = 16000
    max_frames: int = 800
    max_text_len: int = 32
    max_train: int | None = None
    max_valid: int | None = None
    max_test: int | None = None
    num_workers: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> DataConfig:
        raw = _mapping(raw, "data")
        _check_keys(
            raw,
            {
                "root",
                "csv_dir",
                "cache_dir",
                "train_csv",
                "valid_csv",
                "test_csv",
                "cache_tag",
                "sample_rate",
                "max_frames",
                "max_text_len",
                "max_train",
                "max_valid",
                "max_test",
                "num_workers",
            },
            "data",
        )
        return cls(
            root=Path(str(raw.get("root", "data/fluent_speech_commands_dataset"))),
            csv_dir=Path(str(raw.get("csv_dir", "data/fluent_speech_commands_dataset/data"))),
            cache_dir=Path(str(raw.get("cache_dir", "cache"))),
            train_csv=str(raw.get("train_csv", "train_data.csv")),
            valid_csv=str(raw.get("valid_csv", "valid_data.csv")),
            test_csv=str(raw.get("test_csv", "test_data.csv")),
            cache_tag=str(raw.get("cache_tag", "full")),
            sample_rate=_int(raw.get("sample_rate", 16000), "data.sample_rate", 8000),
            max_frames=_int(raw.get("max_frames", 800), "data.max_frames", 20),
            max_text_len=_int(raw.get("max_text_len", 32), "data.max_text_len", 2),
            max_train=_optional_int(raw.get("max_train"), "data.max_train"),
            max_valid=_optional_int(raw.get("max_valid"), "data.max_valid"),
            max_test=_optional_int(raw.get("max_test"), "data.max_test"),
            num_workers=_int(raw.get("num_workers", 0), "data.num_workers", 0),
        )


@dataclass
class AudioConfig:
    n_mels: int = 40
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    fmin: float = 20.0
    fmax: float = 8000.0
    cmvn: bool = True
    conv_strides: tuple[int, ...] = (4, 2)
    conv_kernels: tuple[int, ...] = (5, 3)
    spec_augment: bool = True
    freq_masks: int = 2
    freq_width: int = 4
    time_masks: int = 2
    time_width: int = 10

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AudioConfig:
        raw = _mapping(raw, "audio")
        _check_keys(
            raw,
            {
                "n_mels",
                "n_fft",
                "hop_length",
                "win_length",
                "fmin",
                "fmax",
                "cmvn",
                "conv_strides",
                "conv_kernels",
                "spec_augment",
                "freq_masks",
                "freq_width",
                "time_masks",
                "time_width",
            },
            "audio",
        )
        strides = _int_tuple(raw.get("conv_strides", [4, 2]), "audio.conv_strides")
        kernels = _int_tuple(raw.get("conv_kernels", [5, 3]), "audio.conv_kernels")
        if len(strides) != len(kernels):
            raise ConfigError("audio.conv_strides and audio.conv_kernels must have equal length")
        return cls(
            n_mels=_int(raw.get("n_mels", 40), "audio.n_mels", 8),
            n_fft=_int(raw.get("n_fft", 400), "audio.n_fft", 64),
            hop_length=_int(raw.get("hop_length", 160), "audio.hop_length", 16),
            win_length=_int(raw.get("win_length", 400), "audio.win_length", 16),
            fmin=_float(raw.get("fmin", 20.0), "audio.fmin", 0.0),
            fmax=_float(raw.get("fmax", 8000.0), "audio.fmax", 100.0),
            cmvn=bool(raw.get("cmvn", True)),
            conv_strides=strides,
            conv_kernels=kernels,
            spec_augment=bool(raw.get("spec_augment", True)),
            freq_masks=_int(raw.get("freq_masks", 2), "audio.freq_masks", 0),
            freq_width=_int(raw.get("freq_width", 4), "audio.freq_width", 0),
            time_masks=_int(raw.get("time_masks", 2), "audio.time_masks", 0),
            time_width=_int(raw.get("time_width", 10), "audio.time_width", 0),
        )


@dataclass
class ModelConfig:
    modality: str = "fusion"
    hid_dim: int = 128
    audio_layers: int = 2
    text_layers: int = 2
    cross_layers: int = 2
    n_heads: int = 4
    ff_dim: int = 256
    dropout: float = 0.15
    attn_dropout: float = 0.1
    text_token_dropout: float = 0.1
    text_span_dropout: float = 0.05
    max_len: int = 512

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ModelConfig:
        raw = _mapping(raw, "model")
        _check_keys(
            raw,
            {
                "modality",
                "hid_dim",
                "audio_layers",
                "text_layers",
                "cross_layers",
                "n_heads",
                "ff_dim",
                "dropout",
                "attn_dropout",
                "text_token_dropout",
                "text_span_dropout",
                "max_len",
            },
            "model",
        )
        modality = str(raw.get("modality", "fusion"))
        if modality not in MODALITIES:
            raise ConfigError(f"model.modality must be one of {MODALITIES}")
        return cls(
            modality=modality,
            hid_dim=_int(raw.get("hid_dim", 128), "model.hid_dim", 8),
            audio_layers=_int(raw.get("audio_layers", 2), "model.audio_layers", 0),
            text_layers=_int(raw.get("text_layers", 2), "model.text_layers", 0),
            cross_layers=_int(raw.get("cross_layers", 2), "model.cross_layers", 0),
            n_heads=_int(raw.get("n_heads", 4), "model.n_heads", 1),
            ff_dim=_int(raw.get("ff_dim", 256), "model.ff_dim", 8),
            dropout=_float(raw.get("dropout", 0.15), "model.dropout", 0.0),
            attn_dropout=_float(raw.get("attn_dropout", 0.1), "model.attn_dropout", 0.0),
            text_token_dropout=_float(raw.get("text_token_dropout", 0.1), "model.text_token_dropout", 0.0),
            text_span_dropout=_float(raw.get("text_span_dropout", 0.05), "model.text_span_dropout", 0.0),
            max_len=_int(raw.get("max_len", 512), "model.max_len", 8),
        )


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 128
    eval_batch_size: int = 256
    lr: float = 4e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 4
    scheduler: str = "cosine"
    warmup_epochs: int = 1
    num_threads: int = 8
    head_weights: tuple[float, ...] = (1.0, 1.0, 1.0)
    seeds: tuple[int, ...] = (0, 1, 2)
    modalities: tuple[str, ...] = MODALITIES

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TrainingConfig:
        raw = _mapping(raw, "training")
        _check_keys(
            raw,
            {
                "epochs",
                "batch_size",
                "eval_batch_size",
                "lr",
                "weight_decay",
                "grad_clip",
                "patience",
                "scheduler",
                "warmup_epochs",
                "num_threads",
                "head_weights",
                "seeds",
                "modalities",
            },
            "training",
        )
        modalities = _str_tuple(raw.get("modalities", list(MODALITIES)), "training.modalities")
        for modality in modalities:
            if modality not in MODALITIES:
                raise ConfigError(f"training.modalities contains unknown model {modality!r}")
        head_weights = _float_tuple(raw.get("head_weights", [1.0, 1.0, 1.0]), "training.head_weights")
        if len(head_weights) != 3:
            raise ConfigError("training.head_weights must contain three values")
        scheduler = str(raw.get("scheduler", "cosine"))
        if scheduler not in {"cosine", "constant"}:
            raise ConfigError("training.scheduler must be cosine or constant")
        return cls(
            epochs=_int(raw.get("epochs", 10), "training.epochs", 1),
            batch_size=_int(raw.get("batch_size", 128), "training.batch_size", 1),
            eval_batch_size=_int(raw.get("eval_batch_size", 256), "training.eval_batch_size", 1),
            lr=_float(raw.get("lr", 4e-4), "training.lr", 0.0),
            weight_decay=_float(raw.get("weight_decay", 1e-4), "training.weight_decay", 0.0),
            grad_clip=_float(raw.get("grad_clip", 1.0), "training.grad_clip", 0.0),
            patience=_int(raw.get("patience", 4), "training.patience", 1),
            scheduler=scheduler,
            warmup_epochs=_int(raw.get("warmup_epochs", 1), "training.warmup_epochs", 0),
            num_threads=_int(raw.get("num_threads", 8), "training.num_threads", 1),
            head_weights=head_weights,
            seeds=_int_tuple(raw.get("seeds", [0, 1, 2]), "training.seeds"),
            modalities=modalities,
        )


@dataclass
class EvalConfig:
    conditions: tuple[str, ...] = (
        "clean",
        "text_drop_10",
        "text_drop_25",
        "text_drop_50",
        "text_missing",
        "text_shuffle",
        "audio_noise_5db",
        "audio_noise_0db",
    )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EvalConfig:
        raw = _mapping(raw, "eval")
        _check_keys(raw, {"conditions"}, "eval")
        conditions = _str_tuple(raw.get("conditions", list(EvalConfig().conditions)), "eval.conditions")
        for condition in conditions:
            parse_condition(condition)
        return cls(conditions=conditions)


@dataclass
class ReportConfig:
    output: Path = Path("report/index.html")
    title: str = "Cross-attention fusion of text and audio for intent classification"
    language: str = "ru"
    top_confusions: int = 12
    max_error_examples: int = 20

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ReportConfig:
        raw = _mapping(raw, "report")
        _check_keys(raw, {"output", "title", "language", "top_confusions", "max_error_examples"}, "report")
        language = str(raw.get("language", "ru"))
        if language not in {"ru", "en"}:
            raise ConfigError("report.language must be ru or en")
        return cls(
            output=Path(str(raw.get("output", "report/index.html"))),
            title=str(raw.get("title", "Cross-attention fusion of text and audio")),
            language=language,
            top_confusions=_int(raw.get("top_confusions", 12), "report.top_confusions", 1),
            max_error_examples=_int(raw.get("max_error_examples", 20), "report.max_error_examples", 1),
        )


@dataclass
class AppConfig:
    root: Path
    config_path: Path
    seed: int
    device: str
    results_dir: Path
    data: DataConfig
    audio: AudioConfig
    model: ModelConfig
    training: TrainingConfig
    eval: EvalConfig
    report: ReportConfig

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return cast(dict[str, Any], convert(dataclasses.asdict(self)))

    def for_run(self, modality: str, seed: int) -> AppConfig:
        if modality not in MODALITIES:
            raise ConfigError(f"unknown modality {modality!r}")
        return dataclasses.replace(
            self,
            seed=seed,
            model=dataclasses.replace(self.model, modality=modality),
        )


def _resolve(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value).resolve()


def load_config(path: str | Path = "configs/default.yaml", overrides: Sequence[str] = ()) -> AppConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    root = find_project_root(config_path.parent)
    raw = _load_raw(config_path)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed: Any
        try:
            parsed = yaml.safe_load(value)
        except yaml.YAMLError:
            parsed = value
        _set_dotted(raw, key.strip(), parsed)

    _check_keys(
        raw,
        {"seed", "device", "results_dir", "data", "audio", "model", "training", "eval", "report"},
        "top level",
    )
    seed = _int(raw.get("seed", 0), "seed", 0)
    device = str(raw.get("device", "auto"))
    if device not in {"auto", "cpu", "cuda", "mps"}:
        raise ConfigError("device must be auto, cpu, cuda or mps")
    data = DataConfig.from_dict(_mapping(raw.get("data", {}), "data"))
    audio = AudioConfig.from_dict(_mapping(raw.get("audio", {}), "audio"))
    model = ModelConfig.from_dict(_mapping(raw.get("model", {}), "model"))
    training = TrainingConfig.from_dict(_mapping(raw.get("training", {}), "training"))
    eval_cfg = EvalConfig.from_dict(_mapping(raw.get("eval", {}), "eval"))
    report = ReportConfig.from_dict(_mapping(raw.get("report", {}), "report"))

    if model.hid_dim % model.n_heads != 0:
        raise ConfigError("model.hid_dim must be divisible by model.n_heads")
    if audio.fmax > data.sample_rate / 2:
        raise ConfigError("audio.fmax must not exceed the Nyquist frequency")
    if audio.win_length > audio.n_fft:
        raise ConfigError("audio.win_length must not exceed audio.n_fft")

    return AppConfig(
        root=root,
        config_path=config_path,
        seed=seed,
        device=device,
        results_dir=_resolve(root, Path(str(raw.get("results_dir", "results/")))),
        data=dataclasses.replace(
            data,
            root=_resolve(root, data.root),
            csv_dir=_resolve(root, data.csv_dir),
            cache_dir=_resolve(root, data.cache_dir),
        ),
        audio=audio,
        model=model,
        training=training,
        eval=eval_cfg,
        report=dataclasses.replace(report, output=_resolve(root, report.output)),
    )


@dataclass(frozen=True)
class Condition:
    kind: str
    value: float = 0.0
    label: str = "clean"

    @property
    def needs_audio(self) -> bool:
        return self.kind.startswith("audio_")


def parse_condition(name: str) -> Condition:
    if name == "clean":
        return Condition("clean", 0.0, name)
    if name == "text_missing":
        return Condition("text_missing", 0.0, name)
    if name == "text_shuffle":
        return Condition("text_shuffle", 0.0, name)
    if name == "audio_missing":
        return Condition("audio_missing", 0.0, name)
    if name.startswith("text_drop_"):
        rate = _parse_number(name.removeprefix("text_drop_"), f"condition {name}")
        if rate < 0.0 or rate > 100.0:
            raise ConfigError(f"condition {name}: drop rate must be in [0, 100]")
        return Condition("text_drop", rate / 100.0, name)
    if name.startswith("audio_noise_") and name.endswith("db"):
        snr = _parse_number(name.removeprefix("audio_noise_").removesuffix("db"), f"condition {name}")
        return Condition("audio_noise", snr, name)
    raise ConfigError(f"unknown evaluation condition {name!r}")
