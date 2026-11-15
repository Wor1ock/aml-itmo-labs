"""EDA page: labels, correlations, missing values, split, examples, shortcut audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..analysis.correlations import age_relation, cramers_v, relatedness_matrix
from ..config import Config
from ..data.labels import SUPERCLASSES
from ..data.prepare import load_processed
from ..logging_utils import get_logger
from ..viz import plots
from .html import figure, page, section, table


def _shortcut_audit(cfg: Config, manifest: pd.DataFrame) -> pd.DataFrame:
    """Show that non-clinical metadata is predictive, which is why it is not an input."""
    raw = cfg.data.root / "raw" / "ptbxl_database.csv"
    if not raw.exists():
        return pd.DataFrame()
    db = pd.read_csv(raw)
    target = db["scp_codes"].astype(str).str.contains("NORM").astype(int).to_numpy()
    rows = []
    for column in ("site", "nurse", "device", "validated_by_human", "heart_axis"):
        if column not in db.columns:
            continue
        values = db[column].fillna("na").astype(str).to_numpy()
        rows.append(
            {
                "колонка": column,
                "cramers_v": cramers_v(values, target),
                "уникальных значений": int(pd.Series(values).nunique()),
                "используется как признак": "нет",
            }
        )
    return pd.DataFrame(rows)


def build_eda(cfg: Config) -> Path:
    log = get_logger()
    manifest, signals, meta = load_processed(cfg.data)
    fig_dir = cfg.output.report_dir / "figures"
    blocks: list[str] = []

    overview = pd.DataFrame(
        [
            {"показатель": "записей", "значение": meta["n_records"]},
            {"показатель": "пациентов", "значение": meta["n_patients"]},
            {"показатель": "пропущено при чтении", "значение": meta["skipped"]},
            {"показатель": "версия PTB-XL", "значение": meta["version"]},
            {"показатель": "частота дискретизации, Гц", "значение": meta["hz"]},
            {"показатель": "длина записи, отсчётов", "значение": meta["signal_len"]},
            {"показатель": "отведений", "значение": meta["n_leads"]},
            {"показатель": "классов ритма", "значение": len(meta["rhythm_classes"])},
        ]
    )
    blocks.append(section("Данные", table(overview)))

    missing = pd.DataFrame(
        [
            {"задача": "диагностика", "без метки": 0, "тип": "multi-label, 5 суперклассов"},
            {
                "задача": "ритм",
                "без метки": int((manifest["rhythm_idx"] < 0).sum()),
                "тип": "single-label, класс по максимальной likelihood",
            },
            {"задача": "возраст", "без метки": int(manifest["age"].isna().sum()), "тип": "регрессия"},
        ]
    )
    blocks.append(section("Пропуски по задачам", table(missing)))

    intersection = pd.DataFrame(
        [
            {
                "диагностика + ритм + возраст": int(
                    ((manifest["rhythm_idx"] >= 0) & manifest["age"].notna()).sum()
                )
            },
            {
                "диагностика + ритм": int((manifest["rhythm_idx"] >= 0).sum()),
            },
            {"диагностика + возраст": int(manifest["age"].notna().sum())},
            {"только диагностика": int(((manifest["rhythm_idx"] < 0) & manifest["age"].isna()).sum())},
        ]
    )
    blocks.append(section("Пересечение подвыборок", table(intersection)))

    blocks.append(
        section(
            "Распределение меток",
            figure(plots.class_distribution(manifest, SUPERCLASSES, fig_dir / "eda_classes.png"), "частота суперклассов и классов ритма"),
            figure(plots.age_distribution(manifest, fig_dir / "eda_age.png"), "распределение возраста"),
            figure(plots.split_summary(manifest, fig_dir / "eda_folds.png"), "размеры фолдов strat_fold"),
        )
    )

    matrix = relatedness_matrix(manifest, SUPERCLASSES)
    blocks.append(
        section(
            "Связанность задач",
            "<p>мера Cramér's V между индикаторами меток; для двух бинарных индикаторов совпадает с |phi|. "
            "значения выше 0.4 означают, что задачи делят большую часть информации.</p>",
            figure(plots.label_corr(matrix, fig_dir / "eda_corr.png", "связанность меток"), "матрица связанности"),
            table(age_relation(manifest, SUPERCLASSES).head(12), "{:.3f}"),
        )
    )

    audit = _shortcut_audit(cfg, manifest)
    if not audit.empty:
        blocks.append(
            section(
                "Аудит утечек",
                "<p>эти колонки коррелируют с диагнозом, но не являются сигналом: в модель они не подаются.</p>",
                table(audit, "{:.3f}"),
            )
        )

    blocks.append(
        section(
            "Примеры сигналов",
            figure(plots.ecg_examples(signals, manifest, fig_dir / "eda_signals.png"), "12 отведений, z-score по отведению"),
        )
    )

    html = page(
        "ЛР2 · EDA",
        f"PTB-XL {meta['version']}, {meta['n_records']} записей, {meta['n_patients']} пациентов",
        blocks,
    )
    path = cfg.output.report_dir / "eda.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    log.info("EDA written to %s", path)
    return path
