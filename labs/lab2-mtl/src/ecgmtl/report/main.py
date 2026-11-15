"""Final self-contained HTML report built from ``outputs/``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config
from ..data.labels import SUPERCLASSES
from ..experiments import best_run_dir
from ..logging_utils import get_logger
from ..viz import plots
from .html import figure, page, section, table

METHOD_NOTES = {
    "single_diag": "одна голова на диагностику, общий энкодер",
    "single_rhythm": "одна голова на ритм",
    "single_age": "одна голова на возраст",
    "mtl_equal": "три задачи, равные веса",
    "mtl_uncertainty": "три задачи, uncertainty weighting",
    "mtl_pcgrad": "три задачи, PCGrad на общем энкодере",
    "mtl_age_weight_0.1": "три задачи, вес возраста снижен",
    "loto_no_rhythm": "все задачи минус ритм",
    "loto_no_age": "все задачи минус возраст",
}

METRIC_LABELS = {
    "diag": {"macro_auc": "macro-AUC", "macro_f1": "macro-F1"},
    "rhythm": {"macro_f1": "macro-F1", "accuracy": "accuracy"},
    "age": {"mae": "MAE, лет", "rmse": "RMSE, лет", "r2": "R²"},
}


def _fmt(stats: dict[str, float], digits: int = 3) -> str:
    mean = stats["mean"]
    std = stats["std"]
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def _results_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = {}
    for method, tasks in summary.items():
        row: dict[str, str] = {"метод": method, "что это": METHOD_NOTES.get(method, "")}
        for task, metrics in tasks.items():
            for metric, stats in metrics.items():
                label = METRIC_LABELS.get(task, {}).get(metric, metric)
                row[f"{task}: {label}"] = _fmt(stats)
        rows[method] = row
    return pd.DataFrame(list(rows.values()))


def _ci_table(summary: dict[str, Any], main_metrics: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for method, tasks in summary.items():
        row: dict[str, str] = {"метод": method}
        for task, metric in main_metrics:
            stats = tasks.get(task, {}).get(metric)
            if stats is None:
                row[f"{task}: {METRIC_LABELS[task][metric]}"] = "—"
                continue
            row[f"{task}: {METRIC_LABELS[task][metric]}"] = (
                f"{stats['mean']:.3f} [{stats['ci_low']:.3f}; {stats['ci_high']:.3f}]"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _delta_table(table_dir: Path) -> pd.DataFrame:
    path = table_dir / "deltas.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame[frame["method"].isin(METHOD_NOTES)]
    frame["Δ"] = frame["delta"].map(lambda v: f"{v:+.3f}")
    frame["p"] = frame["p_value"].map(lambda v: f"{v:.3f}")
    frame["metric"] = frame.apply(
        lambda row: f"{row['task']}: {METRIC_LABELS.get(row['task'], {}).get(row['metric'], row['metric'])}",
        axis=1,
    )
    return frame[["method", "baseline", "metric", "Δ", "p", "n"]].rename(
        columns={"method": "метод", "baseline": "baseline"}
    )


def _agreement(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-tab of task-level correctness and a few examples where the tasks disagree."""
    rows = pred[pred["has_rhythm"].astype(bool)].copy()
    diag_true = rows[[f"diag_true_{name}" for name in SUPERCLASSES]].to_numpy()
    diag_prob = rows[[f"diag_prob_{name}" for name in SUPERCLASSES]].to_numpy()
    diag_ok = ((diag_prob >= 0.5).astype(int) == diag_true).all(axis=1)
    rhythm_ok = rows["rhythm_pred"].to_numpy() == rows["rhythm_true"].to_numpy()
    table = pd.crosstab(
        pd.Series(diag_ok, name="диагностика верна"),
        pd.Series(rhythm_ok, name="ритм верен"),
    ).reset_index()
    disagree = rows[diag_ok != rhythm_ok]
    examples = pd.DataFrame(
        {
            "ecg_id": disagree["ecg_id"].to_numpy()[:8],
            "ритм истина": disagree["rhythm_true"].to_numpy()[:8],
            "ритм предсказание": disagree["rhythm_pred"].to_numpy()[:8],
            "возраст истина": disagree["age_true"].to_numpy()[:8],
            "возраст предсказание": disagree["age_pred"].to_numpy()[:8],
        }
    )
    return table, examples


def _per_class_table(metrics: dict[str, Any]) -> pd.DataFrame:
    diag = metrics["test"].get("diag", {})
    rows = []
    for key, value in diag.items():
        if not key.startswith("f1_"):
            continue
        name = key[3:]
        rows.append(
            {
                "суперкласс": name,
                "F1": diag.get(f"f1_{name}", float("nan")),
                "precision": diag.get(f"precision_{name}", float("nan")),
                "recall": diag.get(f"recall_{name}", float("nan")),
                "support": diag.get(f"support_{name}", 0),
            }
        )
    rows.sort(key=lambda row: -row["F1"])
    return pd.DataFrame(rows)


def build_report(cfg: Config) -> Path:
    log = get_logger()
    table_dir = cfg.output.out_dir
    summary_path = table_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no {summary_path}; run the experiments first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fig_dir = cfg.output.report_dir / "figures"
    main_run = best_run_dir(cfg, "mtl_equal") or best_run_dir(cfg, next(iter(summary)))
    blocks: list[str] = []

    task_rows = pd.DataFrame(
        [
            {"задача": "диагностика", "роль": "основная", "метка": "5 суперклассов", "тип": "multi-label", "метрика": "macro-AUC, macro-F1"},
            {"задача": "ритм", "роль": "вспомогательная", "метка": "класс ритма", "тип": "multi-class", "метрика": "macro-F1, accuracy"},
            {"задача": "возраст", "роль": "вспомогательная", "метка": "лет", "тип": "регрессия", "метрика": "MAE, RMSE, R²"},
        ]
    )
    blocks.append(
        section(
            "Постановка задачи",
            "<p>одна общая часть модели решает три задачи разной природы: клиническую "
            "классификацию по 12 отведениям, определение ритма и регрессию возраста. "
            "метки сняты с одного сигнала, но описывают разные явления: морфологию, "
            "генерацию ритма и неклинический признак.</p>",
            table(task_rows),
        )
    )

    metrics: dict[str, Any] = {}
    if main_run is not None:
        metrics = json.loads((main_run / "metrics.json").read_text(encoding="utf-8"))
        split_info = metrics.get("split", {})
        split_table = pd.DataFrame(
            [
                {"часть": "train", "записей": split_info.get("n_train", "—"), "фолды": "все, кроме теста и валидации"},
                {"часть": "валидация", "записей": split_info.get("n_val", "—"), "фолды": split_info.get("val_fold", "—")},
                {"часть": "тест", "записей": split_info.get("n_test", "—"), "фолды": split_info.get("test_fold", "—")},
            ]
        )
        blocks.append(
            section(
                "Данные и split",
                "<p>PTB-XL, стратифицированные по пациенту фолды <code>strat_fold</code>. "
                "тест и валидация — разные фолды, пациенты не пересекаются, тест не "
                "участвует в выборе эпох, весов и стратегии.</p>",
                table(split_table),
            )
        )

    params_table = pd.DataFrame(
        [{"часть модели": key, "параметров": value} for key, value in metrics.get("params", {}).items()]
    )
    blocks.append(
        section(
            "Архитектура и лоссы",
            "<p>общий энкодер (Conv1d + batch norm + attention pooling) даёт вектор "
            "<code>h</code>, каждая голова — двухслойный MLP на этом векторе. "
            "лосс: <code>L = Σ w_i · L_i</code>, где <code>L_diag</code> — BCE по пяти "
            "суперклассам, <code>L_rhythm</code> — cross-entropy, <code>L_age</code> — SmoothL1. "
            "отсутствующие метки маскируются и не дают градиента.</p>"
            "<p>uncertainty weighting: <code>L = Σ L_i / (2σ_i²) + log σ_i</code>, "
            "<code>log σ_i²</code> обучается вместе с моделью. PCGrad: градиенты задач "
            "на общем энкодере проецируются, если их скалярное произведение отрицательно.</p>",
            table(params_table) if not params_table.empty else "",
        )
    )

    blocks.append(
        section(
            "Результаты: задача × метод",
            "<p>тест, среднее ± стандартное отклонение по фолдам.</p>",
            table(_results_table(summary)),
        )
    )

    blocks.append(
        section(
            "Доверительные интервалы",
            "<p>95% интервал по фолдам для основных метрик.</p>",
            table(
                _ci_table(
                    summary,
                    [("diag", "macro_auc"), ("diag", "macro_f1"), ("rhythm", "macro_f1"), ("age", "mae")],
                )
            ),
        )
    )

    deltas = _delta_table(table_dir)
    if not deltas.empty:
        blocks.append(
            section(
                "Что помогло и что навредило",
                "<p>разница с соответствующей single-task моделью по тому же фолду; "
                "p — парный t-тест по фолдам.</p>",
                table(deltas),
            )
        )

    if main_run is not None:
        history = json.loads((main_run / "history.json").read_text(encoding="utf-8"))
        blocks.append(
            section(
                "Кривые обучения и конфликт градиентов",
                figure(plots.history_curves(history, fig_dir / "report_history.png"), f"{main_run.parent.name}: лоссы и метрики по эпохам"),
                figure(plots.conflict_curve(history, fig_dir / "report_conflict.png"), "косинус между градиентами задач"),
            )
        )

    if main_run is not None and (main_run / "predictions.csv").exists():
        predictions = pd.read_csv(main_run / "predictions.csv")
        test_rows = predictions[predictions["split"] == "test"]
        names = [
            column.replace("rhythm_prob_", "")
            for column in test_rows.columns
            if column.startswith("rhythm_prob_")
        ]
        per_class = _per_class_table(json.loads((main_run / "metrics.json").read_text(encoding="utf-8")))
        agreement, examples = _agreement(test_rows)
        blocks.append(
            section(
                "Ошибки",
                "<p>качество по классам диагностики.</p>",
                table(per_class, "{:.3f}"),
                "<p>согласованность задач на тесте: как часто обе задачи решены верно.</p>",
                table(agreement),
                "<p>примеры записей, где одна задача решена верно, а другая нет.</p>",
                table(examples, "{:.1f}"),
                figure(plots.rhythm_confusion(test_rows, names, fig_dir / "report_rhythm_cm.png"), "матрица ошибок ритма"),
                figure(plots.age_scatter(test_rows, fig_dir / "report_age.png"), "возраст: предсказание против истины"),
            )
        )

    blocks.append(
        section(
            "Ограничения",
            "<ul>"
            "<li>одна модальность: модель видит только ЭКГ, метаданные и заключение не используются.</li>"
            "<li>классы ритма выбраны по частоте; редкие ритмы отброшены вместе с их записями.</li>"
            "<li>часть суперклассов разрежена, macro-AUC по ним нестабилен.</li>"
            "<li>абляции идут на меньшем числе фолдов, чем основная таблица.</li>"
            "<li>early stopping и веса выбираются по валидации; разброс между фолдами остается.</li>"
            "</ul>"
        )
    )

    if main_run is not None:
        meta_info = json.loads((main_run / "run_meta.json").read_text(encoding="utf-8"))
        repro = pd.DataFrame(
            [
                {"поле": "git commit", "значение": meta_info["git_commit"]},
                {"поле": "python", "значение": meta_info["python"]},
                {"поле": "torch", "значение": meta_info["torch"]},
                {"поле": "numpy", "значение": meta_info["numpy"]},
                {"поле": "устройство", "значение": meta_info["device_name"]},
                {"поле": "время одного запуска, с", "значение": meta_info["seconds"]},
                {"поле": "эпох", "значение": meta_info["epochs_run"]},
            ]
        )
        blocks.append(
            section(
                "Воспроизведение",
                "<pre>docker compose run --rm prepare\n"
                "docker compose run --rm eda\n"
                "docker compose run --rm experiments\n"
                "docker compose run --rm report</pre>",
                table(repro),
            )
        )

    path = cfg.output.report_dir / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    html = page(
        "ЛР2 · Multi-task learning на PTB-XL",
        "общий энкодер и три головы: диагностика, ритм, возраст",
        blocks,
    )
    path.write_text(html, encoding="utf-8")
    log.info("report written to %s", path)
    return path
