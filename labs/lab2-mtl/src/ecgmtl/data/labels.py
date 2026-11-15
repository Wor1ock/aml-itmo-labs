"""Label construction for the three tasks.

PTB-XL stores one dict of SCP-ECG statements per record, e.g. ``{'NORM': 100.0, 'SR': 0.0}``.
``scp_statements.csv`` says which statement belongs to which family, so the families are
split into separate tasks instead of collapsing everything into one multi-label target.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..logging_utils import get_logger
SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")


def parse_scp_codes(raw: object) -> dict[str, float]:
    """Parse the ``scp_codes`` cell into ``{code: likelihood}``."""
    if not isinstance(raw, str):
        return {}
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(k): float(v) for k, v in value.items()}


def load_statements(path: str | Path) -> pd.DataFrame:
    """Load the statement table indexed by SCP code with three boolean family columns."""
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    df.index.name = "code"
    for column in ("diagnostic", "form", "rhythm"):
        df[column] = df[column].fillna(0).astype(int) if column in df.columns else 0
    if "diagnostic_class" not in df.columns:
        raise ValueError("scp_statements.csv has no diagnostic_class column")
    return df


def statement_families(statements: pd.DataFrame) -> dict[str, set[str]]:
    """Codes grouped by family; used for EDA and for building the tasks."""
    return {
        "diagnostic": set(statements.index[statements["diagnostic"] == 1]),
        "form": set(statements.index[statements["form"] == 1]),
        "rhythm": set(statements.index[statements["rhythm"] == 1]),
    }


def diag_target(codes: dict[str, float], statements: pd.DataFrame) -> np.ndarray:
    """Multi-label superclass target of shape (5,)."""
    target = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for code in codes:
        if code not in statements.index:
            continue
        row = statements.loc[code]
        if int(row["diagnostic"]) != 1:
            continue
        name = row["diagnostic_class"]
        if isinstance(name, str) and name in SUPERCLASSES:
            target[SUPERCLASSES.index(name)] = 1.0
    return target


def rhythm_order(rhythm_codes: list[list[str]], topk: int) -> list[str]:
    """Keep the most frequent rhythm statements, most frequent first.

    The order doubles as a precedence rule for records that carry several rhythm codes.
    """
    counts = Counter(code for codes in rhythm_codes for code in codes)
    return [code for code, _ in counts.most_common(topk)]


def rhythm_target(codes: dict[str, float], order: list[str]) -> int:
    """Single-label rhythm index or -1 when the record has no usable rhythm statement."""
    candidates = [(codes[code], order.index(code)) for code in codes if code in order]
    if not candidates:
        return -1
    # highest likelihood wins; ties are broken by the global frequency order
    return min(candidates, key=lambda item: (-item[0], item[1]))[1]


def build_labels(db: pd.DataFrame, statements: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Turn the PTB-XL metadata table into a flat label frame."""
    log = get_logger()
    rhythm_codes = set(statements.index[statements["rhythm"] == 1])
    per_record = [parse_scp_codes(raw) for raw in db["scp_codes"]]
    order = rhythm_order(
        [sorted(code for code in codes if code in rhythm_codes) for codes in per_record],
        cfg.rhythm_topk,
    )

    ecg_ids: list[int] = [int(value) for value in db["ecg_id"]]
    patient_ids: list[int] = [int(value) for value in db["patient_id"]]
    folds: list[int] = [int(value) for value in db["strat_fold"]]
    ages: list[float] = [float(value) for value in db["age"]]
    sexes: list[int] = [int(value) for value in db["sex"]]
    filenames: list[str] = [str(value) for value in db["filename_lr"]]

    rows = []
    for i, codes in enumerate(per_record):
        target = diag_target(codes, statements)
        age = float(ages[i]) if np.isfinite(ages[i]) else float("nan")
        if not cfg.min_age <= age <= cfg.max_age:
            age = float("nan")
        record = {
            "ecg_id": int(ecg_ids[i]),
            "patient_id": int(patient_ids[i]),
            "strat_fold": int(folds[i]),
            "age": age,
            "sex": int(sexes[i]),
            "rhythm_idx": rhythm_target(codes, order),
            "filename": filenames[i],
            "n_diag_positive": int(target.sum()),
        }
        for j, name in enumerate(SUPERCLASSES):
            record[f"diag_{name}"] = int(target[j])
        rows.append(record)

    labels = pd.DataFrame(rows)
    labels["rhythm_code"] = labels["rhythm_idx"].map(
        lambda idx: order[idx] if idx >= 0 else "none"
    )
    log.info(
        "labels: %d records, %d rhythm classes, %d without rhythm, %d without age",
        len(labels),
        len(order),
        int((labels["rhythm_idx"] < 0).sum()),
        int(labels["age"].isna().sum()),
    )
    log.info("rhythm classes: %s", ", ".join(order))
    return labels


def rhythm_class_names(order: list[str]) -> list[str]:
    return list(order)
