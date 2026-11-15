"""Task relatedness: how often labels of two tasks co-occur.

Cramer's V is used for every pair, so binary superclass indicators and multi-class rhythm
indicators sit in one comparable matrix (for two binary indicators it equals |phi|).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Association between two categorical vectors, 0 = independent, 1 = deterministic."""
    table = pd.crosstab(a, b).to_numpy(dtype=float)
    if table.size == 0 or table.sum() == 0:
        return float("nan")
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / table.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((table - expected) ** 2 / expected)
    n = table.sum()
    phi2 = chi2 / n
    r, k = table.shape
    if min(r - 1, k - 1) <= 0:
        return float("nan")
    return float(np.sqrt(phi2 / min(r - 1, k - 1)))


def indicator_frame(manifest: pd.DataFrame, superclasses: Sequence[str]) -> pd.DataFrame:
    """Binary indicators for every diagnostic superclass and every rhythm class."""
    data = {name: manifest[f"diag_{name}"].to_numpy() for name in superclasses}
    rhythm = manifest["rhythm_code"].astype(str)
    for name in sorted(rhythm.unique()):
        data[f"rhythm:{name}"] = (rhythm == name).to_numpy().astype(int)
    return pd.DataFrame(data, index=manifest.index)


def relatedness_matrix(manifest: pd.DataFrame, superclasses: Sequence[str]) -> pd.DataFrame:
    indicators = indicator_frame(manifest, superclasses)
    names = list(indicators.columns)
    matrix = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            value = cramers_v(indicators[left].to_numpy(), indicators[right].to_numpy())
            matrix.loc[left, right] = value
            matrix.loc[right, left] = value
    return matrix


def age_relation(manifest: pd.DataFrame, superclasses: Sequence[str]) -> pd.DataFrame:
    """Spearman correlation of age with each label indicator and with rhythm classes."""
    indicators = indicator_frame(manifest, superclasses)
    age = manifest["age"]
    rows = []
    for name in indicators.columns:
        mask = age.notna()
        if mask.sum() < 10:
            continue
        value = indicators.loc[mask, name].astype(float).corr(age[mask], method="spearman")
        rows.append({"label": name, "spearman_age": float(value), "n": int(mask.sum())})
    return pd.DataFrame(rows).sort_values("spearman_age", key=lambda s: s.abs(), ascending=False)


def rhythm_vs_diag_table(manifest: pd.DataFrame, superclasses: Sequence[str]) -> pd.DataFrame:
    """Cross-tab of rhythm class versus presence of each superclass."""
    rows = []
    for name in superclasses:
        table = pd.crosstab(manifest["rhythm_code"], manifest[f"diag_{name}"])
        if table.empty:
            continue
        for rhythm in table.index:
            rows.append(
                {
                    "superclass": name,
                    "rhythm": rhythm,
                    "present": int(table.loc[rhythm].get(1, 0)),
                    "absent": int(table.loc[rhythm].get(0, 0)),
                }
            )
    return pd.DataFrame(rows)
