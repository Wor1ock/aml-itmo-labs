"""Shared type aliases for artefacts written to disk."""

from __future__ import annotations

from typing import TypedDict


class DatasetMeta(TypedDict):
    """Contents of ``data/processed/meta.json``."""

    version: str
    hz: int
    signal_len: int
    n_leads: int
    superclasses: list[str]
    rhythm_classes: list[str]
    n_records: int
    n_patients: int
    skipped: int
