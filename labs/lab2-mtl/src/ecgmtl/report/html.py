"""Minimal self-contained HTML rendering: inline CSS, base64 images, no CDN."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..viz.plots import embed

CSS = """
:root { --fg:#1d1d1f; --muted:#6b6b70; --line:#d8d8dc; --accent:#2f6fb3; }
body { margin:0 auto; max-width:1100px; padding:32px 24px 80px; color:var(--fg);
  font:15px/1.55 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size:26px; margin:0 0 6px; }
h2 { font-size:19px; margin:34px 0 10px; border-bottom:1px solid var(--line); padding-bottom:6px; }
h3 { font-size:16px; margin:22px 0 8px; }
p, li { color:var(--fg); }
.meta { color:var(--muted); font-size:13px; margin-bottom:22px; }
table { border-collapse:collapse; width:100%; margin:10px 0 18px; font-size:13px; }
th, td { border:1px solid var(--line); padding:5px 8px; text-align:right; }
th:first-child, td:first-child { text-align:left; }
thead th { background:#f4f6f9; }
figure { margin:14px 0 22px; }
figure img { width:100%; border:1px solid var(--line); }
figcaption { color:var(--muted); font-size:12px; margin-top:4px; }
code { background:#f4f6f9; padding:1px 4px; border-radius:3px; font-size:12.5px; }
pre { background:#f4f6f9; padding:10px 12px; overflow-x:auto; font-size:12.5px; }
ul { padding-left:20px; }
.note { border-left:3px solid var(--accent); padding:6px 12px; background:#f7f9fc; margin:12px 0; }
"""


def esc(value: object) -> str:
    return html.escape(str(value))


def table(frame: pd.DataFrame, float_format: str = "{:.3f}") -> str:
    """DataFrame -> HTML table; floats are formatted, NaN becomes a dash."""
    head = "".join(f"<th>{esc(column)}</th>" for column in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                text = "—" if value != value else float_format.format(value)
            else:
                text = esc(value)
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def figure(path: Path, caption: str) -> str:
    return (
        f'<figure><img src="{embed(path)}" alt="{esc(caption)}">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def section(title: str, *blocks: str) -> str:
    return f"<h2>{esc(title)}</h2>" + "".join(blocks)


def page(title: str, subtitle: str, blocks: Iterable[str]) -> str:
    body = "".join(blocks)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>{esc(title)}</title><style>{CSS}</style></head><body>
<h1>{esc(title)}</h1><div class="meta">{esc(subtitle)}</div>{body}</body></html>"""
