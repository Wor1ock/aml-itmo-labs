"""Dataset download.

PTB-XL is public and needs no token, but physionet.org is not reachable from every
network, so the downloader walks through a list of sources and stops at the first one
that works. Preprocessing happens right after, so training never touches the raw files.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from ..config import DataConfig
from ..logging_utils import get_logger

REQUIRED_FILES = ("ptbxl_database.csv", "scp_statements.csv")
RECORDS_DIR = "records100"
CHUNK = 1 << 20


def _sources(cfg: DataConfig) -> list[tuple[str, str]]:
    """Candidate (label, url) pairs for the raw archive, most canonical first."""
    sources = [
        (
            "physionet",
            "https://physionet.org/static/published-projects/ptb-xl/"
            f"ptb-xl-a-large-publicly-available-electrocardiography-dataset-{cfg.version}.zip",
        ),
        ("huggingface-mirror", f"{cfg.mirror_url}/ptb-xl-data.zip"),
    ]
    return sources[::-1] if cfg.prefer_mirror else sources


def _fetch(url: str, dest: Path, retries: int = 3, timeout: float = 60.0) -> None:
    log = get_logger()
    for attempt in range(1, retries + 1):
        try:
            log.info("downloading %s (attempt %d)", url, attempt)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as out:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    block = response.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if total and done // (128 * CHUNK) != (done - len(block)) // (128 * CHUNK):
                        log.info("  %.0f / %.0f MB", done / 1e6, total / 1e6)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.warning("source failed (%s): %s", type(exc).__name__, exc)
            if attempt == retries:
                raise RuntimeError(f"cannot download {url}") from exc


def _fetch_first(urls: list[str], dest: Path) -> None:
    """Try every url in order and keep the first successful download."""
    errors = []
    for url in urls:
        try:
            _fetch(url, dest)
            return
        except RuntimeError as exc:
            errors.append(str(exc))
    raise RuntimeError("all sources failed:\n" + "\n".join(errors))


def _unpack(archive: Path, target: Path) -> None:
    """Extract the 100 Hz records and the metadata tables only.

    The 500 Hz records are five times larger and are not used by this project, so they are
    skipped on purpose instead of being unpacked and deleted.
    """
    log = get_logger()
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        prefix = names[0] if names and names[0].endswith("/") and names[0].count("/") == 1 else ""
        keep = [
            name
            for name in names
            if not name.endswith("/")
            and not name.startswith("__MACOSX")
            and (
                name.startswith(prefix + RECORDS_DIR + "/")
                or "/" not in name[len(prefix) :]
            )
        ]
        log.info("extracting %d of %d archive members", len(keep), len(names))
        for name in keep:
            member = target / name[len(prefix) :]
            member.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, member.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def ensure_raw(cfg: DataConfig) -> Path:
    """Make sure raw PTB-XL files exist locally and return the raw directory."""
    log = get_logger()
    raw = cfg.root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_FILES:
        if not (raw / name).exists():
            csvs = [f"{cfg.url}/{name}", f"{cfg.mirror_url}/{name}"]
            _fetch_first(csvs[::-1] if cfg.prefer_mirror else csvs, raw / name)
    if not (raw / RECORDS_DIR).is_dir():
        archive = raw / "ptb-xl.zip"
        if not archive.exists():
            _fetch_first([url for _, url in _sources(cfg)], archive)
        _unpack(archive, raw)
    missing = [p for p in (raw / "ptbxl_database.csv", raw / "scp_statements.csv", raw / RECORDS_DIR) if not p.exists()]
    if missing:
        raise RuntimeError(f"raw layout is incomplete, missing: {[str(p) for p in missing]}")
    log.info("raw data ready at %s", raw)
    return raw
