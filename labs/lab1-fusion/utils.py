import argparse
import csv
import logging
import os
import random
import re
import sys
import wave
from functools import partial

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from model import Model

NAMES = ["action", "object", "position"]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--modality", choices=["fusion", "text", "audio"])
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def load_cfg(args: argparse.Namespace) -> dict:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.modality is not None:
        cfg["modality"] = args.modality
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def run_dir(cfg: dict) -> str:
    return os.path.join(cfg["out_dir"], f"{cfg['modality']}_seed{cfg['seed']}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_logger(path: str) -> logging.Logger:
    log = logging.getLogger(path)
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%m-%d-%Y %H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def load_csv(path: str, name: str) -> dict[str, list[str]]:
    # columns: audio path, transcript, action, object, position
    with open(os.path.join(path, name)) as f:
        rows = [row for row in list(csv.reader(f))[1:] if row]
    keys = ["audio", "text", "action", "object", "position"]
    return {k: [row[i] for row in rows] for i, k in enumerate(keys)}


def build_vocabs(df: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    text = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    for s in df["text"]:
        for w in tokenize(s):
            if w not in text:
                text[w] = len(text)
    vocabs = {"text": text}
    for k in NAMES:
        vocabs[k] = {v: i for i, v in enumerate(sorted(set(df[k])))}
    return vocabs


def wav_len(path: str) -> int:
    with wave.open(path, "rb") as f:
        return f.getnframes()


def read_wav(path: str) -> torch.Tensor:
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2 and f.getnchannels() == 1, "expected 16-bit mono wav"
        raw = f.readframes(f.getnframes())
    return torch.frombuffer(bytearray(raw), dtype=torch.int16).float() / 32768.0  # [-1, 1]


class Data(Dataset):
    def __init__(self, df: dict[str, list[str]], vocabs: dict[str, dict[str, int]], wavs: str) -> None:
        self.audio = df["audio"]
        self.wavs = wavs
        sos, eos, unk = (vocabs["text"][t] for t in ("<sos>", "<eos>", "<unk>"))
        self.text = [[sos] + [vocabs["text"].get(w, unk) for w in tokenize(s)] + [eos] for s in df["text"]]
        self.labels = [[vocabs[k][df[k][i]] for k in NAMES] for i in range(len(self.audio))]

    def __len__(self) -> int:
        return len(self.audio)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, list[int], list[int]]:
        return read_wav(os.path.join(self.wavs, self.audio[i])), self.text[i], self.labels[i]


def collate_fn(
    batch: list[tuple[torch.Tensor, list[int], list[int]]], pad_id: int, split: int
) -> dict[str, torch.Tensor]:
    # audio is padded with zeros to a whole number of chunks, audio_mask marks chunks with real samples
    n = len(batch)
    n_chunks = (max(len(b[0]) for b in batch) + split - 1) // split
    max_text = max(len(b[1]) for b in batch)
    audio = torch.zeros(n, n_chunks * split)
    audio_mask = torch.zeros(n, n_chunks, dtype=torch.bool)
    text = torch.full((n, max_text), pad_id, dtype=torch.long)
    for i, (wav, ids, _) in enumerate(batch):
        audio[i, : len(wav)] = wav
        audio_mask[i, : (len(wav) + split - 1) // split] = True
        text[i, : len(ids)] = torch.tensor(ids)
    labels = torch.tensor([b[2] for b in batch])
    return {"audio": audio, "audio_mask": audio_mask, "text": text, "labels": labels}


def make_loader(ds: Data, cfg: dict, pad_id: int, shuffle: bool) -> DataLoader:
    fn = partial(collate_fn, pad_id=pad_id, split=cfg["audio_split_samples"])
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle, collate_fn=fn)


def confusion(pred: torch.Tensor, true: torch.Tensor, n: int) -> torch.Tensor:
    # rows: true class, columns: predicted class
    return torch.bincount(true * n + pred, minlength=n * n).reshape(n, n)


def f1_weighted(pred: torch.Tensor, true: torch.Tensor, n: int) -> float:
    # per-class f1 = 2tp / (support + predicted), averaged with support as weights
    conf = confusion(pred, true, n)
    tp = conf.diag().float()
    support = conf.sum(1).float()
    n_pred = conf.sum(0).float()
    f1 = 2 * tp / (support + n_pred).clamp(min=1)
    return float((f1 * support).sum() / support.sum())


def get_metrics(pred: torch.Tensor, true: torch.Tensor, dims: list[int]) -> dict[str, float]:
    out = {}
    for i, name in enumerate(NAMES):
        out[f"{name}_acc"] = float((pred[:, i] == true[:, i]).float().mean())
        out[f"{name}_f1"] = f1_weighted(pred[:, i], true[:, i], dims[i])
    return out


def run_epoch(
    model: Model,
    loader: DataLoader,
    device: torch.device,
    opt: torch.optim.Optimizer | None = None,
    clip: float = 1.0,
    corrupt: str = "none",
) -> tuple[float, dict[str, float], torch.Tensor, torch.Tensor]:
    """One pass over loader: training when opt is given, evaluation otherwise.

    corrupt (evaluation only): 'audio_noise' replaces audio with noise, 'text_shuffle' mixes texts inside the batch.
    """
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if corrupt == "audio_noise":
                batch["audio"] = torch.randn_like(batch["audio"]) * batch["audio"].std()
            elif corrupt == "text_shuffle":
                batch["text"] = batch["text"][torch.randperm(len(batch["text"]), device=device)]
            out = model(batch["audio"], batch["audio_mask"], batch["text"], batch["labels"])
            if opt is not None:
                opt.zero_grad()
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            k = len(batch["labels"])
            total += out["loss"].item() * k
            n += k
            preds.append(out["pred"].cpu())
            labels.append(batch["labels"].cpu())
    pred, true = torch.cat(preds), torch.cat(labels)
    return total / n, get_metrics(pred, true, model.dims), pred, true
