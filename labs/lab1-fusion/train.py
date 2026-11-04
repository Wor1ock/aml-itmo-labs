import json
import os
import time

import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from model import build_model
from plots import plot_curves
from utils import Data, build_vocabs, get_args, get_logger, load_cfg, load_csv, make_loader, run_dir, run_epoch, set_seed

cfg = load_cfg(get_args())
set_seed(cfg["seed"])
out = run_dir(cfg)
os.makedirs(out, exist_ok=True)
log = get_logger(os.path.join(out, "train.log"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"device {device}, modality {cfg['modality']}, seed {cfg['seed']}")
with open(os.path.join(out, "config.yaml"), "w") as f:
    yaml.safe_dump(cfg, f)

d = cfg["data"]
train_df = load_csv(d["path"], d["train_file"])
valid_df = load_csv(d["path"], d["valid_file"])
vocabs = build_vocabs(train_df)  # built on train only
with open(os.path.join(out, "vocab.json"), "w") as f:
    json.dump(vocabs, f)
log.info(f"train {len(train_df['audio'])}, valid {len(valid_df['audio'])}, text vocab {len(vocabs['text'])}")

pad_id = vocabs["text"]["<pad>"]
train_dl = make_loader(Data(train_df, vocabs, d["wavs_location"]), cfg, pad_id, True)
valid_dl = make_loader(Data(valid_df, vocabs, d["wavs_location"]), cfg, pad_id, False)

model = build_model(cfg, vocabs).to(device)
log.info(f"parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
o = cfg["optimizer"]
opt = torch.optim.Adam(model.parameters(), lr=o["lr"], betas=(o["beta1"], o["beta2"]))
writer = SummaryWriter(os.path.join(out, "tb"))

history: list[dict[str, float]] = []
best = float("inf")
for epoch in range(cfg["epochs"]):
    t0 = time.time()
    tr_loss, tr_m, _, _ = run_epoch(model, train_dl, device, opt, cfg["clip"])
    va_loss, va_m, _, _ = run_epoch(model, valid_dl, device)

    row: dict[str, float] = {"epoch": epoch + 1, "train_loss": tr_loss, "valid_loss": va_loss}
    row.update({f"train_{k}": v for k, v in tr_m.items()})
    row.update({f"valid_{k}": v for k, v in va_m.items()})
    history.append(row)
    for k, v in row.items():
        if k != "epoch":
            writer.add_scalar(k, v, epoch)
    writer.flush()

    if va_loss < best:
        best = va_loss
        torch.save(model.state_dict(), os.path.join(out, cfg["model_name"]))

    acc = " ".join(f"{k}={va_m[f'{k}_acc']:.3f}" for k in ("action", "object", "position"))
    log.info(f"epoch {epoch + 1:02} | {time.time() - t0:.0f}s | train loss {tr_loss:.4f} | valid loss {va_loss:.4f} | valid acc {acc}")

with open(os.path.join(out, "history.json"), "w") as f:
    json.dump(history, f, indent=1)
plot_curves(history, os.path.join(out, "curves.png"))
writer.close()
