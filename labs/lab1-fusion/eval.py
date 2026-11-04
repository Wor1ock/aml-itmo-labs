import json
import os

import torch

from model import build_model
from plots import plot_attention, plot_confusions
from utils import NAMES, Data, confusion, get_args, get_logger, load_cfg, load_csv, make_loader, run_dir, run_epoch, set_seed

cfg = load_cfg(get_args())
set_seed(cfg["seed"])
out = run_dir(cfg)
log = get_logger(os.path.join(out, "eval.log"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

d = cfg["data"]
with open(os.path.join(out, "vocab.json")) as f:
    vocabs = json.load(f)
pad_id = vocabs["text"]["<pad>"]
test_df = load_csv(d["path"], d["test_file"])
test_dl = make_loader(Data(test_df, vocabs, d["wavs_location"]), cfg, pad_id, False)

model = build_model(cfg, vocabs).to(device)
model.load_state_dict(torch.load(os.path.join(out, cfg["model_name"]), map_location=device, weights_only=True))
model.eval()

# ablation is only meaningful for the fusion model
modes = ["none", "audio_noise", "text_shuffle"] if cfg["modality"] == "fusion" else ["none"]
res: dict[str, dict[str, float]] = {}
for mode in modes:
    loss, m, pred, true = run_epoch(model, test_dl, device, corrupt=mode)
    res[mode] = {"loss": loss, **m}
    log.info(f"test {mode}: loss {loss:.4f} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    if mode == "none":
        conf = [confusion(pred[:, i], true[:, i], len(vocabs[k])) for i, k in enumerate(NAMES)]
with open(os.path.join(out, "metrics.json"), "w") as f:
    json.dump(res, f, indent=1)

plot_confusions(conf, [list(vocabs[k]) for k in NAMES], os.path.join(out, "confusion.png"))

if cfg["modality"] == "fusion":
    batch = {k: v.to(device) for k, v in next(iter(test_dl)).items()}
    with torch.no_grad():
        att = model(batch["audio"], batch["audio_mask"], batch["text"], batch["labels"], return_attn=True)["attn"].cpu()
    inv = {i: w for w, i in vocabs["text"].items()}
    items = []
    for i in range(min(4, len(att))):
        n_text = int((batch["text"][i] != pad_id).sum())
        n_audio = int(batch["audio_mask"][i].sum())
        tokens = [inv[int(t)] for t in batch["text"][i, :n_text].cpu()]
        items.append((att[i, :n_text, :n_audio], tokens))
    plot_attention(items, os.path.join(out, "attention.png"))
