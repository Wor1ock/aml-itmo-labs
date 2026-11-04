import glob
import json
import os
from collections import Counter

import torch

from plots import plot_data, plot_summary
from utils import NAMES, Data, build_vocabs, f1_weighted, get_args, load_cfg, load_csv, wav_len

cfg = load_cfg(get_args())
d = cfg["data"]
out_dir = cfg["out_dir"]

runs: dict[str, list[dict]] = {}
for p in sorted(glob.glob(os.path.join(out_dir, "*_seed*", "metrics.json"))):
    mod = os.path.basename(os.path.dirname(p)).split("_seed")[0]
    with open(p) as f:
        runs.setdefault(mod, []).append(json.load(f))

train_df = load_csv(d["path"], d["train_file"])
test_df = load_csv(d["path"], d["test_file"])
vocabs = build_vocabs(train_df)
true = torch.tensor(Data(test_df, vocabs, d["wavs_location"]).labels)

keys = [f"{n}_{s}" for n in NAMES for s in ("acc", "f1")]
summary: dict[str, dict[str, tuple[float, float]]] = {}
for mod, rs in runs.items():
    summary[mod] = {}
    for k in keys:
        t = torch.tensor([r["none"][k] for r in rs])
        summary[mod][k] = (float(t.mean()), float(t.std()) if len(rs) > 1 else 0.0)

# baseline: always predict the most frequent train class
summary["majority"] = {}
for i, name in enumerate(NAMES):
    top = Counter(train_df[name]).most_common(1)[0][0]
    pred = torch.full_like(true[:, i], vocabs[name][top])
    summary["majority"][f"{name}_acc"] = (float((pred == true[:, i]).float().mean()), 0.0)
    summary["majority"][f"{name}_f1"] = (f1_weighted(pred, true[:, i], len(vocabs[name])), 0.0)

lines = ["## test metrics, mean ± std over seeds", "", "| model | seeds | " + " | ".join(keys) + " |"]
lines.append("|---|---|" + "---|" * len(keys))
for mod, s in summary.items():
    n_seeds = len(runs.get(mod, []))
    lines.append(f"| {mod} | {n_seeds} | " + " | ".join(f"{s[k][0]:.3f} ± {s[k][1]:.3f}" for k in keys) + " |")

if "fusion" in runs:
    acc = [f"{n}_acc" for n in NAMES]
    lines += ["", "## fusion ablation, test accuracy, mean over seeds", "", "| input | " + " | ".join(acc) + " |"]
    lines.append("|---|" + "---|" * len(acc))
    for mode in ("none", "audio_noise", "text_shuffle"):
        vals = [float(torch.tensor([r[mode][k] for r in runs["fusion"]]).mean()) for k in acc]
        lines.append(f"| {mode} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")

with open(os.path.join(out_dir, "results.md"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

plot_summary(summary, os.path.join(out_dir, "summary.png"))
lens = [wav_len(os.path.join(d["wavs_location"], p)) for p in train_df["audio"]]
plot_data(train_df, lens, os.path.join(out_dir, "data.png"))
