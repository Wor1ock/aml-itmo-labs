# Audio–text cross-attention fusion for spoken intent classification

Three-slot intent classification (`action`, `object`, `location`) on the
[Fluent Speech Commands](https://www.kaggle.com/datasets/tommyngx/fluent-speech-corpus)
dataset. Transcript tokens query audio tokens through cross-attention; the
`<sos>` state is classified by three linear heads.

The most important artefact is the **self-contained HTML report** in
[`report/index.html`](report/index.html). It covers the task, method, real
results, robustness, error analysis, hypotheses and improvement ideas.

> **Honest framing.** In FSC, the 248 transcripts map bijectively to the 31
> intents, so text-only is almost 100%. The value of fusion is therefore measured
> as robustness to degraded/missing text and as the audio-only gap — not as a
> headline accuracy win over text-only.

## What is implemented

- Hand-written log-mel features (40 mels, 25 ms window, 10 ms hop, CMVN),
  SpecAugment, `float16` mmap cache. No librosa/torchaudio.
- Word-level tokenizer and vocabulary built on the train split.
- Conv1d audio frontend (stride 4 → 2), audio/text self-attention, cross-attention
  (`Q = text`, `K = V = audio`, padding mask), `<sos>` pooling, three heads.
- Baselines: majority class, audio-only, text-only, late fusion.
- Metrics implemented manually with NumPy: confusion matrix, precision/recall/F1,
  exact match, joint-intent accuracy, ECE and reliability curves. **No sklearn.**
- Evaluation without retraining: text drop 10/25/50%, text missing, text shuffle,
  audio noise 5/0 dB (waveform, log-mel recomputed), audio missing.
- Visualisation module: dataset overview, training curves, model comparison,
  robustness curves, confusion matrices, per-class F1, calibration, attention
  heatmaps, error panels.
- Jinja2 report with base64-embedded images.
- Docker (multi-stage, pinned `uv`, non-root, CPU torch) and `docker compose`
  services for full run, smoke run, training, report and tests.
- Typed package checked with strict `mypy`; `pytest` suite for metrics, text,
  audio and model masking.

## Data layout

```
data/fluent_speech_commands_dataset/
  data/train_data.csv      # ,path,speakerId,transcription,action,object,location
  data/valid_data.csv
  data/test_data.csv
  wavs/speakers/.../*.wav  # 16 kHz mono 16-bit PCM
```

Download with the Kaggle CLI:

```bash
uv sync
uv run python scripts/download_data.py
```

The loader ignores the unnamed index column, so the original broken
5-column assumption is gone. `location` is used instead of the old `position`.

## Quick start (uv)

```bash
uv sync
uv run python -m fusion_slu.cli prepare --config configs/default.yaml
uv run python -m fusion_slu.cli all --config configs/smoke.yaml    # ~2 min
uv run python -m fusion_slu.cli all --config configs/default.yaml  # full grid
```

Useful overrides:

```bash
uv run python -m fusion_slu.cli train --config configs/default.yaml \
  --modality fusion --seed 0 --set training.epochs=2
uv run python -m fusion_slu.cli evaluate --config configs/default.yaml --modality fusion
uv run python -m fusion_slu.cli report --config configs/default.yaml
```

Outputs:

- `results/<modality>_seed<N>/`: `model.pt`, `run.json`, `history.json`,
  `metrics.json`, `predictions.csv`, `attention.json`, logs and PNGs.
- `results/summary.json`, `results/analysis.json`.
- `report/figures/*.png` and `report/index.html` (self-contained).

The feature cache lives in `cache/<cache_tag>/` and is not part of the dataset
mount, so `data/` can be mounted read-only.

## Docker

```bash
docker compose build
docker compose run --rm smoke                 # smoke config end-to-end
docker compose run --rm lab all --config configs/default.yaml
docker compose run --rm report                # rebuild HTML from existing runs
docker compose --profile test run --rm test   # pytest inside the image
```

Set `UID`/`GID` if the host user is not 1000, so the non-root container can write
to bind-mounted `cache/`, `results/` and `report/`.

## Configuration

`configs/default.yaml` is the full experiment (4 models x 3 seeds, 6 epochs,
hid 128, 4 heads, 2/2/2 layers, ff 256, batch 128). `configs/smoke.yaml`
extends it with small subsets, 2 epochs and one seed. Any value can be
overridden with `--set section.key=value`.

## Tests and typing

```bash
uv run mypy
uv run pytest
```

`mypy` is configured in strict mode for `src`, `scripts` and `tests`.
`pytest` covers metric corner cases, tokenisation/vocabulary, wav/log-mel
extraction, all four model modalities, attention masking/normalisation and
config merging.

## Design notes

- **Why log-mel + Conv1d instead of raw chunks?** The original raw-waveform
  encoder ran at ~2.4 s per batch of 64 on CPU (~15 min/epoch). Log-mel with
  8× temporal downsampling, a float16 feature cache and FastDropout (uniform-mask dropout,
  2.4x faster than bernoulli-based nn.Dropout on this CPU) bring one step of 128
  samples to ~0.6-0.9 s and one epoch to ~2-3 min, making the full grid feasible.
- **Why `text_span_dropout`?** Without it the fusion model can ignore audio and
  still reach 100% through the transcript. Dropout forces the cross-attention to
  be useful; text-missing evaluation then measures the actual fallback.
- **Why text shuffle is implemented with `torch.roll`?** The legacy code
  permuted texts within a batch, so a sample could occasionally receive its own
  transcript. Rolling by one guarantees that no sample sees its own text.
- **Why no sklearn?** The lab requires the algorithm and metrics to be
  implemented manually; all metric code is in `src/fusion_slu/metrics.py`.

## Project layout

```
src/fusion_slu/
  config.py          dataclass config, YAML extends, --set overrides
  data/              manifest, wav reading, log-mel, vocab, feature cache, dataset
  models/            attention, encoders, fusion heads
  metrics.py         confusion / PRF / exact match / ECE
  engine.py          inference + attention collection
  train.py           training loop, scheduler, early stopping, checkpoints
  evaluate.py        clean + robustness evaluation, calibration, predictions CSV
  analysis.py        aggregation over seeds, error analysis, figures
  viz/               matplotlib plotting module
  report/            Jinja2 template and builder
  cli.py             prepare / train / evaluate / analyze / report / all
configs/             default.yaml and smoke.yaml
scripts/             download_data.py and run_all.py
tests/               pytest suite
```

The original unrefactored scripts remain in git history (they were removed in
this refactor); the full defect list with fixes is in `theory.md`, section 11.

## Dataset licence

Fluent Speech Commands Public License; see the PDF shipped with the dataset.
The dataset is not redistributed in this repository.
