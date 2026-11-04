# Audio-text fusion with cross-attention

Classify the action, object and position of a spoken command from the audio and its transcript.
The two modalities are fused with cross-attention: text tokens are the query, audio chunks are the key and value.

## Data

Expected layout (Fluent Speech Commands style):

```
data/
  train_data.csv
  valid_data.csv
  test_data.csv
  wavs/...            # 16-bit mono wav files
```

Each csv has the columns `path, transcription, action, object, position`; `path` is relative to `data.wavs_location` in `config.yaml`.
The data is not stored in the repository.

## Model

- audio encoder: the waveform is cut into chunks of `audio_split_samples`, each chunk is projected to `hid_dim` and goes through self-attention layers
- text encoder: token embedding plus self-attention layers
- fusion: cross-attention layers, query is text, key and value are audio chunks (padded chunks are masked); the `<sos>` token is classified by three linear heads
- baselines: `--modality text` and `--modality audio` use the same encoders and heads without fusion

Metrics (accuracy, weighted f1) are implemented in `utils.py` without sklearn.

## Run with uv

```
uv sync
uv run python train.py --config config.yaml --modality fusion --seed 0
uv run python eval.py --config config.yaml --modality fusion --seed 0
uv run sh run_all.sh        # all modalities, seeds 0-2, then report.py
uv run mypy .               # type check
uv run tensorboard --logdir results
```

## Run with docker

```
uv lock                      # once, commit uv.lock
docker compose build
docker compose run --rm lab                      # run_all.sh
docker compose run --rm lab python train.py --config config.yaml --modality fusion --seed 0
```

## Outputs

Everything is written to `results/`.

- `<modality>_seed<N>/`: `train.log`, `eval.log`, `model.pt`, `history.json`, `metrics.json`, `tb/`, `curves.png` (loss and accuracy per epoch), `confusion.png` (test confusion matrices), `attention.png` (cross-attention maps, fusion only)
- `results.md`: test metrics as mean ± std over seeds, the majority-class baseline, the fusion ablation
- `summary.png`: weighted f1 per head for every model
- `data.png`: class counts and audio lengths of the train set

The fusion ablation evaluates the same checkpoint with the audio replaced by noise (`audio_noise`) and with texts shuffled inside a batch (`text_shuffle`); it shows which modality the model relies on.
