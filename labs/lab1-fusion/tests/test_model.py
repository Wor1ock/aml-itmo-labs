from __future__ import annotations

import numpy as np
import torch

from fusion_slu.config import AudioConfig, ModelConfig
from fusion_slu.models.fusion import build_model


def _batch(batch: int = 2, frames: int = 64, text_len: int = 6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    audio = torch.randn(batch, frames, 40)
    audio_mask = torch.ones(batch, frames, dtype=torch.bool)
    audio_mask[1, 40:] = False
    text = torch.tensor([[1, 4, 5, 2, 0, 0], [1, 6, 2, 0, 0, 0]][:batch], dtype=torch.long)
    labels = torch.tensor([[0, 1, 2], [1, 2, 3]][:batch], dtype=torch.long)
    return audio, audio_mask, text, labels


def _model(modality: str):  # type: ignore[no-untyped-def]
    audio = AudioConfig(n_mels=40, n_fft=400, hop_length=160, win_length=400, spec_augment=False)
    model_config = ModelConfig(
        modality=modality,
        hid_dim=32,
        audio_layers=1,
        text_layers=1,
        cross_layers=1,
        n_heads=4,
        ff_dim=64,
        dropout=0.0,
        attn_dropout=0.0,
        text_token_dropout=0.0,
        text_span_dropout=0.0,
        max_len=64,
    )
    return build_model(
        model_config,
        audio,
        vocab_size=20,
        pad_id=0,
        dims=(6, 14, 4),
        head_weights=(1.0, 1.0, 1.0),
    )


def test_all_modalities_produce_logits_and_loss() -> None:
    audio, mask, text, labels = _batch()
    for modality in ("fusion", "text", "audio", "late"):
        model = _model(modality)
        output = model(audio, mask, text, labels)
        assert output.loss is not None
        assert torch.isfinite(output.loss)
        assert output.pred.shape == (2, 3)
        assert [tuple(logit.shape) for logit in output.logits] == [(2, 6), (2, 14), (2, 4)]
        total_mass = sum(prob.sum(dim=-1).mean().item() for prob in output.probs)
    assert abs(total_mass - 3.0) < 1e-4


def test_attention_is_masked_and_normalised() -> None:
    audio, mask, text, labels = _batch()
    model = _model("fusion")
    output = model(audio, mask, text, labels, return_attention=True)
    assert output.attention is not None
    assert output.attention.shape[0] == 2
    assert output.attention.shape[1] == text.shape[1]
    sums = output.attention.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)
    # The second utterance is padded after frame 40; attention on padded frames is zero.
    truncated = output.attention[1, :, 5:]
    assert torch.allclose(truncated, torch.zeros_like(truncated), atol=1e-6)


def test_text_missing_still_runs() -> None:
    audio, mask, _, labels = _batch()
    text = torch.tensor([[1, 2, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0]], dtype=torch.long)
    model = _model("fusion")
    output = model(audio, mask, text, labels)
    assert output.pred.shape == (2, 3)
    assert output.loss is not None
    assert bool(np.isfinite(output.loss.detach().item()))
