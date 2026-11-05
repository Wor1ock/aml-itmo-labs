from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Sequence

from fusion_slu.config import AudioConfig, ModelConfig
from fusion_slu.models.attention import TransformerBlock
from fusion_slu.models.encoders import AudioEncoder, TextEncoder
from fusion_slu.utils import masked_mean


@dataclass
class ModelOutput:
    loss: torch.Tensor | None
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    probs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    pred: torch.Tensor
    attention: torch.Tensor | None


class FusionSLU(nn.Module):
    """Three-head spoken language understanding model.

    ``fusion`` uses cross-attention where transcript tokens query audio tokens.
    ``late`` concatenates pooled audio and text representations before the heads.
    ``audio`` and ``text`` are unimodal baselines with the same encoders and heads.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        audio_config: AudioConfig,
        *,
        vocab_size: int,
        pad_id: int,
        dims: tuple[int, int, int],
        head_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    ) -> None:
        super().__init__()
        self.modality = model_config.modality
        self.dims = dims
        self.head_weights = tuple(head_weights)
        hidden = model_config.hid_dim
        if self.modality not in {"fusion", "text", "audio", "late"}:
            raise ValueError(f"unknown modality {self.modality!r}")
        if self.modality in {"fusion", "audio", "late"}:
            self.audio_encoder = AudioEncoder(audio_config, model_config)
        if self.modality in {"fusion", "text", "late"}:
            self.text_encoder = TextEncoder(model_config, vocab_size, pad_id)
        if self.modality == "fusion":
            self.cross_attention = nn.ModuleList(
                [
                    TransformerBlock(hidden, model_config.n_heads, model_config.ff_dim, model_config.dropout)
                    for _ in range(model_config.cross_layers)
                ]
            )
        head_input = hidden * 2 if self.modality == "late" else hidden
        self.heads = nn.ModuleList([nn.Linear(head_input, dim) for dim in dims])

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        text: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_attention: bool = False,
    ) -> ModelOutput:
        attention: torch.Tensor | None = None
        if self.modality == "audio":
            audio_tokens, audio_valid = self.audio_encoder(audio, audio_mask)
            pooled = masked_mean(audio_tokens, audio_valid)
        elif self.modality == "text":
            text_tokens, _ = self.text_encoder(text)
            pooled = text_tokens[:, 0]
        elif self.modality == "late":
            audio_tokens, audio_valid = self.audio_encoder(audio, audio_mask)
            text_tokens, _ = self.text_encoder(text)
            pooled = torch.cat([masked_mean(audio_tokens, audio_valid), text_tokens[:, 0]], dim=-1)
        else:
            text_tokens, _ = self.text_encoder(text)
            audio_tokens, audio_valid = self.audio_encoder(audio, audio_mask)
            for block in self.cross_attention:
                text_tokens, attention = block(
                    text_tokens,
                    key_value=audio_tokens,
                    valid_mask=audio_valid,
                )
            pooled = text_tokens[:, 0]
            if not return_attention:
                attention = None

        logits = tuple(head(pooled) for head in self.heads)
        probs = tuple(torch.softmax(logit, dim=-1) for logit in logits)
        pred = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
        loss: torch.Tensor | None = None
        if labels is not None:
            losses = torch.stack(
                [
                    F.cross_entropy(logit, labels[:, head])
                    for head, logit in enumerate(logits)
                ]
            )
            weights = torch.tensor(self.head_weights, dtype=losses.dtype, device=losses.device)
            loss = (losses * weights).sum() / weights.sum()
        return ModelOutput(
            loss=loss,
            logits=(logits[0], logits[1], logits[2]),
            probs=(probs[0], probs[1], probs[2]),
            pred=pred,
            attention=attention,
        )


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=module.embedding_dim**-0.5)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].fill_(0.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def build_model(
    model_config: ModelConfig,
    audio_config: AudioConfig,
    *,
    vocab_size: int,
    pad_id: int,
    dims: tuple[int, int, int],
    head_weights: Sequence[float] = (1.0, 1.0, 1.0),
) -> FusionSLU:
    model = FusionSLU(
        model_config,
        audio_config,
        vocab_size=vocab_size,
        pad_id=pad_id,
        dims=dims,
        head_weights=tuple(head_weights),
    )
    model.apply(init_weights)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
