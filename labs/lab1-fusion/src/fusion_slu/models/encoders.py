from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_slu.config import AudioConfig, ModelConfig
from fusion_slu.models.attention import FastDropout, PositionalEncoding, TransformerBlock
from fusion_slu.utils import masked_mean


class AudioEncoder(nn.Module):
    """Log-mel frames -> strided convolutions -> self-attention tokens."""

    def __init__(self, audio_config: AudioConfig, model_config: ModelConfig) -> None:
        super().__init__()
        hidden = model_config.hid_dim
        self.input_norm = nn.LayerNorm(audio_config.n_mels)
        layers: list[nn.Module] = []
        channels = audio_config.n_mels
        for index, (stride, kernel) in enumerate(
            zip(audio_config.conv_strides, audio_config.conv_kernels, strict=True)
        ):
            layers.append(
                nn.Conv1d(
                    channels,
                    hidden,
                    kernel_size=kernel,
                    stride=stride,
                    padding=kernel // 2,
                )
            )
            layers.append(nn.GELU())
            if index == 0:
                layers.append(FastDropout(model_config.dropout))
            channels = hidden
        self.convolution = nn.Sequential(*layers)
        self.position = PositionalEncoding(hidden, model_config.dropout, model_config.max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, model_config.n_heads, model_config.ff_dim, model_config.dropout)
                for _ in range(model_config.audio_layers)
            ]
        )
        self.strides = tuple(audio_config.conv_strides)
        self.kernels = tuple(audio_config.conv_kernels)

    def _lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        result = lengths
        for stride, kernel in zip(self.strides, self.kernels, strict=True):
            result = torch.div(result + 2 * (kernel // 2) - kernel, stride, rounding_mode="floor") + 1
        return result.clamp_min(1)

    def forward(self, audio: torch.Tensor, audio_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # audio: [batch, frames, mels], audio_mask: [batch, frames]
        x = self.input_norm(audio).transpose(1, 2)
        x = self.convolution(x).transpose(1, 2)
        lengths = self._lengths(audio_mask.sum(dim=1))
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        valid = positions < lengths.unsqueeze(1)
        x = self.position(x)
        for block in self.blocks:
            x, _ = block(x, valid_mask=valid)
        return x, valid


class TextEncoder(nn.Module):
    def __init__(self, model_config: ModelConfig, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        hidden = model_config.hid_dim
        self.pad_id = pad_id
        self.scale = hidden**0.5
        self.embedding = nn.Embedding(vocab_size, hidden, padding_idx=pad_id)
        self.position = PositionalEncoding(hidden, model_config.dropout, model_config.max_len)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden, model_config.n_heads, model_config.ff_dim, model_config.dropout)
                for _ in range(model_config.text_layers)
            ]
        )

    def forward(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = text != self.pad_id
        x = self.position(self.embedding(text) * self.scale)
        for block in self.blocks:
            x, _ = block(x, valid_mask=valid)
        return x, valid


def pool_text(x: torch.Tensor) -> torch.Tensor:
    # The <sos> token is always at position zero and aggregates the sequence.
    return x[:, 0]
