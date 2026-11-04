import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, hid: int, drop: float, max_len: int) -> None:
        super().__init__()
        assert hid % 2 == 0
        self.drop = nn.Dropout(drop)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(-torch.arange(0, hid, 2) * math.log(10000.0) / hid)
        pe = torch.zeros(1, max_len, hid)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, len, hid]
        return self.drop(x + self.pe[:, : x.size(1)])


class Attention(nn.Module):
    def __init__(self, hid: int, heads: int, drop: float) -> None:
        super().__init__()
        assert hid % heads == 0
        self.heads = heads
        self.dim = hid // heads
        self.scale = math.sqrt(self.dim)
        self.q = nn.Linear(hid, hid)
        self.k = nn.Linear(hid, hid)
        self.v = nn.Linear(hid, hid)
        self.o = nn.Linear(hid, hid)
        self.drop = nn.Dropout(drop)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # q: [batch, q_len, hid], k and v: [batch, k_len, hid]
        # mask: bool [batch, 1, 1, k_len], True = keep
        b = q.shape[0]
        Q = self.q(q).view(b, -1, self.heads, self.dim).transpose(1, 2)
        K = self.k(k).view(b, -1, self.heads, self.dim).transpose(1, 2)
        V = self.v(v).view(b, -1, self.heads, self.dim).transpose(1, 2)

        score = Q @ K.transpose(-1, -2) / self.scale  # [batch, heads, q_len, k_len]
        if mask is not None:
            score = score.masked_fill(~mask, -1e10)
        alpha = torch.softmax(score, dim=-1)

        x = self.drop(alpha) @ V  # [batch, heads, q_len, dim]
        x = x.transpose(1, 2).reshape(b, -1, self.heads * self.dim)
        return self.o(x), alpha


class Layer(nn.Module):
    """Attention + add&norm + feed forward + add&norm.

    Self-attention when kv is None, cross-attention otherwise (x is the query, kv is key and value).
    """

    def __init__(self, hid: int, heads: int, ff: int, drop: float) -> None:
        super().__init__()
        self.attn = Attention(hid, heads, drop)
        self.norm1 = nn.LayerNorm(hid)
        self.norm2 = nn.LayerNorm(hid)
        self.ff = nn.Sequential(nn.Linear(hid, ff), nn.ReLU(), nn.Dropout(drop), nn.Linear(ff, hid))
        self.drop = nn.Dropout(drop)

    def forward(
        self, x: torch.Tensor, kv: torch.Tensor | None = None, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv = x if kv is None else kv
        out, alpha = self.attn(x, kv, kv, mask)
        x = self.norm1(x + self.drop(out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x, alpha


class AudioEncoder(nn.Module):
    """Raw waveform is cut into chunks of audio_split_samples, each chunk is one token."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        hid = cfg["hid_dim"]
        self.split = cfg["audio_split_samples"]
        self.scale = math.sqrt(hid)
        self.proj = nn.Linear(self.split, hid)
        self.pos = PositionalEncoding(hid, cfg["dropout"], cfg["max_len"])
        self.layers = nn.ModuleList(
            [Layer(hid, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"]) for _ in range(cfg["audio_layers"])]
        )

    def forward(self, audio: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # audio: [batch, n_chunks * split], mask: bool [batch, n_chunks]
        x = audio.reshape(len(audio), -1, self.split)
        x = self.pos(self.proj(x) * self.scale)
        m = mask[:, None, None, :]
        for layer in self.layers:
            x, _ = layer(x, mask=m)
        return x


class TextEncoder(nn.Module):
    def __init__(self, cfg: dict, vocab_size: int, pad_id: int) -> None:
        super().__init__()
        hid = cfg["hid_dim"]
        self.pad_id = pad_id
        self.scale = math.sqrt(hid)
        self.emb = nn.Embedding(vocab_size, hid)
        self.pos = PositionalEncoding(hid, cfg["dropout"], cfg["max_len"])
        self.layers = nn.ModuleList(
            [Layer(hid, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"]) for _ in range(cfg["text_layers"])]
        )

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # text: [batch, len]
        m = (text != self.pad_id)[:, None, None, :]
        x = self.pos(self.emb(text) * self.scale)
        for layer in self.layers:
            x, _ = layer(x, mask=m)
        return x


class Model(nn.Module):
    """modality 'fusion': text tokens (query) attend to audio chunks (key, value), the <sos> token is classified.
    modality 'text' and 'audio': unimodal baselines with the same encoders and heads."""

    def __init__(self, cfg: dict, vocab_size: int, pad_id: int, dims: list[int]) -> None:
        super().__init__()
        hid = cfg["hid_dim"]
        self.mode = cfg["modality"]
        self.dims = dims
        self.weighted = cfg["loss_mode"] == "weighted_loss"
        if self.mode in ("fusion", "audio"):
            self.audio_enc = AudioEncoder(cfg)
        if self.mode in ("fusion", "text"):
            self.text_enc = TextEncoder(cfg, vocab_size, pad_id)
        if self.mode == "fusion":
            self.cross = nn.ModuleList(
                [Layer(hid, cfg["n_heads"], cfg["ff_dim"], cfg["dropout"]) for _ in range(cfg["cross_layers"])]
            )
        self.heads = nn.ModuleList([nn.Linear(hid, d) for d in dims])

    def forward(
        self,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
        text: torch.Tensor,
        labels: torch.Tensor,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        # labels: [batch, 3] (action, object, position)
        attn = None
        if self.mode == "audio":
            a = self.audio_enc(audio, audio_mask)
            m = audio_mask.unsqueeze(-1).float()
            pooled = (a * m).sum(1) / m.sum(1)
        else:
            t = self.text_enc(text)
            if self.mode == "fusion":
                a = self.audio_enc(audio, audio_mask)
                for layer in self.cross:
                    t, attn = layer(t, a, audio_mask[:, None, None, :])
            pooled = t[:, 0]

        logits = [h(pooled) for h in self.heads]
        losses = torch.stack([F.cross_entropy(lg, labels[:, i]) for i, lg in enumerate(logits)])
        if self.weighted:
            # heads with more classes get more weight, otherwise small heads shrink the loss
            w = torch.tensor(self.dims, dtype=torch.float, device=losses.device)
            loss = (w * losses).sum() / w.sum()
        else:
            loss = losses.mean()

        out = {"loss": loss, "pred": torch.stack([lg.argmax(-1) for lg in logits], dim=1)}
        if return_attn and attn is not None:
            out["attn"] = attn.mean(1)  # [batch, text_len, n_chunks], mean over heads
        return out


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, 0.0, m.embedding_dim**-0.5)


def build_model(cfg: dict, vocabs: dict[str, dict[str, int]]) -> Model:
    dims = [len(vocabs[k]) for k in ("action", "object", "position")]
    model = Model(cfg, len(vocabs["text"]), vocabs["text"]["<pad>"], dims)
    model.apply(init_weights)
    return model
