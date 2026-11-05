from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from fusion_slu.config import Condition
from fusion_slu.data.dataset import apply_text_condition
from fusion_slu.models.fusion import FusionSLU


@dataclass
class Prediction:
    loss: float
    pred: np.ndarray
    true: np.ndarray
    confidence: np.ndarray
    indices: np.ndarray

    def subset(self, mask: np.ndarray) -> Prediction:
        return Prediction(
            loss=self.loss,
            pred=self.pred[mask],
            true=self.true[mask],
            confidence=self.confidence[mask],
            indices=self.indices[mask],
        )


def run_inference(
    model: FusionSLU,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    condition: Condition | None = None,
    vocab: dict[str, int] | None = None,
    seed: int = 0,
) -> Prediction:
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(seed)
    total_loss = 0.0
    total_rows = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            text = batch["text"]
            if condition is not None and condition.kind not in {"clean", "audio_missing"}:
                if vocab is None:
                    raise ValueError("a vocabulary is required for text corruption")
                text = apply_text_condition(batch["text"], condition, vocab, generator)
            audio = batch["audio"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            labels = batch["labels"].to(device)
            text = text.to(device)
            output = model(audio, audio_mask, text, labels)
            if output.loss is None:
                raise RuntimeError("model did not return a loss")
            rows = int(labels.shape[0])
            total_loss += float(output.loss.item()) * rows
            total_rows += rows
            predictions.append(output.pred.cpu().numpy())
            targets.append(labels.cpu().numpy())
            confidence = torch.stack([prob.max(dim=-1).values for prob in output.probs], dim=1)
            confidences.append(confidence.cpu().numpy())
            indices.append(batch["index"].cpu().numpy())
    if total_rows == 0:
        raise RuntimeError("loader produced no batches")
    return Prediction(
        loss=total_loss / total_rows,
        pred=np.concatenate(predictions, axis=0),
        true=np.concatenate(targets, axis=0),
        confidence=np.concatenate(confidences, axis=0),
        indices=np.concatenate(indices, axis=0),
    )


def collect_attention(
    model: FusionSLU,
    dataset: Any,
    example_indices: list[int],
    device: torch.device,
    *,
    pad_id: int,
) -> list[dict[str, Any]]:
    from fusion_slu.data.dataset import collate_batch

    results: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for index in example_indices:
            item = dataset[index]
            batch = collate_batch([item], pad_id=pad_id)
            audio = batch["audio"].to(device)
            audio_mask = batch["audio_mask"].to(device)
            text = batch["text"].to(device)
            labels = batch["labels"].to(device)
            output = model(audio, audio_mask, text, labels, return_attention=True)
            if output.attention is None:
                continue
            token_ids = text[0].tolist()
            audio_length = int(audio_mask[0].sum().item())
            results.append(
                {
                    "index": int(index),
                    "tokens": token_ids,
                    "attention": output.attention[0].detach().cpu().numpy().tolist(),
                    "audio_frames": audio_length,
                    "pred": output.pred[0].detach().cpu().numpy().tolist(),
                    "true": labels[0].detach().cpu().numpy().tolist(),
                }
            )
    return results
