from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from fusion_slu.config import AudioConfig
from fusion_slu.data.audio import (
    add_white_noise,
    augment_spectrogram,
    log_mel_spectrogram,
    mel_filterbank,
    read_wav,
)


def _write_sine(path: Path, seconds: float = 0.5, sample_rate: int = 16000) -> None:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    samples = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def test_read_wav_returns_mono_float(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    _write_sine(path)
    samples, sample_rate = read_wav(path)
    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.shape == (8000,)
    assert np.max(np.abs(samples)) <= 1.0


def test_log_mel_shape_and_finiteness() -> None:
    config = AudioConfig()
    samples = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    feature = log_mel_spectrogram(samples, sample_rate=16000, config=config, max_frames=800)
    assert feature.shape[1] == config.n_mels
    assert feature.shape[0] <= 800
    assert np.isfinite(feature).all()
    assert float(np.abs(feature.mean())) < 0.1


def test_mel_filterbank_shape() -> None:
    filters = mel_filterbank(16000, 400, 40, 20.0, 8000.0)
    assert filters.shape == (40, 201)
    assert np.all(filters >= 0)


def test_noise_and_augmentation() -> None:
    rng = np.random.default_rng(0)
    samples = np.random.default_rng(1).normal(0, 0.1, 1600).astype(np.float32)
    noisy = add_white_noise(samples, 5.0, rng)
    assert noisy.shape == samples.shape
    feature = np.zeros((100, 40), dtype=np.float32)
    augmented = augment_spectrogram(feature, AudioConfig(freq_masks=1, time_masks=1), rng)
    assert augmented.shape == feature.shape
    assert np.count_nonzero(augmented) < augmented.size


def test_log_mel_rejects_silence_gracefully() -> None:
    config = AudioConfig()
    feature = log_mel_spectrogram(
        np.zeros(1000, dtype=np.float32),
        sample_rate=16000,
        config=config,
        max_frames=800,
    )
    assert np.isfinite(feature).all()
    assert feature.shape[0] >= 1
