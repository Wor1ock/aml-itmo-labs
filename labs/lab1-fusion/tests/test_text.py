from __future__ import annotations

import random

from fusion_slu.data.text import (
    EOS_TOKEN,
    PAD_TOKEN,
    SOS_TOKEN,
    UNK_TOKEN,
    apply_token_dropout,
    build_vocab,
    decode_text,
    encode_text,
    tokenize,
)


def test_tokenize_handles_punctuation_and_case() -> None:
    assert tokenize("Turn on the lights, please!") == ["turn", "on", "the", "lights", ",", "please", "!"]
    assert tokenize("Change language") == ["change", "language"]


def test_vocab_and_encode_decode_roundtrip() -> None:
    vocab = build_vocab(["Turn on the lights", "Turn off the lights"])
    ids = encode_text("Turn on the lights", vocab, max_len=16)
    assert ids[0] == vocab[SOS_TOKEN]
    assert ids[-1] == vocab[EOS_TOKEN]
    assert decode_text(ids, vocab) == "turn on the lights"


def test_unknown_and_truncation() -> None:
    vocab = build_vocab(["turn on"])
    ids = encode_text("turn on totally unknown words", vocab, max_len=5)
    assert len(ids) == 5
    assert vocab[UNK_TOKEN] in ids


def test_token_dropout_preserves_special_tokens() -> None:
    vocab = build_vocab(["turn on the lights"])
    ids = encode_text("turn on the lights", vocab, max_len=16)
    dropped = apply_token_dropout(ids, 1.0, vocab, random.Random(0))
    assert dropped[0] == vocab[SOS_TOKEN]
    assert dropped[-1] == vocab[EOS_TOKEN]
    assert all(token == vocab[UNK_TOKEN] for token in dropped[1:-1])
    assert vocab[PAD_TOKEN] == 0
