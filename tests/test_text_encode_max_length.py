"""Tests for the shared DEFAULT_MAX_TEXT_LENGTH default in text_encode.py (v0.10.0).

Generation previously hardcoded max_length=512 while training tokenized
captions to 32 -- a silent mismatch. ``_tokenize_and_encode`` (shared by both
``FluxFlowTextEncode`` and ``FluxFlowTextEncodeNegative``) now reads the same
constant training reads (via fluxflow.text_length), so this call site can't
diverge from it again.
"""

from unittest.mock import MagicMock

import torch

from comfyui_fluxflow.nodes.text_encode import (
    DEFAULT_MAX_TEXT_LENGTH,
    FluxFlowTextEncode,
    FluxFlowTextEncodeNegative,
    _tokenize_and_encode,
)


def _make_tokenizer_and_encoder(seq_len: int = DEFAULT_MAX_TEXT_LENGTH):
    tok_out = {"input_ids": torch.zeros(1, seq_len, dtype=torch.long)}
    tokenizer = MagicMock(return_value=tok_out)
    tokenizer.pad_token_id = 0

    text_encoder = MagicMock()
    text_encoder.parameters.return_value = iter([torch.zeros(1)])
    text_encoder.return_value = (
        torch.zeros(1, seq_len, 4),
        torch.ones(1, seq_len, dtype=torch.bool),
    )
    return tokenizer, text_encoder


def test_default_is_32():
    assert DEFAULT_MAX_TEXT_LENGTH == 32


def test_tokenize_and_encode_uses_shared_default():
    tokenizer, text_encoder = _make_tokenizer_and_encoder()
    _tokenize_and_encode(text_encoder, tokenizer, "a prompt")
    _, kwargs = tokenizer.call_args
    assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH


def test_positive_node_encode_uses_shared_default():
    tokenizer, text_encoder = _make_tokenizer_and_encoder()
    node = FluxFlowTextEncode()
    node.encode(text_encoder, tokenizer, "a prompt")
    _, kwargs = tokenizer.call_args
    assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH


def test_negative_node_encode_uses_shared_default():
    tokenizer, text_encoder = _make_tokenizer_and_encoder()
    node = FluxFlowTextEncodeNegative()
    node.encode(text_encoder, tokenizer, "a negative prompt")
    _, kwargs = tokenizer.call_args
    assert kwargs["max_length"] == DEFAULT_MAX_TEXT_LENGTH
