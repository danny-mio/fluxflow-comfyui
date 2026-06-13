"""
FluxFlow Text Encoding Nodes for ComfyUI.

Encodes text prompts into the v0.10.0 per-token format ``FLUXFLOW_TEXT``:
a tuple ``(text_seq, text_mask)`` passed as a single ComfyUI connection.

This is a compat-break vs v0.8.x, where the node emitted a pooled
``FLUXFLOW_CONDITIONING`` tensor. Old workflows referencing
``FLUXFLOW_CONDITIONING`` will fail with a clear type mismatch at load time.
"""

import torch


def _tokenize_and_encode(text_encoder, tokenizer, text):
    """Shared tokenization + encoding helper.

    Returns:
        tuple ``(text_seq, text_mask)`` where ``text_seq`` is per-token
        embeddings ``[B, T_txt, E]`` and ``text_mask`` is a bool mask
        ``[B, T_txt]`` (True for valid tokens).
    """
    encodings = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    device = next(text_encoder.parameters()).device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        text_seq, text_mask = text_encoder(input_ids, attention_mask=attention_mask)

    return text_seq, text_mask


class FluxFlowTextEncode:
    """Encode positive text prompt to FluxFlow ``FLUXFLOW_TEXT``."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("FLUXFLOW_TEXT_ENCODER",),
                "tokenizer": ("FLUXFLOW_TOKENIZER",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("FLUXFLOW_TEXT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "encode"
    CATEGORY = "FluxFlow/conditioning"

    def encode(self, text_encoder, tokenizer, text):
        """Encode text to ``(text_seq, text_mask)``.

        Args:
            text_encoder: BertTextEncoder model.
            tokenizer: HuggingFace tokenizer.
            text: Text prompt.

        Returns:
            A single-element tuple ``((text_seq, text_mask),)`` matching the
            ``FLUXFLOW_TEXT`` return type.
        """
        text_seq, text_mask = _tokenize_and_encode(text_encoder, tokenizer, text)
        print(
            f"Encoded text: '{text[:50]}...' to text_seq shape {text_seq.shape}, "
            f"text_mask shape {text_mask.shape}"
        )
        return ((text_seq, text_mask),)


class FluxFlowTextEncodeNegative:
    """Encode negative text prompt for Classifier-Free Guidance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("FLUXFLOW_TEXT_ENCODER",),
                "tokenizer": ("FLUXFLOW_TOKENIZER",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("FLUXFLOW_TEXT",)
    RETURN_NAMES = ("negative_text",)
    FUNCTION = "encode"
    CATEGORY = "FluxFlow/conditioning"

    def encode(self, text_encoder, tokenizer, text):
        """Encode negative text to ``(text_seq, text_mask)``.

        Args:
            text_encoder: BertTextEncoder model.
            tokenizer: HuggingFace tokenizer.
            text: Negative text prompt.

        Returns:
            A single-element tuple ``((text_seq, text_mask),)`` matching the
            ``FLUXFLOW_TEXT`` return type.
        """
        text_seq, text_mask = _tokenize_and_encode(text_encoder, tokenizer, text)
        print(
            f"Encoded negative text: '{text[:50]}...' to text_seq shape {text_seq.shape}, "
            f"text_mask shape {text_mask.shape}"
        )
        return ((text_seq, text_mask),)


NODE_CLASS_MAPPINGS = {
    "FluxFlowTextEncode": FluxFlowTextEncode,
    "FluxFlowTextEncodeNegative": FluxFlowTextEncodeNegative,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFlowTextEncode": "FluxFlow Text Encode",
    "FluxFlowTextEncodeNegative": "FluxFlow Text Encode (Negative)",
}
