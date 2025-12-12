"""
FluxFlow Text Encoding Node for ComfyUI.

Encodes text prompts to conditioning embeddings.
"""

import torch


class FluxFlowTextEncode:
    """Encode text prompt to FluxFlow conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("FLUXFLOW_TEXT_ENCODER",),
                "tokenizer": ("FLUXFLOW_TOKENIZER",),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("FLUXFLOW_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "FluxFlow/conditioning"

    def encode(self, text_encoder, tokenizer, text):
        """
        Encode text to conditioning embeddings.

        Args:
            text_encoder: BertTextEncoder model
            tokenizer: HuggingFace tokenizer
            text: Text prompt

        Returns:
            (conditioning,) - Text embeddings [1, D]
        """
        # Tokenize text
        encodings = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Move to model device
        device = next(text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Encode
        with torch.no_grad():
            conditioning = text_encoder(input_ids, attention_mask=attention_mask)

        print(f"Encoded text: '{text[:50]}...' to conditioning shape {conditioning.shape}")

        return (conditioning,)


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

    RETURN_TYPES = ("FLUXFLOW_CONDITIONING",)
    RETURN_NAMES = ("negative_conditioning",)
    FUNCTION = "encode"
    CATEGORY = "FluxFlow/conditioning"

    def encode(self, text_encoder, tokenizer, text):
        """
        Encode negative text to conditioning embeddings.

        Args:
            text_encoder: BertTextEncoder model
            tokenizer: HuggingFace tokenizer
            text: Negative text prompt

        Returns:
            (negative_conditioning,) - Negative text embeddings [1, D]
        """
        # Tokenize text
        encodings = tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"]
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        # Move to model device
        device = next(text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Encode
        with torch.no_grad():
            conditioning = text_encoder(input_ids, attention_mask=attention_mask)

        print(f"Encoded negative text: '{text[:50]}...' to conditioning shape {conditioning.shape}")

        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "FluxFlowTextEncode": FluxFlowTextEncode,
    "FluxFlowTextEncodeNegative": FluxFlowTextEncodeNegative,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFlowTextEncode": "FluxFlow Text Encode",
    "FluxFlowTextEncodeNegative": "FluxFlow Text Encode (Negative)",
}
