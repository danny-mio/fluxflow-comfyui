"""
FluxFlow Model Loader Node for ComfyUI.

Automatically detects model configuration from checkpoint and initializes all components.
"""

import logging

import safetensors.torch

# Import from installed fluxflow package
from fluxflow.models import (
    BertTextEncoder,
    FluxCompressor,
    FluxExpander,
    FluxFlowProcessor,
    FluxPipeline,
)
from transformers import AutoTokenizer

from comfyui_fluxflow.model_inspector import get_model_info
from comfyui_fluxflow.nodes.utils import parse_device

logger = logging.getLogger(__name__)


class FluxFlowModelLoader:
    """
    Load FluxFlow checkpoint with automatic configuration detection.

    Automatically detects:
    - VAE latent dimensions
    - Flow model dimensions
    - Text embedding dimensions
    - Architecture parameters (downscales, attention layers, etc.)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "checkpoint_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "placeholder": "/path/to/fluxflow_checkpoint.safetensors",
                    },
                ),
            },
            "optional": {
                "device": (
                    ["auto", "cuda", "cpu", "mps"],
                    {"default": "auto"},
                ),
                "tokenizer_name": (
                    "STRING",
                    {
                        "default": "distilbert-base-uncased",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "FLUXFLOW_MODEL",
        "FLUXFLOW_TEXT_ENCODER",
        "FLUXFLOW_TOKENIZER",
        "STRING",
    )
    RETURN_NAMES = ("model", "text_encoder", "tokenizer", "config_info")
    FUNCTION = "load_model"
    CATEGORY = "FluxFlow"

    def load_model(
        self,
        checkpoint_path: str,
        device: str = "auto",
        tokenizer_name: str = "distilbert-base-uncased",
    ):
        """
        Load FluxFlow model from checkpoint.

        Args:
            checkpoint_path: Path to .safetensors checkpoint
            device: Device to load model on (auto, cuda, cpu, mps)
            tokenizer_name: HuggingFace tokenizer name

        Returns:
            (model, text_encoder, tokenizer, config_info)
        """
        logger.info("=" * 60)
        logger.info("FluxFlow Model Loader")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {checkpoint_path}")

        # Auto-detect configuration
        logger.info("Detecting model configuration...")
        config = get_model_info(checkpoint_path, verbose=True)

        # Parse device
        device_obj = parse_device(device)
        logger.info(f"Device: {device_obj}")

        # Initialize models with detected configuration
        logger.info("Initializing models...")

        compressor = FluxCompressor(
            in_channels=3,
            d_model=config["vae_dim"],
            downscales=config["downscales"],
            max_hw=config["max_hw"],
            use_attention=True,
            attn_layers=config.get("vae_attn_layers", 2),
        )

        flow_processor = FluxFlowProcessor(
            d_model=config["flow_dim"],
            vae_dim=config["vae_dim"],
            embedding_size=config["text_embed_dim"],
            n_head=config.get("flow_attn_heads", 8),
            n_layers=config.get("flow_transformer_layers", 10),
            max_hw=config["max_hw"],
        )

        expander = FluxExpander(
            d_model=config["vae_dim"],
            upscales=config["upscales"],
            max_hw=config["max_hw"],
        )

        diffuser = FluxPipeline(compressor, flow_processor, expander)

        text_encoder = BertTextEncoder(embed_dim=config["text_embed_dim"])

        # Load checkpoint weights
        logger.info("Loading checkpoint weights...")
        state_dict = safetensors.torch.load_file(checkpoint_path)

        # Load diffuser state (filter out size mismatches for buffers)
        diffuser_state = {
            k.replace("diffuser.", ""): v
            for k, v in state_dict.items()
            if k.startswith("diffuser.")
        }

        # Load with strict=False and handle mismatches gracefully
        missing_keys, unexpected_keys = diffuser.load_state_dict(diffuser_state, strict=False)

        # Log any issues for debugging
        if missing_keys:
            logger.debug(f"{len(missing_keys)} keys not found in checkpoint (using random init)")
        if unexpected_keys:
            logger.debug(f"{len(unexpected_keys)} unexpected keys in checkpoint (ignored)")

        # Load text encoder state
        text_encoder_state = {
            k.replace("text_encoder.", ""): v
            for k, v in state_dict.items()
            if k.startswith("text_encoder.")
        }
        text_encoder.load_state_dict(text_encoder_state, strict=False)

        # Move to device
        diffuser = diffuser.to(device_obj)
        text_encoder = text_encoder.to(device_obj)

        # Set to eval mode
        diffuser.eval()
        text_encoder.eval()

        # Load tokenizer
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir="./_cache", local_files_only=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Create config info string
        config_info = (
            f"VAE: {config['vae_dim']}d, "
            f"Flow: {config['flow_dim']}d, "
            f"Text: {config['text_embed_dim']}d, "
            f"Compression: {config['compression_ratio']}x"
        )

        logger.info("Model loaded successfully!")
        logger.info("=" * 60)

        return (diffuser, text_encoder, tokenizer, config_info)


NODE_CLASS_MAPPINGS = {"FluxFlowModelLoader": FluxFlowModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxFlowModelLoader": "FluxFlow Model Loader"}
