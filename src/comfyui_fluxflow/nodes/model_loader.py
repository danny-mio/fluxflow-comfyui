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
from fluxflow.models.versioning import load_versioned_checkpoint
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
        Load FluxFlow model from checkpoint with automatic version detection.

        Args:
            checkpoint_path: Path to checkpoint (versioned directory or legacy file)
            device: Device to load model on (auto, cuda, cpu, mps)
            tokenizer_name: HuggingFace tokenizer name

        Returns:
            (model, text_encoder, tokenizer, config_info)
        """
        from pathlib import Path

        logger.info("=" * 60)
        logger.info("FluxFlow Model Loader")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {checkpoint_path}")

        # Parse device
        device_obj = parse_device(device)
        logger.info(f"Device: {device_obj}")

        # Try versioned loading first (new format with metadata)
        try:
            checkpoint_path_obj = Path(checkpoint_path)

            if checkpoint_path_obj.is_dir():
                # Versioned checkpoint (directory with metadata.json + model.safetensors)
                logger.info("Loading versioned checkpoint...")
                pipeline = load_versioned_checkpoint(checkpoint_path, device_obj)
                diffuser = pipeline
                text_encoder = pipeline.text_encoder

                # Extract version info
                version = getattr(pipeline, 'version', 'unknown')
                logger.info(f"Loaded versioned checkpoint (v{version})")

                # Create config info for versioned model
                if hasattr(diffuser, 'compressor') and hasattr(diffuser.compressor, 'd_model'):
                    vae_dim = diffuser.compressor.d_model
                else:
                    vae_dim = "unknown"

                if hasattr(diffuser, 'flow_processor') and hasattr(diffuser.flow_processor, 'd_model'):
                    flow_dim = diffuser.flow_processor.d_model
                else:
                    flow_dim = "unknown"

                if hasattr(text_encoder, 'embed_dim'):
                    text_embed_dim = text_encoder.embed_dim
                else:
                    text_embed_dim = "unknown"

                config_info = f"Version {version} - VAE: {vae_dim}d, Flow: {flow_dim}d, Text: {text_embed_dim}d"

            else:
                # Try legacy versioned loading (single file with metadata)
                try:
                    logger.info("Attempting legacy versioned loading...")
                    pipeline = FluxPipeline.from_pretrained(
                        checkpoint_path, device=device_obj, use_versioning=True
                    )
                    diffuser = pipeline
                    text_encoder = pipeline.text_encoder
                    version = getattr(pipeline, 'version', 'legacy')
                    logger.info(f"Loaded legacy versioned checkpoint (v{version})")

                    # Extract dimensions
                    vae_dim = diffuser.compressor.d_model if hasattr(diffuser, 'compressor') else "unknown"
                    flow_dim = diffuser.flow_processor.d_model if hasattr(diffuser, 'flow_processor') else "unknown"
                    text_embed_dim = text_encoder.embed_dim if hasattr(text_encoder, 'embed_dim') else "unknown"
                    config_info = f"Legacy v{version} - VAE: {vae_dim}d, Flow: {flow_dim}d, Text: {text_embed_dim}d"

                except Exception as legacy_error:
                    logger.info(f"Versioned loading failed: {legacy_error}")
                    logger.info("Falling back to v0.3.0 architecture detection...")
                    raise legacy_error  # Fall through to legacy loading

        except Exception as versioned_error:
            # Fall back to legacy architecture detection with automatic version detection
            logger.info(f"Versioned loading failed ({versioned_error}), inspecting checkpoint for architecture")

            # First, inspect checkpoint to detect architecture
            try:
                import safetensors.torch
                if checkpoint_path.endswith('.safetensors'):
                    state_dict = safetensors.torch.load_file(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location='cpu')
                keys = list(state_dict.keys())

                # Check for v0.7.0 features
                has_v070_features = any(
                    "ctx_mixer" in key or "context_injection" in key or "context_final" in key
                    for key in keys
                )

                if has_v070_features:
                    logger.info("Detected v0.7.0 features in checkpoint, this requires proper metadata")
                    logger.error("Cannot load v0.7.0 model without metadata. Please re-save the model with save_versioned_checkpoint()")
                    return (None, None, None, "Error: v0.7.0 model detected but requires metadata. Please re-save with save_versioned_checkpoint()")
            except Exception as inspect_error:
                logger.debug(f"Checkpoint inspection failed: {inspect_error}")

            # Fall back to v0.3.0 detection
            logger.info("Using legacy v0.3.0 detection")
            config = get_model_info(checkpoint_path, verbose=True)

            # Initialize models with detected configuration (v0.3.0 defaults)
            logger.info("Initializing models with v0.3.0 architecture...")

            # Calculate appropriate attention heads to ensure d_model is divisible
            def get_valid_n_head(d_model, preferred_heads=8):
                """Get number of heads that evenly divides d_model."""
                if d_model % preferred_heads == 0:
                    return preferred_heads
                # Find largest divisor that keeps heads reasonable
                for heads in range(preferred_heads, 0, -1):
                    if d_model % heads == 0:
                        return heads
                return 1  # Fallback, though this shouldn't happen

            # Use flexible attention heads for compatibility
            flow_attn_heads = get_valid_n_head(config["flow_dim"], config.get("flow_attn_heads", 8))

            compressor = FluxCompressor(
                in_channels=3,
                d_model=config["vae_dim"],
                downscales=config["downscales"],
                max_hw=config.get("max_hw", 1024),
                use_attention=True,
                attn_layers=config.get("vae_attn_layers", 2),
            )

            flow_processor = FluxFlowProcessor(
                d_model=config["flow_dim"],
                vae_dim=config["vae_dim"],
                embedding_size=config["text_embed_dim"],
                n_head=flow_attn_heads,  # Use calculated heads for compatibility
                n_layers=config.get("flow_transformer_layers", 10),
                max_hw=config.get("max_hw", 1024),
            )

            flow_processor = FluxFlowProcessor(
                d_model=config["flow_dim"],
                vae_dim=config["vae_dim"],
                embedding_size=config["text_embed_dim"],
                n_head=flow_attn_heads,  # Use calculated heads for compatibility
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

            # Move to device and set eval mode
            diffuser = diffuser.to(device_obj)
            text_encoder = text_encoder.to(device_obj)
            diffuser.eval()
            text_encoder.eval()

            # Create config info for legacy model
            config_info = (
                f"Legacy v0.3.0 - VAE: {config['vae_dim']}d, "
                f"Flow: {config['flow_dim']}d, "
                f"Text: {config['text_embed_dim']}d, "
                f"Compression: {config['compression_ratio']}x"
            )

            version = "0.3.0"

        # Load tokenizer (shared for all versions)
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir="./_cache", local_files_only=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        logger.info("Model loaded successfully!")
        logger.info(f"Version: {version}")
        logger.info(f"Config: {config_info}")
        logger.info("=" * 60)

        return (diffuser, text_encoder, tokenizer, config_info)


NODE_CLASS_MAPPINGS = {"FluxFlowModelLoader": FluxFlowModelLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxFlowModelLoader": "FluxFlow Model Loader"}
