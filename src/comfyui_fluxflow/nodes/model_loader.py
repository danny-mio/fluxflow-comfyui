"""
FluxFlow Model Loader Node for ComfyUI.

Automatically detects model configuration from checkpoint and initializes all components.
"""

import logging
from pathlib import Path

import safetensors.torch
import torch

# Import from installed fluxflow package
from fluxflow.models import (
    BertTextEncoder,
    FluxCompressor,
    FluxExpander,
    FluxFlowProcessor,
    FluxPipeline,
    detect_architecture_version,
)
from fluxflow.models.versioning import load_versioned_checkpoint
from transformers import AutoTokenizer

from comfyui_fluxflow.model_inspector import get_model_info
from comfyui_fluxflow.nodes.utils import parse_device

logger = logging.getLogger(__name__)

# Inference-time precision options, matching the fp16/bf16 options already
# available for training. "fp32" maps to None -- no `.to(dtype=...)` call is
# made, keeping that path behaviorally identical to before this option existed.
_DTYPE_MAP: dict = {
    "fp32": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _detect_display_version(checkpoint_path: str) -> str:
    """Re-detect architecture version from a checkpoint's own keys, for display only.

    No `.version` attribute is ever set on a loaded FluxPipeline, so this
    re-reads the state dict and runs the same key-marker detection used
    during loading. Display-only; failures fall back to "unknown".
    """
    try:
        cp = Path(checkpoint_path)
        weights_path = cp
        if cp.is_dir():
            weights_path = cp / "model.safetensors"
            if not weights_path.exists():
                weights_path = cp / "flxflow_final.safetensors"
        state_dict = safetensors.torch.load_file(str(weights_path))
        return detect_architecture_version(list(state_dict.keys()))
    except Exception:
        return "unknown"


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
                "dtype": (
                    ["fp32", "fp16", "bf16"],
                    {"default": "fp32"},
                ),
                "tokenizer_name": (
                    "STRING",
                    {
                        "default": "distilbert-base-uncased",
                        "multiline": False,
                        "dynamicPrompts": False,
                    },
                ),
                "text_encoder_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "dynamicPrompts": False,
                        "placeholder": "Optional: override path to text_encoder.safetensors",
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

    def load_model(  # noqa: C901
        self,
        checkpoint_path: str,
        device: str = "auto",
        dtype: str = "fp32",
        tokenizer_name: str = "distilbert-base-uncased",
        text_encoder_path: str = "",
    ):
        """
        Load FluxFlow model from checkpoint with automatic version detection.

        Args:
            checkpoint_path: Path to checkpoint (versioned directory or legacy file)
            device: Device to load model on (auto, cuda, cpu, mps)
            dtype: Inference precision to cast the loaded model to
                (fp32, fp16, bf16). Weights are always loaded in their native
                dtype first; the cast happens afterward, same as the device
                move. "fp32" makes no dtype cast at all.
            tokenizer_name: HuggingFace tokenizer name
            text_encoder_path: Optional explicit path to text-encoder weights
                (.safetensors), overriding both the sibling file and any
                bundled copy inside the main checkpoint.

        Returns:
            (model, text_encoder, tokenizer, config_info)

        Raises:
            ValueError: If dtype is not one of fp32, fp16, bf16.
        """
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"Unknown dtype: {dtype}")
        torch_dtype = _DTYPE_MAP[dtype]

        text_encoder_override = text_encoder_path or None
        logger.info("=" * 60)
        logger.info("FluxFlow Model Loader")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {checkpoint_path}")

        # Parse device
        device_obj = parse_device(device)
        logger.info(f"Device: {device_obj}")
        logger.info(f"Dtype: {dtype}")

        # kwargs shared by every `.to()` call below -- dtype is only included
        # when it's not fp32, so the default path stays behaviorally identical.
        move_kwargs: dict = {"device": device_obj}
        if torch_dtype is not None:
            move_kwargs["dtype"] = torch_dtype

        # Try versioned loading first (new format with metadata)
        try:
            checkpoint_path_obj = Path(checkpoint_path)

            if checkpoint_path_obj.is_dir():
                # Versioned checkpoint (directory with metadata.json + model.safetensors)
                logger.info("Loading versioned checkpoint...")
                pipeline = load_versioned_checkpoint(Path(checkpoint_path), str(device_obj))
                diffuser = pipeline

                # Load text_encoder separately (not included in versioned checkpoints)
                logger.info("Loading text encoder...")
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_name, cache_dir="./_cache", local_files_only=False
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

                text_encoder = BertTextEncoder(embed_dim=1024)  # Default for ComfyUI
                text_encoder.load_with_override(
                    checkpoint_path, override_path=text_encoder_override
                )

                # Ensure models are on device and in eval mode
                pipeline.to(**move_kwargs)
                text_encoder.to(**move_kwargs)
                pipeline.eval()
                text_encoder.eval()

                # Extract version info for display (no .version attribute is
                # ever set on a loaded FluxPipeline, so re-detect from the
                # checkpoint's own state-dict key markers).
                version = _detect_display_version(checkpoint_path)
                logger.info(f"Loaded versioned checkpoint (v{version})")

                # Create config info for versioned model
                if hasattr(diffuser, "compressor") and hasattr(diffuser.compressor, "d_model"):
                    vae_dim = diffuser.compressor.d_model
                else:
                    vae_dim = "unknown"

                if hasattr(diffuser, "flow_processor") and hasattr(
                    diffuser.flow_processor, "d_model"
                ):
                    flow_dim = diffuser.flow_processor.d_model
                else:
                    flow_dim = "unknown"

                text_embed_dim = (
                    text_encoder.embed_dim if hasattr(text_encoder, "embed_dim") else "unknown"
                )

                config_info = f"Version {version} - VAE: {vae_dim}d, Flow: {flow_dim}d, Text: {text_embed_dim}d"

            else:
                # Try legacy versioned loading (single file with metadata)
                try:
                    logger.info("Attempting legacy versioned loading...")
                    pipeline = load_versioned_checkpoint(Path(checkpoint_path), str(device_obj))
                    diffuser = pipeline

                    # Load text_encoder separately
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_name, cache_dir="./_cache", local_files_only=False
                    )
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

                    text_encoder = BertTextEncoder(embed_dim=1024)  # Default for ComfyUI
                    text_encoder.load_with_override(
                        checkpoint_path, override_path=text_encoder_override
                    )

                    # Ensure models are on device and in eval mode
                    pipeline.to(**move_kwargs)
                    text_encoder.to(**move_kwargs)
                    pipeline.eval()
                    text_encoder.eval()

                    version = _detect_display_version(checkpoint_path)
                    logger.info(f"Loaded legacy versioned checkpoint (v{version})")

                    # Extract dimensions
                    has_vae_dim = hasattr(diffuser, "compressor") and hasattr(
                        diffuser.compressor, "d_model"
                    )
                    vae_dim = diffuser.compressor.d_model if has_vae_dim else "unknown"
                    has_flow_dim = hasattr(diffuser, "flow_processor") and hasattr(
                        diffuser.flow_processor, "d_model"
                    )
                    flow_dim = diffuser.flow_processor.d_model if has_flow_dim else "unknown"
                    text_embed_dim = (
                        text_encoder.embed_dim if hasattr(text_encoder, "embed_dim") else "unknown"
                    )
                    config_info = f"Legacy v{version} - VAE: {vae_dim}d, Flow: {flow_dim}d, Text: {text_embed_dim}d"

                except Exception as legacy_error:
                    logger.info(f"Versioned loading failed: {legacy_error}")
                    logger.info("Falling back to v0.3.0 architecture detection...")
                    raise legacy_error  # Fall through to legacy loading

        except Exception as versioned_error:
            # Fall back to legacy architecture detection with automatic version detection
            logger.info(
                f"Versioned loading failed ({versioned_error}), inspecting checkpoint for architecture"
            )

            # First, inspect checkpoint to detect architecture
            detected_version = "0.3.0"
            try:
                if checkpoint_path.endswith(".safetensors"):
                    state_dict = safetensors.torch.load_file(checkpoint_path)
                else:
                    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                keys = list(state_dict.keys())

                # detect_architecture_version checks v0.10.0-exclusive markers
                # (ctx_gate_proj/ctx_delta_proj/time_mlp) before the v0.7.0
                # ones (ctx_mixer/context_injection/context_final) that
                # v0.10.0 also happens to contain (inherited names) -- must
                # run first, or v0.10.0 checkpoints get misrouted below.
                detected_version = detect_architecture_version(keys)

                if detected_version == "0.7.0":
                    # v0.8.0 (pillar-attention) is a refinement within the
                    # "0.7.0-family" bucket that detect_architecture_version's
                    # 3-way result doesn't distinguish on its own.
                    has_v080_features = any(
                        "pillar_cross_attn" in key or "film_p0" in key for key in keys
                    )
                    if has_v080_features:
                        logger.info(
                            "Detected v0.8.0 features (pillar-attention) in checkpoint, this requires proper metadata"
                        )
                        logger.error(
                            "Cannot load v0.8.0 model without metadata. Please re-save the model with save_versioned_checkpoint()"
                        )
                        return (
                            None,
                            None,
                            None,
                            "Error: v0.8.0 model detected but requires metadata. Please re-save with save_versioned_checkpoint()",
                        )

                    logger.info(
                        "Detected v0.7.0 features in checkpoint, this requires proper metadata"
                    )
                    logger.error(
                        "Cannot load v0.7.0 model without metadata. Please re-save the model with save_versioned_checkpoint()"
                    )
                    return (
                        None,
                        None,
                        None,
                        "Error: v0.7.0 model detected but requires metadata. Please re-save with save_versioned_checkpoint()",
                    )
            except Exception as inspect_error:
                logger.debug(f"Checkpoint inspection failed: {inspect_error}")

            if detected_version == "0.10.0":
                logger.info("Detected v0.10.0 features in checkpoint structure")
                logger.info("Initializing models with v0.10.0 architecture...")

                from fluxflow.models.v100.flow import FluxFlowProcessor_v100
                from fluxflow.models.v100.vae import FluxCompressor_v100, FluxExpander_v100

                v100_config = FluxPipeline._detect_config(state_dict)
                vae_latent_dim = v100_config["vae_dim"]

                def get_valid_n_head_v100(d_model, preferred_heads=8):
                    if d_model % preferred_heads == 0:
                        return preferred_heads
                    for heads in range(preferred_heads, 0, -1):
                        if d_model % heads == 0:
                            return heads
                    return 1

                flow_attn_heads = get_valid_n_head_v100(
                    v100_config["flow_dim"], v100_config.get("flow_attn_heads", 8)
                )

                compressor = FluxCompressor_v100(
                    in_channels=3,
                    d_model=vae_latent_dim,
                    downscales=v100_config["downscales"],
                    max_hw=v100_config.get("max_hw", 1024),
                )
                flow_processor = FluxFlowProcessor_v100(
                    d_model=v100_config["flow_dim"],
                    vae_dim=vae_latent_dim,
                    embedding_size=v100_config.get("text_embed_dim", 1024),
                    n_head=flow_attn_heads,
                    n_layers=v100_config.get("flow_transformer_layers", 10),
                    max_hw=v100_config.get("max_hw", 1024),
                )
                expander = FluxExpander_v100(
                    d_model=vae_latent_dim,
                    upscales=v100_config.get("upscales", v100_config["downscales"]),
                    max_hw=v100_config.get("max_hw", 1024),
                )

                diffuser = FluxPipeline(compressor, flow_processor, expander)
                text_encoder = BertTextEncoder(embed_dim=v100_config.get("text_embed_dim", 1024))

                diffuser_state = {
                    k.replace("diffuser.", ""): v
                    for k, v in state_dict.items()
                    if k.startswith("diffuser.")
                }
                diffuser.load_state_dict(diffuser_state, strict=False)
                text_encoder.load_with_override(
                    checkpoint_path, override_path=text_encoder_override
                )

                diffuser = diffuser.to(**move_kwargs)
                text_encoder = text_encoder.to(**move_kwargs)
                diffuser.eval()
                text_encoder.eval()

                config_info = (
                    f"v0.10.0 auto-detected - VAE: {vae_latent_dim}d, "
                    f"Flow: {v100_config['flow_dim']}d, "
                    f"Text: {v100_config.get('text_embed_dim', 1024)}d"
                )
                version = "0.10.0"

                # Load tokenizer and return directly -- the v0.3.0 path below
                # (and its shared tail) doesn't apply to this branch.
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
                logger.debug(
                    f"{len(missing_keys)} keys not found in checkpoint (using random init)"
                )
            if unexpected_keys:
                logger.debug(f"{len(unexpected_keys)} unexpected keys in checkpoint (ignored)")

            # Load text encoder weights (override path > sibling file > bundled keys)
            text_encoder.load_with_override(checkpoint_path, override_path=text_encoder_override)

            # Move to device and set eval mode
            diffuser = diffuser.to(**move_kwargs)
            text_encoder = text_encoder.to(**move_kwargs)
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
