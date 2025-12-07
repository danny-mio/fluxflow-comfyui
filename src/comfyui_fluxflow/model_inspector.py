"""
FluxFlow checkpoint inspection and configuration auto-detection.

Automatically detects model architecture parameters from safetensors checkpoints:
- VAE latent dimensions
- Flow model dimensions
- Text embedding dimensions
- Architecture depth (downscales, layers, attention heads)
- Validates checkpoint integrity
"""

from typing import Any, Dict

import safetensors.torch


def detect_fluxflow_config(checkpoint_path: str) -> Dict[str, Any]:  # noqa: C901
    """
    Auto-detect FluxFlow model configuration from safetensors checkpoint.

    Args:
        checkpoint_path: Path to .safetensors checkpoint file

    Returns:
        Configuration dictionary with keys:
            - vae_dim: VAE latent dimension
            - flow_dim: Flow model internal dimension
            - text_embed_dim: Text embedding dimension
            - downscales: Number of downsampling stages
            - upscales: Number of upsampling stages
            - vae_attn_layers: VAE attention layers
            - flow_transformer_layers: Flow transformer depth
            - flow_attn_heads: Flow attention heads
            - max_hw: Maximum spatial dimension
            - compression_ratio: Spatial compression (2^downscales)

    Raises:
        ValueError: If checkpoint format is invalid or configuration cannot be detected
    """
    config = {}

    with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

        if not keys:
            raise ValueError(f"Checkpoint is empty: {checkpoint_path}")

        # 1. Detect VAE dimension from compressor latent projection
        # compressor.latent_proj outputs d_model * 5 channels (for Bezier activation)
        vae_dim_found = False
        for key in keys:
            if "compressor.latent_proj" in key and ".0.0.weight" in key:
                shape = f.get_tensor(key).shape
                if len(shape) == 4:  # Conv2d [out_ch, in_ch, k, k]
                    config["vae_dim"] = shape[0] // 5
                    vae_dim_found = True
                    break

        if not vae_dim_found:
            raise ValueError("Could not detect VAE dimension from checkpoint")

        # 2. Detect flow_processor d_model from vae_to_dmodel projection
        flow_dim_found = False
        for key in keys:
            if "flow_processor.vae_to_dmodel.weight" in key:
                shape = f.get_tensor(key).shape  # [flow_dim, vae_dim]
                config["flow_dim"] = shape[0]
                config["vae_dim_check"] = shape[1]
                flow_dim_found = True
                break

        if not flow_dim_found:
            raise ValueError("Could not detect flow dimension from checkpoint")

        # 3. Detect text embedding dimension from flow_processor.text_proj
        text_embed_found = False
        for key in keys:
            if "flow_processor.text_proj.weight" in key:
                shape = f.get_tensor(key).shape  # [flow_dim, text_embed_dim]
                config["text_embed_dim"] = shape[1]
                text_embed_found = True
                break

        if not text_embed_found:
            # Fallback: try to detect from text_encoder output layer
            for key in keys:
                if "text_encoder.ouput_layer" in key and ".weight" in key:
                    # This is more complex, use default
                    config["text_embed_dim"] = 1024
                    text_embed_found = True
                    break

        if not text_embed_found:
            config["text_embed_dim"] = 1024  # Default fallback

        # 4. Detect downscale/upscale stages
        encoder_stages = [k for k in keys if "compressor.encoder_z." in k and ".0.weight" in k]
        config["downscales"] = len(encoder_stages)

        upscale_layers = [
            k for k in keys if "expander.upscale.layers." in k and ".conv1.0.weight" in k
        ]
        config["upscales"] = len(upscale_layers)

        if config["downscales"] == 0:
            raise ValueError("Could not detect downscale stages from checkpoint")

        # 5. Detect attention configuration
        attn_layers = [
            k for k in keys if "compressor.token_attn." in k and ".attn.in_proj_weight" in k
        ]
        config["vae_attn_layers"] = len(attn_layers)

        transformer_blocks = [
            k
            for k in keys
            if "flow_processor.transformer_blocks." in k and ".self_attn.q_proj.weight" in k
        ]
        config["flow_transformer_layers"] = len(transformer_blocks)

        # 6. Detect attention heads from rotary position encoding inv_freq buffer
        config["flow_attn_heads"] = 8  # Default fallback
        for key in keys:
            if "flow_processor.transformer_blocks.0.rotary_pe.inv_freq" in key:
                inv_freq_shape = f.get_tensor(key).shape  # [head_dim // 2]
                head_dim = inv_freq_shape[0] * 2
                # Get d_model from q_proj to calculate n_head
                for qk in keys:
                    if "flow_processor.transformer_blocks.0.self_attn.q_proj.weight" in qk:
                        d_model = f.get_tensor(qk).shape[0]
                        config["flow_attn_heads"] = d_model // head_dim
                        break
                break

        # 7. Max spatial dimension (default to 1024)
        config["max_hw"] = 1024

        # 8. Calculate compression ratio
        config["compression_ratio"] = 2 ** config["downscales"]

    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate detected configuration for consistency.

    Args:
        config: Configuration dictionary from detect_fluxflow_config

    Returns:
        True if configuration is valid

    Raises:
        ValueError: If configuration is invalid with detailed error message
    """
    # Check VAE dimension consistency
    if "vae_dim" in config and "vae_dim_check" in config:
        if config["vae_dim"] != config["vae_dim_check"]:
            raise ValueError(
                f"VAE dimension mismatch: {config['vae_dim']} vs {config['vae_dim_check']}"
            )

    # Check symmetric VAE architecture
    downscales = config.get("downscales", 0)
    upscales = config.get("upscales", 0)
    if downscales != upscales:
        print(f"Warning: Asymmetric VAE - {downscales} downscales, " f"{upscales} upscales")

    # Validate dimensions are positive
    required_params = ["vae_dim", "flow_dim", "text_embed_dim", "downscales"]
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: {param}")
        if config[param] <= 0:
            raise ValueError(f"Invalid {param}: {config[param]} (must be positive)")

    # Check flow_dim is reasonable relative to vae_dim
    if config["flow_dim"] < config["vae_dim"]:
        print(f"Warning: flow_dim ({config['flow_dim']}) < vae_dim ({config['vae_dim']})")

    return True


def print_config(config: Dict[str, Any], verbose: bool = True) -> None:
    """
    Pretty-print detected configuration.

    Args:
        config: Configuration dictionary
        verbose: If True, print detailed information
    """
    print("\n" + "=" * 60)
    print("FluxFlow Model Configuration (Auto-Detected)")
    print("=" * 60)

    # Core dimensions
    print("\nCore Dimensions:")
    print(f"  VAE Latent Dim:        {config.get('vae_dim', 'N/A')}")
    print(f"  Flow Model Dim:        {config.get('flow_dim', 'N/A')}")
    print(f"  Text Embedding Dim:    {config.get('text_embed_dim', 'N/A')}")

    # Architecture
    print("\nArchitecture:")
    print(f"  Downscales:            {config.get('downscales', 'N/A')}")
    print(f"  Upscales:              {config.get('upscales', 'N/A')}")
    print(f"  Compression Ratio:     {config.get('compression_ratio', 'N/A')}x")
    print(
        f"  Spatial Compression:   {config.get('compression_ratio', 1)**2}x "
        f"({config.get('compression_ratio', 1)}² pixels)"
    )

    # Attention
    if verbose:
        print("\nAttention Configuration:")
        print(f"  VAE Attention Layers:      {config.get('vae_attn_layers', 'N/A')}")
        print(f"  Flow Transformer Layers:   {config.get('flow_transformer_layers', 'N/A')}")
        print(f"  Flow Attention Heads:      {config.get('flow_attn_heads', 'N/A')}")

    # Other
    print("\nOther:")
    print(f"  Max Spatial Dimension: {config.get('max_hw', 'N/A')}")

    # Validation
    print("\nValidation:")
    if "vae_dim" in config and "vae_dim_check" in config:
        if config["vae_dim"] == config["vae_dim_check"]:
            print(f"  ✓ VAE dimension consistent: {config['vae_dim']}")
        else:
            print(f"  ✗ VAE dimension mismatch: {config['vae_dim']} vs {config['vae_dim_check']}")

    if config.get("downscales") == config.get("upscales"):
        print(f"  ✓ Symmetric VAE: {config['downscales']} down/up scales")
    else:
        print(
            f"  ⚠ Asymmetric VAE: {config.get('downscales')} down, " f"{config.get('upscales')} up"
        )

    print("=" * 60 + "\n")


def get_model_info(checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Get complete model information with validation.

    Args:
        checkpoint_path: Path to checkpoint file
        verbose: Print detailed information

    Returns:
        Validated configuration dictionary
    """
    config = detect_fluxflow_config(checkpoint_path)
    validate_config(config)

    if verbose:
        print_config(config, verbose=True)

    return config
