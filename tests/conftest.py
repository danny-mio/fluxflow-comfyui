"""Pytest fixtures for fluxflow-comfyui tests."""

import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_checkpoint_state():
    """Create a mock checkpoint state dict with expected FluxFlow model keys."""
    state_dict = {
        # VAE compressor - vae_dim = 320 / 5 = 64
        # Key pattern: compressor.latent_proj.0.0.weight
        "compressor.latent_proj.0.0.weight": torch.randn(320, 3, 1, 1),
        # Flow processor VAE to d_model projection - flow_dim = 128, vae_dim = 64
        "flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
        # Text embedding projection - text_embed_dim = 256
        "flow_processor.text_proj.weight": torch.randn(128, 256),
        # Downscale stages (encoder_z layers - 2 stages)
        "compressor.encoder_z.0.0.weight": torch.randn(64, 64, 3, 3),
        "compressor.encoder_z.1.0.weight": torch.randn(128, 64, 3, 3),
        # Upscale stages (expander layers - 2 stages)
        "expander.upscale.layers.0.conv1.0.weight": torch.randn(64, 128, 3, 3),
        "expander.upscale.layers.1.conv1.0.weight": torch.randn(64, 64, 3, 3),
        # VAE attention layers (token_attn - 2 layers)
        "compressor.token_attn.0.attn.in_proj_weight": torch.randn(192, 64),
        "compressor.token_attn.1.attn.in_proj_weight": torch.randn(192, 64),
        # Flow transformer blocks (2 blocks)
        "flow_processor.transformer_blocks.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "flow_processor.transformer_blocks.1.self_attn.q_proj.weight": torch.randn(128, 128),
    }
    return state_dict
