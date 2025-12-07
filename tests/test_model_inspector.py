"""Unit tests for model inspector (comfyui_fluxflow/model_inspector.py)."""

import pytest
import safetensors.torch
import torch

from comfyui_fluxflow.model_inspector import (
    detect_fluxflow_config,
    get_model_info,
    print_config,
    validate_config,
)


class TestDetectFluxFlowConfig:
    """Tests for detect_fluxflow_config function."""

    def test_detects_vae_dim(self, mock_checkpoint_state, temp_dir):
        """Should detect VAE dimension from compressor.latent_proj."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # VAE dim should be detected (320 / 5 = 64)
        assert config["vae_dim"] == 64

    def test_detects_flow_dim(self, mock_checkpoint_state, temp_dir):
        """Should detect flow dimension from vae_to_dmodel."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        assert config["flow_dim"] == 128

    def test_detects_text_embed_dim(self, mock_checkpoint_state, temp_dir):
        """Should detect text embedding dimension."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        assert config["text_embed_dim"] == 256

    def test_detects_downscales(self, mock_checkpoint_state, temp_dir):
        """Should detect number of downscale stages."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # Mock has 2 encoder stages
        assert config["downscales"] == 2

    def test_detects_upscales(self, mock_checkpoint_state, temp_dir):
        """Should detect number of upscale stages."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # Mock has 2 upscale layers
        assert config["upscales"] == 2

    def test_detects_vae_attn_layers(self, mock_checkpoint_state, temp_dir):
        """Should detect number of VAE attention layers."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # Mock has 2 attention layers
        assert config["vae_attn_layers"] == 2

    def test_detects_flow_transformer_layers(self, mock_checkpoint_state, temp_dir):
        """Should detect number of transformer layers."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # Mock has 2 transformer blocks
        assert config["flow_transformer_layers"] == 2

    def test_calculates_compression_ratio(self, mock_checkpoint_state, temp_dir):
        """Should calculate compression ratio correctly."""
        checkpoint_path = temp_dir / "test_checkpoint.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # 2 downscales = 2^2 = 4x compression
        assert config["compression_ratio"] == 4

    def test_raises_on_empty_checkpoint(self, temp_dir):
        """Should raise error on empty checkpoint."""
        checkpoint_path = temp_dir / "empty.safetensors"
        safetensors.torch.save_file({}, str(checkpoint_path))

        with pytest.raises(ValueError, match="Checkpoint is empty"):
            detect_fluxflow_config(str(checkpoint_path))

    def test_raises_on_missing_vae_dim(self, temp_dir):
        """Should raise error if VAE dim cannot be detected."""
        # Create checkpoint without latent_proj
        state = {"some.other.weight": torch.randn(10, 10)}
        checkpoint_path = temp_dir / "incomplete.safetensors"
        safetensors.torch.save_file(state, str(checkpoint_path))

        with pytest.raises(ValueError, match="Could not detect VAE dimension"):
            detect_fluxflow_config(str(checkpoint_path))

    def test_raises_on_missing_flow_dim(self, temp_dir):
        """Should raise error if flow dim cannot be detected."""
        # Create checkpoint with only latent_proj
        state = {
            "diffuser.compressor.latent_proj.0.0.weight": torch.randn(320, 32, 1, 1),
        }
        checkpoint_path = temp_dir / "incomplete.safetensors"
        safetensors.torch.save_file(state, str(checkpoint_path))

        with pytest.raises(ValueError, match="Could not detect flow dimension"):
            detect_fluxflow_config(str(checkpoint_path))

    def test_raises_on_missing_downscales(self, temp_dir):
        """Should raise error if downscales cannot be detected."""
        # Create checkpoint with required dims but no encoder stages
        state = {
            "diffuser.compressor.latent_proj.0.0.weight": torch.randn(320, 32, 1, 1),
            "diffuser.flow_processor.vae_to_dmodel.weight": torch.randn(128, 64),
        }
        checkpoint_path = temp_dir / "incomplete.safetensors"
        safetensors.torch.save_file(state, str(checkpoint_path))

        with pytest.raises(ValueError, match="Could not detect downscale stages"):
            detect_fluxflow_config(str(checkpoint_path))


class TestValidateConfig:
    """Tests for validate_config function."""

    def test_valid_config_passes(self):
        """Valid configuration should pass validation."""
        config = {
            "vae_dim": 64,
            "vae_dim_check": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
            "upscales": 4,
        }

        # Should not raise
        assert validate_config(config) is True

    def test_vae_dim_mismatch_raises(self):
        """Mismatched VAE dimensions should raise error."""
        config = {
            "vae_dim": 64,
            "vae_dim_check": 128,  # Mismatch
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
        }

        with pytest.raises(ValueError, match="VAE dimension mismatch"):
            validate_config(config)

    def test_missing_required_param_raises(self):
        """Missing required parameter should raise error."""
        config = {
            "vae_dim": 64,
            "flow_dim": 128,
            # Missing text_embed_dim and downscales
        }

        with pytest.raises(ValueError, match="Missing required parameter"):
            validate_config(config)

    def test_zero_dimension_raises(self):
        """Zero or negative dimensions should raise error."""
        config = {
            "vae_dim": 0,  # Invalid
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
        }

        with pytest.raises(ValueError, match="Invalid vae_dim"):
            validate_config(config)

    def test_negative_dimension_raises(self):
        """Negative dimensions should raise error."""
        config = {
            "vae_dim": 64,
            "flow_dim": -128,  # Invalid
            "text_embed_dim": 256,
            "downscales": 4,
        }

        with pytest.raises(ValueError, match="Invalid flow_dim"):
            validate_config(config)

    def test_asymmetric_vae_warning(self, capsys):
        """Asymmetric VAE should print warning."""
        config = {
            "vae_dim": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
            "upscales": 3,  # Different from downscales
        }

        validate_config(config)

        captured = capsys.readouterr()
        assert "Asymmetric VAE" in captured.out

    def test_flow_dim_less_than_vae_warning(self, capsys):
        """flow_dim < vae_dim should print warning."""
        config = {
            "vae_dim": 256,
            "flow_dim": 128,  # Less than vae_dim
            "text_embed_dim": 512,
            "downscales": 4,
        }

        validate_config(config)

        captured = capsys.readouterr()
        assert "flow_dim" in captured.out and "vae_dim" in captured.out


class TestPrintConfig:
    """Tests for print_config function."""

    def test_prints_core_dimensions(self, capsys):
        """Should print core dimensions."""
        config = {
            "vae_dim": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
            "upscales": 4,
            "compression_ratio": 16,
        }

        print_config(config, verbose=False)

        captured = capsys.readouterr()
        assert "64" in captured.out  # vae_dim
        assert "128" in captured.out  # flow_dim
        assert "256" in captured.out  # text_embed_dim

    def test_prints_architecture_info(self, capsys):
        """Should print architecture information."""
        config = {
            "vae_dim": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
            "upscales": 4,
            "compression_ratio": 16,
        }

        print_config(config, verbose=False)

        captured = capsys.readouterr()
        assert "Downscales" in captured.out
        assert "Upscales" in captured.out
        assert "Compression Ratio" in captured.out

    def test_verbose_prints_attention_config(self, capsys):
        """Verbose mode should print attention configuration."""
        config = {
            "vae_dim": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
            "vae_attn_layers": 4,
            "flow_transformer_layers": 12,
            "flow_attn_heads": 8,
        }

        print_config(config, verbose=True)

        captured = capsys.readouterr()
        assert "Attention Configuration" in captured.out
        assert "VAE Attention Layers" in captured.out
        assert "Flow Transformer Layers" in captured.out

    def test_non_verbose_skips_attention(self, capsys):
        """Non-verbose mode should skip attention details."""
        config = {
            "vae_dim": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
        }

        print_config(config, verbose=False)

        captured = capsys.readouterr()
        assert "Attention Configuration" not in captured.out

    def test_prints_validation_results(self, capsys):
        """Should print validation results."""
        config = {
            "vae_dim": 64,
            "vae_dim_check": 64,
            "flow_dim": 128,
            "text_embed_dim": 256,
            "downscales": 4,
            "upscales": 4,
        }

        print_config(config, verbose=False)

        captured = capsys.readouterr()
        assert "✓" in captured.out or "Validation" in captured.out


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_validated_config(self, mock_checkpoint_state, temp_dir):
        """Should return validated configuration."""
        checkpoint_path = temp_dir / "test.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = get_model_info(str(checkpoint_path), verbose=False)

        # Should have required fields
        assert "vae_dim" in config
        assert "flow_dim" in config
        assert "text_embed_dim" in config
        assert "downscales" in config

    def test_prints_when_verbose(self, mock_checkpoint_state, temp_dir, capsys):
        """Verbose mode should print configuration."""
        checkpoint_path = temp_dir / "test.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        get_model_info(str(checkpoint_path), verbose=True)

        captured = capsys.readouterr()
        assert "FluxFlow Model Configuration" in captured.out

    def test_does_not_print_when_not_verbose(self, mock_checkpoint_state, temp_dir, capsys):
        """Non-verbose mode should not print."""
        checkpoint_path = temp_dir / "test.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        get_model_info(str(checkpoint_path), verbose=False)

        captured = capsys.readouterr()
        # Should not have the banner
        assert "FluxFlow Model Configuration" not in captured.out


class TestModelInspectorIntegration:
    """Integration tests for model inspector."""

    def test_full_workflow(self, mock_checkpoint_state, temp_dir):
        """Test complete detect -> validate -> print workflow."""
        checkpoint_path = temp_dir / "model.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        # Detect
        config = detect_fluxflow_config(str(checkpoint_path))
        assert config is not None

        # Validate
        is_valid = validate_config(config)
        assert is_valid is True

        # Print (should not raise)
        print_config(config, verbose=True)

    def test_get_model_info_wrapper(self, mock_checkpoint_state, temp_dir):
        """Test get_model_info combines all steps."""
        checkpoint_path = temp_dir / "model.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = get_model_info(str(checkpoint_path), verbose=True)

        # Should return complete validated config
        assert config["vae_dim"] == 64
        assert config["flow_dim"] == 128
        assert config["compression_ratio"] == 4

    def test_consistency_check(self, mock_checkpoint_state, temp_dir):
        """Test VAE dimension consistency check."""
        checkpoint_path = temp_dir / "model.safetensors"
        safetensors.torch.save_file(mock_checkpoint_state, str(checkpoint_path))

        config = detect_fluxflow_config(str(checkpoint_path))

        # vae_dim and vae_dim_check should match
        assert config["vae_dim"] == config["vae_dim_check"]

        # Validation should pass
        assert validate_config(config) is True
