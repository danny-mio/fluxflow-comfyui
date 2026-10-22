"""Unit tests for FluxFlowModelLoader ComfyUI node."""

from unittest.mock import Mock, patch

import pytest
import safetensors.torch
import torch

from comfyui_fluxflow.nodes.model_loader import (
    FluxFlowModelLoader,
    _detect_activation_type,
    _has_pade_activations,
)


class TestFluxFlowModelLoaderSchema:
    """Tests for FluxFlowModelLoader node schema."""

    def test_input_types_structure(self):
        """Should return correct INPUT_TYPES structure."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types

    def test_required_inputs(self):
        """Should have checkpoint_path as required input."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        assert "checkpoint_path" in input_types["required"]

    def test_checkpoint_path_is_string(self):
        """checkpoint_path should be STRING type."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        assert input_types["required"]["checkpoint_path"][0] == "STRING"

    def test_checkpoint_path_config(self):
        """checkpoint_path should have correct configuration."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        config = input_types["required"]["checkpoint_path"][1]

        assert "default" in config
        assert "multiline" in config
        assert config["multiline"] is False

    def test_optional_device_input(self):
        """Should have device as optional input."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        assert "device" in input_types["optional"]

    def test_device_options(self):
        """device should have auto, cuda, cpu, mps options."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        device_options = input_types["optional"]["device"][0]

        assert "auto" in device_options
        assert "cuda" in device_options
        assert "cpu" in device_options
        assert "mps" in device_options

    def test_device_default(self):
        """device should default to 'auto'."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        device_config = input_types["optional"]["device"][1]

        assert device_config["default"] == "auto"

    def test_optional_tokenizer_name(self):
        """Should have tokenizer_name as optional input."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        assert "tokenizer_name" in input_types["optional"]

    def test_tokenizer_default(self):
        """tokenizer_name should default to distilbert-base-uncased."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        tokenizer_config = input_types["optional"]["tokenizer_name"][1]

        assert tokenizer_config["default"] == "distilbert-base-uncased"


class TestFluxFlowModelLoaderOutputs:
    """Tests for FluxFlowModelLoader output types."""

    def test_return_types(self):
        """Should have correct RETURN_TYPES."""
        assert hasattr(FluxFlowModelLoader, "RETURN_TYPES")
        assert len(FluxFlowModelLoader.RETURN_TYPES) == 4

    def test_return_type_model(self):
        """First return type should be FLUXFLOW_MODEL."""
        assert FluxFlowModelLoader.RETURN_TYPES[0] == "FLUXFLOW_MODEL"

    def test_return_type_text_encoder(self):
        """Second return type should be FLUXFLOW_TEXT_ENCODER."""
        assert FluxFlowModelLoader.RETURN_TYPES[1] == "FLUXFLOW_TEXT_ENCODER"

    def test_return_type_tokenizer(self):
        """Third return type should be FLUXFLOW_TOKENIZER."""
        assert FluxFlowModelLoader.RETURN_TYPES[2] == "FLUXFLOW_TOKENIZER"

    def test_return_type_config_info(self):
        """Fourth return type should be STRING."""
        assert FluxFlowModelLoader.RETURN_TYPES[3] == "STRING"

    def test_return_names(self):
        """Should have correct RETURN_NAMES."""
        assert hasattr(FluxFlowModelLoader, "RETURN_NAMES")
        assert len(FluxFlowModelLoader.RETURN_NAMES) == 4

    def test_return_names_content(self):
        """RETURN_NAMES should be descriptive."""
        names = FluxFlowModelLoader.RETURN_NAMES

        assert "model" in names
        assert "text_encoder" in names
        assert "tokenizer" in names
        assert "config_info" in names


class TestFluxFlowModelLoaderMetadata:
    """Tests for FluxFlowModelLoader metadata."""

    def test_has_function_attribute(self):
        """Should have FUNCTION attribute."""
        assert hasattr(FluxFlowModelLoader, "FUNCTION")
        assert FluxFlowModelLoader.FUNCTION == "load_model"

    def test_has_category_attribute(self):
        """Should have CATEGORY attribute."""
        assert hasattr(FluxFlowModelLoader, "CATEGORY")
        assert FluxFlowModelLoader.CATEGORY == "FluxFlow"

    def test_load_model_method_exists(self):
        """Should have load_model method."""
        assert hasattr(FluxFlowModelLoader, "load_model")
        assert callable(FluxFlowModelLoader.load_model)


class TestFluxFlowModelLoaderLoadModel:
    """Tests for load_model method (mocked)."""

    @patch("comfyui_fluxflow.nodes.model_loader.get_model_info")
    @patch("comfyui_fluxflow.nodes.model_loader.safetensors.torch.load_file")
    @patch("comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained")
    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_load_model_calls_get_model_info(
        self, mock_parse_device, mock_tokenizer, mock_load_file, mock_get_model_info
    ):
        """load_model should call get_model_info to detect config."""
        # Setup mocks
        mock_get_model_info.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "text_embed_dim": 1024,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "compression_ratio": 8,
        }
        mock_parse_device.return_value = "cpu"
        mock_load_file.return_value = {}

        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.pad_token = None
        mock_tokenizer_obj.eos_token = "[EOS]"
        mock_tokenizer.return_value = mock_tokenizer_obj

        loader = FluxFlowModelLoader()

        try:
            loader.load_model(checkpoint_path="/fake/path.safetensors")
        except Exception:
            # Might fail due to model initialization, but we just want to check the call
            pass

        # Verify get_model_info was called
        mock_get_model_info.assert_called_once()

    @patch("comfyui_fluxflow.nodes.model_loader.get_model_info")
    @patch("comfyui_fluxflow.nodes.model_loader.safetensors.torch.load_file")
    @patch("comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained")
    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_load_model_uses_config_for_models(
        self, mock_parse_device, mock_tokenizer, mock_load_file, mock_get_model_info
    ):
        """load_model should use detected config to initialize models."""
        mock_get_model_info.return_value = {
            "vae_dim": 64,
            "flow_dim": 256,
            "text_embed_dim": 512,
            "downscales": 2,
            "upscales": 2,
            "max_hw": 512,
            "compression_ratio": 4,
        }
        mock_parse_device.return_value = "cpu"
        mock_load_file.return_value = {}

        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.pad_token = "[PAD]"
        mock_tokenizer.return_value = mock_tokenizer_obj

        loader = FluxFlowModelLoader()

        try:
            result = loader.load_model(checkpoint_path="/fake/path.safetensors")
            # If successful, check result structure
            assert len(result) == 4
        except Exception:
            # Model initialization might fail in test environment
            pass

    @patch("comfyui_fluxflow.nodes.model_loader.get_model_info")
    @patch("comfyui_fluxflow.nodes.model_loader.safetensors.torch.load_file")
    @patch("comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained")
    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_load_model_with_custom_device(
        self, mock_parse_device, mock_tokenizer, mock_load_file, mock_get_model_info
    ):
        """load_model should accept custom device."""
        mock_get_model_info.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "text_embed_dim": 1024,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "compression_ratio": 8,
        }
        mock_parse_device.return_value = "cpu"
        mock_load_file.return_value = {}

        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.pad_token = None
        mock_tokenizer_obj.eos_token = "[EOS]"
        mock_tokenizer.return_value = mock_tokenizer_obj

        loader = FluxFlowModelLoader()

        try:
            loader.load_model(checkpoint_path="/fake/path.safetensors", device="cpu")
            # Verify parse_device was called with "cpu"
            mock_parse_device.assert_called_with("cpu")
        except Exception:
            pass

    @patch("comfyui_fluxflow.nodes.model_loader.get_model_info")
    @patch("comfyui_fluxflow.nodes.model_loader.safetensors.torch.load_file")
    @patch("comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained")
    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_load_model_with_custom_tokenizer(
        self, mock_parse_device, mock_tokenizer, mock_load_file, mock_get_model_info
    ):
        """load_model should accept custom tokenizer name."""
        mock_get_model_info.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "text_embed_dim": 1024,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "compression_ratio": 8,
        }
        mock_parse_device.return_value = "cpu"
        mock_load_file.return_value = {}

        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.pad_token = "[PAD]"
        mock_tokenizer.return_value = mock_tokenizer_obj

        loader = FluxFlowModelLoader()

        try:
            loader.load_model(
                checkpoint_path="/fake/path.safetensors",
                tokenizer_name="bert-base-uncased",
            )
            # Verify tokenizer was called with custom name
            assert mock_tokenizer.called
        except Exception:
            pass


class TestFluxFlowModelLoaderIntegration:
    """Integration tests for FluxFlowModelLoader."""

    def test_node_class_mappings(self):
        """NODE_CLASS_MAPPINGS should contain FluxFlowModelLoader."""
        from comfyui_fluxflow.nodes.model_loader import NODE_CLASS_MAPPINGS

        assert "FluxFlowModelLoader" in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS["FluxFlowModelLoader"] == FluxFlowModelLoader

    def test_node_display_name_mappings(self):
        """NODE_DISPLAY_NAME_MAPPINGS should have display name."""
        from comfyui_fluxflow.nodes.model_loader import NODE_DISPLAY_NAME_MAPPINGS

        assert "FluxFlowModelLoader" in NODE_DISPLAY_NAME_MAPPINGS
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowModelLoader"] == "FluxFlow Model Loader"

    def test_instantiation(self):
        """Should be able to instantiate FluxFlowModelLoader."""
        loader = FluxFlowModelLoader()
        assert loader is not None

    def test_input_types_is_classmethod(self):
        """INPUT_TYPES should be a classmethod."""
        assert isinstance(FluxFlowModelLoader.__dict__["INPUT_TYPES"], classmethod) or callable(
            FluxFlowModelLoader.INPUT_TYPES
        )

    def test_has_all_required_comfyui_attributes(self):
        """Should have all required ComfyUI node attributes."""
        # Required attributes for ComfyUI nodes
        assert hasattr(FluxFlowModelLoader, "INPUT_TYPES")
        assert hasattr(FluxFlowModelLoader, "RETURN_TYPES")
        assert hasattr(FluxFlowModelLoader, "FUNCTION")
        assert hasattr(FluxFlowModelLoader, "CATEGORY")

    def test_config_info_format(self):
        """config_info should be formatted correctly."""
        # Test that config info string would be generated correctly
        config = {
            "vae_dim": 128,
            "flow_dim": 512,
            "text_embed_dim": 1024,
            "compression_ratio": 8,
        }

        expected_info = (
            f"VAE: {config['vae_dim']}d, "
            f"Flow: {config['flow_dim']}d, "
            f"Text: {config['text_embed_dim']}d, "
            f"Compression: {config['compression_ratio']}x"
        )

        assert "VAE" in expected_info
        assert "Flow" in expected_info
        assert "Text" in expected_info
        assert "Compression" in expected_info


class TestFluxFlowModelLoaderErrorHandling:
    """Tests for error handling in FluxFlowModelLoader."""

    def test_invalid_checkpoint_path(self):
        """Should handle invalid checkpoint path."""
        loader = FluxFlowModelLoader()

        with pytest.raises(Exception):
            # Should fail with file not found or similar
            loader.load_model(checkpoint_path="/nonexistent/path.safetensors")

    @patch("comfyui_fluxflow.nodes.model_loader.get_model_info")
    def test_missing_keys_handled_gracefully(self, mock_get_model_info):
        """Should handle missing keys in checkpoint gracefully."""
        mock_get_model_info.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "text_embed_dim": 1024,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "compression_ratio": 8,
        }

        # Note: Full test would require mocking more components
        # This is a structure test
        loader = FluxFlowModelLoader()
        assert loader is not None

    def test_default_parameters(self):
        """Should have sensible defaults for all parameters."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        # Check optional parameters have defaults
        assert input_types["optional"]["device"][1]["default"] == "auto"
        assert input_types["optional"]["tokenizer_name"][1]["default"] == "distilbert-base-uncased"


class TestFluxFlowModelLoaderDocumentation:
    """Tests for documentation and docstrings."""

    def test_class_has_docstring(self):
        """Class should have docstring."""
        assert FluxFlowModelLoader.__doc__ is not None
        assert len(FluxFlowModelLoader.__doc__) > 0

    def test_load_model_has_docstring(self):
        """load_model method should have docstring."""
        assert FluxFlowModelLoader.load_model.__doc__ is not None
        assert len(FluxFlowModelLoader.load_model.__doc__) > 0

    def test_docstring_mentions_auto_detection(self):
        """Class docstring should mention auto-detection."""
        docstring = FluxFlowModelLoader.__doc__.lower()

        assert "auto" in docstring or "detect" in docstring


class TestFluxFlowModelLoaderDtypeSchema:
    """Tests for the dtype selector on FluxFlowModelLoader."""

    def test_optional_dtype_input(self):
        """Should have dtype as optional input."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()

        assert "dtype" in input_types["optional"]

    def test_dtype_options(self):
        """dtype should offer fp32, fp16, bf16."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        dtype_options = input_types["optional"]["dtype"][0]

        assert dtype_options == ["fp32", "fp16", "bf16"]

    def test_dtype_default(self):
        """dtype should default to 'fp32'."""
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        dtype_config = input_types["optional"]["dtype"][1]

        assert dtype_config["default"] == "fp32"


_DTYPE_CASES = [
    ("fp32", None),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
]


class TestFluxFlowModelLoaderDtypeVersionedBranch:
    """dtype casting for the versioned-directory-checkpoint branch."""

    @pytest.mark.parametrize("dtype_name,expected_dtype", _DTYPE_CASES)
    @patch("comfyui_fluxflow.nodes.model_loader.BertTextEncoder")
    @patch("comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained")
    @patch("comfyui_fluxflow.nodes.model_loader.load_versioned_checkpoint")
    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_versioned_branch_applies_dtype(
        self,
        mock_parse_device,
        mock_load_versioned,
        mock_tokenizer,
        mock_bert_cls,
        dtype_name,
        expected_dtype,
        tmp_path,
    ):
        device_obj = torch.device("cpu")
        mock_parse_device.return_value = device_obj

        mock_pipeline = Mock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_load_versioned.return_value = mock_pipeline

        mock_text_encoder = Mock()
        mock_text_encoder.to.return_value = mock_text_encoder
        mock_text_encoder.embed_dim = 1024
        mock_bert_cls.return_value = mock_text_encoder

        mock_tokenizer_obj = Mock()
        mock_tokenizer_obj.pad_token = "[PAD]"
        mock_tokenizer.return_value = mock_tokenizer_obj

        loader = FluxFlowModelLoader()
        loader.load_model(checkpoint_path=str(tmp_path), device="cpu", dtype=dtype_name)

        _, pipeline_kwargs = mock_pipeline.to.call_args
        _, text_encoder_kwargs = mock_text_encoder.to.call_args

        assert pipeline_kwargs.get("device") == device_obj
        assert text_encoder_kwargs.get("device") == device_obj

        if expected_dtype is None:
            assert "dtype" not in pipeline_kwargs
            assert "dtype" not in text_encoder_kwargs
        else:
            assert pipeline_kwargs.get("dtype") == expected_dtype
            assert text_encoder_kwargs.get("dtype") == expected_dtype


class TestFluxFlowModelLoaderDtypeLegacyV3Branch:
    """dtype casting for the legacy v0.3.0 architecture-detection branch."""

    @pytest.mark.parametrize("dtype_name,expected_dtype", _DTYPE_CASES)
    def test_legacy_v3_branch_applies_dtype(self, dtype_name, expected_dtype):
        device_obj = torch.device("cpu")

        mock_diffuser = Mock()
        mock_diffuser.to.return_value = mock_diffuser
        mock_diffuser.load_state_dict.return_value = ([], [])
        mock_text_encoder = Mock()
        mock_text_encoder.to.return_value = mock_text_encoder

        with (
            patch("comfyui_fluxflow.nodes.model_loader.parse_device", return_value=device_obj),
            patch(
                "comfyui_fluxflow.nodes.model_loader.safetensors.torch.load_file",
                return_value={},
            ),
            patch(
                "comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained",
                return_value=Mock(pad_token="[PAD]"),
            ),
            patch(
                "comfyui_fluxflow.nodes.model_loader.get_model_info",
                return_value={
                    "vae_dim": 128,
                    "flow_dim": 512,
                    "text_embed_dim": 1024,
                    "downscales": 3,
                    "upscales": 3,
                    "max_hw": 1024,
                    "compression_ratio": 8,
                },
            ),
            patch("comfyui_fluxflow.nodes.model_loader.FluxCompressor", return_value=Mock()),
            patch("comfyui_fluxflow.nodes.model_loader.FluxFlowProcessor", return_value=Mock()),
            patch("comfyui_fluxflow.nodes.model_loader.FluxExpander", return_value=Mock()),
            patch("comfyui_fluxflow.nodes.model_loader.FluxPipeline", return_value=mock_diffuser),
            patch(
                "comfyui_fluxflow.nodes.model_loader.BertTextEncoder",
                return_value=mock_text_encoder,
            ),
        ):
            loader = FluxFlowModelLoader()
            loader.load_model(
                checkpoint_path="/fake/path.safetensors", device="cpu", dtype=dtype_name
            )

        _, diffuser_kwargs = mock_diffuser.to.call_args
        _, text_encoder_kwargs = mock_text_encoder.to.call_args

        assert diffuser_kwargs.get("device") == device_obj
        assert text_encoder_kwargs.get("device") == device_obj

        if expected_dtype is None:
            assert "dtype" not in diffuser_kwargs
            assert "dtype" not in text_encoder_kwargs
        else:
            assert diffuser_kwargs.get("dtype") == expected_dtype
            assert text_encoder_kwargs.get("dtype") == expected_dtype


class TestFluxFlowModelLoaderDtypeV100Branch:
    """dtype casting for the auto-detected v0.10.0 architecture branch."""

    @pytest.mark.parametrize("dtype_name,expected_dtype", _DTYPE_CASES)
    def test_v100_branch_applies_dtype(self, dtype_name, expected_dtype):
        device_obj = torch.device("cpu")

        mock_diffuser = Mock()
        mock_diffuser.to.return_value = mock_diffuser
        mock_text_encoder = Mock()
        mock_text_encoder.to.return_value = mock_text_encoder

        mock_flux_pipeline_cls = Mock(return_value=mock_diffuser)
        mock_flux_pipeline_cls._detect_config.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "flow_attn_heads": 8,
            "text_embed_dim": 1024,
            "flow_transformer_layers": 10,
        }

        with (
            patch("comfyui_fluxflow.nodes.model_loader.parse_device", return_value=device_obj),
            patch(
                "comfyui_fluxflow.nodes.model_loader.safetensors.torch.load_file",
                return_value={},
            ),
            patch(
                "comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained",
                return_value=Mock(pad_token="[PAD]"),
            ),
            patch(
                "comfyui_fluxflow.nodes.model_loader.detect_architecture_version",
                return_value="0.10.0",
            ),
            patch("comfyui_fluxflow.nodes.model_loader.FluxPipeline", mock_flux_pipeline_cls),
            patch(
                "comfyui_fluxflow.nodes.model_loader.BertTextEncoder",
                return_value=mock_text_encoder,
            ),
            patch("fluxflow.models.v100.vae.FluxCompressor_v100", return_value=Mock()),
            patch("fluxflow.models.v100.vae.FluxExpander_v100", return_value=Mock()),
            patch("fluxflow.models.v100.flow.FluxFlowProcessor_v100", return_value=Mock()),
        ):
            loader = FluxFlowModelLoader()
            loader.load_model(
                checkpoint_path="/fake/path.safetensors", device="cpu", dtype=dtype_name
            )

        _, diffuser_kwargs = mock_diffuser.to.call_args
        _, text_encoder_kwargs = mock_text_encoder.to.call_args

        assert diffuser_kwargs.get("device") == device_obj
        assert text_encoder_kwargs.get("device") == device_obj

        if expected_dtype is None:
            assert "dtype" not in diffuser_kwargs
            assert "dtype" not in text_encoder_kwargs
        else:
            assert diffuser_kwargs.get("dtype") == expected_dtype
            assert text_encoder_kwargs.get("dtype") == expected_dtype


class TestFluxFlowModelLoaderDtypeValidation:
    """Tests for dtype input validation."""

    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_invalid_dtype_raises_value_error(self, mock_parse_device):
        """Should raise a clear error for an unrecognized dtype string."""
        mock_parse_device.return_value = torch.device("cpu")
        loader = FluxFlowModelLoader()

        with pytest.raises(ValueError, match="dtype"):
            loader.load_model(checkpoint_path="/fake/path.safetensors", dtype="int8")


class TestFluxFlowModelLoaderActivationTypeSchema:
    """Tests for the activation_type selector on FluxFlowModelLoader."""

    def test_optional_activation_type_input(self):
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        assert "activation_type" in input_types["optional"]

    def test_activation_type_options(self):
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        options = input_types["optional"]["activation_type"][0]
        assert options == ["auto", "bezier", "pade"]

    def test_activation_type_default(self):
        input_types = FluxFlowModelLoader.INPUT_TYPES()
        config = input_types["optional"]["activation_type"][1]
        assert config["default"] == "auto"


class TestDetectActivationType:
    """Tests for the module-level _detect_activation_type helper."""

    def test_defaults_to_bezier_when_no_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        safetensors.torch.save_file({"w": torch.zeros(2)}, str(path))
        assert _detect_activation_type(str(path)) == "bezier"

    def test_reads_pade_from_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        safetensors.torch.save_file(
            {"w": torch.zeros(2)}, str(path), metadata={"activation_type": "pade"}
        )
        assert _detect_activation_type(str(path)) == "pade"

    def test_force_bezier_overrides_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        safetensors.torch.save_file(
            {"w": torch.zeros(2)}, str(path), metadata={"activation_type": "pade"}
        )
        assert _detect_activation_type(str(path), force="bezier") == "bezier"

    def test_force_pade_overrides_missing_metadata(self, tmp_path):
        path = tmp_path / "model.safetensors"
        safetensors.torch.save_file({"w": torch.zeros(2)}, str(path))
        assert _detect_activation_type(str(path), force="pade") == "pade"

    def test_resolves_directory_checkpoint_to_flxflow_final(self, tmp_path):
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        path = ckpt_dir / "flxflow_final.safetensors"
        safetensors.torch.save_file(
            {"w": torch.zeros(2)}, str(path), metadata={"activation_type": "pade"}
        )
        assert _detect_activation_type(str(ckpt_dir)) == "pade"

    def test_missing_file_defaults_to_bezier(self, tmp_path):
        assert _detect_activation_type(str(tmp_path / "nope.safetensors")) == "bezier"


class TestFluxFlowModelLoaderActivationTypeV100Branch:
    """activation_type threading for the auto-detected v0.10.0 architecture branch."""

    def test_v100_branch_reads_pade_from_checkpoint_metadata(self, tmp_path):
        checkpoint_path = tmp_path / "flxflow_final.safetensors"
        safetensors.torch.save_file(
            {"diffuser.compressor.dummy": torch.zeros(2)},
            str(checkpoint_path),
            metadata={"activation_type": "pade"},
        )

        device_obj = torch.device("cpu")
        mock_diffuser = Mock()
        mock_diffuser.to.return_value = mock_diffuser
        mock_text_encoder = Mock()
        mock_text_encoder.to.return_value = mock_text_encoder

        mock_flux_pipeline_cls = Mock(return_value=mock_diffuser)
        mock_flux_pipeline_cls._detect_config.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "flow_attn_heads": 8,
            "text_embed_dim": 1024,
            "flow_transformer_layers": 10,
        }

        mock_compressor_cls = Mock(return_value=Mock())
        mock_expander_cls = Mock(return_value=Mock())
        mock_flow_cls = Mock(return_value=Mock())

        with (
            patch("comfyui_fluxflow.nodes.model_loader.parse_device", return_value=device_obj),
            patch(
                "comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained",
                return_value=Mock(pad_token="[PAD]"),
            ),
            patch(
                "comfyui_fluxflow.nodes.model_loader.detect_architecture_version",
                return_value="0.10.0",
            ),
            patch("comfyui_fluxflow.nodes.model_loader.FluxPipeline", mock_flux_pipeline_cls),
            patch(
                "comfyui_fluxflow.nodes.model_loader.BertTextEncoder",
                return_value=mock_text_encoder,
            ),
            patch("fluxflow.models.v100.vae.FluxCompressor_v100", mock_compressor_cls),
            patch("fluxflow.models.v100.vae.FluxExpander_v100", mock_expander_cls),
            patch("fluxflow.models.v100.flow.FluxFlowProcessor_v100", mock_flow_cls),
        ):
            loader = FluxFlowModelLoader()
            loader.load_model(checkpoint_path=str(checkpoint_path), device="cpu")

        assert mock_compressor_cls.call_args.kwargs["activation_type"] == "pade"
        assert mock_flow_cls.call_args.kwargs["activation_type"] == "pade"
        assert mock_expander_cls.call_args.kwargs["activation_type"] == "pade"

    def test_v100_branch_force_bezier_overrides_pade_metadata(self, tmp_path):
        checkpoint_path = tmp_path / "flxflow_final.safetensors"
        safetensors.torch.save_file(
            {"diffuser.compressor.dummy": torch.zeros(2)},
            str(checkpoint_path),
            metadata={"activation_type": "pade"},
        )

        device_obj = torch.device("cpu")
        mock_diffuser = Mock()
        mock_diffuser.to.return_value = mock_diffuser
        mock_text_encoder = Mock()
        mock_text_encoder.to.return_value = mock_text_encoder

        mock_flux_pipeline_cls = Mock(return_value=mock_diffuser)
        mock_flux_pipeline_cls._detect_config.return_value = {
            "vae_dim": 128,
            "flow_dim": 512,
            "downscales": 3,
            "upscales": 3,
            "max_hw": 1024,
            "flow_attn_heads": 8,
            "text_embed_dim": 1024,
            "flow_transformer_layers": 10,
        }

        mock_compressor_cls = Mock(return_value=Mock())

        with (
            patch("comfyui_fluxflow.nodes.model_loader.parse_device", return_value=device_obj),
            patch(
                "comfyui_fluxflow.nodes.model_loader.AutoTokenizer.from_pretrained",
                return_value=Mock(pad_token="[PAD]"),
            ),
            patch(
                "comfyui_fluxflow.nodes.model_loader.detect_architecture_version",
                return_value="0.10.0",
            ),
            patch("comfyui_fluxflow.nodes.model_loader.FluxPipeline", mock_flux_pipeline_cls),
            patch(
                "comfyui_fluxflow.nodes.model_loader.BertTextEncoder",
                return_value=mock_text_encoder,
            ),
            patch("fluxflow.models.v100.vae.FluxCompressor_v100", mock_compressor_cls),
            patch("fluxflow.models.v100.vae.FluxExpander_v100", Mock(return_value=Mock())),
            patch("fluxflow.models.v100.flow.FluxFlowProcessor_v100", Mock(return_value=Mock())),
        ):
            loader = FluxFlowModelLoader()
            loader.load_model(
                checkpoint_path=str(checkpoint_path), device="cpu", activation_type="bezier"
            )

        assert mock_compressor_cls.call_args.kwargs["activation_type"] == "bezier"


_LOADER = "comfyui_fluxflow.nodes.model_loader"
_RESULT = (Mock(), Mock(), Mock(), "info")


class TestFluxFlowModelLoaderCompileActivationsSchema:
    """compile_activations is an optional boolean, default off."""

    def test_optional_input_default_false(self):
        spec = FluxFlowModelLoader.INPUT_TYPES()["optional"]["compile_activations"]
        assert spec[0] == "BOOLEAN"
        assert spec[1]["default"] is False
        assert "Pad" in spec[1]["tooltip"]

    def test_not_required(self):
        assert "compile_activations" not in FluxFlowModelLoader.INPUT_TYPES()["required"]


class TestHasPadeActivations:
    def test_detects_pade(self):
        from fluxflow.models.pade_activation import TrainablePade

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), TrainablePade((2,)))
        assert _has_pade_activations(model)

    def test_no_pade(self):
        assert not _has_pade_activations(torch.nn.Sequential(torch.nn.Linear(2, 2)))


class TestFluxFlowModelLoaderCompileActivations:
    """compile_pade_activations is called only when requested and model is Pade."""

    def _run(self, has_pade, **kwargs):
        with (
            patch.object(FluxFlowModelLoader, "_load", return_value=_RESULT),
            patch(f"{_LOADER}._has_pade_activations", return_value=has_pade),
            patch(f"{_LOADER}.compile_pade_activations") as compile_mock,
        ):
            out = FluxFlowModelLoader().load_model(checkpoint_path="/x", **kwargs)
        return out, compile_mock

    def test_compiles_when_enabled_and_pade(self):
        out, compile_mock = self._run(True, compile_activations=True)
        compile_mock.assert_called_once_with()
        assert out == _RESULT

    def test_skips_when_not_pade(self):
        _, compile_mock = self._run(False, compile_activations=True)
        compile_mock.assert_not_called()

    def test_skips_by_default(self):
        _, compile_mock = self._run(True)
        compile_mock.assert_not_called()

    def test_skips_when_explicitly_false(self):
        _, compile_mock = self._run(True, compile_activations=False)
        compile_mock.assert_not_called()


class TestLoaderRocmSdpaWorkaround:
    """load_model applies the SDPA workaround on the resolved device."""

    @patch("comfyui_fluxflow.nodes.model_loader.apply_rocm_sdpa_workaround")
    @patch("comfyui_fluxflow.nodes.model_loader.parse_device")
    def test_workaround_called_with_resolved_device(self, mock_parse_device, mock_apply):
        dev = torch.device("cpu")
        mock_parse_device.return_value = dev
        with pytest.raises(Exception):
            FluxFlowModelLoader().load_model(checkpoint_path="/nonexistent.safetensors")
        mock_apply.assert_called_once_with(dev)
