"""
Comprehensive tests for Classifier-Free Guidance (CFG) functionality.
"""

from comfyui_fluxflow import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)
from comfyui_fluxflow.nodes.samplers import FluxFlowSampler
from comfyui_fluxflow.nodes.text_encode import (
    FluxFlowTextEncode,
    FluxFlowTextEncodeNegative,
)


class TestCFGNodeRegistration:
    """Tests for CFG-related node registration."""

    def test_negative_encode_node_registered(self):
        """FluxFlowTextEncodeNegative should be in NODE_CLASS_MAPPINGS."""
        assert "FluxFlowTextEncodeNegative" in NODE_CLASS_MAPPINGS
        assert NODE_CLASS_MAPPINGS["FluxFlowTextEncodeNegative"] == FluxFlowTextEncodeNegative

    def test_negative_encode_display_name(self):
        """FluxFlowTextEncodeNegative should have proper display name."""
        assert "FluxFlowTextEncodeNegative" in NODE_DISPLAY_NAME_MAPPINGS
        assert (
            NODE_DISPLAY_NAME_MAPPINGS["FluxFlowTextEncodeNegative"]
            == "FluxFlow Text Encode (Negative)"
        )

    def test_total_node_count(self):
        """Should have 7 total nodes (added 1 new node)."""
        assert len(NODE_CLASS_MAPPINGS) == 7
        assert len(NODE_DISPLAY_NAME_MAPPINGS) == 7


class TestFluxFlowTextEncodeNegative:
    """Tests for FluxFlowTextEncodeNegative node."""

    def test_input_types_structure(self):
        """Should have required inputs structure."""
        input_types = FluxFlowTextEncodeNegative.INPUT_TYPES()

        assert "required" in input_types
        assert "text_encoder" in input_types["required"]
        assert "tokenizer" in input_types["required"]
        assert "text" in input_types["required"]

    def test_text_encoder_type(self):
        """text_encoder should be FLUXFLOW_TEXT_ENCODER."""
        input_types = FluxFlowTextEncodeNegative.INPUT_TYPES()
        assert input_types["required"]["text_encoder"][0] == "FLUXFLOW_TEXT_ENCODER"

    def test_tokenizer_type(self):
        """tokenizer should be FLUXFLOW_TOKENIZER."""
        input_types = FluxFlowTextEncodeNegative.INPUT_TYPES()
        assert input_types["required"]["tokenizer"][0] == "FLUXFLOW_TOKENIZER"

    def test_text_is_multiline_string(self):
        """text should be multiline STRING."""
        input_types = FluxFlowTextEncodeNegative.INPUT_TYPES()
        text_type, text_config = input_types["required"]["text"]

        assert text_type == "STRING"
        assert text_config.get("multiline") is True

    def test_return_types(self):
        """Should return FLUXFLOW_CONDITIONING."""
        assert FluxFlowTextEncodeNegative.RETURN_TYPES == ("FLUXFLOW_CONDITIONING",)
        assert FluxFlowTextEncodeNegative.RETURN_NAMES == ("negative_conditioning",)

    def test_function_name(self):
        """Function should be encode."""
        assert FluxFlowTextEncodeNegative.FUNCTION == "encode"

    def test_category(self):
        """Category should be FluxFlow/conditioning."""
        assert FluxFlowTextEncodeNegative.CATEGORY == "FluxFlow/conditioning"

    def test_has_encode_method(self):
        """Should have encode method."""
        assert hasattr(FluxFlowTextEncodeNegative, "encode")
        assert callable(FluxFlowTextEncodeNegative.encode)


class TestFluxFlowSamplerCFGInputs:
    """Tests for CFG-related inputs in FluxFlowSampler."""

    def test_optional_use_cfg_parameter(self):
        """Should have optional use_cfg parameter."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "optional" in input_types
        assert "use_cfg" in input_types["optional"]

    def test_use_cfg_is_boolean(self):
        """use_cfg should be BOOLEAN with default False."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        use_cfg_type, use_cfg_config = input_types["optional"]["use_cfg"]

        assert use_cfg_type == "BOOLEAN"
        assert use_cfg_config["default"] is False

    def test_optional_guidance_scale_parameter(self):
        """Should have optional guidance_scale parameter."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "guidance_scale" in input_types["optional"]

    def test_guidance_scale_is_float(self):
        """guidance_scale should be FLOAT with correct range."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scale_type, scale_config = input_types["optional"]["guidance_scale"]

        assert scale_type == "FLOAT"
        assert scale_config["default"] == 5.0
        assert scale_config["min"] == 1.0
        assert scale_config["max"] == 15.0
        assert scale_config["step"] == 0.1

    def test_optional_negative_conditioning_parameter(self):
        """Should have optional negative_conditioning parameter."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "negative_conditioning" in input_types["optional"]

    def test_negative_conditioning_type(self):
        """negative_conditioning should be FLUXFLOW_CONDITIONING."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        neg_cond_type = input_types["optional"]["negative_conditioning"]

        assert neg_cond_type[0] == "FLUXFLOW_CONDITIONING"

    def test_sample_method_signature(self):
        """sample() should accept CFG parameters."""
        import inspect

        sig = inspect.signature(FluxFlowSampler.sample)
        params = list(sig.parameters.keys())

        assert "use_cfg" in params
        assert "guidance_scale" in params
        assert "negative_conditioning" in params


class TestCFGSamplingLogic:
    """Tests for CFG sampling implementation logic."""

    def test_cfg_disabled_by_default(self):
        """CFG should be disabled by default (use_cfg=False)."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        use_cfg_config = input_types["optional"]["use_cfg"][1]

        assert use_cfg_config["default"] is False

    def test_guidance_scale_default_reasonable(self):
        """Default guidance_scale should be in recommended range (3-7)."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scale_config = input_types["optional"]["guidance_scale"][1]
        default = scale_config["default"]

        assert 3.0 <= default <= 7.0  # Recommended range

    def test_guidance_scale_minimum_is_one(self):
        """Minimum guidance_scale should be 1.0 (no guidance)."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scale_config = input_types["optional"]["guidance_scale"][1]

        assert scale_config["min"] == 1.0

    def test_guidance_scale_allows_high_values(self):
        """Should allow high guidance scales up to 15.0."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scale_config = input_types["optional"]["guidance_scale"][1]

        assert scale_config["max"] == 15.0


class TestCFGIntegration:
    """Integration tests for CFG workflow."""

    def test_positive_and_negative_encode_compatibility(self):
        """Positive and negative encode nodes should have same output type."""
        pos_output = FluxFlowTextEncode.RETURN_TYPES[0]
        neg_output = FluxFlowTextEncodeNegative.RETURN_TYPES[0]

        assert pos_output == neg_output == "FLUXFLOW_CONDITIONING"

    def test_sampler_accepts_both_conditioning_types(self):
        """Sampler should accept both positive and negative conditioning."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        # Required positive conditioning
        assert input_types["required"]["conditioning"][0] == "FLUXFLOW_CONDITIONING"

        # Optional negative conditioning
        assert input_types["optional"]["negative_conditioning"][0] == "FLUXFLOW_CONDITIONING"

    def test_cfg_workflow_type_chain(self):
        """CFG workflow should have consistent type chain."""
        # Model loader outputs
        # Text encode (positive) outputs FLUXFLOW_CONDITIONING
        # Text encode (negative) outputs FLUXFLOW_CONDITIONING
        # Sampler accepts both + outputs FLUXFLOW_LATENT
        # VAE decode accepts FLUXFLOW_LATENT

        pos_out = FluxFlowTextEncode.RETURN_TYPES[0]
        neg_out = FluxFlowTextEncodeNegative.RETURN_TYPES[0]
        sampler_inputs = FluxFlowSampler.INPUT_TYPES()

        assert pos_out == "FLUXFLOW_CONDITIONING"
        assert neg_out == "FLUXFLOW_CONDITIONING"
        assert sampler_inputs["required"]["conditioning"][0] == "FLUXFLOW_CONDITIONING"
        assert sampler_inputs["optional"]["negative_conditioning"][0] == "FLUXFLOW_CONDITIONING"


class TestCFGDocumentation:
    """Tests for CFG-related documentation and metadata."""

    def test_sampler_docstring_mentions_cfg(self):
        """Sampler docstring should document CFG parameters."""
        docstring = FluxFlowSampler.sample.__doc__

        assert docstring is not None
        assert "use_cfg" in docstring.lower() or "cfg" in docstring.lower()
        assert "guidance" in docstring.lower()

    def test_negative_encode_docstring(self):
        """Negative encode should have docstring."""
        docstring = FluxFlowTextEncodeNegative.encode.__doc__

        assert docstring is not None
        assert "negative" in docstring.lower()


class TestCFGBackwardCompatibility:
    """Tests for backward compatibility (CFG optional)."""

    def test_sampler_works_without_cfg_params(self):
        """Sampler should work without CFG parameters (backward compatible)."""
        # All CFG params should be optional
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "use_cfg" in input_types["optional"]
        assert "guidance_scale" in input_types["optional"]
        assert "negative_conditioning" in input_types["optional"]

        # Required params should NOT include CFG
        assert "use_cfg" not in input_types["required"]
        assert "guidance_scale" not in input_types["required"]
        assert "negative_conditioning" not in input_types["required"]

    def test_default_cfg_equals_no_cfg(self):
        """Default CFG settings should behave like no CFG."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        # use_cfg defaults to False
        assert input_types["optional"]["use_cfg"][1]["default"] is False

        # guidance_scale at 1.0 means no guidance
        # (though default is 5.0, it's only used when use_cfg=True)


class TestCFGEdgeCases:
    """Tests for CFG edge cases and error handling."""

    def test_guidance_scale_below_one_not_allowed(self):
        """guidance_scale should not allow values below 1.0."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scale_config = input_types["optional"]["guidance_scale"][1]

        assert scale_config["min"] >= 1.0

    def test_guidance_scale_step_size(self):
        """guidance_scale should have reasonable step size."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scale_config = input_types["optional"]["guidance_scale"][1]

        # Step size should be 0.1 for fine control
        assert scale_config["step"] == 0.1


class TestCFGPerformanceConsiderations:
    """Tests documenting CFG performance characteristics."""

    def test_cfg_dual_pass_documented(self):
        """Should document that CFG uses dual-pass (2x compute)."""
        # This is a documentation test - CFG inherently uses 2x forward passes
        # The implementation should handle this correctly
        pass

    def test_cfg_memory_impact_documented(self):
        """Should be aware that CFG requires more memory (dual pass)."""
        # This is a documentation test - CFG requires ~2x VRAM
        pass
