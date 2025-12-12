"""Comprehensive tests for all FluxFlow ComfyUI nodes."""

from comfyui_fluxflow import (
    NODE_CLASS_MAPPINGS,
    NODE_COLORS,
    NODE_DISPLAY_NAME_MAPPINGS,
    FluxFlowEmptyLatent,
    FluxFlowModelLoader,
    FluxFlowSampler,
    FluxFlowTextEncode,
    FluxFlowVAEDecode,
    FluxFlowVAEEncode,
)


class TestNodeRegistration:
    """Tests for node registration in ComfyUI."""

    def test_node_class_mappings_complete(self):
        """NODE_CLASS_MAPPINGS should contain all 7 nodes."""
        assert len(NODE_CLASS_MAPPINGS) == 7
        assert "FluxFlowModelLoader" in NODE_CLASS_MAPPINGS
        assert "FluxFlowEmptyLatent" in NODE_CLASS_MAPPINGS
        assert "FluxFlowVAEEncode" in NODE_CLASS_MAPPINGS
        assert "FluxFlowVAEDecode" in NODE_CLASS_MAPPINGS
        assert "FluxFlowTextEncode" in NODE_CLASS_MAPPINGS
        assert "FluxFlowTextEncodeNegative" in NODE_CLASS_MAPPINGS
        assert "FluxFlowSampler" in NODE_CLASS_MAPPINGS

    def test_node_display_name_mappings_complete(self):
        """NODE_DISPLAY_NAME_MAPPINGS should have display names for all nodes."""
        assert len(NODE_DISPLAY_NAME_MAPPINGS) == 7
        assert "FluxFlowModelLoader" in NODE_DISPLAY_NAME_MAPPINGS
        assert "FluxFlowEmptyLatent" in NODE_DISPLAY_NAME_MAPPINGS
        assert "FluxFlowVAEEncode" in NODE_DISPLAY_NAME_MAPPINGS
        assert "FluxFlowVAEDecode" in NODE_DISPLAY_NAME_MAPPINGS
        assert "FluxFlowTextEncode" in NODE_DISPLAY_NAME_MAPPINGS
        assert "FluxFlowTextEncodeNegative" in NODE_DISPLAY_NAME_MAPPINGS
        assert "FluxFlowSampler" in NODE_DISPLAY_NAME_MAPPINGS

    def test_display_names_descriptive(self):
        """Display names should be human-readable."""
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowModelLoader"] == "FluxFlow Model Loader"
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowEmptyLatent"] == "FluxFlow Empty Latent"
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowVAEEncode"] == "FluxFlow VAE Encode"
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowVAEDecode"] == "FluxFlow VAE Decode"
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowTextEncode"] == "FluxFlow Text Encode"
        assert (
            NODE_DISPLAY_NAME_MAPPINGS["FluxFlowTextEncodeNegative"]
            == "FluxFlow Text Encode (Negative)"
        )
        assert NODE_DISPLAY_NAME_MAPPINGS["FluxFlowSampler"] == "FluxFlow Sampler"

    def test_node_classes_correct(self):
        """NODE_CLASS_MAPPINGS should map to correct classes."""
        assert NODE_CLASS_MAPPINGS["FluxFlowModelLoader"] == FluxFlowModelLoader
        assert NODE_CLASS_MAPPINGS["FluxFlowEmptyLatent"] == FluxFlowEmptyLatent
        assert NODE_CLASS_MAPPINGS["FluxFlowVAEEncode"] == FluxFlowVAEEncode
        assert NODE_CLASS_MAPPINGS["FluxFlowVAEDecode"] == FluxFlowVAEDecode
        assert NODE_CLASS_MAPPINGS["FluxFlowTextEncode"] == FluxFlowTextEncode
        assert NODE_CLASS_MAPPINGS["FluxFlowSampler"] == FluxFlowSampler

    def test_node_colors_defined(self):
        """NODE_COLORS should define colors for custom types."""
        assert "FLUXFLOW_MODEL" in NODE_COLORS
        assert "FLUXFLOW_TEXT_ENCODER" in NODE_COLORS
        assert "FLUXFLOW_TOKENIZER" in NODE_COLORS
        assert "FLUXFLOW_CONDITIONING" in NODE_COLORS
        assert "FLUXFLOW_LATENT" in NODE_COLORS

    def test_node_colors_valid_hex(self):
        """Node colors should be valid hex colors."""
        for type_name, color in NODE_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format


class TestFluxFlowEmptyLatent:
    """Tests for FluxFlowEmptyLatent node."""

    def test_input_types_structure(self):
        """Should have correct INPUT_TYPES structure."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types

    def test_required_inputs(self):
        """Should require model, width, height, batch_size."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()

        assert "model" in input_types["required"]
        assert "width" in input_types["required"]
        assert "height" in input_types["required"]
        assert "batch_size" in input_types["required"]

    def test_model_type(self):
        """model should be FLUXFLOW_MODEL type."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()

        assert input_types["required"]["model"][0] == "FLUXFLOW_MODEL"

    def test_width_config(self):
        """width should be INT with correct defaults."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()
        width_config = input_types["required"]["width"][1]

        assert width_config["default"] == 512
        assert width_config["min"] == 64
        assert width_config["max"] == 2048
        assert width_config["step"] == 64

    def test_height_config(self):
        """height should be INT with correct defaults."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()
        height_config = input_types["required"]["height"][1]

        assert height_config["default"] == 512
        assert height_config["min"] == 64
        assert height_config["max"] == 2048

    def test_batch_size_config(self):
        """batch_size should be INT with correct range."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()
        batch_config = input_types["required"]["batch_size"][1]

        assert batch_config["default"] == 1
        assert batch_config["min"] == 1
        assert batch_config["max"] == 64

    def test_optional_seed(self):
        """Should have optional seed parameter."""
        input_types = FluxFlowEmptyLatent.INPUT_TYPES()

        assert "seed" in input_types["optional"]

    def test_return_types(self):
        """Should return FLUXFLOW_LATENT."""
        assert FluxFlowEmptyLatent.RETURN_TYPES == ("FLUXFLOW_LATENT",)
        assert FluxFlowEmptyLatent.RETURN_NAMES == ("latent",)

    def test_function_name(self):
        """Function should be generate_latent."""
        assert FluxFlowEmptyLatent.FUNCTION == "generate_latent"

    def test_category(self):
        """Category should be FluxFlow/latent."""
        assert FluxFlowEmptyLatent.CATEGORY == "FluxFlow/latent"

    def test_has_generate_latent_method(self):
        """Should have generate_latent method."""
        assert hasattr(FluxFlowEmptyLatent, "generate_latent")
        assert callable(FluxFlowEmptyLatent.generate_latent)


class TestFluxFlowVAEEncode:
    """Tests for FluxFlowVAEEncode node."""

    def test_input_types_structure(self):
        """Should have required inputs."""
        input_types = FluxFlowVAEEncode.INPUT_TYPES()

        assert "required" in input_types

    def test_required_model_and_image(self):
        """Should require model and image."""
        input_types = FluxFlowVAEEncode.INPUT_TYPES()

        assert "model" in input_types["required"]
        assert "image" in input_types["required"]

    def test_model_type(self):
        """model should be FLUXFLOW_MODEL."""
        input_types = FluxFlowVAEEncode.INPUT_TYPES()

        assert input_types["required"]["model"][0] == "FLUXFLOW_MODEL"

    def test_image_type(self):
        """image should be IMAGE type (ComfyUI standard)."""
        input_types = FluxFlowVAEEncode.INPUT_TYPES()

        assert input_types["required"]["image"][0] == "IMAGE"

    def test_return_types(self):
        """Should return FLUXFLOW_LATENT."""
        assert FluxFlowVAEEncode.RETURN_TYPES == ("FLUXFLOW_LATENT",)

    def test_function_name(self):
        """Function should be encode."""
        assert FluxFlowVAEEncode.FUNCTION == "encode"

    def test_category(self):
        """Category should be FluxFlow/latent."""
        assert FluxFlowVAEEncode.CATEGORY == "FluxFlow/latent"


class TestFluxFlowVAEDecode:
    """Tests for FluxFlowVAEDecode node."""

    def test_input_types_structure(self):
        """Should have required inputs."""
        input_types = FluxFlowVAEDecode.INPUT_TYPES()

        assert "required" in input_types

    def test_required_model_and_latent(self):
        """Should require model and latent."""
        input_types = FluxFlowVAEDecode.INPUT_TYPES()

        assert "model" in input_types["required"]
        assert "latent" in input_types["required"]

    def test_model_type(self):
        """model should be FLUXFLOW_MODEL."""
        input_types = FluxFlowVAEDecode.INPUT_TYPES()

        assert input_types["required"]["model"][0] == "FLUXFLOW_MODEL"

    def test_latent_type(self):
        """latent should be FLUXFLOW_LATENT."""
        input_types = FluxFlowVAEDecode.INPUT_TYPES()

        assert input_types["required"]["latent"][0] == "FLUXFLOW_LATENT"

    def test_return_types(self):
        """Should return IMAGE (ComfyUI standard)."""
        assert FluxFlowVAEDecode.RETURN_TYPES == ("IMAGE",)

    def test_function_name(self):
        """Function should be decode."""
        assert FluxFlowVAEDecode.FUNCTION == "decode"

    def test_category(self):
        """Category should be FluxFlow/latent."""
        assert FluxFlowVAEDecode.CATEGORY == "FluxFlow/latent"


class TestFluxFlowTextEncode:
    """Tests for FluxFlowTextEncode node."""

    def test_input_types_structure(self):
        """Should have required inputs."""
        input_types = FluxFlowTextEncode.INPUT_TYPES()

        assert "required" in input_types

    def test_required_inputs(self):
        """Should require text_encoder, tokenizer, and text."""
        input_types = FluxFlowTextEncode.INPUT_TYPES()

        assert "text_encoder" in input_types["required"]
        assert "tokenizer" in input_types["required"]
        assert "text" in input_types["required"]

    def test_text_encoder_type(self):
        """text_encoder should be FLUXFLOW_TEXT_ENCODER."""
        input_types = FluxFlowTextEncode.INPUT_TYPES()

        assert input_types["required"]["text_encoder"][0] == "FLUXFLOW_TEXT_ENCODER"

    def test_tokenizer_type(self):
        """tokenizer should be FLUXFLOW_TOKENIZER."""
        input_types = FluxFlowTextEncode.INPUT_TYPES()

        assert input_types["required"]["tokenizer"][0] == "FLUXFLOW_TOKENIZER"

    def test_text_type(self):
        """text should be STRING with multiline support."""
        input_types = FluxFlowTextEncode.INPUT_TYPES()
        text_type, text_config = input_types["required"]["text"]

        assert text_type == "STRING"
        assert text_config.get("multiline") is True

    def test_return_types(self):
        """Should return FLUXFLOW_CONDITIONING."""
        assert FluxFlowTextEncode.RETURN_TYPES == ("FLUXFLOW_CONDITIONING",)

    def test_function_name(self):
        """Function should be encode."""
        assert FluxFlowTextEncode.FUNCTION == "encode"

    def test_category(self):
        """Category should be FluxFlow/conditioning."""
        assert FluxFlowTextEncode.CATEGORY == "FluxFlow/conditioning"


class TestFluxFlowSampler:
    """Tests for FluxFlowSampler node."""

    def test_input_types_structure(self):
        """Should have required and optional inputs."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "required" in input_types
        assert "optional" in input_types

    def test_required_inputs(self):
        """Should require model, latent, conditioning, steps, scheduler."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "model" in input_types["required"]
        assert "latent" in input_types["required"]
        assert "conditioning" in input_types["required"]
        assert "steps" in input_types["required"]
        assert "scheduler" in input_types["required"]

    def test_model_type(self):
        """model should be FLUXFLOW_MODEL."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert input_types["required"]["model"][0] == "FLUXFLOW_MODEL"

    def test_latent_type(self):
        """latent should be FLUXFLOW_LATENT."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert input_types["required"]["latent"][0] == "FLUXFLOW_LATENT"

    def test_conditioning_type(self):
        """conditioning should be FLUXFLOW_CONDITIONING."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert input_types["required"]["conditioning"][0] == "FLUXFLOW_CONDITIONING"

    def test_steps_config(self):
        """steps should be INT with sensible defaults."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        steps_config = input_types["required"]["steps"][1]

        assert steps_config["default"] == 20
        assert steps_config["min"] == 1
        assert steps_config["max"] == 1000

    def test_scheduler_is_list(self):
        """scheduler should be a list of scheduler names."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scheduler_list = input_types["required"]["scheduler"][0]

        assert isinstance(scheduler_list, list)
        assert len(scheduler_list) > 0
        assert "DPMSolverMultistep" in scheduler_list

    def test_scheduler_default(self):
        """scheduler should default to DPMSolverMultistep."""
        input_types = FluxFlowSampler.INPUT_TYPES()
        scheduler_config = input_types["required"]["scheduler"][1]

        assert scheduler_config["default"] == "DPMSolverMultistep"

    def test_optional_prediction_type(self):
        """Should have optional prediction_type."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "prediction_type" in input_types["optional"]

    def test_optional_seed(self):
        """Should have optional seed."""
        input_types = FluxFlowSampler.INPUT_TYPES()

        assert "seed" in input_types["optional"]

    def test_return_types(self):
        """Should return FLUXFLOW_LATENT."""
        assert FluxFlowSampler.RETURN_TYPES == ("FLUXFLOW_LATENT",)

    def test_function_name(self):
        """Function should be sample."""
        assert FluxFlowSampler.FUNCTION == "sample"

    def test_category(self):
        """Category should be FluxFlow/sampling."""
        assert FluxFlowSampler.CATEGORY == "FluxFlow/sampling"


class TestNodeWorkflow:
    """Integration tests for node workflow."""

    def test_typical_workflow_sequence(self):
        """Nodes should support typical workflow: Load -> Empty/Encode -> TextEncode -> Sample -> Decode."""
        # This is a structure test, not a functional test
        nodes = {
            "loader": FluxFlowModelLoader,
            "empty_latent": FluxFlowEmptyLatent,
            "encode": FluxFlowVAEEncode,
            "text_encode": FluxFlowTextEncode,
            "sampler": FluxFlowSampler,
            "decode": FluxFlowVAEDecode,
        }

        # All nodes should be instantiable
        for node_name, node_cls in nodes.items():
            instance = node_cls()
            assert instance is not None

    def test_loader_outputs_match_downstream_inputs(self):
        """ModelLoader outputs should match expected downstream inputs."""
        loader_outputs = FluxFlowModelLoader.RETURN_TYPES

        # Should output MODEL, TEXT_ENCODER, TOKENIZER
        assert "FLUXFLOW_MODEL" in loader_outputs
        assert "FLUXFLOW_TEXT_ENCODER" in loader_outputs
        assert "FLUXFLOW_TOKENIZER" in loader_outputs

    def test_empty_latent_output_matches_sampler_input(self):
        """EmptyLatent output should match Sampler latent input."""
        empty_output = FluxFlowEmptyLatent.RETURN_TYPES[0]
        sampler_input = FluxFlowSampler.INPUT_TYPES()["required"]["latent"][0]

        assert empty_output == sampler_input == "FLUXFLOW_LATENT"

    def test_text_encode_output_matches_sampler_conditioning(self):
        """TextEncode output should match Sampler conditioning input."""
        text_output = FluxFlowTextEncode.RETURN_TYPES[0]
        sampler_input = FluxFlowSampler.INPUT_TYPES()["required"]["conditioning"][0]

        assert text_output == sampler_input == "FLUXFLOW_CONDITIONING"

    def test_vae_encode_output_matches_sampler_input(self):
        """VAEEncode output should match Sampler latent input."""
        encode_output = FluxFlowVAEEncode.RETURN_TYPES[0]
        sampler_input = FluxFlowSampler.INPUT_TYPES()["required"]["latent"][0]

        assert encode_output == sampler_input == "FLUXFLOW_LATENT"

    def test_sampler_output_matches_decode_input(self):
        """Sampler output should match VAEDecode latent input."""
        sampler_output = FluxFlowSampler.RETURN_TYPES[0]
        decode_input = FluxFlowVAEDecode.INPUT_TYPES()["required"]["latent"][0]

        assert sampler_output == decode_input == "FLUXFLOW_LATENT"

    def test_all_nodes_have_category(self):
        """All nodes should have CATEGORY attribute."""
        nodes = [
            FluxFlowModelLoader,
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowTextEncode,
            FluxFlowSampler,
        ]

        for node in nodes:
            assert hasattr(node, "CATEGORY")
            assert isinstance(node.CATEGORY, str)
            assert len(node.CATEGORY) > 0

    def test_all_categories_start_with_fluxflow(self):
        """All categories should start with 'FluxFlow'."""
        nodes = [
            FluxFlowModelLoader,
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowTextEncode,
            FluxFlowSampler,
        ]

        for node in nodes:
            assert node.CATEGORY.startswith("FluxFlow")


class TestNodeComfyUICompliance:
    """Tests for ComfyUI specification compliance."""

    def test_all_nodes_have_input_types_classmethod(self):
        """All nodes should have INPUT_TYPES as classmethod."""
        nodes = [
            FluxFlowModelLoader,
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowTextEncode,
            FluxFlowSampler,
        ]

        for node in nodes:
            assert hasattr(node, "INPUT_TYPES")
            assert callable(node.INPUT_TYPES)

    def test_all_nodes_have_return_types(self):
        """All nodes should have RETURN_TYPES tuple."""
        nodes = [
            FluxFlowModelLoader,
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowTextEncode,
            FluxFlowSampler,
        ]

        for node in nodes:
            assert hasattr(node, "RETURN_TYPES")
            assert isinstance(node.RETURN_TYPES, tuple)
            assert len(node.RETURN_TYPES) > 0

    def test_all_nodes_have_function_attribute(self):
        """All nodes should have FUNCTION attribute."""
        nodes = [
            FluxFlowModelLoader,
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowTextEncode,
            FluxFlowSampler,
        ]

        for node in nodes:
            assert hasattr(node, "FUNCTION")
            assert isinstance(node.FUNCTION, str)

    def test_function_methods_exist(self):
        """Function methods should exist on each node."""
        node_function_pairs = [
            (FluxFlowModelLoader, "load_model"),
            (FluxFlowEmptyLatent, "generate_latent"),
            (FluxFlowVAEEncode, "encode"),
            (FluxFlowVAEDecode, "decode"),
            (FluxFlowTextEncode, "encode"),
            (FluxFlowSampler, "sample"),
        ]

        for node_cls, function_name in node_function_pairs:
            assert hasattr(node_cls, function_name)
            assert callable(getattr(node_cls, function_name))

    def test_return_types_match_return_names_length(self):
        """RETURN_TYPES and RETURN_NAMES should have same length."""
        nodes = [
            FluxFlowModelLoader,
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowTextEncode,
            FluxFlowSampler,
        ]

        for node in nodes:
            if hasattr(node, "RETURN_NAMES"):
                assert len(node.RETURN_TYPES) == len(node.RETURN_NAMES)


class TestCustomTypeConsistency:
    """Tests for custom type consistency across nodes."""

    def test_fluxflow_model_type_consistent(self):
        """FLUXFLOW_MODEL should be used consistently."""
        # Nodes that produce FLUXFLOW_MODEL
        assert "FLUXFLOW_MODEL" in FluxFlowModelLoader.RETURN_TYPES

        # Nodes that consume FLUXFLOW_MODEL
        consumers = [
            FluxFlowEmptyLatent,
            FluxFlowVAEEncode,
            FluxFlowVAEDecode,
            FluxFlowSampler,
        ]

        for node in consumers:
            inputs = node.INPUT_TYPES()
            assert "model" in inputs["required"]
            assert inputs["required"]["model"][0] == "FLUXFLOW_MODEL"

    def test_fluxflow_latent_type_consistent(self):
        """FLUXFLOW_LATENT should be used consistently."""
        # Nodes that produce FLUXFLOW_LATENT
        producers = [FluxFlowEmptyLatent, FluxFlowVAEEncode, FluxFlowSampler]

        for node in producers:
            assert "FLUXFLOW_LATENT" in node.RETURN_TYPES

        # Nodes that consume FLUXFLOW_LATENT
        consumers = [FluxFlowVAEDecode, FluxFlowSampler]

        for node in consumers:
            inputs = node.INPUT_TYPES()
            assert "latent" in inputs["required"]
            assert inputs["required"]["latent"][0] == "FLUXFLOW_LATENT"

    def test_fluxflow_conditioning_type_consistent(self):
        """FLUXFLOW_CONDITIONING should be used consistently."""
        # Node that produces conditioning
        assert "FLUXFLOW_CONDITIONING" in FluxFlowTextEncode.RETURN_TYPES

        # Node that consumes conditioning
        sampler_inputs = FluxFlowSampler.INPUT_TYPES()
        assert sampler_inputs["required"]["conditioning"][0] == "FLUXFLOW_CONDITIONING"
