"""Unit tests for comfyui_fluxflow.nodes.utils device helpers.

These exercise the REAL (non-mocked) get_device_auto/parse_device behavior
-- both are now thin wrappers around fluxflow.utils.device, and
parse_device's exact string-parsing behavior is depended on by
model_loader.py's public "device" node input, so a regression here would be
user-visible.
"""

import unittest.mock as mock

import pytest
import torch

from comfyui_fluxflow.nodes.utils import (
    flux_image_to_comfy,
    get_device_auto,
    parse_device,
    to_model_dtype,
)


class TestGetDeviceAuto:
    def test_returns_a_torch_device(self):
        assert isinstance(get_device_auto(), torch.device)

    def test_delegates_to_core_device_util(self):
        with mock.patch(
            "fluxflow.utils.device.get_device", return_value=torch.device("cpu")
        ) as mocked:
            result = get_device_auto()
        mocked.assert_called_once()
        assert result == torch.device("cpu")


class TestParseDevice:
    def test_auto_resolves_via_get_device_auto(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert parse_device("auto") == torch.device("cpu")

    def test_cuda_string_passthrough(self):
        assert parse_device("cuda") == torch.device("cuda")

    def test_cuda_indexed_string_passthrough(self):
        assert parse_device("cuda:0") == torch.device("cuda:0")

    def test_cpu_string_passthrough(self):
        assert parse_device("cpu") == torch.device("cpu")

    def test_mps_string_passthrough(self):
        assert parse_device("mps") == torch.device("mps")


class TestToModelDtype:
    """Tests for to_model_dtype: casts floating-point tensors to a module's param dtype."""

    def test_casts_float_tensor_to_reference_dtype(self):
        """A fp32 tensor should be cast to match a bf16 reference module."""
        reference = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        tensor = torch.rand(2, 4, dtype=torch.float32)

        result = to_model_dtype(tensor, reference)

        assert result.dtype == torch.bfloat16

    def test_casts_to_fp16_reference_dtype(self):
        reference = torch.nn.Linear(4, 4).to(dtype=torch.float16)
        tensor = torch.rand(2, 4, dtype=torch.float32)

        result = to_model_dtype(tensor, reference)

        assert result.dtype == torch.float16

    def test_noop_when_already_matching_dtype(self):
        reference = torch.nn.Linear(4, 4)  # default fp32
        tensor = torch.rand(2, 4, dtype=torch.float32)

        result = to_model_dtype(tensor, reference)

        assert result.dtype == torch.float32

    def test_does_not_cast_bool_tensor(self):
        """Masks (bool) must never be dtype-cast, only device-moved elsewhere."""
        reference = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        mask = torch.ones(2, 4, dtype=torch.bool)

        result = to_model_dtype(mask, reference)

        assert result.dtype == torch.bool

    def test_does_not_cast_long_tensor(self):
        """Token-id/timestep-index tensors (long) must never be dtype-cast."""
        reference = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        ids = torch.zeros(2, 4, dtype=torch.long)

        result = to_model_dtype(ids, reference)

        assert result.dtype == torch.long

    def test_does_not_cast_int_tensor(self):
        reference = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        ids = torch.zeros(2, 4, dtype=torch.int32)

        result = to_model_dtype(ids, reference)

        assert result.dtype == torch.int32

    def test_result_usable_by_bf16_linear_layer(self):
        """End-to-end: casting a fp32 tensor lets it flow through a bf16 layer."""
        reference = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        tensor = torch.rand(2, 4, dtype=torch.float32)

        cast = to_model_dtype(tensor, reference)
        output = reference(cast)  # would raise RuntimeError if still fp32

        assert output.dtype == torch.bfloat16

    def test_uncast_fp32_tensor_raises_against_bf16_layer(self):
        """Sanity check proving the bug precondition: fp32 vs bf16 param mismatch."""
        reference = torch.nn.Linear(4, 4).to(dtype=torch.bfloat16)
        tensor = torch.rand(2, 4, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            reference(tensor)


class TestFluxImageToComfy:
    """Tests for flux_image_to_comfy: output must always be float32 for ComfyUI.

    ComfyUI's IMAGE contract is float32 -- its own nodes.py does
    image.cpu().numpy(), and numpy has no bfloat16 support. The model may
    decode in bf16/fp16, so this conversion must always cast to float32
    regardless of input dtype.
    """

    def test_bf16_input_returns_float32(self):
        """bf16 model output must be cast to float32, numerically close to fp32."""
        fp32_image = torch.rand(2, 3, 8, 8, dtype=torch.float32) * 2 - 1
        bf16_image = fp32_image.to(dtype=torch.bfloat16)

        result_from_bf16 = flux_image_to_comfy(bf16_image)
        result_from_fp32 = flux_image_to_comfy(fp32_image)

        assert result_from_bf16.dtype == torch.float32
        assert torch.allclose(result_from_bf16, result_from_fp32, atol=1e-2)

    def test_fp16_input_returns_float32(self):
        """fp16 model output must be cast to float32, numerically close to fp32."""
        fp32_image = torch.rand(2, 3, 8, 8, dtype=torch.float32) * 2 - 1
        fp16_image = fp32_image.to(dtype=torch.float16)

        result_from_fp16 = flux_image_to_comfy(fp16_image)
        result_from_fp32 = flux_image_to_comfy(fp32_image)

        assert result_from_fp16.dtype == torch.float32
        assert torch.allclose(result_from_fp16, result_from_fp32, atol=1e-3)

    def test_fp32_input_stays_float32(self):
        """Regression/no-op case: a plain fp32 input still returns float32."""
        fp32_image = torch.rand(2, 3, 8, 8, dtype=torch.float32) * 2 - 1

        result = flux_image_to_comfy(fp32_image)

        assert result.dtype == torch.float32

    def test_output_shape_is_permuted_to_bhwc(self):
        """Guard against unrelated regressions: [B, C, H, W] -> [B, H, W, C]."""
        fp32_image = torch.rand(2, 3, 8, 16, dtype=torch.float32) * 2 - 1

        result = flux_image_to_comfy(fp32_image)

        assert result.shape == (2, 8, 16, 3)
