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

from comfyui_fluxflow.nodes.utils import get_device_auto, parse_device, to_model_dtype


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
