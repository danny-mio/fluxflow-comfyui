"""Unit tests for comfyui_fluxflow.nodes.utils device helpers.

These exercise the REAL (non-mocked) get_device_auto/parse_device behavior
-- both are now thin wrappers around fluxflow.utils.device, and
parse_device's exact string-parsing behavior is depended on by
model_loader.py's public "device" node input, so a regression here would be
user-visible.
"""

import unittest.mock as mock

import torch

from comfyui_fluxflow.nodes.utils import get_device_auto, parse_device


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
