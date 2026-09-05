"""Unit tests for FluxFlowVAEEncode/FluxFlowVAEDecode dtype + device handling.

These reproduce the dtype-mismatch bug: loading a model with dtype=bf16/fp16
casts its weights, but the VAE encode/decode nodes only moved input tensors
to the target *device*, never the target *dtype* -- so a real bf16/fp16
compressor/expander raises RuntimeError against a fp32 input. Uses real tiny
`torch.nn.Conv2d` submodules cast to a non-fp32 dtype (mirrors this repo's
device-helper tests, which exercise real torch behavior rather than mocking
dtype semantics).
"""

from unittest.mock import Mock

import pytest
import torch

from comfyui_fluxflow.nodes.latent_ops import FluxFlowVAEDecode, FluxFlowVAEEncode


class _FakeExpander(torch.nn.Module):
    """Minimal expander stand-in: a Linear whose dtype mismatches raise like a
    real conv/linear stack, reshaped to the 4D [B, C, H, W] output decode expects."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x)  # [B, T, 8] -- raises RuntimeError on dtype mismatch
        batch = out.shape[0]
        return out.reshape(batch, 2, 2, -1)


class TestFluxFlowVAEEncodeDtype:
    """Tests that encode() casts the image to model.compressor's param dtype."""

    def test_encode_casts_image_to_compressor_dtype_bf16(self):
        """flux_image reaching model.compressor must match its bf16 param dtype."""
        compressor = torch.nn.Conv2d(3, 4, kernel_size=1, dtype=torch.bfloat16)
        model = Mock()
        model.compressor = compressor
        model.parameters.return_value = iter(list(compressor.parameters()))

        image = torch.rand(1, 8, 8, 3, dtype=torch.float32)  # ComfyUI format [B,H,W,C]

        (latent,) = FluxFlowVAEEncode().encode(model, image)

        assert latent.dtype == torch.bfloat16

    def test_encode_casts_image_to_compressor_dtype_fp16(self):
        compressor = torch.nn.Conv2d(3, 4, kernel_size=1, dtype=torch.float16)
        model = Mock()
        model.compressor = compressor
        model.parameters.return_value = iter(list(compressor.parameters()))

        image = torch.rand(1, 8, 8, 3, dtype=torch.float32)

        (latent,) = FluxFlowVAEEncode().encode(model, image)

        assert latent.dtype == torch.float16

    def test_encode_fp32_model_stays_fp32(self):
        """fp32 (default) load path must remain behaviorally unchanged."""
        compressor = torch.nn.Conv2d(3, 4, kernel_size=1)  # fp32 default
        model = Mock()
        model.compressor = compressor
        model.parameters.return_value = iter(list(compressor.parameters()))

        image = torch.rand(1, 8, 8, 3, dtype=torch.float32)

        (latent,) = FluxFlowVAEEncode().encode(model, image)

        assert latent.dtype == torch.float32

    def test_uncast_fp32_image_raises_against_bf16_compressor(self):
        """Sanity check proving the bug precondition without going through the node."""
        compressor = torch.nn.Conv2d(3, 4, kernel_size=1, dtype=torch.bfloat16)
        image = torch.rand(1, 3, 8, 8, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            compressor(image)


class TestFluxFlowVAEDecodeDtype:
    """Tests that decode() casts the latent to model.expander's param dtype."""

    def test_decode_casts_latent_to_expander_dtype_bf16(self):
        expander = _FakeExpander(dtype=torch.bfloat16)
        model = Mock()
        model.expander = expander
        model.parameters.return_value = iter(list(expander.parameters()))

        latent = torch.rand(1, 5, 8, dtype=torch.float32)

        (image,) = FluxFlowVAEDecode().decode(model, latent)

        # decode's output goes through flux_image_to_comfy, which now always
        # casts to float32 for ComfyUI's IMAGE contract, regardless of the
        # expander's (bf16) output dtype.
        assert image.dtype == torch.float32

    def test_decode_casts_latent_to_expander_dtype_fp16(self):
        expander = _FakeExpander(dtype=torch.float16)
        model = Mock()
        model.expander = expander
        model.parameters.return_value = iter(list(expander.parameters()))

        latent = torch.rand(1, 5, 8, dtype=torch.float32)

        (image,) = FluxFlowVAEDecode().decode(model, latent)

        # flux_image_to_comfy always casts to float32 for ComfyUI's IMAGE
        # contract, regardless of the expander's (fp16) output dtype.
        assert image.dtype == torch.float32

    def test_decode_fp32_model_stays_fp32(self):
        expander = _FakeExpander()  # fp32 default
        model = Mock()
        model.expander = expander
        model.parameters.return_value = iter(list(expander.parameters()))

        latent = torch.rand(1, 5, 8, dtype=torch.float32)

        (image,) = FluxFlowVAEDecode().decode(model, latent)

        assert image.dtype == torch.float32

    def test_uncast_fp32_latent_raises_against_bf16_expander(self):
        """Sanity check proving the bug precondition without going through the node."""
        expander = _FakeExpander(dtype=torch.bfloat16)
        latent = torch.rand(1, 5, 8, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            expander(latent)
