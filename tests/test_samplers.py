"""Unit tests for FluxFlowSampler dtype + device handling.

Reproduces the dtype-mismatch bug: loading a model with dtype=bf16/fp16
casts its weights, but the sampler only moved latent/text tensors to the
target *device*, never the target *dtype* -- so a real bf16/fp16
flow_processor raises RuntimeError against fp32 inputs. Uses a tiny real
`torch.nn.Module` (wrapping `torch.nn.Linear`) cast to a non-fp32 dtype,
mirroring this repo's existing dtype-mismatch test pattern in
test_latent_ops.py, plus this repo's Mock-model convention from
test_cfg_sampling.py.
"""

from unittest.mock import Mock

import pytest
import torch

from comfyui_fluxflow.nodes.samplers import FluxFlowSampler


class _FakeFlowProcessor(torch.nn.Module):
    """v0.10.0-shaped flow processor: forward(packed, text_seq, text_mask, timesteps).

    Matches `_flow_processor_takes_pertoken_text`'s signature check (it looks
    for "text_seq"/"text_mask" params) so the sampler dispatches per-token,
    not the legacy pooled path. Routes both `packed` and `text_seq` through a
    dtype-strict Linear so a mismatch raises RuntimeError, exactly like a
    real conv/linear stack would.
    """

    def __init__(self, dim: int = 4, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.packed_proj = torch.nn.Linear(dim, dim, dtype=dtype)
        self.text_proj = torch.nn.Linear(dim, dim, dtype=dtype)
        self.captured_dtypes: dict = {}

    def forward(self, packed, text_seq, text_mask, timesteps):
        self.captured_dtypes["packed"] = packed.dtype
        self.captured_dtypes["text_seq"] = text_seq.dtype
        self.captured_dtypes["text_mask"] = text_mask.dtype
        self.packed_proj(packed)  # raises RuntimeError on dtype mismatch
        self.text_proj(text_seq)  # raises RuntimeError on dtype mismatch
        return packed


def _make_model_and_inputs(
    dtype: torch.dtype, dim: int = 4, t_tokens: int = 2, txt_tokens: int = 3
):
    flow_processor = _FakeFlowProcessor(dim=dim, dtype=dtype)
    model = Mock()
    model.flow_processor = flow_processor
    model.parameters.return_value = iter(list(flow_processor.parameters()))

    latent = torch.rand(1, t_tokens + 1, dim, dtype=torch.float32)
    text_seq = torch.rand(1, txt_tokens, dim, dtype=torch.float32)
    text_mask = torch.ones(1, txt_tokens, dtype=torch.bool)
    return model, flow_processor, latent, text_seq, text_mask


class TestFluxFlowSamplerDtype:
    """Tests that sample() casts floating-point tensors to flow_processor's param dtype."""

    def test_sample_casts_latent_and_text_seq_to_bf16(self):
        model, flow_processor, latent, text_seq, text_mask = _make_model_and_inputs(torch.bfloat16)

        (out_latent,) = FluxFlowSampler().sample(
            model, latent, (text_seq, text_mask), steps=1, scheduler="DPMSolverMultistep"
        )

        assert flow_processor.captured_dtypes["packed"] == torch.bfloat16
        assert flow_processor.captured_dtypes["text_seq"] == torch.bfloat16
        assert out_latent.dtype == torch.bfloat16

    def test_sample_casts_latent_and_text_seq_to_fp16(self):
        model, flow_processor, latent, text_seq, text_mask = _make_model_and_inputs(torch.float16)

        FluxFlowSampler().sample(
            model, latent, (text_seq, text_mask), steps=1, scheduler="DPMSolverMultistep"
        )

        assert flow_processor.captured_dtypes["packed"] == torch.float16
        assert flow_processor.captured_dtypes["text_seq"] == torch.float16

    def test_sample_does_not_cast_text_mask(self):
        """text_mask is bool -- must be device-moved only, never dtype-cast."""
        model, flow_processor, latent, text_seq, text_mask = _make_model_and_inputs(torch.bfloat16)

        FluxFlowSampler().sample(
            model, latent, (text_seq, text_mask), steps=1, scheduler="DPMSolverMultistep"
        )

        assert flow_processor.captured_dtypes["text_mask"] == torch.bool

    def test_sample_fp32_model_stays_fp32(self):
        """fp32 (default) load path must remain behaviorally unchanged."""
        model, flow_processor, latent, text_seq, text_mask = _make_model_and_inputs(torch.float32)

        (out_latent,) = FluxFlowSampler().sample(
            model, latent, (text_seq, text_mask), steps=1, scheduler="DPMSolverMultistep"
        )

        assert flow_processor.captured_dtypes["packed"] == torch.float32
        assert flow_processor.captured_dtypes["text_seq"] == torch.float32
        assert out_latent.dtype == torch.float32

    def test_sample_cfg_casts_negative_text_seq(self):
        """CFG dual-pass: explicit negative_text's text_seq must also be cast."""
        model, flow_processor, latent, text_seq, text_mask = _make_model_and_inputs(torch.bfloat16)
        neg_text_seq = torch.rand(1, 3, 4, dtype=torch.float32)
        neg_text_mask = torch.ones(1, 3, dtype=torch.bool)

        FluxFlowSampler().sample(
            model,
            latent,
            (text_seq, text_mask),
            steps=1,
            scheduler="DPMSolverMultistep",
            use_cfg=True,
            guidance_scale=5.0,
            negative_text=(neg_text_seq, neg_text_mask),
        )

        # Last forward call in the dual-pass loop is the uncond (negative) pass.
        assert flow_processor.captured_dtypes["text_seq"] == torch.bfloat16

    def test_sample_cfg_null_condition_matches_positive_dtype(self):
        """CFG with no negative_text: the zeros_like(text_seq) null condition
        must already be the cast dtype (it derives from the already-cast
        text_seq), so it doesn't need a separate explicit cast."""
        model, flow_processor, latent, text_seq, text_mask = _make_model_and_inputs(torch.bfloat16)

        FluxFlowSampler().sample(
            model,
            latent,
            (text_seq, text_mask),
            steps=1,
            scheduler="DPMSolverMultistep",
            use_cfg=True,
            guidance_scale=5.0,
        )

        assert flow_processor.captured_dtypes["text_seq"] == torch.bfloat16

    def test_uncast_fp32_tensors_raise_against_bf16_flow_processor(self):
        """Sanity check proving the bug precondition without going through the node."""
        flow_processor = _FakeFlowProcessor(dtype=torch.bfloat16)
        packed = torch.rand(1, 3, 4, dtype=torch.float32)
        text_seq = torch.rand(1, 3, 4, dtype=torch.float32)
        text_mask = torch.ones(1, 3, dtype=torch.bool)
        timesteps = torch.zeros(1, dtype=torch.float32)

        with pytest.raises(RuntimeError):
            flow_processor(packed, text_seq, text_mask, timesteps)
