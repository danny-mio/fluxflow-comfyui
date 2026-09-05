"""
FluxFlow Sampler Node for ComfyUI.

Denoises latent using flow model with configurable schedulers.

v0.10.0: consumes ``FLUXFLOW_TEXT`` (a ``(text_seq, text_mask)`` tuple) instead
of the legacy pooled ``FLUXFLOW_CONDITIONING`` tensor. The flow processor is
dispatched per-token vs pooled via
``fluxflow.models.pipeline._flow_processor_takes_pertoken_text``.
"""

from typing import Any, Callable, Optional

import torch

from comfyui_fluxflow.nodes.utils import to_model_dtype
from comfyui_fluxflow.schedulers import PREDICTION_TYPES, create_scheduler, get_scheduler_list

# fluxflow-core >= v0.10.0 (m4-flow-redesign). Declared Optional so the legacy
# fallback path (no fluxflow-core installed) is reachable for mypy.
_flow_processor_takes_pertoken_text: Optional[Callable[[Any], bool]]
_masked_mean_pool: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
try:
    from fluxflow.models.pipeline import _flow_processor_takes_pertoken_text as _ff_takes_pertoken
    from fluxflow.models.pipeline import _masked_mean_pool as _ff_masked_mean_pool

    _flow_processor_takes_pertoken_text = _ff_takes_pertoken
    _masked_mean_pool = _ff_masked_mean_pool
except ImportError:  # pragma: no cover — keep node importable in legacy envs
    _flow_processor_takes_pertoken_text = None
    _masked_mean_pool = None


def _call_flow_processor(flow_processor, packed, text_seq, text_mask, timesteps):
    """Dispatch to the per-token or legacy pooled flow processor signature.

    v0.10.0 processors take ``(packed, text_seq, text_mask, timesteps)``.
    Legacy v0.6/v0.7/v0.8 processors take ``(packed, text_embeddings, timesteps)``
    where ``text_embeddings`` is a pooled ``[B, E]`` tensor.
    """
    if _flow_processor_takes_pertoken_text is not None and _flow_processor_takes_pertoken_text(
        flow_processor
    ):
        return flow_processor(packed, text_seq, text_mask, timesteps)

    # Legacy fallback: pool to [B, E] via masked mean for v0.6/0.7/0.8 processors.
    if _masked_mean_pool is None:  # pragma: no cover
        # If neither helper imported, inline the masked mean so the sampler
        # still works against ancient fluxflow installs.
        mask_f = text_mask.to(text_seq.dtype).unsqueeze(-1)
        pooled = (text_seq * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    else:
        pooled = _masked_mean_pool(text_seq, text_mask)
    return flow_processor(packed, pooled, timesteps)


class FluxFlowSampler:
    """
    Sample/denoise latent using FluxFlow diffusion model.

    Supports 14 schedulers from diffusers with full configuration.

    Inputs the v0.10.0 ``FLUXFLOW_TEXT`` tuple ``(text_seq, text_mask)``.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FLUXFLOW_MODEL",),
                "latent": ("FLUXFLOW_LATENT",),
                "text": ("FLUXFLOW_TEXT",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "scheduler": (get_scheduler_list(), {"default": "DPMSolverMultistep"}),
            },
            "optional": {
                "prediction_type": (
                    PREDICTION_TYPES,
                    {"default": "v_prediction"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
                "use_cfg": ("BOOLEAN", {"default": False}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.0, "max": 15.0, "step": 0.1},
                ),
                "negative_text": ("FLUXFLOW_TEXT",),
            },
        }

    RETURN_TYPES = ("FLUXFLOW_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "sample"
    CATEGORY = "FluxFlow/sampling"

    def sample(
        self,
        model,
        latent,
        text,
        steps,
        scheduler,
        prediction_type="v_prediction",
        seed=0,
        use_cfg=False,
        guidance_scale=5.0,
        negative_text=None,
    ):
        """
        Denoise latent using flow model.

        Args:
            model: FluxFlow pipeline.
            latent: Input latent packet [B, T+1, D].
            text: Positive ``FLUXFLOW_TEXT`` tuple ``(text_seq, text_mask)``.
            steps: Number of denoising steps.
            scheduler: Scheduler name.
            prediction_type: Prediction type (v_prediction, epsilon, sample).
            seed: Random seed.
            use_cfg: Enable Classifier-Free Guidance.
            guidance_scale: CFG guidance scale (1.0 = no guidance).
            negative_text: Optional negative ``FLUXFLOW_TEXT`` tuple. When
                CFG is on and this is ``None``, a zero ``text_seq`` with the
                positive mask is used as the null condition.

        Returns:
            (latent,) - Denoised latent packet [B, T+1, D].
        """
        print("\nFluxFlow Sampler:")
        print(f"  Scheduler: {scheduler}")
        print(f"  Steps: {steps}")
        print(f"  Prediction type: {prediction_type}")
        print(f"  Seed: {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"  CFG enabled: {use_cfg}")
        if use_cfg:
            print(f"  Guidance scale: {guidance_scale}")

        # Unpack FLUXFLOW_TEXT tuple.
        if not (isinstance(text, tuple) and len(text) == 2):
            raise TypeError(
                "FluxFlowSampler expected 'text' to be a FLUXFLOW_TEXT tuple "
                "(text_seq, text_mask); got "
                f"{type(text).__name__}. Reconnect the FluxFlow Text Encode node."
            )
        text_seq, text_mask = text

        # Get device
        device = next(model.parameters()).device

        # Move inputs to device and dtype (text_mask is bool -- device-move only)
        latent = to_model_dtype(latent.to(device), model.flow_processor)
        text_seq = to_model_dtype(text_seq.to(device), model.flow_processor)
        text_mask = text_mask.to(device)

        # Create scheduler
        scheduler_obj = create_scheduler(
            scheduler,
            num_train_timesteps=1000,
            prediction_type=prediction_type,
        )
        scheduler_obj.set_timesteps(steps, device=device)

        # Separate latent tokens and HW vector
        hw_vec = latent[:, -1:, :].clone()
        lat = latent[:, :-1, :].clone()

        # Prepare negative conditioning for CFG
        neg_seq = None
        neg_mask = None
        if use_cfg and guidance_scale > 1.0:
            if negative_text is None:
                # Null condition: zero embeddings, reuse positive mask shape.
                neg_seq = torch.zeros_like(text_seq)
                neg_mask = text_mask.clone()
            else:
                if not (isinstance(negative_text, tuple) and len(negative_text) == 2):
                    raise TypeError(
                        "FluxFlowSampler expected 'negative_text' to be a "
                        "FLUXFLOW_TEXT tuple (text_seq, text_mask)."
                    )
                neg_seq, neg_mask = negative_text
                neg_seq = to_model_dtype(neg_seq.to(device), model.flow_processor)
                neg_mask = neg_mask.to(device)

        # Denoising loop
        with torch.no_grad():
            for i, t in enumerate(scheduler_obj.timesteps):
                # Create timestep batch
                t_batch = torch.full(
                    (lat.size(0),), t.item() / 999.0, device=device, dtype=torch.float32
                )

                # Prepare input for flow processor
                full_input = torch.cat([lat, hw_vec], dim=1)

                if use_cfg and guidance_scale > 1.0:
                    # CFG: Dual-pass sampling
                    model_out_cond = _call_flow_processor(
                        model.flow_processor, full_input, text_seq, text_mask, t_batch
                    )
                    v_cond = model_out_cond[:, :-1, :]

                    model_out_uncond = _call_flow_processor(
                        model.flow_processor, full_input, neg_seq, neg_mask, t_batch
                    )
                    v_uncond = model_out_uncond[:, :-1, :]

                    # Apply guidance: v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                    model_out_lat = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    # Standard sampling (no CFG)
                    model_out = _call_flow_processor(
                        model.flow_processor, full_input, text_seq, text_mask, t_batch
                    )
                    model_out_lat = model_out[:, :-1, :]

                # Scheduler step
                step_output = scheduler_obj.step(
                    model_output=model_out_lat, timestep=int(t.item()), sample=lat
                )

                # Handle both diffusers (.prev_sample) and standalone (tensor)
                if hasattr(step_output, "prev_sample"):
                    lat = step_output.prev_sample
                else:
                    lat = step_output

                if (i + 1) % max(1, steps // 10) == 0 or i == 0 or i == steps - 1:
                    cfg_status = f" [CFG={guidance_scale:.1f}]" if use_cfg else ""
                    print(f"  Step {i+1}/{steps} (t={int(t.item())}){cfg_status}")

        # Reconstruct full latent
        denoised_latent = torch.cat([lat, hw_vec], dim=1)

        print(f"Sampling complete: {denoised_latent.shape}\n")

        return (denoised_latent,)


NODE_CLASS_MAPPINGS = {"FluxFlowSampler": FluxFlowSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxFlowSampler": "FluxFlow Sampler"}
