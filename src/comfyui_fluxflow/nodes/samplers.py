"""
FluxFlow Sampler Node for ComfyUI.

Denoises latent using flow model with configurable schedulers.
"""

import torch

from comfyui_fluxflow.schedulers import (
    PREDICTION_TYPES,
    create_scheduler,
    get_scheduler_list,
)


class FluxFlowSampler:
    """
    Sample/denoise latent using FluxFlow diffusion model.

    Supports 12+ schedulers from diffusers with full configuration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FLUXFLOW_MODEL",),
                "latent": ("FLUXFLOW_LATENT",),
                "conditioning": ("FLUXFLOW_CONDITIONING",),
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
                "negative_conditioning": ("FLUXFLOW_CONDITIONING",),
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
        conditioning,
        steps,
        scheduler,
        prediction_type="v_prediction",
        seed=0,
        use_cfg=False,
        guidance_scale=5.0,
        negative_conditioning=None,
    ):
        """
        Denoise latent using flow model.

        Args:
            model: FluxFlow pipeline
            latent: Input latent packet [B, T+1, D]
            conditioning: Text embeddings [B, D_text]
            steps: Number of denoising steps
            scheduler: Scheduler name
            prediction_type: Prediction type (v_prediction, epsilon, sample)
            seed: Random seed
            use_cfg: Enable Classifier-Free Guidance
            guidance_scale: CFG guidance scale (1.0 = no guidance)
            negative_conditioning: Optional negative prompt embeddings

        Returns:
            (latent,) - Denoised latent packet [B, T+1, D]
        """
        print("\nFluxFlow Sampler:")
        print(f"  Scheduler: {scheduler}")
        print(f"  Steps: {steps}")
        print(f"  Prediction type: {prediction_type}")
        print(f"  Seed: {seed}")
        print(f"  CFG enabled: {use_cfg}")
        if use_cfg:
            print(f"  Guidance scale: {guidance_scale}")

        # Get device
        device = next(model.parameters()).device

        # Move inputs to device
        latent = latent.to(device)
        conditioning = conditioning.to(device)

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

        # Add initial noise if needed (for full denoising from random)
        # Check if latent is already noisy or if it's from VAE encoding
        # For now, assume input latent is already properly prepared

        # Prepare negative conditioning for CFG
        neg_cond = None
        if use_cfg and guidance_scale > 1.0:
            if negative_conditioning is None:
                # Create null embeddings (unconditional)
                neg_cond = torch.zeros_like(conditioning)
            else:
                neg_cond = negative_conditioning.to(device)

        # Denoising loop
        with torch.no_grad():
            for i, t in enumerate(scheduler_obj.timesteps):
                # Create timestep batch
                t_batch = torch.full((lat.size(0),), t.item(), device=device, dtype=torch.long)

                # Prepare input for flow processor
                full_input = torch.cat([lat, hw_vec], dim=1)

                if use_cfg and guidance_scale > 1.0:
                    # CFG: Dual-pass sampling
                    # Conditional pass
                    model_out_cond = model.flow_processor(full_input, conditioning, t_batch)
                    v_cond = model_out_cond[:, :-1, :]

                    # Unconditional pass
                    model_out_uncond = model.flow_processor(full_input, neg_cond, t_batch)
                    v_uncond = model_out_uncond[:, :-1, :]

                    # Apply guidance: v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
                    model_out_lat = v_uncond + guidance_scale * (v_cond - v_uncond)
                else:
                    # Standard sampling (no CFG)
                    model_out = model.flow_processor(full_input, conditioning, t_batch)
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

        print(f"✓ Sampling complete: {denoised_latent.shape}\n")

        return (denoised_latent,)


NODE_CLASS_MAPPINGS = {"FluxFlowSampler": FluxFlowSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"FluxFlowSampler": "FluxFlow Sampler"}
