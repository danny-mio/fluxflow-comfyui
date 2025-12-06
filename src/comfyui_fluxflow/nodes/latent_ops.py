"""
FluxFlow Latent Operations for ComfyUI.

Handles empty latent generation, VAE encode, and VAE decode.
"""

import torch

from comfyui_fluxflow.nodes.utils import comfy_image_to_flux, flux_image_to_comfy


class FluxFlowEmptyLatent:
    """
    Generate random latent packet for target image dimensions.

    Automatically inherits vae_dim, downscales, and max_hw from the model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FLUXFLOW_MODEL",),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
        }

    RETURN_TYPES = ("FLUXFLOW_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate_latent"
    CATEGORY = "FluxFlow/latent"

    def generate_latent(
        self,
        model,
        width: int,
        height: int,
        batch_size: int,
        seed: int = 0,
    ):
        """
        Generate random latent packet.

        Args:
            model: FluxFlow pipeline (parameters auto-detected)
            width: Target image width
            height: Target image height
            batch_size: Batch size
            seed: Random seed

        Returns:
            (latent,) - Random latent packet [B, T+1, D]
        """
        # Auto-detect parameters from model
        vae_dim = model.compressor.d_model
        downscales = model.compressor.downscales
        max_hw = model.compressor.max_hw

        print(f"Auto-detected: vae_dim={vae_dim}, downscales={downscales}, max_hw={max_hw}")

        # Set random seed
        generator = torch.Generator().manual_seed(seed)

        # Calculate latent dimensions after downscaling
        compression = 2**downscales
        H_lat = max(height // compression, 1)
        W_lat = max(width // compression, 1)
        T = H_lat * W_lat

        # Generate random tokens
        tokens = torch.randn(batch_size, T, vae_dim, generator=generator, dtype=torch.float32)

        # Create HW vector
        hw_vec = torch.zeros(batch_size, 1, vae_dim, dtype=torch.float32)
        hw_vec[:, 0, 0] = H_lat / float(max_hw)
        hw_vec[:, 0, 1] = W_lat / float(max_hw)

        # Pack latent
        latent = torch.cat([tokens, hw_vec], dim=1)  # [B, T+1, D]

        print(
            f"Generated latent: {latent.shape} for image size {width}x{height} "
            f"(latent: {H_lat}x{W_lat})"
        )

        return (latent,)


class FluxFlowVAEEncode:
    """Encode image to FluxFlow latent space."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FLUXFLOW_MODEL",),
                "image": ("IMAGE",),  # ComfyUI format: [B, H, W, C]
            },
        }

    RETURN_TYPES = ("FLUXFLOW_LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "encode"
    CATEGORY = "FluxFlow/latent"

    def encode(self, model, image):
        """
        Encode image to latent.

        Args:
            model: FluxFlow pipeline
            image: ComfyUI image [B, H, W, C] in [0, 1]

        Returns:
            (latent,) - Latent packet [B, T+1, D]
        """
        # Convert ComfyUI format to FluxFlow format
        flux_image = comfy_image_to_flux(image)  # [B, C, H, W] in [-1, 1]

        # Move to model device
        device = next(model.parameters()).device
        flux_image = flux_image.to(device)

        # Encode
        with torch.no_grad():
            latent = model.compressor(flux_image)  # [B, T+1, D]

        print(f"Encoded image {flux_image.shape} to latent {latent.shape}")

        return (latent,)


class FluxFlowVAEDecode:
    """Decode FluxFlow latent to image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("FLUXFLOW_MODEL",),
                "latent": ("FLUXFLOW_LATENT",),
            },
            "optional": {
                "use_context": (["true", "false"], {"default": "true"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "FluxFlow/latent"

    def decode(self, model, latent, use_context="true"):
        """
        Decode latent to image.

        Args:
            model: FluxFlow pipeline
            latent: Latent packet [B, T+1, D]
            use_context: Enable context conditioning ("true" or "false")

        Returns:
            (image,) - ComfyUI image [B, H, W, C] in [0, 1]
        """
        # Convert string to boolean
        use_context_bool = use_context == "true"

        # Move to model device
        device = next(model.parameters()).device
        latent = latent.to(device)

        # Decode
        with torch.no_grad():
            flux_image = model.expander(latent, use_context=use_context_bool)  # [B, C, H, W]

        # Convert FluxFlow format to ComfyUI format
        comfy_image = flux_image_to_comfy(flux_image)  # [B, H, W, C] in [0, 1]

        print(f"Decoded latent {latent.shape} to image {comfy_image.shape}")

        return (comfy_image,)


NODE_CLASS_MAPPINGS = {
    "FluxFlowEmptyLatent": FluxFlowEmptyLatent,
    "FluxFlowVAEEncode": FluxFlowVAEEncode,
    "FluxFlowVAEDecode": FluxFlowVAEDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFlowEmptyLatent": "FluxFlow Empty Latent",
    "FluxFlowVAEEncode": "FluxFlow VAE Encode",
    "FluxFlowVAEDecode": "FluxFlow VAE Decode",
}
