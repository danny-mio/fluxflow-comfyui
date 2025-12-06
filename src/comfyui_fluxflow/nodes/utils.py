"""
Utility functions for ComfyUI FluxFlow plugin.

Handles image format conversion between ComfyUI and FluxFlow formats.
"""

import torch


def comfy_image_to_flux(image: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI image format to FluxFlow format.

    ComfyUI format: [B, H, W, C] in range [0, 1]
    FluxFlow format: [B, C, H, W] in range [-1, 1]

    Args:
        image: ComfyUI image tensor [B, H, W, C]

    Returns:
        FluxFlow image tensor [B, C, H, W]
    """
    # Permute from [B, H, W, C] to [B, C, H, W]
    image = image.permute(0, 3, 1, 2)

    # Scale from [0, 1] to [-1, 1]
    image = (image * 2.0) - 1.0

    return image.contiguous()


def flux_image_to_comfy(image: torch.Tensor) -> torch.Tensor:
    """
    Convert FluxFlow image format to ComfyUI format.

    FluxFlow format: [B, C, H, W] in range [-1, 1]
    ComfyUI format: [B, H, W, C] in range [0, 1]

    Args:
        image: FluxFlow image tensor [B, C, H, W]

    Returns:
        ComfyUI image tensor [B, H, W, C]
    """
    # Scale from [-1, 1] to [0, 1]
    image = (image + 1.0) / 2.0

    # Clamp to valid range
    image = torch.clamp(image, 0.0, 1.0)

    # Permute from [B, C, H, W] to [B, H, W, C]
    image = image.permute(0, 2, 3, 1)

    return image.contiguous()


def get_device_auto() -> torch.device:
    """
    Auto-detect best available device.

    Returns:
        torch.device (cuda > mps > cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def parse_device(device_str: str) -> torch.device:
    """
    Parse device string to torch.device.

    Args:
        device_str: "auto", "cuda", "cpu", "mps", or "cuda:0" format

    Returns:
        torch.device instance
    """
    if device_str == "auto":
        return get_device_auto()
    else:
        return torch.device(device_str)
