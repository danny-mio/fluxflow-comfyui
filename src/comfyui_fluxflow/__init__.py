"""
ComfyUI FluxFlow Plugin

A comprehensive plugin for using FluxFlow diffusion models in ComfyUI.

Features:
- Automatic model configuration detection from checkpoints
- 14 scheduler support (DPM++, Euler, DDIM, LCM, etc.)
- Full VAE encode/decode support
- Text conditioning with DistilBERT
- Native ComfyUI tensor format integration

Nodes:
- FluxFlowModelLoader: Load checkpoint with auto-detection
- FluxFlowEmptyLatent: Generate random latent for target dimensions
- FluxFlowVAEEncode: Encode image to latent
- FluxFlowVAEDecode: Decode latent to image
- FluxFlowTextEncode: Encode text prompt to conditioning
- FluxFlowSampler: Denoise latent with configurable scheduler
"""

from .nodes.latent_ops import FluxFlowEmptyLatent, FluxFlowVAEDecode, FluxFlowVAEEncode
from .nodes.model_loader import FluxFlowModelLoader
from .nodes.samplers import FluxFlowSampler
from .nodes.text_encode import FluxFlowTextEncode

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "FluxFlowModelLoader": FluxFlowModelLoader,
    "FluxFlowEmptyLatent": FluxFlowEmptyLatent,
    "FluxFlowVAEEncode": FluxFlowVAEEncode,
    "FluxFlowVAEDecode": FluxFlowVAEDecode,
    "FluxFlowTextEncode": FluxFlowTextEncode,
    "FluxFlowSampler": FluxFlowSampler,
}

# Display name mappings for ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFlowModelLoader": "FluxFlow Model Loader",
    "FluxFlowEmptyLatent": "FluxFlow Empty Latent",
    "FluxFlowVAEEncode": "FluxFlow VAE Encode",
    "FluxFlowVAEDecode": "FluxFlow VAE Decode",
    "FluxFlowTextEncode": "FluxFlow Text Encode",
    "FluxFlowSampler": "FluxFlow Sampler",
}

# Connector colors for FluxFlow custom types
# Colors are in RGB hex format for ComfyUI's node editor
NODE_COLORS = {
    "FLUXFLOW_MODEL": "#8B5CF6",  # Purple - main model
    "FLUXFLOW_TEXT_ENCODER": "#10B981",  # Green - text processing
    "FLUXFLOW_TOKENIZER": "#059669",  # Dark green - text processing
    "FLUXFLOW_CONDITIONING": "#F59E0B",  # Amber - conditioning data
    "FLUXFLOW_LATENT": "#3B82F6",  # Blue - latent space
}

# Export web directory for ComfyUI JavaScript extensions
import os

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "NODE_COLORS",
    "FluxFlowModelLoader",
    "FluxFlowEmptyLatent",
    "FluxFlowVAEEncode",
    "FluxFlowVAEDecode",
    "FluxFlowTextEncode",
    "FluxFlowSampler",
]

__version__ = "0.1.0"
__author__ = "Daniele Camisani"
