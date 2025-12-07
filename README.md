# FluxFlow ComfyUI

ComfyUI custom nodes for FluxFlow text-to-image generation.

## 🚧 Checkpoint Availability

**Training In Progress**: FluxFlow model checkpoints are currently being trained and are not yet available for download.

**Status**: The ComfyUI nodes are fully implemented and tested, but require trained FluxFlow checkpoints to generate images.

**When Available**: Trained checkpoints will be published to [MODEL_ZOO.md](https://github.com/danny-mio/fluxflow-core/blob/main/MODEL_ZOO.md) upon completion of training validation. You will then be able to load them using the FluxFlowModelLoader node.

**For Developers**: You can use this plugin with your own trained FluxFlow checkpoints if you're conducting custom training experiments.

---

## Installation

### Production Install (ComfyUI Users)

**Recommended for ComfyUI users**: Clone directly into ComfyUI's custom_nodes directory for automatic discovery:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/danny-mio/fluxflow-comfyui.git
cd fluxflow-comfyui
pip install -e .
```

This method requires no additional symlink setup.

### Production Install (via PyPI)

For advanced users who want to manage the package separately:

```bash
pip install fluxflow-comfyui
```

**What gets installed:**
- `fluxflow-comfyui` - ComfyUI custom nodes for FluxFlow
- `fluxflow` core package (automatically installed as dependency)
- **Note**: Does NOT include training capabilities. Only inference/generation.

**Package available on PyPI**: [fluxflow-comfyui v0.1.1](https://pypi.org/project/fluxflow-comfyui/)

**Additional Setup Required**: You must symlink the package into ComfyUI's `custom_nodes` directory:

```bash
# Find where fluxflow-comfyui was installed
PACKAGE_PATH=$(python -c "import comfyui_fluxflow; print(comfyui_fluxflow.__path__[0])")

# Create symlink in ComfyUI's custom_nodes directory
ln -s "$PACKAGE_PATH" ~/ComfyUI/custom_nodes/comfyui_fluxflow
```

Adjust the `~/ComfyUI` path to match your ComfyUI installation location.

### Development Install

```bash
git clone https://github.com/danny-mio/fluxflow-comfyui.git
cd fluxflow-comfyui
pip install -e ".[dev]"
```

## Features

- **Model Loader**: Load FluxFlow checkpoints with auto-configuration
- **Text Encoding**: BERT-based text encoding for prompts
- **Sampling**: Multiple sampling algorithms (Euler, DPM++, DDIM, etc.)
- **VAE Operations**: Encode/decode latents
- **Latent Generation**: Create empty latents at various resolutions

## Available Nodes

### FluxFlowModelLoader
Load FluxFlow model checkpoints (.safetensors or .pth files).

### FluxFlowTextEncode
Encode text prompts using DistilBERT.

### FluxFlowSampler
Sample from the diffusion model with 14 schedulers:
- Euler, Euler Ancestral
- DPM++ 2M, DPM++ 2M Karras
- DPM++ SDE, DPM++ SDE Karras
- DDIM, DDPM
- LCM (Latent Consistency Model)
- And more...

### FluxFlowVAEEncode / FluxFlowVAEDecode
Encode images to latents and decode latents to images.

### FluxFlowEmptyLatent
Generate empty latent tensors at specified dimensions.

## Quick Start

1. Load a FluxFlow model using **FluxFlowModelLoader**
2. Encode your prompt with **FluxFlowTextEncode**
3. Create empty latents with **FluxFlowEmptyLatent**
4. Generate with **FluxFlowSampler**
5. Decode latents with **FluxFlowVAEDecode**

## Example Workflow

```
[FluxFlowModelLoader] → model
[FluxFlowTextEncode] → conditioning
[FluxFlowEmptyLatent] → latent
[FluxFlowSampler] (model + conditioning + latent) → sampled_latent
[FluxFlowVAEDecode] (model + sampled_latent) → image
```

## Package Contents

- `comfyui_fluxflow.nodes` - Custom node implementations
- `comfyui_fluxflow.schedulers` - Sampling scheduler implementations
- `comfyui_fluxflow.web` - JavaScript extensions for ComfyUI UI

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-comfyui)
- [ComfyUI Documentation](https://github.com/danny-mio/fluxflow-comfyui/tree/main/comfyui_fluxflow)
- [Core Package](https://pypi.org/project/fluxflow/)
- [Training Package](https://pypi.org/project/fluxflow-training/)

## License

MIT License - see LICENSE file for details.
