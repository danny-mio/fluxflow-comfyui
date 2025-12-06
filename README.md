# FluxFlow ComfyUI

ComfyUI custom nodes for FluxFlow text-to-image generation.

## 🚧 Checkpoint Availability

**Training In Progress**: FluxFlow model checkpoints are currently being trained and are not yet available for download.

**Status**: The ComfyUI nodes are fully implemented and tested, but require trained FluxFlow checkpoints to generate images.

**When Available**: Trained checkpoints will be published to [MODEL_ZOO.md](../MODEL_ZOO.md) upon completion of training validation. You will then be able to load them using the FluxFlowModelLoader node.

**For Developers**: You can use this plugin with your own trained FluxFlow checkpoints if you're conducting custom training experiments.

---

## Installation

**Note**: Package not yet published to PyPI - use symlink installation method below.

For development installation, see [INSTALL.md](src/comfyui_fluxflow/INSTALL.md) for symlink setup instructions.

### Alternative: Manual Installation for ComfyUI

If you're using ComfyUI, you can install directly into your ComfyUI custom_nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/danny-mio/fluxflow-comfyui.git
cd fluxflow-comfyui
pip install -e .
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
