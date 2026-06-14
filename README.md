# FluxFlow ComfyUI

ComfyUI custom nodes for FluxFlow text-to-image generation.

## Checkpoint Availability

FluxFlow model checkpoints are still in training and not yet available for
public download. Once validation completes, they will be published to
[MODEL_ZOO.md](https://github.com/danny-mio/fluxflow-core/blob/main/MODEL_ZOO.md).
If you are running custom training experiments, you can use this plugin with
your own checkpoints.

---

## v0.10.0 — workflow-breaking change

v0.10.0 introduces a new ComfyUI socket type `FLUXFLOW_TEXT` carrying a
per-token `(text_seq, text_mask)` tuple between `FluxFlowTextEncode` and
`FluxFlowSampler`, replacing the v0.8.x pooled `FLUXFLOW_CONDITIONING` type
and renaming the sampler input `conditioning` → `text`. Workflows saved
with v0.8.x will fail to load with a clear ComfyUI type-mismatch error and
must be re-wired.

See [the package CHANGELOG](src/comfyui_fluxflow/CHANGELOG.md) and
[TROUBLESHOOTING](src/comfyui_fluxflow/TROUBLESHOOTING.md) for the full
migration note.

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
pip install comfyui-fluxflow
```

**What gets installed:**
- `comfyui-fluxflow` - ComfyUI custom nodes for FluxFlow
- `fluxflow` core package (automatically installed as dependency)
- **Note**: Does NOT include training capabilities. Only inference/generation.

**Package available on PyPI**: [comfyui-fluxflow v0.10.0](https://pypi.org/project/comfyui-fluxflow/)

**Additional Setup Required**: You must symlink the package into ComfyUI's `custom_nodes` directory:

```bash
# Find where comfyui-fluxflow was installed
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
- **Classifier-Free Guidance (CFG)**: Improved prompt adherence with guidance scaling
- **Negative Prompts**: Support for negative text conditioning
- **Sampling**: Multiple sampling algorithms (Euler, DPM++, DDIM, etc.)
- **VAE Operations**: Encode/decode latents
- **Latent Generation**: Create empty latents at various resolutions

## Available Nodes

### FluxFlowModelLoader
Load FluxFlow model checkpoints (.safetensors or .pth files).

**v0.8.0 model detection**: The loader automatically detects v0.8.0 pillar-attention checkpoints (identified by `pillar_cross_attn` or `film_p0` keys in the state dict). When a v0.8.0 checkpoint is detected, the loader returns a clear error directing you to use versioned loading via `load_versioned_checkpoint()` instead of the legacy loader. This prevents silent architecture mismatches.

### FluxFlowTextEncode
Encode text prompts using DistilBERT.

### FluxFlowTextEncodeNegative
Encode negative text prompts for Classifier-Free Guidance (CFG).

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

## Example Workflows

> **v0.10.0**: the text sockets between encoder and sampler now use the
> `FLUXFLOW_TEXT` type (per-token `(text_seq, text_mask)` tuple) and the
> sampler input is named `text` (positive) / `negative_text` (optional).
> Workflows saved against the older `FLUXFLOW_CONDITIONING` socket type
> must be re-wired — see [TROUBLESHOOTING](src/comfyui_fluxflow/TROUBLESHOOTING.md).

### Basic Workflow (No CFG)

```
[FluxFlowModelLoader] → model
[FluxFlowTextEncode] → text (FLUXFLOW_TEXT)
[FluxFlowEmptyLatent] → latent
[FluxFlowSampler] (model + text + latent) → sampled_latent
[FluxFlowVAEDecode] (model + sampled_latent) → image
```

### CFG Workflow (Recommended)

```
[FluxFlowModelLoader] → model
[FluxFlowTextEncode] (positive prompt) → text (FLUXFLOW_TEXT)
[FluxFlowEmptyLatent] → latent
[FluxFlowSampler] (model + text + latent + use_cfg=True + guidance_scale=5.0) → sampled_latent
[FluxFlowVAEDecode] (model + sampled_latent) → image
```

### Advanced CFG with Negative Prompt

```
[FluxFlowModelLoader] → model
[FluxFlowTextEncode] (positive prompt) → text (FLUXFLOW_TEXT)
[FluxFlowTextEncodeNegative] (negative prompt) → negative_text (FLUXFLOW_TEXT)
[FluxFlowEmptyLatent] → latent
[FluxFlowSampler] (model + text + negative_text + latent + use_cfg=True + guidance_scale=5.0) → sampled_latent
[FluxFlowVAEDecode] (model + sampled_latent) → image
```

## Classifier-Free Guidance (CFG)

FluxFlow supports CFG for improved prompt adherence and higher quality outputs.

### How CFG Works

CFG performs two forward passes during sampling:
1. **Conditional pass**: Using your positive prompt
2. **Unconditional pass**: Using null/negative embeddings

The final prediction is guided by: `v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)`

### Using CFG

**Basic CFG** (recommended for most use cases):
1. Load a FluxFlow checkpoint trained with CFG support
2. Encode your positive prompt with `FluxFlowTextEncode`
3. In `FluxFlowSampler`:
   - Set `use_cfg` to `True`
   - Set `guidance_scale` between 1.0-15.0 (recommended: 3.0-7.0)
   - Leave `negative_text` empty (uses a `zeros_like(text_seq)` null with the
     positive mask; see CHANGELOG known follow-up)

**Advanced CFG with Negative Prompts**:
1. Encode positive prompt with `FluxFlowTextEncode`
2. Encode negative prompt with `FluxFlowTextEncodeNegative`
3. Connect both to `FluxFlowSampler`
4. Set `use_cfg=True` and adjust `guidance_scale`

### Guidance Scale Guidelines

Recommended range: 3.0–7.0. At 1.0, guidance has no amplification effect (standard conditional generation). Above 7.0 may oversaturate or reduce diversity. Higher values increase computation (2x forward passes per step). See [fluxflow-core CFG documentation](https://github.com/danny-mio/fluxflow-core#classifier-free-guidance-cfg) for full guidance.

### CFG Performance

- **Memory**: CFG requires ~2x VRAM due to dual forward passes
- **Speed**: CFG sampling takes ~2x longer than standard sampling
- **Quality**: Improves prompt adherence and output coherence
- **Compatibility**: Requires checkpoints trained with CFG dropout

## Package Contents

- `comfyui_fluxflow.nodes` - Custom node implementations
- `comfyui_fluxflow.schedulers` - Sampling scheduler implementations
- `comfyui_fluxflow.web` - JavaScript extensions for ComfyUI UI

## Links

- [GitHub Repository](https://github.com/danny-mio/fluxflow-comfyui)
- [INSTALL / QUICKSTART](https://github.com/danny-mio/fluxflow-comfyui/tree/main/src/comfyui_fluxflow)
- [Core Package](https://pypi.org/project/fluxflow/)
- [Training Package](https://pypi.org/project/fluxflow-training/)

## License

MIT License - see LICENSE file for details.
