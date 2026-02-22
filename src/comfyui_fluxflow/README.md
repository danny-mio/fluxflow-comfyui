# ComfyUI FluxFlow Plugin

A comprehensive ComfyUI plugin for using FluxFlow diffusion models with automatic configuration detection and advanced scheduler support.

## Features

### Automatic Configuration Detection
- **Zero Manual Configuration**: Automatically detects all model dimensions from checkpoint
- **Multi-Model Support**: Works with any FluxFlow model size (32d to 128d+ VAE dimensions)
- **Intelligent Validation**: Cross-checks detected parameters for consistency

### Complete FluxFlow Workflow
- **Model Loading**: Auto-detecting checkpoint loader
- **Empty Latent Generation**: Create random latents for target dimensions
- **VAE Encode/Decode**: Full image ↔ latent conversion
- **Text Conditioning**: DistilBERT text encoding
- **Flow Sampling**: Advanced denoising with 14 schedulers

### Advanced Sampling
- **14 Schedulers**: DPM++, DPM++ Karras, Euler, DDIM, LCM, UniPC, and more
- **Standalone Fallback**: Works even with broken diffusers installations
- **Prediction Types**: v_prediction, epsilon, sample
- **Reproducible**: Seed control for deterministic generation
- **Flexible**: Configurable steps, scheduler parameters

### ComfyUI Native
- **Proper Tensor Formats**: Automatic conversion between ComfyUI and FluxFlow formats
- **Native Integration**: Works seamlessly with other ComfyUI nodes
- **Progress Logging**: Detailed console output for debugging

---

## Installation

### Method 1: Git Clone (Recommended)

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/danny-mio/fluxflow-comfyui.git
cd fluxflow-comfyui
pip install -r requirements.txt
```

### Method 2: Symlink for Development

```bash
cd ComfyUI/custom_nodes/
ln -s /path/to/fluxflow-comfyui ./fluxflow-comfyui
pip install -r fluxflow-comfyui/requirements.txt
```

---

## Nodes Overview

### 1. FluxFlow Model Loader

**Purpose**: Load FluxFlow checkpoint with automatic configuration detection

**Inputs**:
- `checkpoint_path` (STRING): Path to .safetensors file
- `device` (COMBO): auto, cuda, cpu, mps (default: auto)
- `tokenizer_name` (STRING): HuggingFace tokenizer (default: distilbert-base-uncased)

**Outputs**:
- `model`: FluxFlow pipeline (compressor + flow + expander)
- `text_encoder`: BertTextEncoder
- `tokenizer`: DistilBERT tokenizer
- `config_info`: Detected configuration summary

**Auto-Detected Parameters**:
- VAE latent dimension (vae_dim)
- Flow model dimension (flow_dim)
- Text embedding dimension (text_embed_dim)
- Downscale/upscale stages
- Attention layers and heads
- Compression ratio

---

### 2. FluxFlow Empty Latent

**Purpose**: Generate random latent for target image dimensions

**Inputs**:
- `model`: FluxFlow pipeline (auto-detects vae_dim, downscales, max_hw)
- `width` (INT): Target image width (default: 512)
- `height` (INT): Target image height (default: 512)
- `batch_size` (INT): Batch size (default: 1)
- `seed` (INT): Random seed (optional)

**Outputs**:
- `latent`: Random latent packet [B, T+1, D]

**Notes**:
- Automatically inherits parameters from model (no manual configuration needed)
- Latent format: [B, T+1, D] where T depends on model's downscale settings
- Last token (+1) encodes spatial dimensions

---

### 3. FluxFlow VAE Encode

**Purpose**: Encode image to latent space

**Inputs**:
- `model`: FluxFlow pipeline
- `image`: ComfyUI image [B, H, W, C] in [0, 1]

**Outputs**:
- `latent`: Encoded latent packet [B, T+1, D]

**Notes**:
- Automatically converts ComfyUI format to FluxFlow format
- Uses VAE compressor with variational encoding

---

### 4. FluxFlow VAE Decode

**Purpose**: Decode latent to image

**Inputs**:
- `model`: FluxFlow pipeline
- `latent`: Latent packet [B, T+1, D]
- `use_context` (BOOLEAN): Enable context conditioning (default: True)

**Outputs**:
- `image`: ComfyUI image [B, H, W, C] in [0, 1]

**Notes**:
- Automatically converts FluxFlow format to ComfyUI format
- Context conditioning improves reconstruction quality

---

### 5. FluxFlow Text Encode

**Purpose**: Encode text prompt to conditioning

**Inputs**:
- `text_encoder`: BertTextEncoder from loader
- `tokenizer`: Tokenizer from loader
- `text` (STRING, multiline): Text prompt

**Outputs**:
- `conditioning`: Text embeddings [B, D]

**Notes**:
- Uses DistilBERT for text encoding
- Max sequence length: 512 tokens
- Automatically pads/truncates

---

### 6. FluxFlow Sampler

**Purpose**: Denoise latent using flow model

**Inputs**:
- `model`: FluxFlow pipeline
- `latent`: Noisy latent packet
- `conditioning`: Text embeddings
- `steps` (INT): Sampling steps (1-1000, default: 20)
- `scheduler` (COMBO): Scheduler selection
- `prediction_type` (COMBO): v_prediction, epsilon, sample
- `seed` (INT): Random seed

**Outputs**:
- `latent`: Denoised latent packet

**Available Schedulers** (14 total):
1. **DPMSolverMultistep** (default) - Fast, high quality
2. **DPMPlusPlusKarras** ⭐ NEW - Premium quality with Karras schedule
3. **DPMSolverSinglestep** - Single-step variant
4. **DPMSolverSDE** - Stochastic variant
5. **EulerDiscrete** - Simple, stable
6. **EulerAncestralDiscrete** - Stochastic Euler
7. **HeunDiscrete** - Second-order method
8. **DDIM** - Classic DDIM sampler
9. **DDPM** - Original DDPM
10. **LCM** - Latent Consistency Model (fast!)
11. **UniPCMultistep** - Unified predictor-corrector
12. **KDPM2Discrete** - Karras DPMPP 2M
13. **KDPM2AncestralDiscrete** - Karras ancestral
14. **DEISMultistep** - Diffusion exponential integrator

**Note**: All schedulers work with standalone fallback if diffusers is broken

---

## Example Workflow

### Basic Text-to-Image Generation

```
1. FluxFlowModelLoader
   └─ checkpoint_path: "outputs/flux/flxflow_final.safetensors"
   └─ device: "auto"
   ↓
   ├─ model → 2, 3, 4
   ├─ text_encoder → 3
   └─ tokenizer → 3

2. FluxFlowEmptyLatent
   └─ width: 512
   └─ height: 512
   └─ seed: 42
   ↓
   └─ latent → 4

3. FluxFlowTextEncode
   ├─ text_encoder (from 1)
   ├─ tokenizer (from 1)
   └─ text: "A beautiful sunset over mountains"
   ↓
   └─ conditioning → 4

4. FluxFlowSampler
   ├─ model (from 1)
   ├─ latent (from 2)
   ├─ conditioning (from 3)
   ├─ steps: 20
   ├─ scheduler: "DPMSolverMultistep"
   └─ prediction_type: "v_prediction"
   ↓
   └─ latent → 5

5. FluxFlowVAEDecode
   ├─ model (from 1)
   └─ latent (from 4)
   ↓
   └─ image (final output)
```

### Image-to-Image with Different Scheduler

```
1. Load Image (ComfyUI native)
   ↓
   └─ image → 2

2. FluxFlowModelLoader
   └─ checkpoint_path: "..."
   ↓
   ├─ model → 3, 5
   ├─ text_encoder → 4
   └─ tokenizer → 4

3. FluxFlowVAEEncode
   ├─ model (from 2)
   └─ image (from 1)
   ↓
   └─ latent → 5

4. FluxFlowTextEncode
   ├─ text_encoder (from 2)
   ├─ tokenizer (from 2)
   └─ text: "Transform into oil painting style"
   ↓
   └─ conditioning → 5

5. FluxFlowSampler
   ├─ model (from 2)
   ├─ latent (from 3)
   ├─ conditioning (from 4)
   ├─ steps: 30
   └─ scheduler: "EulerAncestralDiscrete"
   ↓
   └─ latent → 6

6. FluxFlowVAEDecode
   ├─ model (from 2)
   └─ latent (from 5)
   ↓
   └─ image (final output)
```

---

## Technical Details

### Latent Format

FluxFlow uses a packed latent representation:
- **Format**: [B, T+1, D]
- **T**: Number of spatial tokens = (H//2^downscales) * (W//2^downscales)
- **D**: Latent dimension (e.g., 32, 128)
- **+1**: Last token encodes spatial dimensions (H/max_hw, W/max_hw)

Example for 512x512 image with downscales=4:
- Compression: 16x (2^4)
- Latent spatial: 32x32 = 1024 tokens
- Shape: [1, 1025, 128] for vae_dim=128

### Image Format Conversion

**ComfyUI Format** → **FluxFlow Format**:
- [B, H, W, C] → [B, C, H, W]
- [0, 1] → [-1, 1]

**FluxFlow Format** → **ComfyUI Format**:
- [B, C, H, W] → [B, H, W, C]
- [-1, 1] → [0, 1]

### Scheduler Configuration

Each scheduler has sensible defaults:
- **DPMSolverMultistep**: algorithm_type="dpmsolver++", solver_order=2
- **EulerDiscrete**: timestep_spacing="trailing"
- **LCM**: Optimized for fast generation (4-8 steps)

All schedulers support:
- `num_train_timesteps`: 1000 (default)
- `prediction_type`: v_prediction, epsilon, or sample

---

## Troubleshooting

### Import Errors

If you see import errors on ComfyUI startup:

```bash
# Install missing dependencies
cd ComfyUI/custom_nodes/comfyui_fluxflow
pip install -r requirements.txt
```

### Checkpoint Not Found

Ensure the checkpoint path is absolute or relative to ComfyUI root:

```
✓ Good: "/absolute/path/to/outputs/flux/flxflow_final.safetensors"
✓ Good: "outputs/flux/flxflow_final.safetensors" (relative to FluxFlow project)
✗ Bad: "flux/model.safetensors" (ambiguous)
```

### Device Errors

If CUDA out of memory:
- Set `device: "cpu"` in Model Loader
- Reduce batch_size
- Use smaller image dimensions

### Dimension Mismatch

If you get dimension errors:
- Let auto-detection handle it (don't override vae_dim manually)
- Check console output for detected configuration
- Ensure checkpoint is a valid FluxFlow model

---

## Performance Tips

### Fast Generation
- Use `LCM` scheduler with 4-8 steps
- Use smaller image sizes (256x256, 384x384)
- Set `device: "cuda"` if available

### High Quality
- Use `DPMSolverMultistep` or `UniPCMultistep` with 20-50 steps
- Enable `use_context: True` in VAE Decode
- Use larger models (vae_dim=128)

### Reproducibility
- Always set the same seed in Empty Latent and Sampler
- Use deterministic schedulers (avoid ancestral variants for exact reproduction)

---

## Development

### Project Structure

Structure of the `src/comfyui_fluxflow/` package directory:

```
comfyui_fluxflow/
├── __init__.py                   # Plugin entry point
├── model_inspector.py            # Auto-detection system
├── schedulers.py                 # Scheduler factory
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── nodes/
    ├── __init__.py              # Node exports
    ├── model_loader.py          # Model loader node
    ├── latent_ops.py            # Latent operations
    ├── text_encode.py           # Text encoding
    ├── samplers.py              # Sampling node
    └── utils.py                 # Image conversion utilities
```

### Adding New Schedulers

To add a new scheduler:

1. Import in `schedulers.py`:
```python
from diffusers import NewScheduler
```

2. Add to `SCHEDULER_MAP`:
```python
SCHEDULER_MAP = {
    ...
    "NewScheduler": NewScheduler,
}
```

3. Add defaults to `SCHEDULER_DEFAULTS`:
```python
SCHEDULER_DEFAULTS = {
    ...
    "NewScheduler": {
        "prediction_type": "v_prediction",
        ...
    },
}
```

---

## License

Same license as FluxFlow project.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

### v0.8.0 (2026-02-21)
- v0.8.0 checkpoint detection in `FluxFlowModelLoader` (pillar-attention weights)
- Updated `fluxflow` dependency to `>=0.8.0`

### v0.1.0 (2025-01-13)
- Initial release: 6 nodes, 14 schedulers, automatic configuration detection
