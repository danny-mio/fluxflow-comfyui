# ComfyUI FluxFlow Plugin - Installation Guide

## Quick Start

### 1. Prerequisites

- ComfyUI installed and working
- Python 3.10+ with torch
- FluxFlow trained checkpoint (.safetensors)

### 2. Installation

**Symlink Installation (Recommended)**

This is the recommended method for development.

```bash
# Navigate to ComfyUI custom_nodes directory
cd /path/to/ComfyUI/custom_nodes/

# Create symlink to plugin (MUST use absolute path)
ln -s /absolute/path/to/fluxflow-comfyui/src/comfyui_fluxflow ./comfyui_fluxflow
```

### 3. Verify Installation

```bash
# Check dependencies (usually already installed with ComfyUI)
cd /path/to/ComfyUI/custom_nodes/comfyui_fluxflow
pip install -r requirements.txt
```

### 4. Restart ComfyUI

```bash
# Restart ComfyUI server
# The plugin will be auto-loaded
```

### 5. Verify Nodes Loaded

In ComfyUI interface, search for "FluxFlow" in the node browser.

You should see:
- FluxFlow Model Loader
- FluxFlow Empty Latent
- FluxFlow VAE Encode
- FluxFlow VAE Decode
- FluxFlow Text Encode
- FluxFlow Sampler

## First Workflow

1. **Add FluxFlowModelLoader node**
   - Set `checkpoint_path` to your trained model
   - Leave `device` as "auto"

2. **Add FluxFlowEmptyLatent node**
   - Set desired width/height (e.g., 512x512)
   - Connect to sampler

3. **Add FluxFlowTextEncode node**
   - Connect text_encoder and tokenizer from loader
   - Enter your prompt

4. **Add FluxFlowSampler node**
   - Connect model, latent, and conditioning
   - Choose scheduler (default: DPMSolverMultistep)
   - Set steps (20-50 recommended)

5. **Add FluxFlowVAEDecode node**
   - Connect model and sampled latent
   - Output connects to SaveImage or Preview

6. **Queue Prompt!**

## Troubleshooting

### "Module not found" errors

```bash
# Install missing dependencies
pip install torch safetensors transformers diffusers einops Pillow
```

### "Checkpoint not found"

Use absolute paths or paths relative to FluxFlow project root:

```
✓ /absolute/path/to/outputs/flux/flxflow_final.safetensors
✓ outputs/flux/flxflow_final.safetensors
```

### CUDA out of memory

- Set `device: "cpu"` in Model Loader
- Use smaller batch sizes
- Reduce image dimensions

### Import errors on startup

Check ComfyUI console for specific error messages.
Ensure FluxFlow `src/` directory is accessible from plugin.

## Next Steps

See [README.md](README.md) for:
- Complete node documentation
- Example workflows
- Advanced configuration
- Scheduler options
- Performance tips

## Support

For issues specific to:
- **Plugin**: Check FluxFlow repository issues
- **ComfyUI**: Check ComfyUI repository
- **Models**: Verify checkpoint training completed successfully
