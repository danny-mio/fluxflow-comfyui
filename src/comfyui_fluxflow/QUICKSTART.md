# FluxFlow ComfyUI Plugin - Quick Start

## Installation (60 seconds)

```bash
# 1. Navigate to ComfyUI custom nodes
cd /path/to/ComfyUI/custom_nodes/

# 2. Create symlink (use YOUR actual path!)
ln -s /path/to/fluxflow-comfyui/src/comfyui_fluxflow ./comfyui_fluxflow

# 3. Restart ComfyUI
# Kill existing: Ctrl+C or pkill -f "python.*main.py"
# Start: cd /path/to/ComfyUI && python main.py
```text
## First Generation (2 minutes)

### 1. Add Nodes

In ComfyUI interface, add these nodes (search for "FluxFlow"):

1. **FluxFlow Model Loader**
1. **FluxFlow Empty Latent**  
1. **FluxFlow Text Encode**
1. **FluxFlow Sampler**
1. **FluxFlow VAE Decode**
1. **Preview Image** (standard ComfyUI node)

### 2. Configure

**FluxFlow Model Loader:**
- checkpoint_path: `outputs/flux/flxflow_final.safetensors`
- device: `auto`

**FluxFlow Empty Latent:**
- width: `512`
- height: `512`
- seed: `42`

**FluxFlow Text Encode:**
- text: `A beautiful sunset over mountains`

**FluxFlow Sampler:**
- steps: `20`
- scheduler: `DPMSolverMultistep`

### 3. Connect

```text
Model Loader outputs:
├─ model → Text Encode, Sampler, VAE Decode
├─ text_encoder → Text Encode
└─ tokenizer → Text Encode

Empty Latent:
└─ latent → Sampler

Text Encode:
└─ conditioning → Sampler

Sampler:
└─ latent → VAE Decode

VAE Decode:
└─ image → Preview Image
```text
### 4. Generate!

Click "Queue Prompt" and watch the magic happen!

## Troubleshooting

**Nodes don't appear?**
- Restart ComfyUI completely
- Check console for import errors
- Verify symlink: `ls -la /path/to/ComfyUI/custom_nodes/comfyui_fluxflow`

**Import errors?**
- See TROUBLESHOOTING.md for detailed solutions
- Ensure symlink uses absolute path
- Check dependencies are installed

**Other issues?**
- See TROUBLESHOOTING.md for detailed solutions

## Next Steps

- Try different schedulers (EulerDiscrete, LCM for fast)
- Experiment with image sizes
- Try image-to-image (add FluxFlow VAE Encode)
- Read README.md for all features
- Check out 14 available schedulers!

## Configuration Detected

When you load a checkpoint, the console shows:

```text
FluxFlow Model Configuration (Auto-Detected)
VAE Latent Dim:        32
Flow Model Dim:        32  
Text Embedding Dim:    1024
Compression Ratio:     16x
✓ All dimensions auto-configured!
```text
No manual configuration needed!

