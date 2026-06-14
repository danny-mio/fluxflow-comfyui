# FluxFlow ComfyUI Plugin - Quick Start

## Installation (60 seconds)

1. Navigate to the ComfyUI `custom_nodes` directory and clone the plugin:

   ```bash
   cd /path/to/ComfyUI/custom_nodes/
   git clone https://github.com/danny-mio/fluxflow-comfyui.git
   cd fluxflow-comfyui
   pip install -r requirements.txt
   ```

2. Restart ComfyUI so the new nodes are picked up:

   ```bash
   pkill -f "python.*main.py"
   cd /path/to/ComfyUI && python main.py
   ```

## First Generation (2 minutes)

### 1. Add Nodes

In ComfyUI interface, add these nodes (search for "FluxFlow"):

1. **FluxFlow Model Loader**
2. **FluxFlow Empty Latent**
3. **FluxFlow Text Encode**
4. **FluxFlow Sampler**
5. **FluxFlow VAE Decode**
6. **Preview Image** (standard ComfyUI node)

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

> **v0.10.0**: the sampler's text input is named `text` (was `conditioning`
> in v0.8.x) and the socket type is `FLUXFLOW_TEXT` carrying a
> `(text_seq, text_mask)` tuple. Old workflows fail at load until they are
> re-wired to the new socket — see `TROUBLESHOOTING.md`.

```
Model Loader outputs:
├─ model → Text Encode, Sampler, VAE Decode
├─ text_encoder → Text Encode
└─ tokenizer → Text Encode

Empty Latent:
└─ latent → Sampler

Text Encode:
└─ text (FLUXFLOW_TEXT) → Sampler.text

(Optional, for CFG)
Text Encode (Negative):
└─ negative_text (FLUXFLOW_TEXT) → Sampler.negative_text

Sampler:
└─ latent → VAE Decode

VAE Decode:
└─ image → Preview Image
```

### 4. Generate!

Click "Queue Prompt" to start generation. Check the console for progress output.

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

```
FluxFlow Model Configuration (Auto-Detected)
VAE Latent Dim:        128
Flow Model Dim:        128
Text Embedding Dim:    1024
Compression Ratio:     16x
✓ All dimensions auto-configured!
```

No manual configuration needed!
