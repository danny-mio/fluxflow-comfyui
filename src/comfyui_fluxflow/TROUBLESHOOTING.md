# ComfyUI FluxFlow Plugin - Troubleshooting

## Common Issues and Solutions

### 1. "ImportError: cannot import name 'cached_download' from 'huggingface_hub'"

**Problem**: Dependency version conflict between ComfyUI's diffusers and huggingface_hub.

**Solution**: This has been fixed in v1.0.1+ with lazy scheduler loading. Update the plugin:

```bash
cd /path/to/fluxflow
git pull  # if using git
# or re-copy the plugin files
```

If still having issues, update dependencies in ComfyUI:
```bash
cd /path/to/ComfyUI
source venv/bin/activate  # if using venv
pip install --upgrade diffusers huggingface_hub
```

**Note**: The plugin now lazy-loads schedulers to avoid import conflicts.

---

### 2. "ModuleNotFoundError: No module named 'fluxflow'"

**Problem**: Plugin cannot find FluxFlow core package.

**Solutions**:

#### Option A: Install FluxFlow Core Package
```bash
# Install fluxflow package (if available)
pip install fluxflow

# Or install from local source
cd /path/to/fluxflow-core
pip install -e .
```

#### Option B: Use Symlink (Development)
```bash
# Remove copied plugin if it exists
rm -rf /path/to/ComfyUI/custom_nodes/comfyui_fluxflow

# Create symlink with ABSOLUTE path to the package directory
cd /path/to/ComfyUI/custom_nodes/
ln -s /absolute/path/to/fluxflow-comfyui/src/comfyui_fluxflow ./comfyui_fluxflow

# Verify symlink
ls -la comfyui_fluxflow
# Should show: comfyui_fluxflow -> /absolute/path/to/fluxflow-comfyui/src/comfyui_fluxflow
```

### 3. Verify Installation

**Check symlink resolution**:

```bash
cd /path/to/ComfyUI/custom_nodes/
ls -la comfyui_fluxflow
readlink comfyui_fluxflow
```

**Test imports**:
```python
# Verify fluxflow core is accessible
import fluxflow
print(f'FluxFlow version: {fluxflow.__version__}')

# Verify plugin imports
from comfyui_fluxflow.nodes.model_loader import FluxFlowModelLoader
print('Plugin imports successful')
```

### 4. Import Errors on Other Nodes

**Problem**: Other imports failing after fixing src import.

**Solution**: Ensure all dependencies are installed:

```bash
cd /path/to/ComfyUI
pip install safetensors transformers diffusers einops
```

### 5. Scheduler Import Warnings

**Problem**: Deprecation warnings about diffusers scheduler imports.

**Solution**: These are warnings, not errors. They don't affect functionality.
The schedulers work correctly despite the warnings.

### 6. Checkpoint Not Found

**Problem**: Model loader can't find checkpoint file.

**Solutions**:

1. Use absolute path:
```
/absolute/path/to/outputs/flux/flxflow_final.safetensors
```

2. Or path relative to FluxFlow project:
```
outputs/flux/flxflow_final.safetensors
```

3. Verify file exists:
```bash
ls -lh /path/to/checkpoint.safetensors
```

### 7. CUDA Out of Memory

**Problem**: GPU runs out of memory during generation.

**Solutions**:

1. Set device to CPU in Model Loader:
   - device: "cpu"

2. Reduce image dimensions:
   - Use 256x256 or 384x384 instead of 512x512

3. Reduce batch size to 1

4. Close other GPU applications

### 8. "No module named 'comfyui_fluxflow'"

**Problem**: Plugin not in Python path.

**Solution**: Restart ComfyUI completely:
```bash
# Kill ComfyUI process
pkill -f "python.*main.py"

# Restart
cd /path/to/ComfyUI
python main.py
```

### 9. Nodes Don't Appear in ComfyUI

**Problem**: Plugin loaded but nodes not visible.

**Solutions**:

1. Check ComfyUI console for import errors

2. Refresh node list in ComfyUI:
   - Right-click → Refresh
   - Or restart ComfyUI

3. Search for "FluxFlow" in node browser

4. Check plugin loaded:
```bash
# In ComfyUI console, you should see:
# "Import times for custom nodes:"
# "   0.X seconds: /path/to/comfyui_fluxflow"
```

### 10. Type Checking Errors in IDE

**Problem**: IDE shows import errors but code runs fine.

**Solution**: These are static type checking issues and can be ignored.
The plugin works correctly at runtime.

### 11. Black Formatting Errors

**Problem**: Code not formatted correctly.

**Solution**:
```bash
cd /path/to/fluxflow
python -m black comfyui_fluxflow/ --line-length 100
```

## Getting More Help

### Debug Information to Provide

When reporting issues, include:

1. ComfyUI console output (full error traceback)
2. Plugin installation method (symlink/pip install)
3. Operating system
4. Python version: `python --version`
5. PyTorch version: `python -c "import torch; print(torch.__version__)"`
6. FluxFlow version: `python -c "import fluxflow; print(fluxflow.__version__)"`

### Test Import Manually

```python
# Test if src can be imported
import sys
from pathlib import Path

# Adjust path to your setup
sys.path.insert(0, '/path/to/fluxflow')

try:
    from fluxflow.models import FluxPipeline
    print("✓ FluxFlow imports work")
except Exception as e:
    print(f"✗ Import failed: {e}")

try:
    from comfyui_fluxflow.nodes.model_loader import FluxFlowModelLoader
    print("✓ Plugin imports work")
except Exception as e:
    print(f"✗ Import failed: {e}")
```

### Verify Plugin Structure

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui_fluxflow
tree -L 2
# or
find . -name "*.py" | sort
```

Should show:
```
./\_\_init\_\_.py
./model_inspector.py
./schedulers.py
./nodes/\_\_init\_\_.py
./nodes/model_loader.py
./nodes/latent_ops.py
./nodes/text_encode.py
./nodes/samplers.py
./nodes/utils.py
```

## Still Having Issues?

1. Check ComfyUI logs for detailed error messages
2. Ensure FluxFlow training completed successfully
3. Verify checkpoint file is valid (.safetensors format)
4. Try the example workflow from README.md
5. Check FluxFlow repository issues on GitHub
