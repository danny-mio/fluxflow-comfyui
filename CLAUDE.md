# CLAUDE.md - FluxFlow ComfyUI

This document provides comprehensive guidance for AI assistants working with the FluxFlow ComfyUI integration codebase.

## Project Overview

FluxFlow ComfyUI is a custom nodes package for ComfyUI that integrates FluxFlow text-to-image generation. The package provides:

- **Model Loader**: Load FluxFlow checkpoints (.safetensors or .pth)
- **Text Encoding**: DistilBERT-based text encoding for prompts
- **Sampling**: 14 different schedulers (Euler, DPM++, DDIM, LCM, etc.)
- **VAE Operations**: Encode images to latents, decode latents to images
- **Latent Generation**: Create empty latent tensors at various resolutions
- **6 Custom Nodes**: Complete workflow for text-to-image generation

**Version**: 0.1.0 (Beta)

## Quick Commands

```bash
# Installation
pip install -e ".[dev]"           # Development mode (quotes required!)

# Testing
make test                         # Run all tests
pytest tests/test_foo.py -v                    # Single test file
pytest tests/test_foo.py::test_bar -v          # Single test function
pytest tests/test_foo.py::TestClass::test_bar  # Single method in class

# Code Quality
make lint                         # Run flake8, black --check, isort --check
make format                       # Format with black + isort
mypy src/                         # Type checking (run directly)
pre-commit run --all-files        # Run all pre-commit hooks

# Build
make build                        # Build distribution package
make clean                        # Clean build artifacts
```text
**Available Makefile Targets** (run `make help` to verify):
- `install` - Install package
- `install-dev` - Install with dev dependencies
- `test` - Run tests
- `lint` - Run linting checks
- `format` - Format code
- `clean` - Clean build artifacts
- `build` - Build distribution package

## Architecture

FluxFlow ComfyUI provides 6 custom nodes that integrate with ComfyUI's node graph system:

```text
[FluxFlowModelLoader] → model
[FluxFlowTextEncode] → conditioning
[FluxFlowEmptyLatent] → latent
↓
[FluxFlowSampler] (model + conditioning + latent) → sampled_latent
↓
[FluxFlowVAEDecode] (model + sampled_latent) → image
```text
**Node Workflow**:
1. Load FluxFlow checkpoint with `FluxFlowModelLoader`
1. Encode text prompts with `FluxFlowTextEncode`
1. Create empty latents with `FluxFlowEmptyLatent`
1. Sample/denoise with `FluxFlowSampler` (14 scheduler options)
1. Decode latents to images with `FluxFlowVAEDecode`

## Directory Structure

```text
fluxflow-comfyui/
├── src/comfyui_fluxflow/
│   ├── __init__.py              # Package init, node registration
│   ├── nodes/
│   │   ├── __init__.py          # Node exports
│   │   ├── model_loader.py      # FluxFlowModelLoader
│   │   ├── text_encode.py       # FluxFlowTextEncode
│   │   ├── latent_ops.py        # EmptyLatent, VAEEncode, VAEDecode
│   │   ├── samplers.py          # FluxFlowSampler
│   │   └── utils.py             # Node utilities
│   ├── schedulers.py            # Scheduler implementations
│   ├── standalone_schedulers.py # Standalone scheduler variants
│   ├── model_inspector.py       # Checkpoint inspection utilities
│   └── web/
│       └── fluxflow_types.js    # JavaScript UI extensions
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── test_nodes.py            # Node tests
│   ├── test_schedulers.py       # Scheduler tests
│   ├── test_model_loader.py     # Model loading tests
│   └── test_model_inspector.py  # Inspector tests
├── src/comfyui_fluxflow/
│   ├── INSTALL.md               # Installation instructions
│   ├── QUICKSTART.md            # Quick start guide
│   ├── TROUBLESHOOTING.md       # Common issues
│   └── README.md                # Package documentation
├── pyproject.toml               # Project configuration
├── Makefile                     # Build automation
└── .pre-commit-config.yaml      # Pre-commit hooks
```text
## Code Conventions

### Python Version and Dependencies

- **Python >= 3.10** required (supports 3.10, 3.11, 3.12)
- Use type hints on all public APIs
- Key dependencies: fluxflow, torch, transformers, pillow, numpy

### Formatting and Linting

- **Black**: line-length=100, target Python 3.10+
- **isort**: profile=black, line_length=100
- **flake8**: max-line-length=100, max-complexity=15
- **mypy**: type checking enabled (not strict)

### Import Order

```python
# stdlib
import math
from pathlib import Path

# third-party
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer

# local
from comfyui_fluxflow.schedulers import create_scheduler
from .utils import validate_latent_shape
```text
### Docstrings (Google-style)

```python
def node_function(arg1: str, arg2: int = 10) -> dict:
    """
    Short description of the node function.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something is wrong

    Example:
        >>> node_function("test")
        {'result': 'test'}
    """
```text
### Naming Conventions

- **Functions/variables**: snake_case
- **Classes**: PascalCase (ComfyUI nodes use PascalCase)
- **Constants**: UPPER_SNAKE_CASE
- **Private methods**: _leading_underscore

### Error Handling

Use descriptive error messages for ComfyUI users:

```python
if checkpoint_path is None or not checkpoint_path.strip():
    raise ValueError("Checkpoint path cannot be empty")

if not Path(checkpoint_path).exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
```text
### Logging

Use print statements for ComfyUI console output:

```python
print(f"\nFluxFlow Model Loader:")
print(f"  Checkpoint: {checkpoint_path}")
print(f"  Device: {device}")
print(f"  Model loaded successfully")
```text
## Testing Guidelines

### Test Structure

```python
import pytest

class TestFluxFlowNode:
    """Tests for FluxFlow custom nodes."""

    def test_node_input_types(self):
        """Test node INPUT_TYPES structure."""
        inputs = FluxFlowModelLoader.INPUT_TYPES()
        assert "required" in inputs
        assert "checkpoint_path" in inputs["required"]

    def test_node_execution(self, mock_checkpoint):
        """Test node execution."""
        node = FluxFlowModelLoader()
        result = node.load_model(mock_checkpoint)
        assert result is not None
```text
### Test Markers

- `@pytest.mark.slow` - Tests > 1s execution time
- `@pytest.mark.gpu` - Tests requiring GPU/CUDA
- `@pytest.mark.integration` - Integration tests with ComfyUI

### Available Fixtures (conftest.py)

- **Device fixtures**: `device`, `cpu_device`
- **Mock fixtures**: `mock_checkpoint_path`, `mock_tokenizer`, `mock_fluxflow_model`
- **Tensor fixtures**: `sample_latent`, `sample_conditioning`

### Test Coverage

Tests are run with pytest-cov. Aim for high coverage on node logic.

## ComfyUI Integration Patterns

### Node Class Structure

All FluxFlow nodes follow this pattern:

```python
class FluxFlowNodeName:
    """Node description for ComfyUI UI."""

    @classmethod
    def INPUT_TYPES(cls):
        """Define node inputs."""
        return {
            "required": {
                "param1": ("TYPE", {"default": value}),
            },
            "optional": {
                "param2": ("TYPE", {"default": value}),
            },
        }

    RETURN_TYPES = ("FLUXFLOW_TYPE",)
    RETURN_NAMES = ("output_name",)
    FUNCTION = "process"
    CATEGORY = "FluxFlow/category"

    def process(self, param1, param2=None):
        """Node processing logic."""
        # Implementation
        return (result,)  # Always return tuple
```text
### Custom Data Types

FluxFlow defines custom types for ComfyUI:

- `FLUXFLOW_MODEL` - Loaded FluxFlow pipeline
- `FLUXFLOW_CONDITIONING` - Text embeddings tensor
- `FLUXFLOW_LATENT` - Latent packet tensor [B, T+1, D]

### Node Categories

All nodes are organized under `FluxFlow/`:

- `FluxFlow/loaders` - Model and checkpoint loading
- `FluxFlow/conditioning` - Text encoding
- `FluxFlow/latent` - Latent operations
- `FluxFlow/sampling` - Sampling and denoising

## Key Implementation Details

### Checkpoint Loading

FluxFlow supports `.safetensors` and `.pth` checkpoint formats:

```python
from fluxflow.models import FluxPipeline

# Load checkpoint
pipeline = FluxPipeline.from_pretrained(
    checkpoint_path,
    device=device,
    torch_dtype=torch.float16,  # Optional
)
```text
### Text Encoding

Uses DistilBERT tokenizer and encoder:

```python
from transformers import DistilBertTokenizer
from fluxflow.models import BertTextEncoder

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
encoder = BertTextEncoder(embed_dim=256)

tokens = tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
embeddings = encoder(tokens['input_ids'], tokens['attention_mask'])
```text
### Scheduler Configuration

14 schedulers available via `create_scheduler()`:

```python
from comfyui_fluxflow.schedulers import create_scheduler

scheduler = create_scheduler(
    scheduler_name="DPMSolverMultistep",
    num_train_timesteps=1000,
    prediction_type="v_prediction",
)
scheduler.set_timesteps(steps=20, device=device)
```text
**Available Schedulers**:
- Euler, EulerAncestral
- DPMSolverMultistep, DPMSolverSinglestep
- DPMSolverMultistepKarras, DPMSolverSinglestepKarras
- DDIM, DDPM
- LCM (Latent Consistency Model)
- PNDM, UniPCMultistep
- HeunDiscrete, DPM2, DPM2Ancestral

### Latent Format

FluxFlow uses packet-based latent representation:

```python
# Shape: [B, T+1, D]
# - B: batch size
# - T: number of tokens (H*W/256)
# - +1: HW dimension token
# - D: latent dimension (vae_dim)

# Create empty latent
batch_size = 1
height, width = 512, 512
tokens = (height // 16) * (width // 16)  # 16x spatial compression
latent_dim = 128

latent = torch.randn(batch_size, tokens + 1, latent_dim)
latent[:, -1, 0] = height  # Encode height
latent[:, -1, 1] = width   # Encode width
```text
## ComfyUI Node Development

### Testing Nodes in ComfyUI

1. **Symlink Installation** (recommended for development):

```bash
cd ComfyUI/custom_nodes
ln -s /path/to/fluxflow-comfyui fluxflow-comfyui
cd fluxflow-comfyui
pip install -e ".[dev]"
```text
1. **Restart ComfyUI** to load nodes

1. **Verify nodes appear** in node browser under "FluxFlow" category

### Node Registration

Nodes are automatically registered via `NODE_CLASS_MAPPINGS` in `__init__.py`:

```python
NODE_CLASS_MAPPINGS = {
    "FluxFlowModelLoader": FluxFlowModelLoader,
    "FluxFlowTextEncode": FluxFlowTextEncode,
    # ... other nodes
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxFlowModelLoader": "FluxFlow Model Loader",
    "FluxFlowTextEncode": "FluxFlow Text Encode",
    # ... other nodes
}
```text
### Web UI Extensions

JavaScript extensions in `web/fluxflow_types.js` customize ComfyUI UI for FluxFlow nodes.

## Model Checkpoint Requirements

FluxFlow checkpoints must contain:

- VAE encoder/decoder weights
- Flow transformer weights
- Text encoder projection weights (DistilBERT loaded separately)
- Configuration metadata (latent_dim, embed_dim, etc.)

**Checkpoint Inspection**:

```python
from comfyui_fluxflow.model_inspector import inspect_checkpoint

info = inspect_checkpoint("path/to/checkpoint.safetensors")
print(f"Latent dim: {info['latent_dim']}")
print(f"Embed dim: {info['embed_dim']}")
print(f"Total params: {info['total_params']}")
```text
## Development Workflow

### Before Making Changes

1. Create a feature branch
1. Install dev dependencies: `make install-dev`
1. Run existing tests: `make test`

### Making Changes

1. Write code following conventions above
1. Add/update tests for new functionality
1. Keep functions < 50 lines, complexity < 15
1. Use type hints on public APIs
1. Test nodes in ComfyUI before committing

### Before Committing

1. Format code: `make format`
1. Run linting: `make lint`
1. Run tests: `make test`
1. Pre-commit hooks: `pre-commit run --all-files`
1. Test in actual ComfyUI instance if nodes were modified

### Commit Messages

Use clear, descriptive commit messages:
- `Add node X for Y functionality`
- `Fix scheduler Z initialization bug`
- `Refactor latent validation for better error messages`

## Common Tasks

### Adding a New Node

1. Create node class in `src/comfyui_fluxflow/nodes/`
1. Follow ComfyUI node pattern (see above)
1. Export in `nodes/__init__.py`
1. Register in top-level `__init__.py`
1. Add tests in `tests/test_nodes.py`
1. Update documentation in package README

### Adding a New Scheduler

1. Add scheduler name to `SCHEDULER_CLASSES` in `schedulers.py`
1. Update `get_scheduler_list()` return value
1. Test with `create_scheduler()` function
1. Add tests in `tests/test_schedulers.py`

### Modifying Node Inputs

1. Update `INPUT_TYPES()` class method
1. Update node function signature
1. Update docstring with new parameters
1. Add validation for new inputs
1. Update tests to cover new inputs

### Adding Dependencies

1. Add to `dependencies` in `pyproject.toml` (runtime)
1. Or add to `dev` optional dependencies (development only)
1. Update `requirements.txt` for non-editable installs

## CI/CD

GitHub Actions runs on push/PR to main and develop branches:

1. Linting (flake8)
1. Format checking (black)
1. Type checking (mypy)
1. Tests with coverage (pytest-cov)
1. Coverage upload (codecov)

Publishing to PyPI triggers on version tags (`v*`) - currently disabled until package is ready for public release.

## Related Repositories

- **fluxflow-core**: Base model implementations
- **fluxflow-training**: Training tools and scripts
- **fluxflow-ui**: Web interface for training and generation

## Installation Methods

### Development (Recommended)

```bash
# Symlink into ComfyUI custom_nodes
cd ComfyUI/custom_nodes
ln -s /path/to/fluxflow-comfyui fluxflow-comfyui
cd fluxflow-comfyui
pip install -e ".[dev]"
```text
### Production (Git Clone)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/danny-mio/fluxflow-comfyui.git
cd fluxflow-comfyui
pip install -e .
```text
### Package Installation (Not Yet Available)

Package not yet published to PyPI. Use development or git clone methods above.

## Known Issues

**setup.sh and Makefile potential issues:**
- Both may use unquoted `.[dev]` which can fail in zsh and some other shells
- Use `pip install -e ".[dev]"` with quotes instead

**ComfyUI Integration:**
- Nodes require ComfyUI to be installed and properly configured
- JavaScript extensions may need cache clearing after updates
- Checkpoint paths must be absolute or relative to ComfyUI root

## Additional Documentation

- `README.md` - Project overview and features
- `src/comfyui_fluxflow/INSTALL.md` - Detailed installation instructions
- `src/comfyui_fluxflow/QUICKSTART.md` - Quick start guide with workflows
- `src/comfyui_fluxflow/TROUBLESHOOTING.md` - Common issues and solutions
- `CONTRIBUTING.md` - Contribution guidelines
- `AGENTS.md` - Quick reference for AI assistants
