#!/bin/bash
# FluxFlow ComfyUI - Setup Script
# Sets up the ComfyUI plugin for FluxFlow

set -e

echo "=== FluxFlow ComfyUI Plugin Setup ==="

# Check if running inside ComfyUI custom_nodes
if [[ "$PWD" == *"custom_nodes"* ]]; then
    echo "Detected ComfyUI custom_nodes directory"
    COMFYUI_MODE=true
else
    COMFYUI_MODE=false
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required (found $python_version)"
    exit 1
fi

echo "Python version: $python_version"

if [ "$COMFYUI_MODE" = true ]; then
    # Install directly into ComfyUI's environment
    echo "Installing into ComfyUI environment..."
    pip install -e .
else
    # Create virtual environment for standalone development
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate virtual environment
    source venv/bin/activate

    # Upgrade pip
    pip install --upgrade pip

    # Install PyTorch
    echo "Installing PyTorch..."
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected, installing CUDA version..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "No GPU detected, installing CPU version..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    # Install package
    if [ "$1" == "--dev" ]; then
        echo "Installing in development mode with dev dependencies..."
        pip install -e .[dev]
    else
        echo "Installing package..."
        pip install -e .
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""

if [ "$COMFYUI_MODE" = true ]; then
    echo "Plugin installed into ComfyUI."
    echo "Restart ComfyUI to load the FluxFlow nodes."
else
    echo "To activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
    echo "For ComfyUI integration, copy or symlink this directory"
    echo "to your ComfyUI/custom_nodes/ folder:"
    echo ""
    echo "  ln -s \$(pwd) /path/to/ComfyUI/custom_nodes/fluxflow-comfyui"
fi
echo ""
