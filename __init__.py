import os
import sys

# Ensure src/ is on the path so comfyui_fluxflow is importable
# regardless of whether the package is pip-installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from comfyui_fluxflow import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
