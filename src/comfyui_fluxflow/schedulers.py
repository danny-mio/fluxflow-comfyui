"""
Scheduler factory and configuration for FluxFlow sampling.

Provides access to 12+ diffusers schedulers with sensible defaults.
Falls back to standalone implementations if diffusers is broken.
"""

from typing import Any, Dict, Optional, cast

# Lazy import schedulers to avoid dependency conflicts
# Import only when needed to prevent version incompatibilities
_SCHEDULER_CLASSES: Dict[str, Any] = {}
_DIFFUSERS_AVAILABLE: Optional[bool] = None  # Tri-state: None=unknown, True=working, False=broken


def _check_diffusers_available():
    """Check if diffusers package is available and working."""
    global _DIFFUSERS_AVAILABLE
    if _DIFFUSERS_AVAILABLE is not None:
        return _DIFFUSERS_AVAILABLE

    try:
        # Try importing diffusers AND a scheduler to verify it works
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler  # noqa: F401

        _DIFFUSERS_AVAILABLE = True
    except (ImportError, Exception):
        # If any error occurs (including huggingface_hub issues), mark as unavailable
        _DIFFUSERS_AVAILABLE = False

    return _DIFFUSERS_AVAILABLE


def _get_scheduler_class(name: str):  # noqa: C901
    """Lazy load scheduler class to avoid import errors."""
    if name in _SCHEDULER_CLASSES:
        return _SCHEDULER_CLASSES[name]

    # Check if diffusers is available
    if not _check_diffusers_available():
        # Fall back to standalone schedulers
        from .standalone_schedulers import STANDALONE_SCHEDULERS

        # Map diffusers names to standalone schedulers
        fallback_map = {
            "DDIMScheduler": "DDIM",
            "EulerDiscreteScheduler": "Euler",
            "DPMSolverMultistepScheduler": "DPMPlusPlus",  # Use high-quality DPM++
            "DPMSolverSinglestepScheduler": "FlowMatching",
            "DPMSolverSDEScheduler": "FlowMatching",
            "DPMPlusPlusKarrasScheduler": "DPMPlusPlus",  # New scheduler
            "EulerAncestralDiscreteScheduler": "Euler",
            "HeunDiscreteScheduler": "Euler",
            "DDPMScheduler": "DDIM",
            "LCMScheduler": "FlowMatching",
            "UniPCMultistepScheduler": "DPMPlusPlus",  # Use DPM++ for quality
            "KDPM2DiscreteScheduler": "Euler",
            "KDPM2AncestralDiscreteScheduler": "Euler",
            "DEISMultistepScheduler": "FlowMatching",
        }

        fallback_name = fallback_map.get(name, "FlowMatching")
        if fallback_name not in STANDALONE_SCHEDULERS:
            raise ValueError(f"No fallback scheduler for {name}")

        _SCHEDULER_CLASSES[name] = STANDALONE_SCHEDULERS[fallback_name]
        return _SCHEDULER_CLASSES[name]

    # Import on first use
    try:
        if name == "DDIMScheduler":
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler

            _SCHEDULER_CLASSES[name] = DDIMScheduler
        elif name == "DDPMScheduler":
            from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

            _SCHEDULER_CLASSES[name] = DDPMScheduler
        elif name == "DPMSolverMultistepScheduler":
            from diffusers.schedulers.scheduling_dpmsolver_multistep import (
                DPMSolverMultistepScheduler,
            )

            _SCHEDULER_CLASSES[name] = DPMSolverMultistepScheduler
        elif name == "DPMPlusPlusKarrasScheduler":
            # DPM++ with Karras schedule uses the same class as DPMSolverMultistep
            # with use_karras_sigmas=True (set in SCHEDULER_DEFAULTS)
            from diffusers.schedulers.scheduling_dpmsolver_multistep import (
                DPMSolverMultistepScheduler,
            )

            _SCHEDULER_CLASSES[name] = DPMSolverMultistepScheduler
        elif name == "DPMSolverSinglestepScheduler":
            from diffusers.schedulers.scheduling_dpmsolver_singlestep import (
                DPMSolverSinglestepScheduler,
            )

            _SCHEDULER_CLASSES[name] = DPMSolverSinglestepScheduler
        elif name == "DPMSolverSDEScheduler":
            try:
                from diffusers.schedulers.scheduling_dpmsolver_sde import (
                    DPMSolverSDEScheduler,
                )

                _SCHEDULER_CLASSES[name] = DPMSolverSDEScheduler
            except ImportError as e:
                # DPMSolverSDE requires torchsde which might not be installed
                # Fall back to standalone scheduler
                if "torchsde" in str(e):
                    from .standalone_schedulers import STANDALONE_SCHEDULERS

                    _SCHEDULER_CLASSES[name] = STANDALONE_SCHEDULERS["FlowMatching"]
                else:
                    raise
        elif name == "EulerDiscreteScheduler":
            from diffusers.schedulers.scheduling_euler_discrete import (
                EulerDiscreteScheduler,
            )

            _SCHEDULER_CLASSES[name] = EulerDiscreteScheduler
        elif name == "EulerAncestralDiscreteScheduler":
            from diffusers.schedulers.scheduling_euler_ancestral_discrete import (
                EulerAncestralDiscreteScheduler,
            )

            _SCHEDULER_CLASSES[name] = EulerAncestralDiscreteScheduler
        elif name == "HeunDiscreteScheduler":
            from diffusers.schedulers.scheduling_heun_discrete import (
                HeunDiscreteScheduler,
            )

            _SCHEDULER_CLASSES[name] = HeunDiscreteScheduler
        elif name == "KDPM2DiscreteScheduler":
            from diffusers.schedulers.scheduling_k_dpm_2_discrete import (
                KDPM2DiscreteScheduler,
            )

            _SCHEDULER_CLASSES[name] = KDPM2DiscreteScheduler
        elif name == "KDPM2AncestralDiscreteScheduler":
            from diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete import (
                KDPM2AncestralDiscreteScheduler,
            )

            _SCHEDULER_CLASSES[name] = KDPM2AncestralDiscreteScheduler
        elif name == "LCMScheduler":
            from diffusers.schedulers.scheduling_lcm import LCMScheduler

            _SCHEDULER_CLASSES[name] = LCMScheduler
        elif name == "UniPCMultistepScheduler":
            from diffusers.schedulers.scheduling_unipc_multistep import (
                UniPCMultistepScheduler,
            )

            _SCHEDULER_CLASSES[name] = UniPCMultistepScheduler
        elif name == "DEISMultistepScheduler":
            from diffusers.schedulers.scheduling_deis_multistep import (
                DEISMultistepScheduler,
            )

            _SCHEDULER_CLASSES[name] = DEISMultistepScheduler
        else:
            raise ValueError(f"Unknown scheduler: {name}")

        return _SCHEDULER_CLASSES[name]
    except ImportError as e:
        raise ImportError(
            f"Could not import {name}. "
            f"Please update diffusers: pip install --upgrade diffusers\n"
            f"Error: {e}"
        )


# Scheduler name mapping (loaded lazily)
SCHEDULER_NAMES = [
    "DPMSolverMultistep",
    "DPMSolverSinglestep",
    "DPMSolverSDE",
    "DPMPlusPlusKarras",  # Added: High quality Karras schedule
    "EulerDiscrete",
    "EulerAncestralDiscrete",
    "HeunDiscrete",
    "DDIM",
    "DDPM",
    "LCM",
    "UniPCMultistep",
    "KDPM2Discrete",
    "KDPM2AncestralDiscrete",
    "DEISMultistep",
]

# Default configurations for each scheduler
SCHEDULER_DEFAULTS = {
    "DPMSolverMultistep": {
        "algorithm_type": "dpmsolver++",
        "solver_order": 2,
        "prediction_type": "v_prediction",
        "lower_order_final": True,
        "timestep_spacing": "trailing",
    },
    "DPMSolverSinglestep": {
        "solver_order": 2,
        "prediction_type": "v_prediction",
        # Note: timestep_spacing not supported in all diffusers versions
    },
    "DPMSolverSDE": {
        "prediction_type": "v_prediction",
        "noise_sampler_seed": 0,
    },
    "DPMPlusPlusKarras": {
        "algorithm_type": "dpmsolver++",
        "solver_order": 2,
        "prediction_type": "v_prediction",
        "use_karras_sigmas": True,
        "lower_order_final": True,
    },
    "EulerDiscrete": {
        "prediction_type": "v_prediction",
        "timestep_spacing": "trailing",
    },
    "EulerAncestralDiscrete": {
        "prediction_type": "v_prediction",
    },
    "HeunDiscrete": {
        "prediction_type": "v_prediction",
    },
    "DDIM": {
        "prediction_type": "v_prediction",
        "clip_sample": False,
    },
    "DDPM": {
        "prediction_type": "v_prediction",
        "clip_sample": False,
    },
    "LCM": {
        "prediction_type": "v_prediction",
    },
    "UniPCMultistep": {
        "solver_order": 2,
        "prediction_type": "v_prediction",
    },
    "KDPM2Discrete": {
        "prediction_type": "v_prediction",
    },
    "KDPM2AncestralDiscrete": {
        "prediction_type": "v_prediction",
    },
    "DEISMultistep": {
        "solver_order": 2,
        "prediction_type": "v_prediction",
    },
}


def get_scheduler_list() -> list:
    """Get list of available scheduler names for ComfyUI dropdowns."""
    return SCHEDULER_NAMES


def create_scheduler(
    scheduler_name: str,
    num_train_timesteps: int = 1000,
    prediction_type: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Create a scheduler instance with default configuration.

    Args:
        scheduler_name: Name of scheduler
        num_train_timesteps: Number of training timesteps (default: 1000)
        prediction_type: Override prediction type (v_prediction, epsilon, sample)
        **kwargs: Additional scheduler-specific parameters

    Returns:
        Configured scheduler instance

    Raises:
        ValueError: If scheduler_name is not recognized
    """
    if scheduler_name not in SCHEDULER_NAMES:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. " f"Available: {', '.join(SCHEDULER_NAMES)}"
        )

    # Lazy load scheduler class
    scheduler_cls = _get_scheduler_class(scheduler_name + "Scheduler")
    config = cast(Dict[str, Any], SCHEDULER_DEFAULTS.get(scheduler_name, {})).copy()

    # Override prediction type if specified
    if prediction_type is not None:
        config["prediction_type"] = prediction_type

    # Merge user kwargs
    config.update(kwargs)

    # Add num_train_timesteps
    config["num_train_timesteps"] = num_train_timesteps

    # Check if standalone scheduler (accepts num_train_timesteps, prediction_type)
    from .standalone_schedulers import STANDALONE_SCHEDULERS

    if scheduler_cls in STANDALONE_SCHEDULERS.values():
        # Standalone schedulers only accept these two parameters
        return scheduler_cls(
            num_train_timesteps=num_train_timesteps,
            prediction_type=config.get("prediction_type", "v_prediction"),
        )

    return scheduler_cls(**config)


def get_scheduler_info(scheduler_name: str) -> Dict[str, Any]:
    """
    Get information about a specific scheduler.

    Args:
        scheduler_name: Name of scheduler

    Returns:
        Dictionary with scheduler class and default config
    """
    if scheduler_name not in SCHEDULER_NAMES:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return {
        "class": _get_scheduler_class(scheduler_name + "Scheduler"),
        "defaults": SCHEDULER_DEFAULTS.get(scheduler_name, {}),
    }


# Prediction type options
PREDICTION_TYPES = ["v_prediction", "epsilon", "sample"]

# Algorithm type options (for DPM solvers)
ALGORITHM_TYPES = ["dpmsolver++", "dpmsolver", "sde-dpmsolver++"]
