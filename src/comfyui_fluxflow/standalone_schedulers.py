"""
Standalone scheduler implementations for FluxFlow.

These schedulers don't depend on diffusers, avoiding version conflicts.
Implements the most commonly used schedulers for flow matching.
"""

from typing import Optional

import torch


class FlowMatchingScheduler:
    """
    Simple flow matching scheduler for FluxFlow.

    Uses linear interpolation between noise and data: x_t = (1-t)*x_0 + t*x_1
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v_prediction",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Set the discrete timesteps for inference."""
        self.timesteps = torch.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps, device=device
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the sample at the previous timestep.

        Args:
            model_output: Direct output from model
            timestep: Current timestep
            sample: Current sample

        Returns:
            Previous sample
        """
        if self.timesteps is None:
            raise ValueError("Must call set_timesteps before step")

        # Find current timestep index
        t_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(t_idx) == 0:
            raise ValueError(f"Timestep {timestep} not in schedule")
        t_idx = t_idx[0]

        # Get next timestep
        if t_idx == len(self.timesteps) - 1:
            return sample  # Last step

        # t = timestep / self.num_train_timesteps  # noqa: F841
        dt = 1.0 / len(self.timesteps)

        # Euler step: x_{t-dt} = x_t - dt * v_t
        prev_sample = sample - dt * self.num_train_timesteps * model_output

        return prev_sample


class EulerScheduler:
    """Euler method scheduler for flow matching."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v_prediction",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Set timesteps from T to 0."""
        self.timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps, device=device
        )

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Single Euler step."""
        if self.timesteps is None:
            raise ValueError("Must call set_timesteps before step")

        # Find current index
        t_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(t_idx) == 0:
            # Approximate match
            t_idx = torch.argmin(torch.abs(self.timesteps - timestep))
        else:
            t_idx = t_idx[0]

        if t_idx == len(self.timesteps) - 1:
            return sample

        # Calculate step size
        dt = (self.timesteps[t_idx + 1] - self.timesteps[t_idx]) / self.num_train_timesteps

        # Euler step
        prev_sample = sample + dt * self.num_train_timesteps * model_output

        return prev_sample


class DDIMScheduler:
    """DDIM-style scheduler for deterministic sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v_prediction",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timesteps: Optional[torch.Tensor] = None

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Set evenly spaced timesteps."""
        # Evenly spaced timesteps
        step = self.num_train_timesteps // num_inference_steps
        self.timesteps = torch.arange(0, self.num_train_timesteps, step, device=device).flip(0)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM deterministic step."""
        if self.timesteps is None:
            raise ValueError("Must call set_timesteps before step")

        t_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        if len(t_idx) == 0:
            t_idx = torch.argmin(torch.abs(self.timesteps - timestep))
        else:
            t_idx = t_idx[0]

        if t_idx == len(self.timesteps) - 1:
            return sample

        # Simple linear interpolation step
        alpha_t = timestep / self.num_train_timesteps
        alpha_prev = self.timesteps[t_idx + 1] / self.num_train_timesteps

        # v-prediction: v = alpha * epsilon - sigma * x0
        # Simplified DDIM step
        prev_sample = sample - (alpha_t - alpha_prev) * model_output

        return prev_sample


class DPMPlusPlusKarrasScheduler:
    """
    DPM++ 2M Karras scheduler.

    High-quality scheduler with Karras noise schedule.
    Popular for its excellent quality-to-speed ratio.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        prediction_type: str = "v_prediction",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.timesteps: Optional[torch.Tensor] = None
        self.sigmas: Optional[torch.Tensor] = None
        self.prev_sample: Optional[torch.Tensor] = None

    def _karras_sigmas(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Generate Karras noise schedule."""
        # Karras schedule parameters
        sigma_min = 0.1
        sigma_max = 10.0
        rho = 7.0

        # Generate schedule
        ramp = torch.linspace(0, 1, num_inference_steps, device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho

        return sigmas

    def set_timesteps(self, num_inference_steps: int, device: Optional[torch.device] = None):
        """Set timesteps with Karras noise schedule."""
        self.sigmas = self._karras_sigmas(num_inference_steps + 1, device=device)
        self.timesteps = (
            self.sigmas[:-1] * self.num_train_timesteps / 10.0
        )  # Scale to timestep range
        self.prev_sample = None

    def step(
        self,
        model_output: torch.Tensor,
        timestep: float,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """DPM++ 2M step with second-order multistep."""
        if self.timesteps is None or self.sigmas is None:
            raise ValueError("Must call set_timesteps before step")

        # Find timestep index
        t_idx = torch.argmin(torch.abs(self.timesteps - timestep))

        if t_idx >= len(self.sigmas) - 2:
            return sample

        sigma = self.sigmas[t_idx]
        sigma_next = self.sigmas[t_idx + 1]

        # DPM++ 2M (second order)
        if self.prev_sample is None:
            # First step: Euler
            dt = sigma_next - sigma
            prev_sample = sample + model_output * dt
        else:
            # Second order step
            dt = sigma_next - sigma
            # Linear multistep coefficient
            h = dt / sigma
            prev_sample = sample + model_output * dt * (1 + h / 2)

        self.prev_sample = prev_sample
        return prev_sample


# Scheduler registry
STANDALONE_SCHEDULERS = {
    "FlowMatching": FlowMatchingScheduler,
    "Euler": EulerScheduler,
    "DDIM": DDIMScheduler,
    "DPMPlusPlus": DPMPlusPlusKarrasScheduler,
}


def create_standalone_scheduler(
    scheduler_name: str,
    num_train_timesteps: int = 1000,
    prediction_type: str = "v_prediction",
):
    """
    Create a standalone scheduler instance.

    Args:
        scheduler_name: Name of scheduler (FlowMatching, Euler, DDIM)
        num_train_timesteps: Number of training timesteps
        prediction_type: Prediction type (v_prediction, epsilon, sample)

    Returns:
        Scheduler instance
    """
    if scheduler_name not in STANDALONE_SCHEDULERS:
        raise ValueError(
            f"Unknown scheduler: {scheduler_name}. "
            f"Available: {', '.join(STANDALONE_SCHEDULERS.keys())}"
        )

    scheduler_cls = STANDALONE_SCHEDULERS[scheduler_name]
    return scheduler_cls(
        num_train_timesteps=num_train_timesteps,
        prediction_type=prediction_type,
    )
