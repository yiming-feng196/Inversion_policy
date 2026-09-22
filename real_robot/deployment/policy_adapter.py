"""Hardware-neutral adapter for closed-loop Prior Flow inference."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from prior_policy.runtime.prior_flow_inference import predict_action_with_prior


@dataclass(frozen=True)
class ActionSafetyLimits:
    """Position and per-command limits in the same units as the training action."""

    lower: np.ndarray
    upper: np.ndarray
    max_delta: np.ndarray

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float32).reshape(-1)
        upper = np.asarray(self.upper, dtype=np.float32).reshape(-1)
        max_delta = np.asarray(self.max_delta, dtype=np.float32).reshape(-1)
        if lower.shape != upper.shape or lower.shape != max_delta.shape:
            raise ValueError("lower, upper, and max_delta must have the same shape")
        if (
            not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or not np.isfinite(max_delta).all()
        ):
            raise ValueError("Position and delta limits must be finite")
        if np.any(lower >= upper) or np.any(max_delta <= 0):
            raise ValueError("Invalid position or delta limits")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "max_delta", max_delta)

    @property
    def action_dim(self) -> int:
        return int(self.lower.size)

    def apply(self, action_chunk: np.ndarray, current_action_reference: np.ndarray) -> np.ndarray:
        actions = np.asarray(action_chunk, dtype=np.float32).copy()
        reference = np.asarray(current_action_reference, dtype=np.float32).reshape(-1)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            raise ValueError(f"Expected action chunk [T, {self.action_dim}], got {actions.shape}")
        if reference.shape != (self.action_dim,):
            raise ValueError(f"Expected action reference [{self.action_dim}], got {reference.shape}")
        if not np.isfinite(actions).all() or not np.isfinite(reference).all():
            raise ValueError("Policy output or action reference contains NaN or Inf")
        previous = np.clip(reference, self.lower, self.upper)
        for index in range(len(actions)):
            target = np.clip(actions[index], self.lower, self.upper)
            target = np.clip(target, previous - self.max_delta, previous + self.max_delta)
            actions[index] = np.clip(target, self.lower, self.upper)
            previous = actions[index]
        return actions


class RealRobotPriorPolicy:
    """Maintain observation history and return bounded action targets.

    The caller owns camera/state synchronization and hardware actuation. The
    adapter never sends commands to a robot.
    """

    def __init__(
        self,
        action_policy,
        prior_flow,
        safety_limits: ActionSafetyLimits,
        image_size: int = 256,
        prior_steps: int = 8,
        action_steps: int = 10,
        execute_steps: int = 1,
        device: str | torch.device = "cuda:0",
    ) -> None:
        self.safety_limits = safety_limits
        self.image_size = int(image_size)
        self.prior_steps = int(prior_steps)
        self.action_steps = int(action_steps)
        self.execute_steps = int(execute_steps)
        self.device = torch.device(device)
        self.action_policy = action_policy.to(self.device).eval().requires_grad_(False)
        self.prior_flow = prior_flow.to(self.device).eval().requires_grad_(False)
        self.history_length = int(action_policy.n_obs_steps)
        if self.image_size <= 0 or self.execute_steps <= 0:
            raise ValueError("image_size and execute_steps must be positive")
        self._images: deque[torch.Tensor] = deque(maxlen=self.history_length)
        self._states: deque[torch.Tensor] = deque(maxlen=self.history_length)

    def reset(self) -> None:
        self._images.clear()
        self._states.clear()

    def _prepare_image(self, rgb_image: np.ndarray) -> torch.Tensor:
        image = np.asarray(rgb_image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected RGB HWC image, got {image.shape}")
        if not np.isfinite(image).all():
            raise ValueError("RGB image contains NaN or Inf")
        tensor = torch.as_tensor(np.ascontiguousarray(image), device=self.device)
        tensor = tensor.permute(2, 0, 1).float()[None]
        if image.dtype == np.uint8 or float(tensor.max()) > 1.0 + 1e-6:
            tensor = tensor / 255.0
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = F.interpolate(
                tensor,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        return tensor[0]

    def observe(self, rgb_image: np.ndarray, state: np.ndarray) -> None:
        image_tensor = self._prepare_image(rgb_image)
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if not np.isfinite(state_array).all():
            raise ValueError("Robot state contains NaN or Inf")
        state_tensor = torch.as_tensor(state_array, device=self.device)
        if not self._images:
            for _ in range(self.history_length - 1):
                self._images.append(image_tensor.clone())
                self._states.append(state_tensor.clone())
        self._images.append(image_tensor)
        self._states.append(state_tensor)

    @torch.inference_mode()
    def predict(self, current_action_reference: np.ndarray) -> np.ndarray:
        if len(self._images) != self.history_length:
            raise RuntimeError("Observation history is not initialized; call observe first")
        observations = {
            "head_cam": torch.stack(tuple(self._images), dim=0)[None],
            "agent_pos": torch.stack(tuple(self._states), dim=0)[None],
        }
        result = predict_action_with_prior(
            self.action_policy,
            self.prior_flow,
            observations,
            prior_steps=self.prior_steps,
            action_steps=self.action_steps,
        )
        action_chunk = result["action"][0].detach().cpu().numpy()
        bounded = self.safety_limits.apply(action_chunk, current_action_reference)
        return bounded[: self.execute_steps]
